<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX models: vision-language models

- `NFKMLXSmolVLM` / `NFKMLXSigLIPNet` / `NFKMLXSmolVLMConnector` / `NFKMLXSmolVLMNet` — the package's
  **first vision-language model**, SmolVLM2-500M: an image and a question in, an answer out, which is
  the dominant 2026 on-device use (captioning, VQA, doc/screen understanding). Three parts: a **SigLIP
  vision encoder** (`NFKMLXSigLIPNet`: patch-embedding convolution + learned position embedding + 12
  bidirectional layers + post-norm; separate q/k/v/out projections with bias, gelu-tanh, LayerNorm eps
  1e-6), a **pixel-shuffle connector** (`NFKMLXSmolVLMConnector`: the Idefics3 pixel shuffle folds a
  4×4 patch neighborhood into one token at 16× the channels — 1024 patches per tile → 64 — then one
  bias-free `proj` to the decoder width), and a **Llama decoder** the dense `NFKMLXLanguageNet` already
  runs (`text_config` model_type llama, hidden 960, 32 layers, loaded from the checkpoint's
  `model.text_model.` subtree remapped onto `model.`). The **fusion** embeds the text, then splices the
  flattened projected vision tokens into the decoder's input embeddings at the `image_token_id` (49190)
  positions with a `where` over a gathered feature index (no per-row scatter), and the causal decoder
  reads the whole fused sequence. Reference parity against transformers' own
  `SmolVLMForConditionalGeneration` (`run_reference.py smolvlm`, the `llm` oracle — needs Pillow,
  torchvision, num2words), staged by the isolation harness: SigLIP embeddings 0.9999999999, layer 0
  0.9999999999, the full encoder 0.99999999997, the connector 0.9999999999, and the fused decoder
  logits predicting the reference's token at every one of the 1140 positions (last-position cosine
  0.9999999999) with the greedy continuation **token for token**. **Two bugs the isolation harness
  located, neither guessable:** SigLIP's position ids are not the row-major `0 … 1023` — the reference
  buckets each patch's fractional coordinate against `1/side … (side-1)/side` with a `1 - 1e-6` factor,
  so a full 32-patch row maps to `[0, 0, 1, …, 30]` (`positionIds` reproduces it, held as a Swift array
  so the stored constant does not enter `parameters()` — a stored `MLXArray` would, and the loader would
  then report it as an uncovered weight); and SmolVLM's `lm_head` is not tied to the embedding
  (byte-diff 0.85 despite the tied geometry), so the decoder loads its own `lm_head.weight` from the top
  level (`tiesWordEmbeddings: false`) rather than reusing the embedding — loading it tied scored logit
  cosine 0.83 and drifted after the sixth continuation token. **The consumer path:**
  `NFKMLXSmolVLMImageProcessor` tiles a `CGImage` the way SmolVLM does (longest edge scaled to 2048,
  split into `⌈h/512⌉ × ⌈w/512⌉` 512×512 sub-tiles plus a global 512×512 thumbnail appended last, each
  normalized to `-1 … 1`), and `NFKMLXSmolVLM.prompt(rows:cols:question:)` builds the expanded
  `User:` + per-tile `<fake_token_around_image><row_r_col_c>` + 64 `<image>` + `<global-img>` tile +
  question + `<end_of_utterance>\nAssistant:` string, which the byte-level BPE tokenizer (every added
  token registered as a special so the string segments on them) turns into ids **token-exactly** against
  the processor. The resize is CoreGraphics, not the reference's PIL LANCZOS, so a consumer caption is
  not token-identical to the reference — a documented approximation; the network is at parity on the
  reference's own pixel values, and end to end on the real Apollo-astronaut validation photo the model
  answers "a man is standing in front of a backdrop that resembles the moon. He is dressed in a white
  …". ObjC reaches it through `smolVLMWithDirectoryURL:error:` and `answerForImage:question:`.
  The port reads the release rather than hard-coding 500M: `NFKMLXSmolVLM.release(directoryURL:)` builds
  the SigLIP geometry, the pixel-shuffle factor, the tile size and the tokens per tile from the
  directory's own `config.json`, and the weights load sharded through `NFKMLXReleaseWeights`, so the
  other two released sizes run the same code. SmolVLM2-256M and SmolVLM2-2.2B are both at parity
  (256M: vision 0.9999999999398742, connector 0.9999999999894968, logits 0.9999999999585247, argmax
  1140/1140; 2.2B: vision 0.9999999997401828, connector 0.9999999998995716, logits 0.9999999959030899,
  argmax 1428/1429, where the one disagreeing position is a 1.4e-05 tie in the REFERENCE's own logits
  and the test asserts that gap rather than the token). The 2.2B config states no
  `num_attention_heads`, so the language reader derives the head count from `head_dim` instead of
  falling back to 16; transformers derives 32, and the wrong count would have loaded silently.
- `NFKMLXQwen3VLVisionNet` / `NFKMLXQwen3VL` — the vision tower of a **second VLM**, Qwen3-VL-2B, and a
  second vision architecture beside SmolVLM's SigLIP. Qwen3-VL's encoder is a **2D-rotary ViT** whose
  patches are laid out in **2×2 merge blocks** (not row-major): a patch embedding (the reference's
  full-kernel `Conv3d` written as one `Linear` over the flattened `3·2·16·16` patch), a bilinearly
  interpolated position embedding (the learned 48×48 grid resampled to the image's grid, then
  reordered to merge-block order to line up with the patches), 24 blocks with 2D rotary and gelu-tanh
  MLP, a **merger** that folds each 2×2 block to `out_hidden` (2048), and a three-layer **deepstack**
  (feature maps from vision layers 5/11/17, each through its own merger with a post-shuffle norm). The
  2D rope pairs channel `i` with `i+32` over a `[row·freqs, col·freqs]` table; the merger groups four
  Consecutive patches, which is why the merge-block patch order is load-bearing. Reference parity on
  the first numeric run against transformers' own Qwen3-VL vision model (`run_reference.py qwen3vl`,
  the `llm` oracle — needs Pillow/torchvision): patch embedding 0.9999999999, position embedding
  0.9999999999, merged output 0.9999999997, and every one of the three deepstack features
  0.9999999999. The decoder is the Qwen3 dense stack `NFKMLXLanguageNet` already runs, loaded from the
  checkpoint's `model.language_model.` subtree. Qwen3-VL is much more than "reuses the Qwen3 decoder":
  the decoder adds interleaved M-RoPE (3D positions, `mrope_section [24,20,20]`) and deepstack injection
  at its first three layers, and `get_rope_index` computes the 3D positions from the token layout.
  That decoder integration is shipped, in the shared `NFKMLXLanguageNet` (the dense Qwen3, the
  embedders, SmolVLM, the music AR stage, and the Z-Image text step all reuse it, kept byte-identical
  through nil defaults): an opt-in M-RoPE (precomputed interleaved 3-D cos/sin in `NFKLMAttention`,
  rotate-half; the 64 frequency pairs interleave the T/H/W axes by `c % 3` for `c < 60`, then T, which is
  exactly `mrope_section [24, 20, 20]`) and deepstack injection (adding `features[i]` at the image-token
  positions after layers 0/1/2), both carried by an `NFKLMMultimodal` struct through
  `hiddenStates(fromEmbeddings:multimodal:)`. `NFKMLXQwen3VL` gained `decoder(directoryURL:)` (the
  `model.language_model.` subtree; Qwen3-1.7B geometry at `ropeTheta` 5e6), `ropePositionIds`
  (`get_rope_index`, single image), `mropeCosSin`, and `logits(...)`. Reference parity on the released
  4.25 GB Qwen3-VL-2B (`testQwen3VLDecoderMatchesTheReferenceOnReleasedWeights`, the recorded parity
  vision features fed in): logit cosine 0.99999999998, argmax 80/80, the first continuation token
  matching. The shared-decoder change was re-verified against the dense Qwen3 (0.99999999999), SmolVLM
  (argmax 1140/1140), and Qwen3-Embedding (0.99999999999) records. `NFKMLXQwen3VLImageProcessor` is the
  `smart_resize` + patchify input adapter. Its resize is `Qwen2VLImageProcessorFast`'s, torchvision's
  antialiased bicubic on the 8-bit image (`int16` weights), exact to float rounding (see Sa2VA's
  preprocessing below).
  The larger sizes are read from their own releases: `NFKMLXQwen3VLVisionConfiguration.configuration(fromHuggingFace:)`
  (model_type `qwen3_vl` or `qwen3_vl_moe`; the position grid's side is the square root of
  `num_position_embeddings`), `visionNet(directoryURL:)` (sharded through `NFKMLXReleaseWeights.arrays`),
  `decoderConfiguration(directoryURL:)` (the `text_config` through `NFKMLXLanguage.configuration(fromJSON:)`,
  tied unless the release ships a head), and `decoder(directoryURL:)`, which splits the 30B-A3B's fused
  `gate_up_proj [E, hidden, 2·inter]` into the module's `gate_proj` / `up_proj [E, inter, hidden]` and
  transposes `down_proj`. The 4B keeps the 2B's 24-block tower; the 8B, 32B, and 30B-A3B run the
  27-block one. Held to their released headers by shape, tower and decoder together: 713 / 750 /
  1058 / 930 tensors, 0 missing, 0 mismatched, 0 unaccounted.
- `NFKMLXPixtralVisionNet` / `NFKMLXPixtralConnector` / `NFKMLXPixtral` — the vision tower of a third
  VLM, Pixtral 12B (`mistral-experimental/pixtral-12b`, Mistral, Apache-2.0), and a third vision
  architecture beside SmolVLM's SigLIP and Qwen3-VL's merge-block ViT. Pixtral's encoder is a
  from-scratch **2D-rotary ViT** with native variable resolution: a patch convolution (kernel = stride
  = patch, so one linear projection over each flattened `[channel, patch, patch]` patch) embeds the
  image at its own aspect ratio, an RMSNorm normalizes, and 24 pre-normalized blocks read the patches
  with a **2D rotary** over the (height, width) grid. The rotary indexes a precomputed table by
  `row · maxSide + column`: a patch takes the first quarter of the head dimension from the row's
  frequencies and the second quarter from the column's, doubled, so `NFKMLXPixtralVisionNet` builds the
  table per patch in row-major order rather than materializing the reference's `image_size²` grid. The
  attention is full over one image's patches (the reference's block-diagonal mask degenerates to full
  for a single image), q/k/v/o carry no bias, and the feed-forward is a SiLU-gated MLP. A two-layer
  GELU connector (`NFKMLXPixtralConnector`, the Llava `multi_modal_projector`, both layers biased)
  projects the last hidden state to the decoder width, and the projected patch features splice into the
  decoder's input embeddings at the `[IMG]` placeholder id (`image_token_index` 10). The decoder is a
  **Mistral-Nemo dense** stack (40 layers, hidden 5120, 32 heads at head width 128, 8 key-value heads,
  rotary base 1e9) the dense `NFKMLXLanguageNet` already runs, loaded from the checkpoint's
  `language_model.` subtree. Reference parity against transformers' own `LlavaForConditionalGeneration`
  (`run_reference.py pixtral`, the `llm` oracle), the vision seams measured in float32: patch embedding
  1.0, ln_pre 1.0, first block 0.9999999, vision output 0.9999995, connector 0.99999994. The whole
  fused pipeline — vision tower, connector, the scatter into the `[IMG]` positions, and the Mistral
  decoder over the fused sequence — is measured against a tiny float32 oracle through the ordinary
  release builders (`run_reference.py pixtral_tiny`): vision output, projected features, and the fused
  logits all at parity with the reference's argmax at every position. The released 12B decoder needs
  ~24 GB resident for the fused pass, which does not fit alongside other work on a 32 GB machine, so its
  released-weight fused-logit check is a separate test behind the `IK_PIXTRAL_FUSED` opt-in. **The
  head count is not derivable from the config:** Pixtral's `text_config` states `head_dim` 128 but omits
  `num_attention_heads`, and 5120 / 128 = 40 is wrong — the release has 32 heads at a 4096-wide query
  projection, the transformers `MistralConfig` default it inherits, which `decoderConfiguration` applies
  when the count is absent. `NFKMLXPixtralImageProcessor` resizes a `CGImage` within the longest edge
  and to a multiple of the patch, then normalizes with the CLIP mean and standard deviation; the resize
  is CoreGraphics rather than the reference's bicubic, the documented approximation the SmolVLM and
  Qwen3-VL processors also carry. ObjC reaches the model through `modelWithDirectoryURL:error:` and
  `answerForImage:question:maxTokens:`. Customization is **offline-only**: the 12B decoder is a language
  model above 4B, past what a device holds at float precision for training, so the reference route is a
  LoRA fine-tune in Python, merged, then loaded through the ordinary factory.

## Florence-2

- `NFKMLXFlorence2` (`microsoft/Florence-2-large` and `-base`, Microsoft, MIT), a unified vision model: one image
  plus a task token in, text out — a caption, detected objects, or grounded regions, with boxes carried
  as `<loc_0..999>` location tokens. Three parts, ported at reference parity (all measured on the
  released weights, first numeric run): the **DaViT vision tower** (`NFKMLXFlorence2VisionNet`), whose
  every block pairs a windowed **spatial** attention (`NFKMLXFlorence2WindowAttention`, scale
  head_dim^-0.5) with a grouped **channel** attention (`NFKMLXFlorence2ChannelAttention`, attention
  across the channel axis within groups, scale num_tokens^-0.5) — a dual-attention form new to the
  toolkit; the **projector** (`NFKMLXFlorence2Projector`, a learned 2-D position embedding + a cosine
  temporal row + a mean-pooled token prepended + a bare `[C, projectionDim]` `image_projection`
  parameter and LayerNorm); and the **BART** encoder-decoder (the shared `NFKMLXSeq2SeqTransformer`,
  `NFKMLXFlorence2Net.bartLarge`). The fusion CONCATENATES `[image tokens, prompt embeddings]` image
  first (the reference `_merge_input_ids_with_image_features`; it is NOT the transformers-native
  `image_token_id` scatter), the encoder runs over the join, the decoder cross-attends, and the head
  adds `final_logits_bias`. Cosines: conv0 1.0000001, block0 0.9999998, vision 0.99999976, projector
  1.0, encoder 1.0000001, first-step logits 0.9999998; greedy `<OD>` matches the reference token for
  token.
- **The oracle is the repo's own code via `trust_remote_code`** (`run_reference.py florence2`, needs
  `timm` in the llm env), because the transformers-NATIVE `Florence2ForConditionalGeneration` does not
  map the released original-davit checkpoint — it loads all-random (the config uses `dim_embed` not
  `embed_dim`, the blocks wrap conv/attention in `.fn`/PreNorm containers, and the keys lack the `model.`
  prefix). The oracle canonicalizes every seam to `[tokens, C]`; the ConvEmbed and blocks return
  `(tokens, size)` tuples, so the hooks unwrap `o[0]`. Record with `--image inputs/face960.jpg` (the
  same photo the seam test loads) so the image-processor comparison is the same image on both sides; a
  synthetic plate makes the reference caption degenerate.
- **Weight load** (`NFKMLXFlorence2Weights` / `NFKMLXFlorence2Net.loadWeights`): the raw davit keys map
  with `.fn`/PreNorm flattening (`spatial_block.window_attn.norm` → `spatial.norm1`,
  `window_attn.fn.qkv` → `spatial.attn.qkv`, `conv1.fn.dw` → `conv1.conv`, `ffn.fn.net.fc1` →
  `ffn.fc1`); the projector's `image_pos_embed.*` → `row/column_embeddings`, `image_projection` is a
  bare `[2048,1024]` matmul parameter; the BART subtree drops its `language_model.model.` prefix and
  routes through `NFKMLXSeq2SeqNet.moduleKey` (which drops the tied embed/head copies). Nested
  `blocks: [[NFKFlorence2VisionBlock]]` gives the `blocks.N.M` keys (the Gemma3nVision pattern).
- **Consumer surface** (`NFKMLXFlorence2Backend`): `NFKInputImage` + a task token under `NFKInputPrompt`
  → `NFKOutputText`, plus `NFKOutputDetections` for the localization tasks (`<OD>`,
  `<DENSE_REGION_CAPTION>`, `<REGION_PROPOSAL>`, grounding). The processor expands the task token to its
  natural-language prompt (both dicts from `processing_florence2.py`), tokenizes with the base BART
  byte-level BPE (the core `NFKTokenizer`) EXTENDED with the 1024 added tokens as specials (four OD/OCR
  markers, `<loc_0..999>`, twenty structural markers, ids continuing from 50265), wraps the prompt
  `<s> … </s>` (`encodePrompt` — the core tokenizer returns bare ids), resizes the image through the
  validated `NFKMLXImageBridge.tensor` + `NFKMLXResample.resizeBilinear` and ImageNet-normalizes, then
  generates through a `NFKMLXSeq2SeqDecodable` wrapper whose `encodeSource` returns the fused memory.
  `<loc_>` quads dequantize to normalized boxes as `(bin + 0.5) / 1000`.
- **Traps** (all found by seam localization): the image-processor comparison failed at cosine −0.40
  because the test loaded `IK_VAL_IMAGE` (a 384×384 photo) while the oracle used face960 — two different
  images, not a processor bug; use the same photo. **Captioning degenerates to a run of `<s>` under a
  raw argmax** (the `florence2` record's greedy loop, which applies none of the release's processors).
  The release's generation settings live in `text_config`, where transformers reads them for the
  language model's `generate`: three beams, early stopping, `no_repeat_ngram_size` 3,
  `forced_bos_token_id` 0, and `forced_eos_token_id` 2. `NFKMLXFlorence2.generationDefaults` reads
  them into the backend's `defaultDecoding`; a request overrides the length (`NFKParameterMaxTokens`)
  and the beams (`NFKMLXTranslationParameterKey.beamCount`). `run_reference.py florence2_generate`
  records the release's own `generate` on five tasks, a 16-token cut, and constrained greedy; both
  releases match token for token (`IK_PARITY_FLORENCE2[_BASE]_GENERATE`). The remote code's cached
  decode fails under transformers 4.57, so the oracle generates with `use_cache=False`.
- **A forced token scores 0** (found by this measurement). transformers' `ForcedBOS`/`ForcedEOS`
  processors replace the scores, so the forced token contributes nothing to a hypothesis's sum. The
  shared decoder had masked the other tokens and kept the forced token's own log-probability, which
  after length normalization penalized short hypotheses: Florence-2-large `<OCR>` returned
  `<s><s><s>-` where the reference returns `<s>-` (reference beam scores −0.483 against −0.536). The
  same path serves M2M-100's forced language token, whose parity holds.
- **Customization: LoRA.** Microsoft publishes no fine-tuning script, so the objective is the release's
  own `labels=` loss (`Florence2LanguageForConditionalGeneration`: the answer shifted right behind the
  decoder start token, `CrossEntropyLoss` over every position), and the level is the translators': LoRA
  on the BART decoder's query and value projections, both attentions (`rank: nil` trains the language
  model and projector with the DaViT tower frozen). The answer is `<s> … </s>` as the processor's
  tokenizer gives it (`NFKMLXFlorence2Processor.encodePrompt`), which is what the releases generate after
  their start token. `NFKMLXFlorence2Objective`; `NFKMLXFlorence2.network(directoryURL:)`; `fineTune`
  (AdamW, bias-corrected, 1e-4, no decay, clip 1: the translators' defaults, the rate this package's
  choice); `NFKMLXLoRA.merge(into:)` then `save(_:toDirectoryURL:release:)`, which writes the weights in
  the module's layout beside the release's `config.json` and `tokenizer.json`, and which
  `backendWithDirectoryURL:` loads. `NFKMLXTrainer.train(_:optimizer:steps:arrays:loss:)` is the
  trainer's form for three arrays a step. Measured on Florence-2-base-ft (`run_reference.py
  florence2_loss`, `IK_PARITY_FLORENCE2_BASE_FT_LOSS`) on the release's own caption, *"A man in a space
  suit holding a helmet."*: the loss on its logits 0.4842829 and the port's forward 0.4842833 against the
  criterion in float64, 0.48428318 (the release's float32 value, 0.48423862, is the rounding of a
  log-sum-exp over 51k logits). `NFKMLXFlorence2TrainingTests` holds the parity and a LoRA run that
  trains only the decoder adapters, lowers the loss, and reloads merged through the factory with the
  same score.
- **Both sizes build from `config.json`** (`NFKMLXFlorence2Net.configuration(fromConfigURL:)`): the DaViT
  widths, heads, groups, depths, patches, and window from `vision_config`, the projection from its
  `projection_dim`, and the BART sizes from `text_config`. Until 2026-09-23 the directory factory built
  the large geometry whatever the directory held, so base could not load, and the `.base` vision preset
  kept the large model's 1024-wide projection where base's is 768. Florence-2-base (`FLORENCE2_BASE`,
  pinned `5ca5edf5`, `run_reference.py florence2` with `FLORENCE2_REPO=microsoft/Florence-2-base`):
  projector, encoder, and first-step logits 1.0000002, greedy `<OD>` token for token.
- **Every release** (the four on the Hugging Face API, 2026-09-23) is at reference parity:
  `Florence-2-base`, `-large`, and the fine-tuned `-base-ft` and `-large-ft`. The fine-tuned releases
  share their bases' geometry; `-large-ft` declares 1,024 positions where `-large` declares 4,096, which
  the reader takes from `config.json`. `-large-ft` also ships a top-level `generation_config.json` with
  early stopping off, but the language model's `generate` reads `text_config` (early stopping on), and
  the port's token-exact match on all five tasks confirms it. The `-ft` directories carry the repos'
  own remote code, so the oracle loads them in place (`FLORENCE2_REPO=<dir>`). Seams: base-ft projector
  0.99999946, encoder 0.9999991, logits 1.0000001; large-ft projector 1.0, encoder 0.99999994, logits
  1.0000004. Generation is token-exact on all five tasks, the 16-token cut, and constrained greedy for
  all four releases.

## TrOCR

- `NFKMLXTrOCR` (`microsoft/trocr-base-handwritten`, Microsoft, MIT), a handwriting-line reader: one
  image in, its transcription out. A `VisionEncoderDecoder` of two parts, ported at reference parity
  (measured on the released weights, first numeric run): the **ViT-base image encoder**
  (`NFKMLXTrOCRVisionNet`, a plain `google/vit`: a stride-16 patch convolution over the 384-square
  image, a prepended class token, a learned 577-position table, twelve pre-normalized blocks with NO
  query/key/value bias, and a final layer norm) and a **BART-style `trocr` decoder** (the shared
  `NFKMLXSeq2SeqNet` in its decoder-only shape: `encoderLayers` 0 builds no encoder, and `decode`
  reads the ViT output as its memory). The decoder is 1024-wide over the 768-wide image features, so
  its cross-attention projects the memory from 768 (`crossAttentionWidth`,
  `cross_attention_hidden_size` in the decoder config); the output projection is tied to the token
  embedding (the checkpoint has no `output_projection`). Cosines: embeddings 1.0, first block
  0.99999976, encoder output 0.999999, first-step logits 0.9999999; greedy transcription matches the
  reference token for token, and the backend reads the rendered line end to end.
- **The oracle** is the transformers-NATIVE `VisionEncoderDecoderModel` (`run_reference.py trocr`, in
  the llm env; no `trust_remote_code`, unlike Florence). It records the pixel values (NHWC), the ViT
  embeddings output, the first encoder block, the encoder last hidden state (the decoder memory), the
  first-step logits, and the greedy ids. `decoder_start_token_id` and `eos_token_id` are None at the
  top-level config; read them from `model.config.decoder`.
- **Weight load** (`NFKMLXTrOCRNet.loadWeights(fromDirectory:)`): one checkpoint pass partitioned by
  prefix — `encoder.*` (the ViT) routes through `NFKMLXTrOCRVisionWeights.visionKey` (strip the
  `encoder.` prefix, drop the unused `pooler`, transpose the 4-D patch convolution to MLX layout);
  everything else (`decoder.model.decoder.*`) routes through `NFKMLXSeq2SeqNet.moduleKey(for:
  configuration:hasShared:)`, which strips the `decoder.model.` prefix, maps the first
  `embed_tokens.weight` to `shared` (the checkpoint carries no `shared`), and drops the tied output
  projection. The ViT loads into the `vision` child, the decoder into the `language` child.
- **Consumer surface** (`NFKMLXTrOCRBackend`): `NFKInputImage` → `NFKOutputText`. The processor
  resizes with PIL's own filter (`NFKMLXPILResample`, the one `preprocessor_config.json` names:
  bilinear, or bicubic at a = −0.5) and scales to `[-1, 1]` (mean 0.5, std 0.5, the reference
  `ViTImageProcessor`); its pixels equal the reference's exactly (max abs difference 0.0 on every
  release). The tokenizer is the release's: RoBERTa byte-level BPE from `vocab.json` + `merges.txt`
  (the five specials declared so they decode to their literals), or, for the small releases, XLM-R's
  SentencePiece model with fairseq's id offset. Generation is greedy through a
  `NFKMLXSeq2SeqDecodable` wrapper whose `encodeSource` returns the ViT memory. Start and end tokens
  are both 2 (`</s>`).
- **Reuse**: the decoder is entirely the translation session's shared seq2seq; the new work is the ViT
  encoder and the key partitioning. The shared net grew `crossAttentionWidth`, `untiedOutputProjection`,
  and the decoder-only shape for TrOCR (Florence's BART path is untouched).
- **Customization: full** (the reference's own recipe). The releases were fine-tuned in fairseq
  (`microsoft/unilm/trocr`: `fairseq-train --task text_recognition --finetune-from-model`), every
  weight trained with the default `cross_entropy` criterion. `NFKMLXTrOCRObjective` is that criterion:
  the token sum the trainer divides by the token count, the mean over the target. The target is the
  text's pieces with no start token, then the end token (fairseq's `encode_line`), which is exactly what
  the releases generate after their decoder start token; `NFKMLXTrOCRProcessor.targetIds(for:tokenizer:endToken:)`
  builds it. `NFKMLXTrOCRTrainable` is `.everything` (the reference) or `.decoder`. `fineTune` runs fairseq's
  `adam` (decoupled weight decay 1e-4, bias-corrected) at 2e-5 under its `inverse_sqrt` schedule, a
  500-update warm-up from 1e-8 then `√(500 / k)` (`NFKMLXLearningRateSchedule.fairseqInverseSquareRoot`),
  with no clipping: the IAM recipe; SROIE's is 5e-5 with 800 warm-up updates. `NFKMLXTrOCR.network(directoryURL:)`
  builds the network, and `save(_:toDirectoryURL:release:)` writes `model.safetensors` in the module's
  layout beside the release's configuration and tokenizer files, which `backendWithDirectoryURL:` loads.
  Not reproduced: fairseq's Adam adds its epsilon before the second-moment bias correction (the Adam
  paper's form) where torch and MLX add it after, which differs only where `√v` is near 1e-8; and the
  fp16 flag. Measured (`run_reference.py trocr_loss`, `IK_PARITY_TROCR_<RELEASE>_LOSS`, on each
  release's own greedy transcription): the target ids equal the release tokenizer's; the loss on
  identical logits is within 3e-7 of the criterion computed in float64 (small-handwritten 0.27900872
  against 0.27900857, base-printed 0.17066547 against 0.17066573; torch's float32 value is 2e-5 off,
  the rounding of a log-sum-exp over 64k logits); the port's own forward lands within 3e-6; and the
  schedule equals fairseq's `InverseSquareRootSchedule`, run from the file the manifest pins, at every
  update count from 0 to 1199. **transformers 4.57's `labels=` loss is wrong for this model**: a
  `VisionEncoderDecoderModel` scores labels with `ForCausalLMLoss`, which shifts logits against labels
  a second time, and returns 25.1 and 22.3 on the same sequences. `NFKMLXTrOCRTrainingTests` holds the
  parity, a tiny run (the loss falls, a frozen encoder stays), and the save and factory reload.
- **Geometry per size.** Base and large are ViT encoders (768 and 1024 wide) under a 1024-wide decoder; the stage-1 releases and `trocr-large-printed` use sinusoidal decoder positions, whose
  `_float_tensor` placeholder the loader drops. Small is a DeiT encoder (384 wide, a distillation token
  after the class token, so 578 positions) under a 256-wide decoder. Every factory reads the geometry
  from `config.json`.
- **Every release** (the eleven on the Hugging Face API, 2026-09-23) is at reference parity, each
  against its own `run_reference.py trocr` record on the same rendered line: processor pixels,
  embeddings, first block, encoder output, first-step logits, the greedy ids and their decoded text, and
  the backend's transcription through `NFKMLXTrOCR.backend(directoryURL:)`:

  | Release | Embeddings | First block | Encoder | First-step logits |
  |---|---|---|---|---|
  | `trocr-base-handwritten` | 1.0 | 0.99999976 | 0.999999 | 0.9999999 |
  | `trocr-base-printed` | 0.99999994 | 0.9999998 | 1.0 | 1.0000001 |
  | `trocr-base-str` | 0.99999976 | 1.0000004 | 0.9999999 | 1.0000002 |
  | `trocr-base-stage1` | 1.0000001 | 0.99999994 | 1.0 | 0.9999997 |
  | `trocr-large-handwritten` | 1.0000001 | 1.0000002 | 0.9999999 | 0.9999996 |
  | `trocr-large-printed` | 1.0 | 1.0 | 1.0000002 | 1.0000001 |
  | `trocr-large-str` | 0.9999998 | 1.0000002 | 0.9999997 | 0.9999999 |
  | `trocr-large-stage1` | 1.0 | 1.0000002 | 1.000001 | 1.0000004 |
  | `trocr-small-handwritten` | 1.0000002 | 1.0 | 0.9999995 | 0.999999 |
  | `trocr-small-printed` | 0.9999997 | 0.99999964 | 0.9999994 | 1.0000001 |
  | `trocr-small-stage1` | 1.0 | 1.0000002 | 0.99999994 | 1.0000001 |

  Greedy transcription, its decode, and the backend's text equal the reference's on all eleven.
  The small releases' oracle needs protobuf 3.20 or later, because transformers builds their fast
  tokenizer by converting the slow XLM-R one, and the `llm` environment pins 3.19.6 for
  `descript-audiotools`: `PYTHONPATH=~/.inferkit-validation/protobuf-4` (protobuf 4.25.3, installed with
  `pip install --target`).

## Sa2VA

- `NFKMLXSa2VA` (`ByteDance/Sa2VA-4B`, ByteDance, Apache-2.0), a segmentation VLM: one image plus a
  referring prompt in, an answer out, and a segmentation mask when the answer carries `[SEG]`. Four
  parts, ported at reference parity (float32 on both sides, all seams first numeric run): an **InternViT-300M image encoder** (`NFKMLXSa2VAVisionNet`, a pre-norm ViT — a stride-14 patch
  convolution over the 448-square tile, a class token, a learned 1025-position table, sixteen-head
  attention with qkv bias and NO query/key normalization, and per-channel **LayerScale** on each
  residual; there is no final layer norm, the features are the last block's output); a **pixel-shuffle +
  2-layer MLP projector** (`NFKSa2VAProjector`, `mlp1`: InternVL's own `ps_version` v2 shuffle folds a
  0.5 scale into the channel axis and transposes the spatial axes back, then `LayerNorm → Linear → GELU
  → Linear` maps 4096 → 2048, 256 tokens per tile); the **Qwen2.5-3B decoder** (the shared
  `NFKMLXLanguageNet`, a plain qwen2 dense decoder, GQA 16/2, qkv bias, untied head); and a **SAM 2
  Hiera-Large grounding encoder** (the shared `NFKMLXSAM2TrackerNet`, `grounding_encoder.sam2_model.*`).
  The `[SEG]` bridge (`NFKSa2VATextHiddenFCS`, `Linear → ReLU → Linear`, 2048 → 256) maps the decoder's
  hidden state at each `[SEG]` position to a SAM sparse-prompt token. Cosines at float32: InternViT
  1.0000001, projector 0.99999994, image/text fusion 1.0, `[SEG]` bridge 1.0, and the **mask** 0.9999997
  (IoU 1.0 on the thresholded masks); greedy generation matches the reference token for token
  (*"Sure, it is [SEG]"*). The bf16 load the backend runs by default measured InternViT 0.99959,
  projector 0.99973, `[SEG]` 0.99995, and mask IoU 0.99981 against the float32 oracle; the float32
  figures rule out a defect behind that gap.
- **The single-image mask path** is SAM 2's conditioning-frame path. The reference's
  `language_embd_inference` runs `init_state → add_language_embd → propagate_in_video`, but for one image
  that reduces to the first-frame branch: `directly_add_no_mem_embed` adds the tracker's `no_mem_embed`
  to the top vision feature (no memory attention runs), the `[SEG]` embedding is concatenated onto the
  empty-point sparse prompt, and the mask decoder emits three multimask candidates whose best (by IoU) is
  the mask. `NFKSa2VAGroundingEncoder.segment` drives the shared tracker's `imageEncoder` / `promptEncoder`
  / `maskDecoder` directly, dropping the tracker's click-frame object-score suppression, which Sa2VA
  comments out.
- **The oracle** loads the repo's own custom code (`run_reference.py sa2va`, in the llm env with `peft`
  and `timm`, `trust_remote_code`, CPU float32). It records the pixel values, the grounding image, the
  input ids and generated sequence, the InternViT last hidden state, the projected vision tokens, the
  fused embeddings, the `[SEG]` embedding, the conditioned SAM feature, and the three multimask logits.
  The SAM branch is replicated through the first-frame glue directly because the released video predictor
  hardcodes CUDA; that glue is exactly the conditioning-frame path a single image takes.
- **Weight load** (`NFKMLXSa2VANet.loadWeights(fromDirectory:)`): the four-shard release is partitioned
  by prefix. The understanding stream (`vision_model.*`, `language_model.*`) keeps its checkpoint names;
  the InternViT patch convolution transposes to channels-last. The projector and `[SEG]` bridge are
  numeric `nn.Sequential`s — MLX reads a module whose child keys are all integers as an array, so
  `mlp1.{0,1,3}` and `text_hidden_fcs.{0,2}` are renamed `norm`/`fc1`/`fc2`. The grounding subtree strips
  `grounding_encoder.sam2_model.`, runs the SAM 2 tracker's own key remap and convolution transposes, and
  is put back under the prefix. Two Sa2VA-specific deltas from stock SAM 2: the checkpoint renames the
  memory-encoder ConvNeXt layer scale `gamma` to `g_weight` (mapped back), and it is a SAM 2.0 checkpoint
  (no `obj_ptr_tpos_proj` / `no_obj_embed_spatial`), so the grounding geometry is `.geometry(.large,
  release: .sam2)`.
- **Consumer surface** (`NFKMLXSa2VABackend`): `NFKInputImage` + a referring prompt under `NFKInputPrompt`
  → the answer under `NFKOutputText`, and the first object's mask under `NFKOutputMask` when the decoder
  emits `[SEG]`. The processor tiles the image the InternVL way (up to 12 448-tiles plus a thumbnail),
  builds the 1024-square grounding image, and reads the release's own tokenizer. The instruction
  follows the template `config.json` names (`NFKMLXSa2VATemplate`: `phi3_chat`, `qwen_chat`,
  `internlm2_chat`, `vicuna`). Generation stops at the tokenizer's end token or when the decoded text
  ends with a stop word, which is the test the reference's `StopWordStoppingCriteria` applies to text.
  A stop word can span tokens no fixed run matches: phi3's `<|end|>` merges with the text before it in
  a Qwen vocabulary. The answer is `predict_forward`'s: the generated tokens decoded with their special
  tokens and trimmed, and the end token `generate` stopped on is kept (Sa2VA-1B answers
  `Sure, [SEG].<|im_end|>`).
- **Reuse**: the decoder is the shared `NFKMLXLanguageNet` (Qwen2.5 needs only a config); the whole SAM 2
  stack is the shared `NFKMLXSAM2TrackerNet` at `NFKMLXSAM2Configuration.large`. The new work is the
  InternViT tower, the InternVL projector, the `[SEG]` bridge, and the loader. `NFKMLXSa2VA.backend(directoryURL:)`
  is a directory factory (the geometry lives in the release's `config.json`), so Sa2VA is not in
  `registerAll` / `MLXModelGalleryExamples`, the pattern Florence-2 and TrOCR follow. The model loads
  bfloat16 (its declared dtype; the release stores float32), which halves a 4B model's resident
  footprint.
- **Customization: LoRA, the reference's own recipe.** bytedance/Sa2VA publishes its fine-tuning
  configuration (`projects/sa2va/configs/sa2va_finetune.py`, `sa2va_qwen_finetune.py`) and model
  (`Sa2VAModel.forward`), pinned in the manifest's sources. The trained set: LoRA (rank 128, alpha 256)
  on every linear layer of the language model but its head; the embeddings and head whole
  (`modules_to_save`); the `[SEG]` bridge; SAM 2's (or SAM 3's) mask decoder; and, for InternVL, the
  `mlp1` projector. The vision tower and the rest of the grounding encoder are frozen, and so are the
  Qwen-VL merger (inside `model.visual`) and LLaVA's projector, which those wrappers freeze. The
  objective (`NFKMLXSa2VAObjective`): the language model's shifted cross-entropy over the answer (the
  prompt labeled -100); each sample fixed at five objects (`check_obj_number`: fewer repeated in order,
  more subsampled); each `[SEG]`'s low-resolution mask through the training extension's
  `_forward_sam_heads` (no object-score suppression, the best of three by IoU); and, on 12,544 points
  per mask (three times as many uniform candidates, the 75% least certain kept, the rest uniform),
  2 × mmdet's sigmoid cross-entropy on soft targets plus 0.5 × its naive dice (eps 1), each summed and
  divided by its average factor plus mmdet's float32 epsilon. `fineTune` (one overload per family:
  `NFKMLXSa2VANet`, `NFKMLXSa2VAQwenNet`, `NFKMLXSa2VALLaVANet`) runs AdamW at 4e-5, weight decay 0.05
  on every trained parameter, clip 1, under mmengine's `LinearLR` warm-up (5%, from 1e-5) and
  `CosineAnnealingLR` to zero (`NFKMLXLearningRateSchedule.mmengineWarmupCosine`; `LinearParamScheduler`
  counts `end − begin − 1` warm-up steps, so the full rate arrives one step before the cosine starts).
  `NFKMLXSa2VAExample` carries the tiles (and a Qwen-VL grid), ids, labels, grounding image, and masks.
  `NFKMLXLoRA.merge(into:)` then `NFKMLXSa2VA.save(_:toDirectoryURL:release:)` writes the weights in
  the module's layout beside the release's other files; `backendWithDirectoryURL:` and each family's
  loader read it. Not reproduced: LoRA dropout (0.05), the bfloat16 autocast, and the point draws, which
  come from this process's generator. Measured on Sa2VA-1B (`run_reference.py sa2va_loss`,
  `IK_PARITY_SA2VA_1B_LOSS`: the authors' own `_compute_loss`, `check_obj_number`, `sample_points`,
  `SAM2TrainRunner`, extension `_forward_sam_heads`, and vendored mmdet losses, run on the release,
  the sampler's `torch.rand` draws recorded): on the reference's masks and points, the mask term
  0.34273592 against 0.34273607 and the dice 0.14848039 against 0.1484804; the selection from the
  reference's draws scores the same; through the port's forward at float32, the language loss 0.26715532
  against 0.26715347 in float64 (float32 0.26714978), the five masks at cosine 1.0000001, the mask term
  0.34273693 and the dice 0.14848095; and the schedule equal to mmengine's own schedulers at every step of
  40- and 200-iteration runs. `NFKMLXSa2VATrainingTests` holds those and the trained set and merged
  round trip on the 1B; `NFKMLXSa2VAFamilyTrainingTests` holds the Qwen-VL and LLaVA trained sets and
  round trips, measured on `Sa2VA-Qwen3-VL-2B` (the untied head trained and reloaded; about 22 GB peak,
  one float32 network alive at a time) and the `Sa2VA-LLaVA-1.5-7B` cut. A float32 4B run needs about 16 GB for the weights before optimizer state.
- **Preprocessing** (`run_reference.py sa2va_processor`, the release's own code with its parameters on
  the meta device, into `<release>/processor.safetensors`; `testEveryFamilysPreprocessingMatchesTheReference`).
  On a 640×360 picture, every family's understanding pixels and grounding image match the release's:
  InternVL tiles 0.0, Qwen3-VL 5.9e-8, Qwen2.5-VL 2.4e-7, LLaVA 0.0, and every grounding image 0.0
  (SAM 3's at 1008). The 448 plate the seam tests use resizes nothing on the InternVL path. Three rules
  hold it there:
  - InternVL's `dynamic_preprocess` and every family's `DirectResize` call PIL's `resize` with its
    default filter, bicubic, on 8-bit pixels (`NFKMLXPILResample`).
  - `preprocess_image` builds the ImageNet mean and deviation as bfloat16 tensors, so the grounding image
    is normalized by 0.484375, 0.455078125, 0.40625 over 0.228515625, 0.2236328125, 0.224609375.
  - The Qwen-VL releases ship `Qwen2VLImageProcessorFast` with `resample: 3`: torchvision's antialiased
    bicubic on a `uint8` tensor. It takes PIL's windows and normalized weights, quantizes every weight of
    an axis to `int16` at the most fractional bits that axis's largest weight allows (PIL keeps 22), runs
    the width first, and skips an axis whose size is unchanged (`NFKMLXPILResample.Precision.torchvision`).
    PIL's rounding differs by one or two levels in thousands of pixels.
- **Oracle notes for the Qwen-VL releases.** `_stage_remote_code` copies every `.py` of a release into
  transformers' module folder before loading, because transformers copies only the entry file's direct
  imports and Sa2VA-Qwen3-VL-4B-SAM3's `sam3pkg_*` chain nests deeper. The staged copies place on the CPU
  what the SAM 3 package places on `"cuda"` by name, and keep its fused `addmm_act` at the input dtype
  (it casts the ViT MLP's first projection to bfloat16 for the CUDA runtime); on the CPU its GELU is the
  exact erf form the port uses.
- **Precision.** `NFKMLXSa2VANet.loadWeights(fromDirectory:dtype:)` loads at bfloat16 by default, the
  dtype the releases declare (they store float32), or at float32. Parity is measured at float32 on
  both sides. The bf16 gaps the 4B first showed (InternViT 0.99959, mask IoU 0.99981) close to within
  3e-7 of 1 at float32, which rules out a defect behind them. The Qwen-VL releases' backend loads
  bfloat16 too (`NFKMLXSa2VAQwenNet.load(directoryURL:dtype:)`; their `text_config` declares it), each
  tensor converted as it is read (`NFKMLXReleaseWeights.arrays(inDirectory:converting:)`), since a
  converted list bound beside the stored one held both through `apply` and peaked above float32. The
  backend peaks at 7.8 GB (Qwen3-VL-2B), 10.5 GB (Qwen2.5-VL-3B), 12.6 GB (Qwen3-VL-4B), and 13.0 GB
  (-4B-SAM3), where float32 reached 24.5 GB, and answers each release's float32 reference text exactly.
  Its floor against the float32 records (`testTheBFloat16LoadStaysNearTheFloat32Reference`): decoder
  0.99996 / 0.99989 / 0.99994 / 0.99991 (2B, 3B, 4B, SAM3), `[SEG]` 0.99999 or closer, mask 0.99999 or
  closer, IoU 0.99981 on the 2B and 1.0 on the rest, generation token for token on all four. The LLaVA
  net loads at float32. A test class that loads several float32 releases clears MLX's cache in `tearDown`; without
  it the fourth 4B load starved the next forward into a Metal command-buffer timeout.
- **Four architectures behind one factory.** `NFKMLXSa2VA.backend(directoryURL:)` reads `config.json`
  and builds the matching net:
  - InternVL (`Sa2VA-1B`/`-4B`/`-8B`/`-26B`, `Sa2VA-InternVL3-2B`/`-8B`/`-14B`) →
    `NFKMLXSa2VANet`: InternViT-300M, or InternViT-6B for the 26B (RMSNorm and per-token query/key
    RMS normalization across all heads), under a qwen2, phi3, or InternLM2 decoder.
  - Qwen-VL (`Sa2VA-Qwen3-VL-2B`/`-4B`/`-4B-SAM3`, `Sa2VA-Qwen2_5-VL-3B`/`-7B`) →
    `NFKMLXSa2VAQwenNet`: transformers' Qwen3-VL (interleaved M-RoPE, deepstack) or Qwen2.5-VL
    (`NFKMLXQwen25VLVisionNet`, a windowed tower whose full-attention layers `fullatt_block_indexes`
    names, and M-RoPE in contiguous `[16, 24, 24]` sections, `NFKMLXMRoPELayout.chunked`), loaded under
    the checkpoint's outer `model.`. The processor runs the reference's `512·28²` to `2048·28²` pixel
    bounds; Qwen2.5-VL patches at 14 with CLIP's normalization. The `[SEG]` bridge reads what Sa2VA
    reads, `hidden_states[-1]`, in training and in `predict_forward` alike. Under transformers 4.57,
    which the releases declare, `Qwen3VLForConditionalGeneration` is wrapped in `@check_model_inputs`,
    and its output has no `last_hidden_state`. That last entry is therefore the last decoder layer's
    output before the final norm, and the Qwen3-VL releases' bridge is trained on it. Qwen2.5-VL appends
    the normalized state. `NFKMLXSa2VAQwenNet.segmentationStates` follows each family, and the language
    loss reads the normalized states. Reading the normalized state on Qwen3-VL-2B measured decoder 0.66,
    `[SEG]` 0.975, and IoU 0.995, while greedy generation stayed token for token. The Qwen3-VL-2B release
    ships an untied head that differs from the embeddings, and a saved fine-tune keeps it untied.
  - LLaVA-1.5 (`Sa2VA-LLaVA-1.5-7B`) → `NFKMLXSa2VALLaVANet`: CLIP ViT-L/336 read at
    `hidden_states[-2]` with the class token dropped, a two-layer GELU projector, and a Vicuna
    (Llama) decoder. The image is resized to 336 with PIL's bicubic (`NFKMLXPILResample`) and
    CLIP-normalized; the prompt carries 576 `<image>` tokens.
  - SAM 3 grounding (`Sa2VA-Qwen3-VL-4B-SAM3`) → `NFKSa2VASAM3GroundingEncoder`: the toolkit's
    SAM 3 ViT at 1008 (a 72-token grid), the neck's tracker branch (`sam2_convs`, levels at scales
    4, 2, and 1), `no_mem_embed` on the coarsest level, and SAM 2's prompt encoder and mask decoder.
    The checkpoint keeps the original `sam3` package's names (a fused `qkv`, `ln_pre`, a
    `pos_embed` with a class-token row, and `mlp.lin1`/`lin2` in the two-way transformer).
    `testTheSAM3GroundingMatchesTheReleasedShapes` holds the map to the release's safetensors
    headers (`shapes.py`): all 741 grounding tensors, every parameter supplied at its shape, only the
    detector's neck and the tracker's memory path dropped. Written before the weights arrived, it
    found two defects: the split projections were named `attention.weightq_proj.weight`, and the
    `lin1`/`lin2` names were unmapped.
- **InternLM2** (`Sa2VA-8B`, `-26B`): `NFKMLXInternLM2` maps the decoder onto the dense one. The fused
  `wqkv` stores, per key-value head, its query heads followed by the key and the value, so the split
  regroups rather than slices. The dynamic-NTK `rope_scaling` changes nothing below 32,768 positions
  and is dropped. `NFKMLXInternLM2Tokenizer` is the slow SentencePiece tokenizer the releases ship (no
  `tokenizer.json`); with `.llamaFast` decoding the same class is LLaVA-1.5's Llama tokenizer, whose fast
  form encodes identically and decodes each token in place. Two slow-decode quirks reproduce: the prefix
  space it prepends cancels out, and `clean_up_tokenization` always runs. The oracle needs sentencepiece 0.2.0
  (`PYTHONPATH=~/.inferkit-validation/sentencepiece-0.2.0`), because 0.2.2 rejects the model's
  null-character piece (id 354), and transformers 4.57's `AutoTokenizer` returns a bool for the
  release, so `run_reference.py` builds the tokenizer class from the remote code directly.
- **Released sizes** (Hugging Face API, 2026-09-23), all fourteen measured at float32 on both sides:
  - `Sa2VA-4B`: the figures above.
  - `Sa2VA-1B` (Qwen2.5-0.5B, qwen template): InternViT 1.0, projector 1.0000001, fusion
    0.99999994, `[SEG]` 1.0000002, mask 0.9999994 (IoU 1.0), greedy token for token, and the backend
    answers with the reference's text and a mask.
  - `Sa2VA-Qwen3-VL-2B` (`run_reference.py sa2va_qwen`): vision tower 1.0000005, deepstack
    1.0 / 0.99999946 / 1.0000002, decoder (the bridge's input) 1.0000002, `[SEG]` 0.9999995, mask
    0.99999946 (IoU 1.0), greedy generation token for token, and the backend answers
    `Sure, it is [SEG].<|im_end|>`, the reference's text, with a mask. The first real-weight
    measurement of the interleaved M-RoPE and the deepstack under Sa2VA.
  - `Sa2VA-Qwen3-VL-4B`: vision tower 1.0, deepstack 0.9999999 / 0.9999996 / 0.99999964, decoder
    1.0000023, `[SEG]` 1.0, mask 1.0000005 (IoU 1.0), greedy generation token for token, and the
    backend answers `Sure, [SEG].<|im_end|>`, the reference's text, with a mask. The float32 release
    is 20.2 GB. The Sa2VA-Qwen loader plans the decoder `.automatic`, because a resident plan refuses
    it against a 32 GB machine's 21.2 GiB working set. The float32 parity seams and the backend run as
    two tests (`testEveryQwenVLReleaseAnswersThroughTheBackend`); the parity test's footprint is about
    22.5 GB, and the bfloat16 backend's MLX peak 12.6 GB. `IK_SA2VA_ONLY=<name>` selects one release.
  - `Sa2VA-Qwen3-VL-4B-SAM3` (SAM 3 grounding at 1008): vision tower 1.0, deepstack 0.9999999 /
    0.9999996 / 0.99999964, decoder 0.9999959, `[SEG]` 0.9999995, mask 1.0000004 (IoU 1.0), greedy
    generation token for token, and the backend answers `Sure, the segmentation result is [SEG].<|im_end|>`,
    the reference's text, with a mask. The first real-weight measurement of `NFKSa2VASAM3GroundingEncoder`.
    The parity test runs in two phases (`NFKMLXSa2VAQwenNet.load(directoryURL:parts:)`): the tower and
    decoder, then the bridge and grounding encoder from the copied `[SEG]` hidden state, because the
    float32 decoder beside SAM 3's 1008-pixel trunk paged a 32 GB machine; about a 21.5 GB footprint.
    The bfloat16 backend holds both at an MLX peak of 13.0 GB.
  - `Sa2VA-Qwen2_5-VL-3B` (36 layers): windowed vision tower 0.9999996, decoder 1.0000008, `[SEG]`
    0.99999964, mask 0.9999999 (IoU 1.0), greedy generation token for token under the default system
    turn, and the backend answers `Sure, it is [SEG].<|im_end|>`, the reference's text, with a mask.
    The backend test's footprint reaches about 24.5 GB here as on the 4B, 3–4 GB above the parity
    test's.
  - `Sa2VA-InternVL3-2B` (Qwen2.5-1.5B, qwen template): InternViT 1.0000005, projector 1.0000002,
    fusion 1.0000002, `[SEG]` 0.9999998, mask 0.9999999 (IoU 1.0), greedy token for token, and the
    backend answers `Sure, the segmentation result is [SEG].<|im_end|>`, the reference's text, with a
    mask.

  The releases too large for a float32 oracle on this machine (`-8B`, `-26B`,
  `Sa2VA-InternVL3-8B`/`-14B`, `Sa2VA-Qwen2_5-VL-7B`, `Sa2VA-LLaVA-1.5-7B`) are cut to their first
  four decoder layers (`truncate.py`, the 26B's InternViT-6B to four layers too) and recorded
  teacher-forced (`run_reference.py sa2va_teacher`, `sa2va_llava_teacher`, `SA2VA_TEACHER=1
  sa2va_qwen`). The cut keeps every geometry-specific tensor and drops repeated depth. Measured cuts:
  - `Sa2VA-8B` (InternLM2, four layers): InternViT 0.99999976, projector 0.99999964, fusion
    0.9999996, decoder 0.9999999, logits 1.0000001, `[SEG]` 1.0, mask 1.0000001 (IoU 1.0); the slow
    SentencePiece tokenizer's prompt ids and decode exact.
  - `Sa2VA-InternVL3-8B` (Qwen2.5-7B, four layers): InternViT 0.9999994, projector 0.9999994, fusion
    0.9999995, decoder 1.0, logits 0.9999995, `[SEG]` 1.0000001, mask 0.99999946 (IoU 1.0).
  - `Sa2VA-26B` (InternViT-6B and InternLM2, each four layers): InternViT-6B 1.0000002, projector
    1.0000001, fusion 1.0000001, decoder 1.0000002, logits 0.99999994, `[SEG]` 0.99999994, mask
    1.0000001 (IoU 1.0). The first real-weight measurement of InternViT-6B's RMSNorm and per-token
    query/key normalization.
  - `Sa2VA-InternVL3-14B` (Qwen2.5-14B, four layers): InternViT 1.0000001, projector 1.0000008,
    fusion 1.0000007, decoder 0.9999997, logits 1.0000004, `[SEG]` 1.0000001, mask 1.0000005 (IoU 1.0).
  - `Sa2VA-Qwen2_5-VL-7B` (Qwen2.5-7B, four layers; `SA2VA_TEACHER=1 run_reference.py sa2va_qwen`):
    windowed vision tower 0.9999999, decoder 0.9999995, `[SEG]` 1.0000002, mask 1.0000002 (IoU 1.0);
    the prompt ids exact. The first real-weight measurement of the windowed tower and the chunked
    `[16, 24, 24]` M-RoPE. Two release details the loader and the prompt follow: transformers 4.56
    writes the wrapper's name into `text_config.architectures`, which the decoder reader ignores, and
    Qwen2.5-VL's chat template opens with `<|im_start|>system\nYou are a helpful assistant.<|im_end|>`
    when the conversation has no system turn (`NFKMLXSa2VAQwenNet.promptText`). Qwen3-VL's template
    has no default system turn.
  - `Sa2VA-LLaVA-1.5-7B` (Vicuna, four layers): processor max abs 0.0, CLIP tower 1.0000002,
    projected features 0.99999994, fusion 1.0, decoder 1.0000005, logits 1.0000002, `[SEG]`
    0.9999997, mask 1.0000002; prompt ids and the answer's decode exact. IoU 0.9993932: one of the
    low-resolution mask's pixels flips, and its reference logit is 7.1e-6 from zero, so the decision
    difference is float rounding. `NFKMLXSa2VATests.maskFlips` reports the flipped count and that
    magnitude.

  Every released size is measured.

## Phi-4-multimodal

- `NFKMLXPhi4MM` (`@objc`) / `NFKMLXPhi4MMModel` / `NFKMLXPhi4MMBackend` (`@objc`) — Phi-4-multimodal
  (`microsoft/Phi-4-multimodal-instruct`, Microsoft, MIT), a text, image, and speech model. One
  Phi-4-mini decoder serves every input mode. A SigLIP image tower and a Conformer speech tower place
  their embeddings at placeholder token positions (`<|endoftext10|>` 200010 for image, `<|endoftext11|>`
  200011 for audio), and a per-modality LoRA rides on the decoder's projections: the release's mixture
  of LoRAs. At reference parity on the released weights (5.6 billion parameters, float32) against the
  release's own remote code, in all four modes it serves. Text: logits 0.9999999999965636, continuation
  exact (*"The capital of France is Paris. It is not only the"*). Speech: encoder 0.9999999999982836;
  projector 0.9999999999969423; logits 0.9999999999994174; transcription exact (*"The quick brown fox
  jumps over the lazy dog."*). Vision: SigLIP 0.9999999999468102; projector 0.9999999999585922; logits
  0.9999999999986776; caption exact. A 900×500 picture that pads into a 2×3 crop grid: SigLIP
  0.9999999999281103; projector 0.9999999999470011; logits 0.9999999999979092; caption exact. Through
  the assembled model from raw inputs: text 0.9999999999996578, speech 0.9999999999994575, vision
  0.9999999999987333, vision with speech 0.9999999999993611, every greedy answer exact, and the backend
  transcribes the validation WAV and captions the photo as the reference does.
- **The decoder** is the shared `NFKMLXLanguageNet` at Phi-4-mini's geometry: 3072 wide, 32 layers,
  24 query and 8 key/value heads of 128, a SwiGLU of 8192, RMSNorm, tied embeddings, and the o200k
  vocabulary (200,064 with Phi's specials). Two features entered the shared decoder for it, both
  byte-identical for every other model. `NFKMLXLanguageConfiguration.rotaryDimensions` rotates only the
  first `partial_rotary_factor · head_dim` channels (96 of 128) and passes the rest through.
  `NFKMLXRoPEScaling.Kind.longrope` reads the per-pair `short_factor` / `long_factor` tables, chooses by
  sequence length against the trained window, and multiplies the rotated queries and keys by
  `sqrt(1 + ln(max / original) / ln(original))` (1.19024 here). Two traps held the first text run at
  0.99730. The attention factor scales only the ROTATED channels: the reference carries it on the
  cosines and sines, which the pass-through tail never meets, so pre-scaling the whole head is wrong.
  And `original_max_position_embeddings` (4096) sits at the TOP level of Phi's config, outside
  `rope_scaling`; read from the block it defaults to the extended window, the ratio becomes one, and the
  factor silently vanishes. `NFKMLXRoPEScalingTests` measures the LongRoPE formula against
  transformers' `ROPE_INIT_FUNCTIONS` for the short table, the long table, and a declared factor.
- **The mixture of LoRAs** is baked into the base shards: each of `qkv_proj`, `o_proj`, `gate_up_proj`,
  and `down_proj` is stored as `base_layer` plus `lora_A`/`lora_B` for `vision` (r 256, α 512) and
  `speech` (r 320, α 640), both at scale α/r = 2. The `vision-lora/` and `speech-lora/` directories
  duplicate them and are not read. The input chooses the adapter: an image (with or without audio)
  runs `vision`, audio alone runs `speech`, text alone runs the base. `NFKMLXPhi4MM.mixtureDecoder`
  keeps the base once and both adapters live beside it (`NFKPhi4MMMixtureLoRALinear`, `Wx +
  scale·(x·A)·B` for the active adapter), and `select(_:in:)` switches mode at no cost.
  `NFKMLXPhi4MM.decoder(directoryURL:modality:)` folds one adapter into the base instead. The fused
  projections split on load (q/k/v 3072/1024/1024, gate/up 8192/8192), and each adapter splits with its
  projection: q, k, and v share the down-projection `A` and take their rows of `B`.
- **The image tower** (`NFKMLXPhi4MMImageNet`) is SigLIP-so400m (1152 wide, 16 heads, patch 14, 448
  crops) read at its PENULTIMATE hidden state: 26 of 27 layers run, and the final layer, the
  post-layer-norm, and the attention-pooling head are released but reached by nothing. The tower is
  NaViT, and its position ids stretch each crop's valid patch region over the full 32×32 grid: row or
  column `k` of `n` valid maps to the count of boundaries `j/32` at or below `k/n`. A full crop reduces
  to plain row-major ids. SmolVLM's shared embedding multiplies the coordinate by `1 - 1e-6`, which shifts
  its ids to `[0, 0, 1, …, 30]`; reused here it held the first vision run at 0.583, so Phi has its own
  `NFKPhi4MMSigLIPEmbeddings`. Padded patches take id 0 and are masked as attention keys, through an
  optional mask that `NFKSigLIPAttention` gained (nil for SmolVLM and SigLIP 2). The features
  average-pool 2× to 16×16 per crop and take Phi-3.5's HD layout in `sub_glb` order: the sub-image
  grid trimmed to its useful rows and columns with a learned row separator (`sub_GN`) per row, an image
  separator (`glb_GN`), then the global 16×16 grid with its row separators. Average pooling leaves the
  patch channels unmerged, so the layout's 2×2 channel merge is the identity. The image reserves
  `273 + rows·columns + rows` tokens (545 for one full crop), and `Linear → GELU → Linear` projects
  1152 → 3072.
- **The speech tower** (`NFKMLXPhi4MMAudioNet`) is Microsoft's `ConformerEncoder`: per-feature mean and
  variance normalization from stored statistics, a NeMo depthwise-striding subsampler (non-causal
  despite the config's `causal`, 80 mels → 1024, eight-fold in time), and 24 macaron blocks
  (`0.5·FF → attention → gated convolution → 0.5·FF → LayerNorm`). Attention is full: `chunk_size = -1`
  makes the streaming mask span the sequence, and the only positional signal is a T5 relative bias
  (1000 learned distances per head, clipped to ±500, no bucketing). The convolution module is the
  causal part: a pointwise GLU with separate value and gate biases, a depthwise kernel-3 convolution
  left-padded and right-trimmed, Swish, and a pointwise output. The projector has a `speech` and a
  `vision` head, and the head follows the mode, so audio beside an image goes through `vision`.
- **Long clips and several clips** follow two encoder behaviors the single-clip records never reached.
  Past 500 subsampled frames (40 s) the encoder unfolds the sequence into independent 500-frame windows
  (its fixed `max_seq_len`, a separate constant from the T5 range), each with its own 500-long bias and
  its own convolution start; the last window is zero-padded, and for a lone clip the reference passes no
  mask, so that padding stays VISIBLE to attention. Several clips run as one batch: their log-mel
  features pad with zeros to the longest, normalize and subsample together (the subsampler's receptive
  field reads into the padding, so a short clip batched beside a long one differs from the clip alone),
  and each clip's padded frames are masked out of the keys; a short clip's windows past its end are
  wholly masked, and the reference's masked softmax turns their NaN rows to zeros.
  `NFKMLXPhi4MMAudioNet.projected(clips:mode:)` reproduces both (`attentionWindow` in the
  configuration). The encoder output measures 0.9999999999982198 on a 45 s clip alone, and 0.9999999999984652 and 0.9999999999982334 on the 45 s and 3.5 s clips batched; the first-token logits 0.9999999999995438 and 0.9999999999995042, with both transcriptions exact.
- **Preprocessing** (`NFKMLXPhi4MMProcessor.swift`). Images take the dynamic-HD layout: a grid of 448
  crops (up to 36), a PIL bilinear resize (`NFKMLXPILBilinear`, PIL's fixed-point resampler with the
  triangle filter), white padding on the right and bottom, normalization to [-1, 1], and a 448×448
  global view by torch's un-antialiased bicubic. The pixels match the reference processor exactly
  (max |Δ| 0.0) for the photo and for the padded 2×3 picture, with the same image-token counts. Audio
  takes SpeechLib's filterbank: 25 ms frames every 10 ms with no centering, pre-emphasis within each
  frame, a symmetric Hamming window, and SpeechLib's own 80-band mel scale to 7690 Hz; 0.999999999999802
  at 16 kHz. The reference's sample-rate handling is reproduced as written: above 16 kHz it decimates by
  the integer `rate / 16000` with SciPy's Kaiser `resample_poly` and reads the result as 16 kHz, and
  between 8 and 16 kHz by `rate / 8000`, read as 8 kHz, whose spectrum fills the lower half of the 16 kHz
  one. 44.1 kHz therefore becomes 22.05 kHz read as 16 kHz, the reference's own behavior. Measured at
  44.1 kHz 0.9999999999993252, 48 kHz 0.9999999999998265, 8 kHz 0.9999999999999564, and 11.025 kHz
  0.9999999999999464, each with the reference's token count.
- **The oracle** is `run_reference.py phi4mm`, the release's remote code run with `trust_remote_code`,
  eager attention, float32, on the CPU. The remote code targets transformers 4.46.1, which the llm
  environment's 4.57 breaks, so it runs in its own environment (`phi4mm` in the manifest's
  `oracle_environments`). It records every mode's seams, first-token logits, and continuation, the
  processor's raw inputs (the clip's samples and the PIL-decoded bytes, so the Swift preprocessors are
  measured on identical values; CoreGraphics decodes a JPEG slightly differently), and the feature
  extractor at each rate branch. Structural: all 2,047 released tensors have exactly one destination,
  706 in the decoder, 425 in the image tower, 887 in the speech tower, and the 29 SigLIP tensors the
  penultimate feature never reaches; every loader applies strictly with shapes verified. A second
  run, `run_reference.py phi4mm_conversation` (`IK_PARITY_PHI4MM_CONVERSATION`), records the requests
  past one turn with one picture and one clip. Its seam hooks return None: a PyTorch forward hook that
  returns a value REPLACES the module's output, and a `dict.setdefault` hook swapped the encoder's
  `(features, masks)` tuple for the bare features, which the reference then unpacked along the batch.
- **At bf16.** The backend loads the decoder at the release's bf16, its mixture of LoRAs beside the
  base as the reference runs it, and the two towers at float32. Against the remote code at bf16
  (`run_reference.py phi4mm_bf16`, `IK_PHI4MM_DTYPE=bfloat16`, eager), every probed decoder piece reads
  at most 0.002 of the reference's bf16-versus-float32 distance, and the text and speech logits from
  the float32 towers sit 0.66 and 0.9 of it from float32 (`testPhi4MultimodalInBFloat16MatchesTheBFloat16Reference`).
  The towers' own half-precision paths, cast to bf16, place their roundings as the reference does:
  the Conformer's `Swish` is `x * sigmoid(x)` with two roundings, its convolutions and norms round
  once, and its attention scales the query in float32 and keeps its softmax in float32; the SigLIP
  tower and its 2×2 average pool widen and round once. Every Conformer sub-piece reads under 0.01 of
  the floor. The last Conformer layer as a whole reads 0.29, because its final LayerNorm amplifies
  the pieces' one-ulp GEMM differences, so the test holds sub-pieces. The release wraps each Conformer
  layer as `_checkpoint_wrapped_module` for activation checkpointing, which is where a hook finds it.
  The remote attention divides the scores by `sqrt(head_dim)` where the shared decoder multiplies by
  its reciprocal; at `head_dim` 128 no probed element differs.
- **Consumer surface** (`NFKMLXPhi4MMBackend`): `NFKInputMessages` (or `NFKInputPrompt` as one user
  turn), pictures under `NFKInputImage` then `NFKInputImages`, clips under `NFKInputAudio` then
  `NFKInputAudios` (the core key added for it) → the answer under `NFKOutputText` and the token counts
  under `NFKOutputUsage`. `NFKMLXPhi4MMPrompt` renders the release's chat template,
  `<|role|>content<|end|>` per message (a system message's `tools` string in `<|tool|>…<|/tool|>`) and
  a closing `<|assistant|>`. Two processor rules carry over. The `<|image_N|>` / `<|audio_N|>`
  references become placeholder tokens consumed in the order they appear, whatever N says, and a count
  that disagrees with the request is refused; unreferenced media open the first user turn, pictures
  first. And the role and end markers are `rstrip` in the tokenizer, so whitespace after one is absorbed
  (the renderer strips it before encoding). Pictures embed one by one (the reference's padded crops are
  discarded, so batching them changes nothing); clips encode as one batch. `NFKParameterTemperature`,
  `NFKParameterTopP`, and `NFKParameterSeed` sample through `NFKMLXLanguageNet.sample`; the default is
  greedy, as the release's `generation_config.json` sets no sampling. The job form reports the partial
  answer and stops at cancellation. The adapter choice is state on the shared decoder, so the model
  serializes requests. Measured on a multi-turn chat with a system turn, two pictures, and the 45 s and
  3.5 s clips: prompt ids exact, logits 0.9999999999964129, the clips' vision-head projections 0.9999999999966644 and 0.9999999999968732, the
  answer exact through the model and through the backend's message, image, and audio keys
  (*"The second clip says the same thing as the first one."*). Audio with no text runs the release's
  transcription instruction. The decoder loads at the released bfloat16 by default
  (`backendWithDirectoryURL:precision:error:` chooses float32, which the tests measure), and the towers,
  under a billion parameters together, run at float32.
- **Reuse and customization**: the decoder is `NFKMLXLanguageNet` and the image encoder layers are the
  shared SigLIP blocks. The new work is LongRoPE, partial rotary, the NaViT embedding and mask, the HD
  layout, the Conformer, the mixture-of-LoRAs layer, and both preprocessors. The factories take a release
  directory (the geometry lives in its `config.json`) or download one
  (`backendWithRepo:revision:cacheDirectoryURL:precision:` and its `completionHandler:` form, from
  `NFKMLXPhi4MM.releaseRepo`), and `register()` names it to `NFKMLXModelRegistry` over a directory URL, so
  Phi-4-multimodal is not in `registerAll`; `MLXModelGalleryExamples.testPhi4Multimodal` runs both
  preprocessors, tiny towers (two clips as one batch), and a tiny LongRoPE decoder fusing an image and the
  clips, the way Pixtral's entry runs a tiny tower.
  Customization is offline-only: 5.6 billion parameters (the 3.8B decoder, 0.8B across the two adapters,
  0.9B of towers) is past the 4B float line a device trains at, and the release's recipes
  (`sample_finetune_speech.py`, `sample_finetune_vision.py`) train a modality's adapter and tower in
  bfloat16 on large-memory GPUs. A release fine-tuned there keeps the `base_layer` /
  `lora_A|B.{vision,speech}` layout and loads through the ordinary factory.
