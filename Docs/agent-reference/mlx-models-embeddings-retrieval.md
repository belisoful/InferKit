<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX models: image-text embeddings, text embeddings, reranking

- `NFKMLXCLIP` (`@objc`) — real image+text embeddings (CLIP ViT-B/32): a ViT image tower (`visual.*`)
  and a causal text transformer (`token_embedding`/`transformer`/`ln_final`/`text_projection`), both
  L2-normalized into a shared space. Attention keeps the reference fused `in_proj_weight`/`out_proj`.
  Reference parity on both towers against transformers' `CLIPModel` (image cosine 0.9999965,
  **text cosine 0.9999999999988** — the text record carries the reference's own token ids, since the
  port embeds ids rather than text).
  `NFKMLXCLIPBackend` reads `NFKInputImage` → embedding under the new core key `NFKOutputEmbedding`; a
  text prompt encodes when a tokenizer is supplied (byte-level-BPE vocab is a load-time artifact — a
  caller can pass token ids through `encodeText`). `+register` under `clip-vit-b-32`, and the other
  released towers under `clip-vit-b-16` / `-l-14` / `-l-14-336` (`NFKMLXCLIPVariant`: `.vitB16`,
  `.vitL14` — vision 1024 wide, 24 blocks, 16 heads, embedding 768, text 768 / 12 / 12 — and
  `.vitL14At336`); B/16 and L/14 load their released checkpoints strictly and return unit embeddings,
  and being the B/32 blocks at another geometry they carry the numeric parity above.
  `Tools/clip-to-safetensors/convert.py` targets the OpenAI JIT/state-dict (names match). Forward,
  round-trip, and unit-length embedding tested.
- `NFKMLXSigLIP2` (`@objc`) — real image+text embeddings (SigLIP 2, base-patch16-224), the CLIP upgrade
  and the vision tower a VLM reads. The vision and text towers are the same transformer the SmolVLM
  SigLIP encoder uses (`NFKSigLIPLayer`/`NFKSigLIPAttention`/`NFKSigLIPMLP`/`NFKSigLIPEncoder` are reused
  directly); SigLIP 2 adds an **attention-pooling head** over the vision patches (`NFKSigLIP2ProbeAttention`:
  a learned probe token cross-attends over the patch features through `nn.MultiheadAttention`'s fused
  `in_proj_weight`, then a residual LayerNorm and MLP), a **text tower** over a 256k multilingual
  vocabulary (last-token pooled through a `head` projection), and learned `logit_scale`/`logit_bias` for
  the sigmoid similarity. The image embedding is the pooling head's output; the text embedding is the
  last token's; both are L2-normalized and `logit = scale·(text·image) + bias`. The vision embeddings
  read the position table row-major (SigLIP 2 does not use SmolVLM's fractional position buckets, which
  is why `NFKSigLIP2VisionEmbeddings` is separate from the SmolVLM one). `NFKMLXSigLIP2Backend` reads
  `NFKInputImage` → the image embedding under `NFKOutputEmbedding`; `imageEmbedding`/`textEmbedding` are
  the object accessors. `+register` under `siglip2-base-patch16-224`. The MAP head's `attention` is a
  real submodule, not a dotted parameter key — MLX splits parameter keys on `.`, so `@ParameterInfo(key:
  "attention.in_proj_weight")` would not nest into an `attention` child and the head weights load as
  random (measured: image cosine collapses to ~0.5 while the text tower stays exact, the seam that
  localized it). The release is already a PyTorch-layout safetensors the loader reads directly (it
  transposes the 4-D patch conv and maps the `vision_model.`/`text_model.` prefixes; Linear/embedding
  weights and the fused attention projection are 2-D and pass through). Reference parity against
  transformers on the released weights (`run_reference.py siglip2`, llm oracle env, transformers ≥ 4.51):
  image embedding cosine 0.999999999999, every text embedding 0.999999999999, and the sigmoid logits to
  1e-5. The architecture is SigLIP v1 (`model_type` "siglip"); the "2" is the training. Most of the
  1.5 GB checkpoint is the 256k text embedding table. Converter `Tools/siglip2-to-safetensors` is a
  passthrough normalizer. Every SigLIP 2 release is a preset (`NFKMLXSigLIP2Family` × patch × image
  size through `towers(_:patchSize:imageSize:)`: base at patch 16 / 32, large, so400m at patch 14 / 16,
  giant-opt, each at its released resolutions; the giant-opt text tower is 1152 wide and projects to
  1536, which is what `projectionSize` carries), with `NFKMLXSigLIP2Variant` (15 cases) selecting one
  from Objective-C and `register()` naming each under its release name. The fourteen beyond the
  measured base-224 are held to the module by shape against their released headers (408 / 792 / 888 /
  1096 tensors per family, 0 missing, 0 mismatched, 0 unaccounted).
- `NFKMLXTextEmbedder` / `NFKMLXQwen3Embedding` / `NFKMLXTextEmbeddingBackend` — on-device **text
  embeddings**, the capability the package lacked: it embedded images (CLIP) with no path for text, so
  no semantic search, retrieval, clustering, or reranking over a consumer's corpus. A text embedder is
  the decoder-only model with its output projection removed — the post-final-norm hidden states pooled
  to one vector and L2-normalized — which is exactly the seam `NFKMLXLanguageNet.hiddenStates(fromEmbeddings:)`
  already exposes, so nothing about the transformer is re-implemented. Qwen3-Embedding-0.6B is the
  Qwen3-0.6B dense decoder this package already runs (`configuration(fromHuggingFace:)` reads its
  `Qwen3ForCausalLM`/`qwen3` config unchanged; geometry equals `.qwen3_0_6B` but for `vocab_size` 151669
  and the tie), pooled at the **last token** over an appended `<|endoftext|>` (id 151643, not the chat
  `eos_token` 151645) and L2-normalized. `NFKMLXTextEmbedderConfiguration` carries the pooling
  (`.lastToken`/`.mean`), the appended token, normalization, and the Matryoshka `dimensions` a leading
  slice is a usable embedding at. Reference parity against the model card's own transformers recipe
  (`AutoModel` last hidden state, last-token pool, `F.normalize`) on the released 0.6B weights: query
  embedding cosine 0.99999999999, document 0.99999999999, retrieval score 0.76456 reproduced to 1e-6 end
  to end (`run_reference.py qwen3_embedding`, the `llm` oracle interpreter). A separate
  tokenizer-agreement test reproduces the reference's ids from the shared text — the **`qwen2`
  pre-tokenization** is what makes them right, the same trap the music tokenizer hit.
  Two release facts are load-bearing. The tokenizer appends `<|endoftext|>` and its hidden state is
  what the pooling reads, so the append is the model's geometry rather than the tokenizer's; and the
  released checkpoint is the **base model** (`AutoModel`/`Qwen3Model`), so its keys carry no `model.`
  prefix and no `lm_head` — the decoder keeps the causal-LM layout, so `NFKMLXQwen3Embedding.loadWeights`
  prepends `model.` and drops the absent projection (the shared `NFKMLXLanguage.loadedRelease` loader
  expects the prefix and would reject it, which is exactly what the first parity run reported). ObjC
  reaches it through `backendWithDirectoryURL:error:` and `backendWithDirectoryURL:outputDimensions:error:`
  (0 = full width); the `*Configuration` structs stay Swift-only, per the parity rule, and a Swift
  caller with token ids reads them through `embedding(forTokens:)`. `NFKMLXLanguage.releaseTokenizer(inDirectory:)`
  was extracted from `loadedRelease` so the embedder and the tokenizer-agreement test build the release
  tokenizer without loading the 1.2 GB of weights.
- `NFKMLXEmbeddingGemma` / `NFKMLXGemma3EncoderNet` / `NFKMLXGemmaTokenizer` — a second text embedder
  over a second architecture: EmbeddingGemma-300M, a bidirectional encoder where Qwen3-Embedding is a
  causal decoder. The backbone is the Gemma 3 text model (`gemma3_text`, `use_bidirectional_attention`),
  Not the causal Gemma 4 (`gemma4_text`) `NFKMLXGemmaLanguage` implements, so it is its own
  implementation `NFKMLXGemma3EncoderNet`: `(1 + w)` RMS normalization (Gemma 3; Gemma 4 uses `x · w`, the
  difference that first broke a Gemma port here), the sandwich norm (a norm before and after each of
  attention and the feed-forward), dual RoPE (local base 10000 for the sliding layers, global 1000000 for
  the full ones, the full head turned entirely — Gemma 3 carries no partial factor), per-head QK-norm
  before the rotary, a GeGLU `gelu_pytorch_tanh` feed-forward, the query scaled by `queryPreAttnScalar ^
  -0.5`, an embedding scaled by `√hidden`, no value norm, no per-layer embeddings, no softcapping. Every
  layer is bidirectional; a sliding layer sees a symmetric window (512), a full layer everything, and for
  an input shorter than the window (the common case) the mask is inert and every layer is full attention.
  The sentence-transformers head is mean pooling over every token → Dense 768→3072 → Dense 3072→768
  (both no-bias, Identity activation) → L2, with Matryoshka truncation before the final normalize.
  The Dense projections live in `2_Dense/`/`3_Dense/` subdirectories keyed `linear.weight`, which
  `NFKMLXReleaseWeights.files` does not read, so the loader takes them separately; the backbone's keys are
  the base-model checkpoint's (no `model.` prefix). Reference parity on the first numeric run against
  the sentence-transformers pipeline over transformers' own `Gemma3TextModel` (`run_reference.py
  embeddinggemma`, the `llm` oracle): every one of the 24 layers exact by the per-layer isolation harness,
  query and document embedding cosine 0.99999999999, retrieval score 0.60923 to 1e-7.
  The tokenizer is BPE, not unigram, and that was measured rather than assumed. Gemma's
  `tokenizer.model` scores are merge ranks (score ≈ −(id − constant)), so `NFKUnigramTokenizer`'s unigram
  Viterbi picks the wrong pieces (`ta`+`sk` over `task`) — caught by a tokenizer-agreement test before it
  reached anything else. `NFKMLXGemmaTokenizer` reads Gemma's byte-fallback BPE `tokenizer.json` Directly
  (no offline conversion): a metaspace normalizer (space → `▁`), the whole normalized string as one
  pre-token (the space split is a no-op after normalization), each character or its UTF-8 byte-fallback
  `<0xHH>` pieces, then the greedy merge-by-rank loop. It is neither the byte-level BPE the GPT-2/Qwen
  path uses nor unigram, so it is its own reader; token-for-token agreement with the reference is tested.
  The gated `google/embeddinggemma-300m` is mirrored ungated at `unsloth/embeddinggemma-300m` (the
  project uses mirrors for gated repos, as SD 2.1 does). ObjC reaches it through
  `backendWithDirectoryURL:error:` / `backendWithDirectoryURL:outputDimensions:error:` and the
  `query:`/`document:` prompt helpers; `NFKMLXTextEmbeddingBackend` now serves both embedders through the
  internal `NFKTextEmbedding` protocol and a tokenization closure (so a family with its own pooling,
  projection, and tokenizer plugs into one backend). The `*Configuration` structs stay Swift-only.
- `NFKMLXModernBERTReranker` / `NFKMLXModernBertRerankerNet` — a **cross-encoder reranker**, the third
  piece of the retrieval story after the two embedders. An embedder scores a query and a document
  independently and compares the vectors; a cross-encoder reads the pair together —
  `[CLS] query [SEP] document [SEP]` — through one bidirectional pass and predicts a single relevance
  logit, which is more accurate and is what reorders an embedder's shortlist. It is the released
  `gte-reranker-modernbert-base` (`ModernBertForSequenceClassification`). Because it takes a query and a
  List of documents rather than one input, it is a scoring object, not an `NFKInferenceBackend`:
  `scores(query:documents:)` / `rankedIndices(query:documents:)` (ObjC `scoresForQuery:documents:` /
  `rankedIndicesForQuery:documents:` / `scoreForQuery:document:`), built by `rerankerWithDirectoryURL:error:`.
  ModernBERT is a modernized BERT encoder: **RoPE** (a global base 160000 every third layer — `i %
  globalAttentionEvery == 0` — and a local base 10000 with a 128-token bidirectional sliding window
  elsewhere), a **GeGLU** feed-forward (`Wi` → input,gate; `gelu(input) * gate`; exact erf GELU, not
  tanh), LayerNorm throughout with no biases (`norm_bias` false), bias-free attention and MLP, no
  absolute position embeddings, and layer 0's `attn_norm` is the identity (the embeddings are
  pre-normed, so the checkpoint carries no weight for it and the module's is nil). The reranker head is
  mean pooling over the pair → a prediction head (dense + gelu + LayerNorm) → a single-logit classifier
  (with bias, though `classifier_bias` reads false — the checkpoint has `classifier.bias`, so trust the
  checkpoint). Module keys are the checkpoint's under `model.`/`head`/`classifier`, so nothing is
  remapped. Reference parity against transformers' own `ModernBertForSequenceClassification`
  (`run_reference.py modernbert_reranker`, the `llm` oracle): every one of the 22 layers exact by the
  per-layer isolation harness, and both the relevant and irrelevant scores to within 5e-3, plus the
  reranking order. Two things were measured, not assumed. The parity pair is deliberately long (95
  tokens) so it exceeds the 64-either-side local window and actually exercises the sliding layers — a
  short pair would leave a wrong window inert and pass silently. And ModernBERT's `output_hidden_states`
  does not apply the final norm to its last entry (the final norm goes only into `last_hidden_state`),
  unlike the Llama/Gemma convention, so `layerStates` returns the raw last layer output and the score test
  covers the final norm — the isolation reported a lone divergence at the last state until this was
  matched, while the score already agreed to 2e-6. The tokenizer is GPT-2-family byte-level BPE (not
  Gemma's char-BPE), which the core `NFKByteLevelBPETokenizer` reads; the release ships only
  `tokenizer.json`, so `byteLevelTokenizer(inDirectory:)` extracts its vocabulary and merges into the
  `vocab.json`/`merges.txt` the core reader takes (a temp directory), then wraps the pair in
  `[CLS]`/`[SEP]`; token-for-token agreement with the reference is tested.
