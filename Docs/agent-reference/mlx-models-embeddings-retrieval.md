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
  round-trip, and unit-length embedding tested. The towers also load MetaCLIP, which is CLIP's
  architecture on a re-curated training set: the authors' released `b32_400m.pt` carries the same key
  names, so the converter takes it unchanged (its `state_dict` is nested under a training checkpoint,
  which the converter already unwraps once its torch 2.6 fallback passes `weights_only=False`). The
  reference is the same `CLIPModel` path pointed at `facebook/metaclip-b32-400m`, whose tensors are
  bitwise the authors' own release; at parity through the public backend (image embedding
  0.9999031286241951, the same 8-bit image bridge the OpenAI row measures through).
  **Customization is a PROBE and it ships** (`NFKMLXCLIPProbe.swift`): both towers stay frozen,
  `NFKMLXCLIP.embeddings(for:using:colorSpace:)` encodes a consumer's images once, and
  `trainProbe` fits an `NFKMLXCLIPProbe` linear classifier over the cached vectors with cross entropy.
  `NFKMLXCLIPProbe` and `NFKMLXCLIPProbeBackend` are CLIP's names for the shared
  `NFKMLXEmbeddingProbe` and `NFKMLXEmbeddingProbeBackend` (`NFKMLXEmbeddingProbe.swift`), which SigLIP 2
  trains too.
  CLIP's own linear probe is an L-BFGS logistic regression, so the optimizer (bias-corrected AdamW at
  1e-3) is this package's choice. `probeBackend(net:probe:labels:)` answers under
  `NFKOutputClassifications`, and `NFKMLXCLIPProbeTests.testATrainedProbeSavesAndReloads` reloads it.
  A contrastive fine-tune of the towers is offline: it needs large batches for its negatives.
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
  **Customization is a PROBE and it ships** (`NFKMLXSigLIP2Probe.swift`): the shared linear probe,
  `NFKMLXEmbeddingProbe`, over the frozen attention-pooled image embedding. `model(variant:weightsURL:)`
  (`@objc modelWithVariant:weightsURL:error:`) builds the model object, `imageEmbeddings(for:)` encodes a
  consumer's images once, `NFKMLXEmbeddingProbe.train` fits the probe with cross entropy, and
  `probeBackend(probe:labels:)` answers under `NFKOutputClassifications`. A saved probe reloads through
  `probeBackend(probeURL:labels:)` (`@objc probeBackendWithProbeURL:labels:error:`), which reads its
  width and class count from the file and refuses a probe of another width. big_vision's own few-shot
  probe is a closed-form regularized least squares, so the objective and the optimizer (bias-corrected
  AdamW at 1e-3) are this package's, as for CLIP. `NFKMLXSigLIP2ProbeTests` covers the round trip. An
  unloaded tower pools every image to the same embedding, because `in_proj_weight` is built as zeros for
  a checkpoint to fill, so a weight-free test randomizes it first. The sigmoid contrastive fine-tune of
  the towers is offline: it needs large batches of image-text pairs.
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
  **Customization is a PROBE and it ships** (`NFKMLXEmbeddingAdapter.swift`, `NFKMLXTextEmbedder.swift`):
  an identity-initialized linear adapter over the frozen embedding, trained with sentence-transformers'
  `MultipleNegativesRankingLoss` at its defaults (`NFKMLXEmbeddingRankingObjective`, the objective the
  Qwen3-VL embedder measured on identical tensors). `NFKMLXTextEmbeddingBackend` carries it for every
  embedder it serves: `embeddings(for:)` / `embeddings(forTokenSequences:)` encode a corpus once without
  the adapter, `makeAdapter(weightsURL:)` starts at the identity, `fineTune(adapter:queries:documents:…)`
  trains it with bias-corrected Adam at 1e-3, and `loadAdapter(from:)` (`@objc loadAdapterFromURL:error:`)
  installs a saved one so every later embedding is adapted; `removeAdapter` restores the release.
  The adapter applies after Matryoshka truncation, so its width is the backend's `embeddingDimensions`.
  `NFKMLXTextEmbeddingAdapterTests` runs the recipe over both embedder families. A full fine-tune of the
  0.6B decoder is LoRA-feasible and not written.
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
  **Customization is a PROBE and it ships**: the backend's adapter, as for Qwen3-Embedding above, over
  the mean-pooled, Dense-projected, normalized embedding. The release's own Dense head is the other
  place a probe could train; the adapter over the output is the one recipe every text embedder shares,
  and it keeps the released Dense weights untouched.
- `NFKMLXQwen3VLEmbedder` (`@objc`) / `NFKMLXQwen3VLReranker` (`@objc`) — **multimodal retrieval**, an
  embedder and a reranker that read text and images in one space (`Qwen3VLForEmbedding` and
  `Qwen3VLReranker`, Qwen, Apache 2.0). The two text embedders above score text against text, so a
  corpus of images was reachable only through CLIP's shared space, which carries neither an instruction
  nor document text. The released `Qwen/Qwen3-VL-Embedding-2B` and `Qwen/Qwen3-VL-Reranker-2B` are the
  Qwen3-VL backbone `NFKMLXQwen3VL` already runs, under a retrieval fine-tune, so the port is the
  pooling, the prompt, and the score: `NFKMLXQwen3VL.hiddenStates(decoder:inputIds:visionFeatures:…)`
  was extracted from the existing `logits(…)` so both models read the post-norm states the generative
  path projects, and a nil `visionFeatures` is the text-only sequence. The instruction goes in a system
  turn and the content in a user turn, ending at the chat template's generation prompt.
  Four release facts are load-bearing. The **pooled position is decided by the tokenizer file rather
  than by any config**: the embedding release's `tokenizer.json` carries a `TemplateProcessing`
  post-processor that appends `<|endoftext|>` to every encoding, and that token's hidden state is what
  the pooling reads, while the reranker release carries no post-processor and reads the newline after
  `<|im_start|>assistant`. The core tokenizer implements no post-processor, so the embedder appends the
  id itself (`NFKMLXQwen3VLEmbedder.promptTokens(text:imageTokens:instruction:)`), which is the same
  arrangement Qwen3-Embedding has. The reranker's score is `sigmoid(logit[yes] − logit[no])` at the last
  position, which equals the reference's one-output linear layer over `lm_head[yes] − lm_head[no]`
  because the output projection carries no bias; the two ids are the release's own
  (`1_LogitScore/config.json`, 9693 and 2152), read by `scoredTokens(inDirectory:)`. An instruction is
  punctuated before it is used, the reference's rule for an instruction that ends in anything but a
  punctuation category. And the **pixel bounds differ per release**, which decides how many tokens an
  image occupies: the instruct model holds an image between 65,536 and 16,777,216 pixels, the retrieval
  models between 4,096 and 1,310,720, so `NFKMLXQwen3VLImageProcessor.processor(inDirectory:)` reads
  `preprocessor_config.json` rather than carrying one release's constants.
  Reference parity on the released weights against each repo's own script (`run_reference.py
  qwen3vl_embedding` and `qwen3vl_reranker`, the `llm` oracle): the port's prompt tokenizes to the
  reference's ids exactly, text embedding cosine 0.9999999999866735, vision tower 0.9999999997925626,
  image embedding 0.9999999999305262; reranker scores 0.7347071466634647 vs 0.7347047924995422
  (relevant), 0.052644203261412975 vs 0.052642300724983215 (irrelevant), and 0.48971297489501875 vs
  0.4897109866142273 (an image document). The 8B pair is the same architecture at the deeper 27-block
  tower, held to the module by shape against its released headers (749 and 750 tensors consumed, 0
  missing, 0 mismatched, 0 unaccounted). The 8B reranker ships `lm_head.weight` although its config
  ties the embeddings, which is the case `NFKMLXQwen3VL.decoderConfiguration(directoryURL:)` already
  settles from the weights rather than the config, and which `scoringDirection()` reads through:
  the release's head rows when it ships them, the tied embedding rows otherwise.
  Customization is a **probe** (`NFKMLXQwen3VLRetrievalTraining.swift`): the backbone is frozen and
  produces its embeddings once, and a small module over them trains. The embedder's is
  `NFKMLXQwen3VLEmbeddingAdapter`, a linear adapter initialized to the identity so an untrained one
  reproduces the released space, trained with `NFKMLXQwen3VLEmbeddingObjective` — sentence-transformers'
  `MultipleNegativesRankingLoss` at its defaults (cosine similarity, scale 20, in-batch negatives,
  optional hard-negative groups), which is the library the releases are packaged for (`modules.json`
  names its Transformer, Pooling, and Normalize modules). Both names alias the shared
  `NFKMLXEmbeddingAdapter` and `NFKMLXEmbeddingRankingObjective`, which the text embedders train too. The reranker's is
  `NFKMLXQwen3VLRerankerHead`, initialized from the release's own scoring direction, trained with
  `NFKMLXQwen3VLRerankerObjective` — the same library's `BinaryCrossEntropyLoss` over the raw pair
  logit. Objective parity on identical tensors (`run_reference.py qwen3vl_retrieval_loss`): ranking
  3.7002993 vs 3.7002993, with a hard-negative group 5.3649597 vs 5.3649597, binary 0.87207115 vs
  0.8720712. A trained probe saves through `NFKMLXWeights.save` and installs through
  `loadAdapterFromURL:error:` / `loadHeadFromURL:error:`, which is the Objective-C reach into a
  fine-tune. A full fine-tune of the 2B backbone is offline: it needs the optimizer state of 2 billion
  parameters and a batch of negatives large enough for the contrastive objective to mean anything.
  ObjC reaches both models through `embedderWithDirectoryURL:error:` / `rerankerWithDirectoryURL:error:`,
  the embedding and scoring methods, and `backendWithDirectoryURL:error:` for the embedder's text path
  through `NFKMLXTextEmbeddingBackend`; the `Module` types and the `MLXArray` seams stay Swift-only,
  per the parity rule.
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
- `NFKMLXLaya` (`@objc`) / `NFKMLXLayaNet` / `NFKMLXLayaBackend` (`@objc`) — Laya (`convaiinnovations/laya`,
  Apache-2.0), a **typed-decision model**: the open reproduction of TypeSafe's Jev, released three days
  after it. It answers the same three question types about a state in one bidirectional pass and never
  generates text: a choice among named options, a score on an ordered scale, and a noul (the probability
  a statement holds). It is the on-device counterpart of the core's `NFKTypeSafeBackend` and takes the
  same objects (`NFKDecisionQuestion` in, `NFKDecisionAnswer` out; `NFKInputState` / `NFKInputQuestions`
  / `NFKOutputAnswers` through the backend, with the reply under `NFKOutputStructured` in the hosted
  service's shape). The network is the reranker's ModernBERT encoder (`NFKModernBertModel`, under the
  checkpoint's `encoder.` prefix, so the module is reused unchanged) plus a decision head trained from
  scratch (`rl_common.DecisionModel`): a 3-row `type_emb` added to every token, two pre-norm
  `nn.TransformerEncoderLayer`s (fused `self_attn.in_proj_weight`/`bias` as a raw parameter with an
  `out_proj` Linear, a **ReLU** feed-forward at 4d, `norm1`/`norm2` with biases), a `scorer` Sequential
  (LayerNorm, Linear, GELU, Linear→1) read at each option's `[MASK]` marker, and an `act_head`
  (Linear(d+4, 256), GELU, Linear→2) over the first token beside a detached summary of the answer
  distribution (top probability, margin, normalized entropy, count/255). The two Sequentials are
  `[Module]` arrays with a parameter-free `NFKLayaGELU` at the gap index, the TAESD trick, so the numeric
  keys load with no remap; the checkpoint's `temperature` buffer loads as a frozen parameter and is not
  read, because the calibration the release fitted lives in `rl_agent_config.json` (`temperature` per
  type, `temperature_by_options` per type-and-cardinality bucket such as `choice:6-10`), which the
  configuration carries. The prompt is the reference's `build_sequence` exactly:
  `[CLS] "<type> question: <instructions>" [SEP] ([MASK] " " + option)* [SEP] state [SEP]`, each option
  cut to 48 tokens, the options shrunk evenly when they leave under 16 of `head_max_len`, the
  instructions to what is left (at least 8), the state to the room under `max_len`; a noul's options are
  always `[false, true]`. A non-string state is serialized as Python's `json.dumps` writes it (`", "` /
  `": "`, non-ASCII kept) with sorted keys, and the record is checked byte for byte. Three released
  variants, each a directory with `rl_agent_config.json`, `encoder/config.json`, `tokenizer/`, and
  `model.safetensors` (fp16, loaded at float32): the root (ModernBERT-large: 28 layers, 1024 wide, 16
  heads, 2624 intermediate, 421M), `typed-decisions` (the same geometry fine-tuned on that benchmark),
  and `multilingual` (mmBERT-base: 22 layers, 768 wide, a 256k Gemma vocabulary, 322M). Three traps,
  each measured: the encoder configs are transformers 5 files whose rotary bases sit under
  `rope_parameters`, which a 4.x `ModernBertConfig` ignores, and mmBERT's **local base is 160000**, not
  the 10000 default, so the reader takes `rope_parameters` first and the oracle copies them into the
  4.x fields; the multilingual tokenizer is Gemma's BPE under a **Metaspace pre-tokenizer with
  `prepend_scheme: always` and `split: true`**, so a metaspace is prepended and the text is split at
  every metaspace into pieces that each start with one before the merge (the plain Gemma reader treats
  the whole string as one pre-token); and the multilingual **classifier token is the tokenizer's `<bos>`
  (2)**, while its encoder config's `cls_token_id` says 1, so the special ids come from the tokenizer
  files, as the reference reads them. Reference parity against the release's own inference code
  (`run_reference.py laya`, the llmvenv oracle importing `rl_agent_api.RLAgent` from the release; four
  questions, a described 3-way choice, an undescribed 6-way choice, a 3-level score, and a noul with
  meanings, over a 130-token string state and a record state): every prompt token for token and marker
  for marker on all three variants; the isolation harness exact (every layer and the decision head at
  1.0000000000, a few mid-stack layers of the large encoder at 0.9999999999); worst logit cosine over the
  reference's ids 0.9999999999903213 (root), 0.9999999999915642 (typed-decisions),
  0.9999999999988376 (multilingual); worst calibrated-probability gap 1.9669533e-06, 6.556511e-07, and
  1.4603138e-06; the act probability within 2e-3. Customization is a **head fine-tune** (`.head`: the
  decision head, the type embedding, and the scorer; `.all` adds the encoder, which is how the reference
  trains; the act head and the temperature buffer stay frozen because the act head's cost-weighted
  objective is not published). The objective is the reference's `rl_common.proper_reward` ported as
  `NFKMLXLayaObjective`: the log score (floored at −9.21) plus 0.5 times the spherical score for every
  type, minus the ranked probability score over the cumulative distributions for a score question; the
  reference maximizes it by REINFORCE over noised logits with a group-mean baseline, which is described
  and not published, so the recipe takes the gradient of the expected reward directly, which has the
  same optimum. Objective parity on identical tensors (`run_reference.py laya_loss`): rewards
  [−5.272212, −2.2349505, −3.7501607, −2.8746824, −3.3278863, −0.2989947] vs [−5.2722116, −2.2349505,
  −3.7501602, −2.8746824, −3.327886, −0.29899463], loss 2.9598145 vs 2.959814. The set: a public
  `network(weightsURL:configuration:)`, `NFKMLXLayaTrainable`, `NFKMLXLayaExample` (a state, a
  question, and a one-hot, soft, or noul target, encoded through the release tokenizer),
  `fineTune(examples:steps:learningRate:trainable:)` over `NFKMLXTrainer`, and the round trip through
  `NFKMLXWeights.save` and `laya(directoryURL:weightsURL:)` (ObjC `layaWithDirectoryURL:weightsURL:error:`).
  The release's README says the base checkpoints are near chance on a new decision task and the
  capability comes from fine-tuning, which is what the recipe is for. The **conversation-prefix path**
  (`rl_common.encode_record` for an `episode` record, `episode_prefix_lengths`, `td_lambda_targets`)
  ships as `NFKMLXLayaEpisode` + `fineTune(episodes:steps:learningRate:trainable:lambda:)`: an
  episode is a context, the turns oldest first, one noul, and the outcome; its prefixes are every
  length up to `max_prefixes` (6, from `rl_agent_config.json`) and beyond that `max_prefixes` lengths
  from `linspace(1, n)` rounded and deduplicated (`NFKMLXLayaPrompt.prefixLengths`); each prefix's
  state is the context's fields (sorted) then the turns so far under `conversation`
  (`serialize(context:turns:)`, the reference appends `conversation` last), built with
  `truncateLeft` so the newest turns survive (and Python's `st[-0:]` quirk reproduced: a head that
  fills the budget keeps the whole state), and a prefix whose two markers do not fit is dropped. The
  targets are `NFKMLXLayaObjective.temporalDifferenceTargets`: the last prefix's is the outcome and
  each earlier one's return is `(1 − λ)·p_true[next] + λ·G[next]`, with the model's own next-prefix
  predictions taken under `stopGradient` in the step (the reference passes them in as a tensor, so
  whether they carry a gradient is not published; a TD target is used as a constant). λ = 1 is the
  release's setting. Measured against `rl_common` (`run_reference.py laya_episode`, the root
  tokenizer): an eight-turn episode samples to `[1, 2, 4, 5, 7, 8]`, every prefix token for token and
  marker for marker with its state byte for byte, and the target tables at λ = 1 and λ = 0.5 to 1e-6.
  ObjC reaches `layaWithDirectoryURL:error:`,
  `answersForState:questions:`, `answerForState:question:`, `backendWithDirectoryURL:error:`, and
  `makeBackend`; the `Module`, the prompt builder, and the objective stay Swift-only per the parity rule.
  Not registered by name: it loads a whole release directory, like the reranker.
  **Downloading** (`NFKMLXLayaRelease.swift`): `NFKMLXLayaVariant` (`@objc`, root / typedDecisions /
  multilingual) names the folders, `releaseFiles(for:)` the five files each needs, and
  `download(variant:revision:cacheDirectoryURL:)`, `laya(variant:…)`, and `backend(variant:…)` fetch
  them through `NFKHFHub` and build, each with a completion-handler peer. The folder returned is the
  hub cache's `<cache>/convaiinnovations/laya/<revision>/<folder>`, found from where
  `rl_agent_config.json` landed. `measuredRevision` is `1c5edc17a7ac…`, the repository head when the
  parity store was fetched (2026-09-22; the head commit is dated 2026-09-20), and all 15 file URLs
  resolve at it. The three variants share one hub snapshot, so a cache size limit evicts them together.
  Tested with a recording `NFKHFHub` subclass (paths fetched once, folder returned) and with the
  validation store symlinked into a cache layout (the variant factory answers exactly as the
  directory factory does).
  The standalone repositories `convaiinnovations/laya-typed-decisions` and `laya-multilingual` hold
  weights identical to the root repository's folders and are not used. The multilingual mirror's
  `tokenizer_config.json` stores `extra_special_tokens` as a list, the folder's as a dictionary; the
  reader takes only the cls, sep, mask, and pad tokens, so either loads.
- `NFKMLXOpenJevDeBERTa` (`@objc`) / `NFKMLXOpenJevDeBERTaNet` / `NFKMLXDeBERTaV2Net` —
  open-jev-deberta-v3-large (`com-kotobalabs/open-jev-deberta-v3-large` at `19bf9a64`, Apache-2.0;
  base `microsoft/deberta-v3-large`, MIT), a community Jev reproduction. One pass over
  `[CLS] [STATE] state ([Q] instructions ([OPT] option)*)* [SEP]`; the head
  (`Linear(3H, H)`, GELU, `Linear(H, 1)`, saved as `head.safetensors` keys `0.*` / `2.*`, loaded through
  the `[Module]`-array trick) scores `[mean(question text); mean(option text); product]`, markers
  excluded from both spans, softmaxed within each question at `temperature` 1.05. The pooling is a
  `[slots, length]` matrix of `1/count` rows (options question-major, then each question), so a padded
  batch pools only real tokens and an empty span averages to zero as the reference's count floor does.
  `NFKMLXDeBERTaV2Net` is the package's first DeBERTa: disentangled attention (content-to-position and
  position-to-content terms through the shared query/key projections of the LayerNormed relative
  embeddings, all three scores over `√(3·d)`), log buckets (256 each side, scaled to 512), no absolute
  positions, no token types. A config outside v3's options is refused, not approximated. The bucket
  table is computed in float32 as the reference does; offsets −600…600 match exactly.
  **Reference parity** against the release's bundled `typed_decisions` package (`run_reference.py
  open_jev_deberta`, llmvenv, transformers 4.57.6, the version that saved it): three cases (a message
  with a 5-way choice, a 5-level score, and a noul; a sorted-key record state with a noul and a 3-way
  choice; a long state cut at 256 tokens with a 10-way choice and a score). Tokens exact, every one of
  the 25 encoder states at worst cosine 0.9999999999917955, logit cosines 0.9999999999992606 /
  0.9999999999947546 / 0.9999999999926424 (max gaps 1.1e-5 / 7.5e-6 / 2.1e-5), answers to 1e-4, and
  cases 0 and 1 padded together through the collator within 1.1e-5.
  **The tokenizer trap:** the release's code loads `tokenizer.json` (the Rust `tokenizers` Unigram),
  which sums piece scores in DOUBLE; SentencePiece and the port's segmenter summed in float. Two
  segmentations that tie to within float rounding (40 × `a`: `▁a aaa a⁹…` versus `▁a a⁹ a⁹ aaa…`) then
  resolve differently. `NFKMLXSentencePieceSegmenter` gained `accumulatesInDoublePrecision` (default
  off, so the translators are unchanged); this model sets it and matches 14 of 14 awkward strings.
  The slow `DebertaV2Tokenizer` disagrees with the fast one on that same string.
  **The state budget:** the bundled collator raises when the state and questions overflow 512; the
  repository's current collator (`kotoba-lang/typed-decisions` `e4ed8076`) cuts the state to
  `min(256, 512 − 3 − question tokens)`. The port follows the current rule, which agrees with the
  bundled one wherever the bundled one succeeds; `open_jev_deberta_budget` measures ten 10-way choices
  over the long state, exactly 512 tokens, token for token.
  **Customization ships:** `NFKMLXOpenJevDeBERTaTrainable` (`.head`, `.all`), the objective
  `NFKMLXOpenJevDeBERTaObjective` (mean cross-entropy + `brierWeight` × mean Brier over existing options,
  the release's `decision_loss`), measured on the padded batch at Brier weights 1 and 0.5 (loss
  1.4505672 vs 1.450567, cross-entropy 0.8694962 vs 0.86949617, Brier 0.5810711 vs 0.5810709), and
  `fineTune(examples:)` with the reference's two AdamW groups (head 1e-3, encoder 3e-5, weight decay
  0.01, bias correction on, clip 1) through `MultiOptimizer`. No dropout, no warm-up schedule. Round trip
  through `network(weightsURL:)` / `openJev(directoryURL:weightsURL:)`. A dictionary of questions is
  read in sorted identifier order, since the answers depend on position; the list API keeps order.
- `NFKMLXOpenJev` (`@objc`) / `NFKMLXOpenJevNet` — Open-Jev (`ZefanCai/Open-Jev-2B` at `0c7aa498`,
  `-9B` at `47e96688`, `-27B-v1.1` at `28cf7306`; weights Apache-2.0, loader MIT,
  `github.com/Zefan-Cai/Open-Jev` at `be624f36`; its `main` on 2026-09-23 changes only device placement
  and quantized loading, not the arithmetic).
  Each candidate is `<|im_start|>user\n{Context: state / Question: … / Proposed answer: option / Is this
  proposed answer correct? Answer Yes or No.}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
  (a noul asks once, meanings appended as `Yes means:` / `No means:`, and its logits are `[0, s]`); the
  Qwen3.5 text model (`NFKMLXHybridLanguageNet`, `normalizedHidden` skips the output projection, which is
  never built) reads it, and a float32 `Linear(H, 1)` reads the last token. The adapter is rank 8,
  α 16, over `q_proj`/`k_proj`/`v_proj`/`o_proj` and `in_proj_qkv`/`out_proj`, loaded into
  `NFKMLXLoRALinear` (PEFT's `A`/`B` transposed) at the decoder's element type; the head is `head.pt`
  (the native torch reader) or a fine-tune's `head.safetensors`. It needs the base at the revision
  `model.json` pins; the download fetches that revision, and the shards from its index.
  **The tokenizer:** Qwen3.5's `Split` regex is Qwen2's with `\p{M}` inside letter runs, plus an NFC
  normalizer. The core gained a `qwen35` pre-tokenization, selected from the regex; the 2B record's
  decomposed-accent state tokenizes exactly because of it.
  **Reference parity (2B, float32)** against the loader's own `DecisionModel.load` with its
  `from_pretrained` calls redirected to the local pinned base (`run_reference.py open_jev`,
  openjevvenv: transformers 5.10.2, PEFT 0.19.1): five questions (a described choice, a score, a noul
  with meanings over a record; a 4-way choice and a plain noul over accented text). Every candidate's
  tokens exact, all 25 hidden states of the adapted text model at worst cosine 0.9999999999956228,
  logits within 1.03e-5, probabilities and answers to 1e-3.
  **The 9B release** does not fit at float32 (about 36 GB). It is checked structurally (160 adapter
  tensors and 426 base parameters against the local safetensors headers, 0 mismatched) and end to end
  at bfloat16 against the loader at bfloat16, its own precision: tokens exact, every question decides
  the same, logits within 0.14, probabilities within 0.0037. Both sides round every layer, so that bound
  is bfloat16's, not the port's; the float32 2B measurement is what carries the arithmetic.
  The bfloat16 run builds only the text model (12.89 GiB of layers and 1.89 GiB of embeddings; the tied
  output projection and the vision tower are never built). Its xctest process peaked at about 16 GB
  (`footprint`, whole-GB resolution, 2026-09-24) with swap flat; schedule it as needing 18 GB free. It
  takes about 61 s with the weights in the page cache and about 32 min when they page in from
  WindowsBoot.
  **The 27B release** (`-27B-v1.1`) is the same recipe over `Qwen/Qwen3.8-27B` at `1d4bf0f2`, a
  `qwen3_5` model (64 layers, 48 recurrent and 16 attention, hidden 5120). Its base is about 54 GB and
  is not downloaded here: the structure test reads the 62 MB adapter package (`OPEN_JEV_27B`) and the
  base's `config.json` and shard-header shapes (`IK_CONFIG_QWEN3_8`, `IK_SHAPES_QWEN3_8`, both
  byte-identical to that revision): 320 adapter tensors and 850 base parameters, 0 mismatched.
  Qwen3.8's tokenizer adds seven audio and TTS tokens (ids 248070–248076) to Qwen3.5's; the vocabulary,
  merges, and pre-tokenizer are the same. Its chat template injects a reasoning-effort system turn
  unless `enable_thinking` is false; the loader passes false, and that rendering is byte-identical to
  Qwen3.5's, so `NFKMLXOpenJevPrompt.chat` holds. It trains at 2e-5 (adapter) and 5e-5 (head), per its
  `provenance.json`; `fineTune`'s defaults are the 2B and 9B rates.
  **Customization ships:** `NFKMLXOpenJevObjective` (per record `−Σ t·log softmax + 0.1·Σ (softmax −
  t)²`, the loader's loss, matched on every question of the 2B record to 1e-5), `freeze` (adapter and
  head only), `fineTune(examples:)` with the loader's AdamW groups (adapter 5e-5, head 1e-4, weight
  decay 0.01, bias correction, clip 1, batch = its accumulation of 4), and `save(to:)` writing the
  release's own layout (PEFT adapter, `head.safetensors`, `model.json`, `temperature.json`), which the
  same factory reloads; a tiny base written to disk proves that round trip offline. A fine-tune needs
  the base at `.float32`. The temperature is not refitted. The loader's prefix cache (opt-in there) is
  not ported: each candidate is its own pass.
- `NFKMLXChronos` / `NFKMLXChronosNet` — Chronos-Bolt (`amazon/chronos-bolt-base`, Amazon, Apache-2.0),
  a **time-series forecaster** (a distinct modality that lives here as a standalone object, the way the
  reranker does). A patched T5 encoder-decoder reads a numeric context window and emits a quantile
  forecast. Flow (`ChronosBoltModelForForecasting`): standardize the series (InstanceNorm: subtract the
  mean, divide by the standard deviation), patchify into non-overlapping 16-sample patches with an
  observed-mask, embed each `[patch, mask]` through the `input_patch_embedding` ResidualBlock, append a
  learned REG token, run the T5 **encoder**, then a T5 **decoder over a single `decoder_start` token**
  that cross-attends to the encoder output, and map that one vector through `output_patch_embedding` to
  `quantiles × prediction_length`, un-scaled by the InstanceNorm statistics. The transformer is plain T5
  v1.0 (single-`wi` ReLU feed-forward, unscaled attention, a bucketed relative-position bias on the first
  block of each stack, RMS `T5LayerNorm`), so it does not reuse the gated `NFKMLXT5Encoder`; it is its own
  compact encoder + one-token decoder. The released `model.safetensors` loads directly: the module keys
  mirror the HF T5 names, nothing is transposed, and the tied `embed_tokens` aliases plus the unused
  `lm_head`/`quantiles` buffer are dropped. `NFKMLXChronos.forecast(context:horizon:)` returns one `[Float]`
  row per quantile level (0.1 … 0.9); `medianForecastForContext:horizon:` is the Objective-C point
  forecast. Reference parity against the `chronos` package's own `ChronosBoltPipeline` (`run_reference.py
  chronos`, the llmvenv oracle with `chronos-forecasting`), seam by seam on the first numeric run: input
  embeddings 1.0000001; the encoder, the one-token decoder, the quantile head, and the un-scaled forecast
  all 1.0; every quantile row matches. `NFKMLXChronosConfiguration` carries the base geometry (768-wide,
  12 + 12 layers, 16-sample patches, 64-step horizon). Customization (a pinball-loss regression fine-tune
  on a consumer's own series) is implementable on this small model and is NOT yet shipped; it is the
  remaining half per the customization-is-part-of-parity rule, named here rather than claimed done.
