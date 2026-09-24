<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX models: language models

The dense decoder and its generation runtime, the hybrid decoder, Qwen4-Exp, DeepSeek, rotary scaling, chat templates.

A `.checkpoint` (bf16) load of the dense decoder, Granite 4.0-H, and the Mamba-2 mixer places its
roundings as transformers does at bf16 (`NFKMLXReferenceRounding`; the method is in
`mlx-parity-checklist.md`, "Half precision against the half-precision reference"). The fused
`MLXFast.RoPE` and fused attention rounded once where transformers rounds its tables, its scores and
its probabilities, and `silu` rounded twice where torch rounds once: Qwen3-0.6B's first layer, run on
the reference's own bf16 input, sat at the reference's whole bf16-versus-float32 distance and now sits
at 0.0006 of it. MLX's fused `RMSNorm` rounds twice, which is what a Llama-style norm does, so it
stays. The multimodal M-RoPE tables were float32 and promoted a bf16 layer to float32; they now take
the queries' type. The Granite Speech, Qwen3-VL, Voxtral and Mistral decoders run through the same
attention and feed-forward.

- `NFKMLXRoPEScaling` — the rotary frequency scaling a release declares (`rope_scaling`), shared by the
  dense decoder and DeepSeek. At reference parity against `transformers`' own `ROPE_INIT_FUNCTIONS`
  for `linear`, `yarn`, `llama3`, and `longrope` across ten configurations (worst relative frequency
  difference < 1e-5), driven by `run_reference.py rope_scaling`, which needs no weights: the scaling is
  a function of the rotary geometry and the config alone. `dynamic` is not implemented and is refused
  rather than approximated; it appears in released configs, computes different frequencies, and
  loading it under the wrong rotary runs and is wrong.
  `longrope` (Phi-3, Phi-4) carries two per-pair factor tables, `short_factor` for a sequence within the
  trained window and `long_factor` past it, chosen by length rather than blended, and multiplies the
  rotated queries and keys by `sqrt(1 + ln(max / original) / ln(original))` at every length.
  `NFKLMRotary` precomputes both period tables and switches when `offset + length` passes the window.
  Two details decide it on a real release. Phi keeps `original_max_position_embeddings` at the config's
  top level, outside `rope_scaling`, so the loader lifts it from there. And under a partial rotary the
  factor scales only the rotated channels, since the reference carries it on the cosines and sines.
  YaRN's blend runs the opposite way to the intuitive guess, and the first draft here had it
  backwards: the fast channels are left unscaled and the slow ones are interpolated. A fast channel
  completes many turns inside the trained window, so it encodes local offset and a longer sequence does
  not change its meaning; a slow channel does not complete a turn even at the trained length, so past
  that length it reaches angles the model never saw. The parity record caught the error — in the prose
  and the assertions, not in the arithmetic, because the formula was ported rather than reasoned out.
  The attention factor is `0.1·ln(factor) + 1` unless the config states one, and it multiplies the
  queries and the keys alike, so a score carries its square.
- `NFKMLXLanguage` / `NFKMLXLanguageBackend` — on-device **text generation** through MLX, which the
  package had no path for: the core runs a Core ML language model and the Foundation Models companion
  wraps Apple's, and nothing here ran a Qwen or Llama. `NFKMLXLanguageNet` is the modern dense decoder
  — grouped-query attention with rotary embeddings, a SwiGLU feed-forward, RMS normalization
  throughout — which is what Qwen3 and Llama both are; they differ in a **configuration**, not in
  structure (`normalizesQueryAndKey` is Qwen3's per-head query/key norm, `attentionBias` is Qwen2's,
  `tiesWordEmbeddings` is the smaller sizes'). Module keys are the released checkpoint's names
  (`model.layers.N.self_attn.q_proj`), so a release loads with no remapping at all, and every weight is
  at most 2-D so none of the convolution transposes apply. `NFKMLXKeyValueCache` is what makes a token
  cost one step's work instead of the whole sequence's; `testACachedStepMatchesRecomputingThePrefix`
  is the assertion the generation path rests on. It holds its rows in a buffer that grows in blocks
  with a cursor at each end, rather than concatenating — concatenating copies the whole cache on every
  token, which turns decoding back into quadratic work in the one place that exists to avoid it.
  A `window` bounds it (`NFKMLXGenerationOptions.contextWindow`): the oldest positions are dropped,
  so memory stops growing with the conversation. The trim goes to `window - 1` Before the append, which
  is what lets a single-token step read exactly `window` positions and need no sliding mask at all.
  `offset` stays the absolute position count — a rotary angle depends on where a token is in the
  sequence, not where it sits in the buffer — while `maskCacheLength` is what a multi-token pass
  actually sees, and the mask is built against that: a mask sized to the offset would be wider than the
  keys it is applied to. The bound is off by default, because for a model whose attention is not
  natively windowed it is an approximation rather than a configuration — exact while the conversation
  fits inside the window, and dropping the beginning past that.
  The cache also quantizes (`NFKMLXGenerationOptions.cacheQuantization`, `NFKMLXKeyValueCache.Quantization`):
  keys and values are stored affine-packed (`quantized`/`dequantized`) beside per-group scales in the
  same block-growing buffers, and a step dequantizes the retained span for attention. It changes
  storage, not positions, so the offset/window/mask accounting is untouched — the float path is byte
  for byte the same, gated behind `guard let quantization`. It is lossy, so off by default; measured,
  8-bit decoding tracks the full-precision logits while shrinking the resident cache, which is what
  lets a long conversation reach further before the cache is the ceiling. `groupSize` must divide the
  head dimension. Measured on the released Qwen3-0.6B against the reference record
  (`testTheCacheQuantizationBitWidthsAgainstTheQwen3Record`, a token-by-token decode through the packed
  cache so every key and value is read back packed): 8-bit/64 last-logit cosine 0.99956, same argmax,
  the greedy continuation 16/16 with the reference — and **4-bit per-token collapses**: 0.577 at group
  64, 0.928 at group 32, a different argmax, 0/16 of the continuation. The axis is the cause, not the
  bit width (`testTheKeyValueQuantizationAxisDiagnostic`, 4-bit relative reconstruction error over 28
  layers and 132 positions): the keys lose 0.133 per token at g64 against 0.045 per channel (grouped
  along the sequence, the KIVI axis; worst layer 0.155 → 0.043), while the values barely care (0.101 →
  0.088). Qwen3's keys carry per-channel outliers that a per-token group has to span. So the cache
  now stores keys per channel by default (`Quantization.keyAxis`, `.sequence`; `.headDimension` is
  the old per-position layout; ObjC `cacheQuantizationPerChannelKeys`, default true): each channel's
  keys over `groupSize` consecutive positions share one scale, packed as `[B, H, D, groups · words]`
  in a buffer that grows along the group axis, with the positions of the unfinished group held in a
  full-precision residual `[B, H, r, D]` — so a prompt shorter than a group costs no precision at all.
  A window drops whole groups by moving the group cursor and a partial group by a `skip` count on the
  first retained one; a rollback returns to the residual first, then whole groups, and a group cut
  part way is dequantized back into the residual (lossy by construction, held to a cosine in the tests).
  Export/restore carry `key_groups` / `key_group_scales` / `key_group_biases` / `key_residual` /
  `key_skip`, and the prompt cache's metadata gains a `:sequence` suffix, so an older per-position
  file still loads as what it was. Values stay per position. Measured end to end on the 132-token
  prompt (`testThePerChannelKeyCacheOnQwen3`, one prefill then a 24-token greedy continuation
  against the float cache): 8-bit per-channel keys reproduce the float cache — last-logit cosine
  0.99997, continuation 24/24 — where 8-bit per-position keys read 0.994 and 1/24; 4-bit per-channel
  reads 0.994 at group 64 and 0.997 at group 32 against 0.68 / 0.97 per position (continuations 9/24
  and 1/24 — a greedy continuation compounds a near-tie flip, so the cosine is the stable reading).
  So 8-bit is now near-lossless and 4-bit per-channel is usable where 4-bit per-position was not.
  MLX packs groups of 32, 64, or 128 only — a group of 16 aborts the process at the first append, and
  the truncated xcodebuild run still printed "0 failures" (read the exit code). **Prefill chunks** (`prefillChunkSize`): a long prompt runs through
  the cache in slices, so the attention peak is bounded by the chunk rather than the prompt. It is
  Exact — each chunk attends through the cache to exactly the keys a single pass would — pinned by
  `testChunkedPrefillMatchesASinglePass`. The backend applies a chat template on request
  (`chatTemplate: .chatML`): an instruct release is trained on `<|im_start|>role … <|im_end|>` turns,
  and the old message-flattening prompted it outside that format; `.chatML` renders the turns with the
  release's own special tokens (which the byte-level tokenizer resolves), off by default because a base
  model wants the plain text. **A fit-before-load predicate** (`NFKMLXReleaseWeights.verifyFits`)
  refuses a release whose weights exceed the memory budget before materializing any, so a load that
  would kill the process becomes an error naming the shortfall; it is wired into the dense loader,
  where weights are ~all the file (Gemma and the hybrid load a subtree, so the file size over-counts,
  and they are left to a model-aware check).
  `NFKMLXWeights.apply` gained an opt-in `verifyShapes` (default off): a parameter supplied at a
  shape the module does not expect is normally adopted wholesale by `update(parameters:)` — the
  checkpoint loads clean and computes wrong numbers. Checking shapes turns that into a load-time error,
  but only where every built shape already equals the checkpoint's, which is the dense decoder alone.
  Shape adoption is load-bearing in several builders — Conv-TasNet's placeholder `.base` widths and
  Gemma E4B's feed-forward-doubling heuristic both load right only because adoption reshapes the module
  to the checkpoint — so a global shape check false-positives (both were caught turning it on), and it
  stays scoped to the Qwen3 dense loader where config.json widths are exact. Per-layer weight streaming
  (making a 27B that does not fit run, rather than fail cleanly) needs lazy module weights and is a
  larger separate change; the fit predicate delivers the clean-error half. All of these reach Objective-C: an ObjC consumer builds the LLM through the `@objc` `NFKMLXLanguage.backendWithDirectoryURL:error:` (reads config.json + tokenizer + shards) and sets every generation option per request through the `NFKMLXGenerationParameterKey` string constants (`contextWindow`, `cacheQuantizationBits`/`GroupSize`, `prefillChunkSize`, `chatTemplate`), the same mechanism the core `NFKParameter*` keys use — parity with the Swift `NFKMLXGenerationOptions` struct. The struct-taking factories stay Swift-only because `NFKMLXLanguageConfiguration` is Swift-only (the directory factory reads config.json instead), and the cache class stays Swift-only because it takes `MLXArray`; the config knob is what bridges, per the package's expose-what-bridges rule.
  Gemma 4 and the hybrid do not use this cache: they take no cache argument at all and run
  prefill-only, so the window reaches the dense decoder alone. DeepSeek has a cache of its own
  (`NFKMLXDeepSeekCache`) and takes it as an argument; it shares none of this one's policy knobs,
  because the mechanisms those knobs configure are ones that architecture supplies itself. Sampling is greedy at temperature 0, otherwise
  temperature with optional nucleus (`topP`) and a seed for repeatability. The backend reads
  `NFKInputPrompt` / `NFKInputMessages` → `NFKOutputText` and honors the core's temperature, top-p,
  max-tokens, and seed parameters. Reference parity against transformers' own `Qwen3ForCausalLM` on
  the released Qwen3-0.6B: prefill logit cosine 0.9999999999943 with the same argmax at every
  position, and greedy generation reproducing the reference's continuation **token for token** —
  which is what proves the cache and the rotary offsets, since a single forward pass does not exercise
  them. A tied release still ships `lm_head.weight`; in Qwen3-0.6B it is **byte-identical** to
  `model.embed_tokens.weight` (verified, not assumed), so the loader drops the duplicate.
  Qwen3.5 and 3.6 are not this architecture — they are `Qwen3_5ForConditionalGeneration`,
  multimodal, interleaving `linear_attention` layers with full attention every fourth layer and gating
  the attention output. `configuration(fromHuggingFace:)` rejects them, and any mixture-of-experts
  config, rather than loading their weights into a dense stack and producing fluent nonsense. The
  oracle needs transformers >= 4.51, which is newer than the vision oracles run under, so it has its
  own interpreter recorded in the manifest's `oracle_environments`.
  1.7B and 4B are at parity too (logit cosine 0.9999999999975 and 0.999999999987, each reproducing
  the reference's greedy continuation token for token), which is what shows the family scales by
  configuration. 14B and 32B are presets (`.qwen3_14B`: 5120 / 40 layers / 40 heads / 8 kv /
  head 128 / 17408, untied; `.qwen3_32B`: 5120 / 64 / 64 / 8 / 128 / 25600, untied), each held to its
  released headers by shape (443 / 707 tensors, 0 missing / mismatched / unaccounted), as are
  Qwen3-Embedding-4B and -8B (398 each). `.mistralSmall3` is the same reader on a different family:
  the `mistralai/Mistral-Small-3.1/3.2-24B` decoder at 5120 / 40 / 32 / 8 / head 128 / 32768, untied,
  rope base 1e9. Two things about it are worth naming. Its head width is STATED, not implied — 5120
  over 32 heads divides to 160 while the release sets 128, so the attention projections are 4096 wide
  and narrower than the residual, and a reader that infers the head width builds projections the
  checkpoint does not fit. And the release is multimodal, so its decoder sits under `text_config`
  while the top-level `architectures` names the WRAPPER (`Mistral3ForConditionalGeneration`):
  `configuration(fromHuggingFace:)` unwraps `text_config` before the causal-model guard sees it,
  which is the idiom every Gemma and DeepSeek reader here already uses. Held to its released headers
  by shape: 363 decoder tensors consumed, 0 missing / mismatched / unaccounted, with the 218-tensor
  vision tower and its 4-tensor connector named as dropped. The 24 billion parameters are 48 GB at
  the released bf16, above what a 32 GB machine holds, so the whole release is checked at a small
  configuration of the same shape. Its released weights are measured on its first four layers
  (`mistralai/Mistral-Small-24B-Instruct-2501` cut by `truncate.py`, `IK_VAL_MISTRAL_SMALL_CUT4`):
  float32 0.9999999999948546 over every state, bf16 at 0.0172 of the floor per layer, logits
  2.681e-05 against a floor of 2.640e-05. The float32 record is streamed from a bf16 load
  (`IK_PROBE_STREAM_F32=1`, 12.6 GB); the test peaks at about 20 GB. It is the text front end FLUX.2 [dev]
  conditions on; see `mlx-models-dit-generation.md`. Both are **sharded**: every release above 0.6B splits its weights across files with a
  `model.safetensors.index.json` naming which shard holds each tensor, so a loader reading only
  `model.safetensors` covers the smallest model and nothing else. 4B is the largest size this machine
  holds at float32 — about 16 GB of weights on each side, measured, with the oracle and the test run as
  separate processes.
  Prompt cache, speculative decoding, mixture of experts, constrained decoding (all 0.3.0,
  each measured). `NFKMLXKeyValueCache.rollback(by:)` moves the end cursors back and copies nothing;
  it returns false where a window has dropped what it would reach. `NFKMLXPromptCache` keeps the
  cache and its token ids between generations, `align(to:)` rolls back to the shared prefix (capped
  one short of the prompt so a token runs and produces logits), and `save(to:)`/`load(from:)` persist
  it — float or packed rows alike. `reusesPromptCache` makes the backend keep one; the backend
  serializes generation through a lock because two runs through one cache interleave their rows.
  **Speculative decoding** (`generate(prompt:options:draft:promptCache:report:onToken:)`,
  `backend(directoryURL:draftDirectoryURL:)`, ObjC `backendWithDirectoryURL:draftDirectoryURL:error:`,
  request key `draftTokens`) verifies `[next] + proposals` in one cached pass, keeps the leading
  agreeing run, rolls both caches back by the rejected count — through the prompt cache when one is
  present, or the KV cache is rolled back twice — and is greedy-exact by construction; above
  temperature 0 it is the standard rejection scheme. Measured on Qwen3-1.7B←0.6B at float32:
  token-identical, 73.5% acceptance, 1.01× wall clock — a 28-layer step here is launch-bound, so
  the draft costs nearly a target step. The bandwidth-bound case is measured too and does not pay
  (`testSpeculativeDecodingPaysOnABandwidthBoundTarget`, Qwen3-4B bf16 ← 0.6B bf16, warmed up, best of
  two): plain 26.3 tok/s, speculative 14.9 tok/s, **0.57×**, acceptance 0.435. At bf16 the two runs
  can part at a near-tie — at token 39 the target's own top two were the two divergent tokens, margin
  0.125 — because the batched verification pass and the single-token pass round differently; greedy
  exactness holds at float32 and up to that rounding at bf16, which the test asserts rather than
  assumes (an earlier 1.44× reading was a warm-up artifact: the plain run went first and paid the
  kernel compilation).
  **The routed feed-forward** (`NFKLMMixtureFeedForward`: a router `gate`, experts stacked as one
  `[E, out, in]` tensor per projection in `NFKLMSwitchLinear`, dispatched through `gatherMM` /
  `gatherQuantizedMM`; `NFKLMQuantizedSwitchLinear` conforms to `Quantized` so `save` records it and
  `matchStructure` rebuilds it) reads `qwen3_moe` (`num_experts`, `moe_intermediate_size`,
  `norm_topk_prob`; dense-interleaved layers refused) and `mixtral` (`num_local_experts`,
  `intermediate_size`, always renormalized; a sliding window refused). Softmax over all experts then
  renormalizing the selected is Mixtral's softmax over the selected, so one implementation serves
  both. `moduleKey(forRelease:)` maps `block_sparse_moe.experts.N.w1/w3/w2` onto
  `mlp.experts.N.gate_proj/up_proj/down_proj`, and `stackingExperts` stacks the per-expert tensors in
  index order. Reference parity against transformers' own Qwen3MoeForCausalLM and
  MixtralForCausalLM at tiny random configurations (`run_reference.py qwen3_moe` / `mixtral`,
  `IK_PARITY_QWEN3_MOE_TINY` / `IK_PARITY_MIXTRAL_TINY`): every hidden state exact layer by layer,
  logit cosine 0.99999999999999 / 0.9999999999999903, on the first numeric run. The released
  Qwen3-30B-A3B is accounted for by shape: `Tools/validation-assets/shapes.py` reads every shard's
  safetensors header by HTTP range request (config + 18,867 shapes, no weights), and
  `testEveryParameterMatchesTheReleasedQwen3MoeCheckpoint` consumes all 18,867 with 0 missing, 0
  mismatched, 0 unaccounted. The released sizes need quantized experts to fit 32 GB.
  Qwen2-MoE is read too (`qwen2_moe`: Qwen1.5-MoE-A2.7B, Qwen2-57B-A14B): the same routed
  feed-forward plus a shared expert every token runs — a dense SwiGLU of
  `shared_expert_intermediate_size` gated by `sigmoid(shared_expert_gate(x))`, summed with the routed
  output (`NFKLMMixtureFeedForward.sharedExpert` / `sharedExpertGate`, present only when the width is
  set so the other families' strict loads stay strict). Its releases leave `norm_topk_prob` False
  (the default the reader applies for this type) and carry query/key/value biases spelled `qkv_bias`,
  absent from the released config because true is its default — the reader now defaults the bias to
  true for `qwen2` and `qwen2_moe`, which also means a dense Qwen2 release loads where the old
  `attention_bias ?? false` default had refused its bias tensors. Reference parity against
  transformers' own Qwen2MoeForCausalLM at a tiny configuration (`run_reference.py qwen2_moe`,
  `IK_PARITY_QWEN2_MOE_TINY`): every hidden state exact, logit cosine 0.9999999999999813, first
  numeric run.
  gpt-oss is read too (`gpt_oss`: gpt-oss-20b / 120b, Apache-2.0), the fourth expert family and
  the one that needed new mechanisms rather than a configuration. Four differences from the other
  mixtures, each a flag on the shared dense decoder so the other families are byte-identical:
  alternating sliding-window and full attention (`slidingWindows`, per layer from `layer_types`;
  a sliding layer keeps every key in the cache and masks the ones further back than its window
  through a banded additive mask built from absolute positions, so a single-token step against a
  cache longer than the window is bounded too, and the cache accounting is unchanged); a learned
  attention sink per head (`sinks`, one extra softmax logit that drains mass and contributes no
  value — the fused kernel has no slot for it, so `explicitAttention` writes the softmax out, spreading
  the kv heads to the query heads as `repeat_kv` does; the same explicit path serves the sliding
  layers); biases on every attention projection and the output projection (`outputProjectionBias`)
  and **on the router** (`routerBias`) — its softmax over the selected top-k logits is the
  renormalized form the module already computes; and fused, interleaved, clamped experts
  (`NFKLMFusedSwitchGLU`, `clampedSwiGLU`): one `gate_up_proj` whose even columns gate and odd
  columns lift, read back with a stride-2 slice, biases on both projections (gathered per chosen
  expert), the gate clamped above at 7 and the up clamped to ±7, `(up + 1) · gate · sigmoid(1.702 ·
  gate)`. The release stores `gate_up_proj` as `[E, hidden, 2·width]` (`x @ W`), which
  `releaseWeights` transposes to the switch linear's `[E, out, in]`, keeping the interleave so a saved
  checkpoint round-trips through the same loader; `router.` maps to the module's `gate.`. Its YaRN
  leaves the correction band fractional (`truncate: false`), now a field of `NFKMLXRoPEScaling`
  (`truncatesCorrectionRange`, default true). Reference parity against transformers' own
  GptOssForCausalLM at a tiny configuration (`run_reference.py gpt_oss`, `IK_PARITY_GPT_OSS_TINY`,
  eager attention forced since the fused kernels take no sink, a window of 4 over 8 tokens so the
  sliding layers see less than the full ones): every hidden state exact, logit cosine
  0.9999999999999721, first numeric run.
  The released experts are MXFP4 and stay packed. `*_blocks` (`uint8 [E, out, in/32, 16]`) viewed
  as little-endian `uint32` Are MLX's `mxfp4` words in its own element order — measured two ways:
  `testMXFP4PackingIsTheOpenComputeLayout` hand-decodes MLX's packing (element i in bits 4·(i mod 8)
  of word i/8, the sixteen e2m1 values, an e8m0 scale byte per 32 biased by 127) and matches
  `dequantized` exactly, and `run_reference.py gpt_oss_quant` range-fetches the first 64 rows of the
  released layer-0 `gate_up_proj` and decodes them through transformers' own
  `convert_moe_packed_tensors`, which MLX's decode of the same bytes matches at **worst |difference|
  0.0**. So `releaseWeights` maps `_blocks` → `.weight` (viewed) and `_scales` → `.scales`, and
  `installPackedExperts` swaps each fused projection for an `NFKLMQuantizedSwitchLinear(packed:…, mode:
  .mxfp4)` (that class now carries a stored `mode` and a prepacked init) before the strict apply, so
  the packed arrays land on matching structure instead of being adopted into a float layer; the
  mxfp4 `gatherQuantizedMM` runs them as they are. A release whose `quantization_config.quant_method`
  is `mxfp4` loads at `.checkpoint` precision (bf16 attention, embeddings, and head; packed experts),
  which is also what makes `verifyFits` count the bytes that will be resident — 13.8 GB, where the
  float32 doubling would have refused it. The checkpoint contract records the mode
  (`inferkit.quantization` = `bits:groupSize[:mode]`; a module mixing affine layers with MXFP4 experts
  records the affine geometry, and `matchStructure` rebuilds only affine structure — the packed
  experts are recognized by their `uint8` scales on load, an affine save's float scales being the
  tell). `testAnMXFP4ExpertModuleRoundTripsThroughTheCheckpoint` saves and reloads a packed module
  to identical logits. The tokenizer is o200k_harmony, shipped as `tokenizer.json` alone: the core
  `NFKByteLevelBPETokenizer` gained the `o200k` pre-tokenization (words split by their case pattern —
  lower-led or one-capital-led, each optionally led by one non-letter and followed by a
  case-insensitive contraction; digits in runs of at most three; a punctuation run absorbing trailing
  newlines or slashes), `releaseTokenizer(inDirectory:)` picks it by the `\p{Lu}\p{Lt}` classes in the
  release's `Split` regex and extracts the vocabulary and merges when no `vocab.json` exists
  (`byteLevelFiles(fromTokenizerJSON:)`, the extraction ModernBERT's loader now shares). Token-exact
  against the `tokenizers` library over seven strings (`testTheReleaseTokenizerAgreesWithTokenizers`),
  the harmony markers included; eos is `<|return|>`. The released 20B is held to the module by shape
  from its own local shard headers (`testEveryParameterMatchesTheReleasedGPTOSSCheckpoint`, the
  fused projections against their `_blocks`/`_scales` geometry: 459 released tensors consumed, 0
  missing, 0 mismatched, 0 unaccounted) and generates through the ordinary backend
  (`testGPTOSSGeneratesOnTheReleasedWeights`, `IK_VAL_GPT_OSS`): "The capital of France is" → " Paris."
  in 4.6 s for 12 tokens, the 20B resident at 13.8 GB with its experts packed. A config key
  registered by hand while `fetch.py` is running is lost: it loads `~/.inferkit-validation.json` at
  start and rewrites it at the end, so register keys before or after a fetch, never during. Not ported: the harmony
  chat template's tool-calling structure (a raw prompt or a caller-rendered template works), and
  gpt-oss-120b, the same architecture at 65 GB.
  **Constrained decoding** (`NFKMLXConstrainedDecoding.swift`): `NFKMLXVocabulary` holds every id's
  bytes, read through the core's new `NFKTokenizer.bytesForTokenId:`; `NFKMLXByteConstraint<State>`
  walks a grammar byte by byte and caches the admissible mask per state (the uncached cost is the
  vocabulary times a few bytes; a run revisits a handful of states); `NFKMLXJSONConstraint` is JSON
  syntax with `root` (`.container`/`.object`/`.array`/`.any`), `NFKMLXChoiceConstraint` a fixed set.
  `NFKMLXJSONSchemaConstraint` (`NFKMLXJSONSchemaConstraint.swift`, 0.3.1) is the schema grammar:
  `NFKMLXJSONSchema` compiles a JSON Schema dictionary into nodes (`type` as a name or a list,
  `properties`/`required`/`additionalProperties`, `items`/`minItems`/`maxItems`, `enum`/`const` as
  byte-matched compact serializations, `anyOf`/`oneOf`, `$ref` into `$defs`/`definitions` with
  recursion, an empty schema or `true` as `any`), and the constraint walks it byte by byte over the same
  engine. The state is a set of deterministic machines, so an `anyOf` forks one machine per alternative
  the byte can open and the survivors rejoin as the bytes decide; an `any` value pushes a frame that
  delegates to the free JSON grammar. Keys are matched as raw bytes against the unwritten properties
  (a 64-bit seen mask, so keys come in any order and a duplicate is refused), an object closes only
  once every `required` key is written, a comma is refused once every key is written and unlisted ones
  are forbidden, and a key that outgrows every property becomes an unlisted one where
  `additionalProperties` allows it. `integer` refuses the dot and the exponent. Keywords that only
  narrow content (`pattern`, `format`, `minimum`, `minLength`) are ignored; ones that change what is
  admissible (`allOf`, `not`, `if`, `patternProperties`) are refused at compile time, as is a
  `required` name not under `properties`. Wired through the core's own `NFKParameterJSONSchema`
  (the key the remote backends read, so a structured-output request is engine-agnostic): the backend
  compiles it per request, throws on a schema it cannot enforce rather than running unconstrained, and
  hands the parsed document back under `NFKOutputStructured` beside the text whenever JSON was asked for
  (schema or `outputFormat`) — never guessed from JSON-looking text. Measured live on Qwen3-0.6B
  (`testASchemaConstrainedRequestOnQwen3ConformsAndReturnsStructuredOutput`): a `{city, country,
  population: integer, landlocked?}` schema comes back with exactly those keys and types. The free
  grammar's byte helpers (`advanceNumber`, `isTerminal`, `word`, `isWhitespace`, …) are module-internal
  statics so both grammars share one spelling of JSON's lexical rules. Request keys:
  `outputFormat` (`"json"`/`"json-object"`/`"json-array"`), `choices`, and the core `NFKParameterJSONSchema`. Two traps, both measured
  on Qwen3-0.6B: JSON admits unbounded whitespace, and with its preamble forbidden the greedy
  model emitted 96 tokens of blank lines — `maximumWhitespaceRun` (8 bytes) caps the detour; and
  a thinking model wants its `<think>` block, which the grammar forbids, and the leftover mass
  gave `{}` — the prompt closes the block (`<think>\n\n</think>\n\n` after the assistant marker),
  as the release's own no-think template does. **Two latent defects fixed on the way:** the release
  path passed no special tokens to the tokenizer (they live in `tokenizer_config.json`'s
  `added_tokens_decoder`, not `vocab.json`), so a ChatML marker encoded as plain text — now
  `specialTokens(inDirectory:)` supplies them and the `eos_token`; and no end-of-sequence stop was
  ever set, so generation ran to `maxTokens` — the release's eos is now the default stop when a
  request names none (a behavior change, recorded in the changelog). `NFKMLXLanguageBackend` is now
  `@objc(NFKMLXLanguageBackend)` with `hasDraftModel`, `promptCacheLength`, `resetPromptCache`.
  `Tools/reference-parity/run_reference.py` must stay parseable by Python 3.9: the LLM oracle
  environment is 3.9, and a backslash inside an f-string expression (legal from 3.12, written for
  the music oracle) had made every mode there unrunnable — found the first time the qwen3_moe mode
  ran, fixed by hoisting the literal.
- `NFKMLXHybridLanguage` — the hybrid decoder Qwen3.5, Qwen3.6, and **Qwen3.8** are built from
  (`Qwen3_5ForConditionalGeneration`), at reference parity on the released Qwen3.5-4B (logit
  cosine 0.9999999999962, every one of the 33 hidden states exact layer by layer). 4B is the smallest
  release of the family and the only one that fits here; Qwen3.8-27B is the same architecture at
  ~54 GB, so it is covered structurally (851 parameters, 0 mismatched) and its numerics rest on the
  4B measurement rather than on a run of its own.
  The per-layer isolation harness found three defects a shape check could not:
  the family normalizes with `x · (1 + w)` where the dense Qwen3 stack and Gemma 4 both scale by
  the weight directly — two conventions from the same vendor, indistinguishable by shape, different in
  every number (the gated norm inside the recurrence is the plain kind even here). The full-attention
  projection interleaves queries and gate per head: it is viewed as `[.., heads, 2·headDim]` and
  split on the last axis, so taking two contiguous halves of the flat width takes the wrong channels
  entirely. And the gate is applied as a plain `sigmoid`, despite the config field being named
  `output_gate_type: swish` — the implementation is what the weights were trained against. The
  recurrence itself also needed the query scaled by `1/√headDim` and the decay applied before reading
  the state rather than after.
  Three quarters of its layers replace attention with a **gated delta-rule recurrence** — a fixed-size
  state instead of a growing key-value cache, so cost is linear in sequence length — and every fourth
  layer is full attention whose output is gated. The shapes decode the design: `q_proj` is
  `[12288, 5120]` where 24 heads × 256 would be 6144, because the query projection also emits the
  output gate, which is applied as a plain sigmoid; `in_proj_qkv` is 10240 = two key streams of 16×128 plus a value stream of
  48×128; `A_log` and `dt_bias` are `[48]`, one decay and one step per value head; and
  `partial_rotary_factor` 0.25 turns only 64 of each head's 256 channels. Beside the 4B measurement,
  `NFKMLXHybridLanguageTests` checks every one of the 27B decoder's 851 parameters against that
  checkpoint's own safetensors headers, name by name and shape by shape — read with HTTP range
  requests, about a megabyte instead of 54 GB — with zero missing and zero mismatched, which is what
  carries the 4B result across to the size that cannot be run. The converse is asserted too, so the parts deliberately absent are named rather
  than overlooked: 333 tensors are the vision tower and 15 the multi-token-prediction head, and
  851 + 333 + 15 is the checkpoint's full 1199. A small configuration also runs end to end, and the
  recurrence is checked to be causal (appending tokens cannot change an earlier token's output).
  The one layout difference is the depthwise convolution: PyTorch stores `[channels, 1, kernel]` and
  MLX `[channels, kernel, 1]`, which the structural test compares as a loader would.
  Qwen3.5-2B and -9B run under this decoder as Open-Jev's base (`NFKMLXOpenJev`, in
  `mlx-models-embeddings-retrieval.md`), which measured the 2B release through a LoRA adapter at
  float32: all 25 hidden states at worst cosine 0.9999999999956228. Those releases tokenize with the
  core's `qwen35` pre-tokenization, which the text backend now selects from their regex. The recurrence
  runs token by token, so a 90-token prompt is 90 sequential steps per linear layer; that, not the
  parameter count, dominates a short prompt's cost and a fine-tune's graph.
- `NFKMLXQwen4Exp` — the Qwen4-Exp decoder (`Qwen4ExpForConditionalGeneration`), which
  **Qwen3.8-Flash-Next** is the released 180B instance of. It keeps the hybrid family's skeleton and
  adds four mechanisms nothing else in the package uses, so it is a port rather than a configuration
  of `NFKMLXHybridLanguage`.
  **Hyper-connections** replace the residual stream: the state is carried `hc_count` times over
  (4 x 2560 = 10240 wide), and each block reads a learned mixture of the streams and writes a learned
  share back to each. The usual pre-normalization is gone with it — the mixer's own grouped norm does
  that work, which is why a layer carries no `input_layernorm`. The read is
  `sigmoid(up(silu(down(x̂) / hc)))` against the normalized streams, averaged; the write is
  `2 · sigmoid(inject(x̂) / hc)` per stream. The residual added back is the RAW input, not the
  normalized one.
  **Per-layer embeddings over hashed n-grams** sit on one layer (`ple_layer_ids` [2], one-indexed).
  Each token's 2- and 3-gram hash into a table whose every head has a distinct PRIME vocabulary size,
  taken in order across the layers' heads from `ngram_vocab_size_base`; the multipliers come from a
  SplitMix64 stream seeded by the layer index. The release ships both as buffers and this module
  derives them, which the parity test holds to the shipped values and the structural test confirms on
  the release: sixteen primes above twenty million, padded to a multiple of 128, give the released
  table's 320,001,536 rows exactly. The hash refuses to read across an end-of-sequence token, so a
  segment's first tokens hash against the end token rather than the previous document. The features
  gate per stream by a sign-preserving square root of the key-query score, and a DILATED depthwise
  convolution (dilation = `ngram_size`, so each tap reaches one n-gram further back rather than one
  token) adds local context.
  **A query-sparse-attention indexer** fronts every full-attention layer. A small side network pools
  each complete run of `indexer_compress_ratio` keys into one block key positioned at the block's
  FIRST token, scores the blocks as a sum of rectified head dot products, and keeps the best
  `indexer_budget / compress_ratio`; the trailing tokens that fill no block are always kept. The
  **Ties among the rectified scores are arbitrary in the reference and cannot be matched by
  construction.** `relu` floors a block's score at exactly zero, so two blocks can tie, and
  `torch.topk`'s choice among equal values is a partial-sort artifact with no reproducible rule. This
  port selects by the sorted indices (`argSort`, which resolves a tie toward the lower block index),
  which is the same FAMILY as the reference's index-based selection and the reason the measured mask
  agrees — selecting by a threshold at the k-th value instead would keep every tied entry and diverge.
  The measured configuration does not depend on it: instrumenting the oracle's eleven `topk` calls
  shows seven zero scores and the k-th value landing on zero twice, but NO call where more entries
  tied the k-th value than there were slots, so tie-breaking never had to choose. At released scale it
  may. A tie means two blocks of identical zero relevance, so a divergence there selects a different
  irrelevant block rather than a wrong one, but it is unmeasured and should be read that way.
  result is a mask, so the attention itself is the family's ordinary gated one. The release's
  `config.json` calls these layers `full_attention`, and the reference rewrites every such entry to
  the sparse kind, because a layer that carries an indexer never attends to the whole prefix — the
  config reader does the same rather than building twelve layers that would quietly attend to
  everything.
  **512 experts** with ten active and a shared expert beside them, the experts stored as the bare
  `gate_up_proj` / `down_proj` parameters Qwen3-Next uses, with the gate in the first half of the
  fused rows rather than interleaved as gpt-oss's are.
  The released weights are 360 GB across 131 shards, so the arithmetic is measured at the size the
  oracle builds (`run_reference.py qwen4_exp`, no checkpoint): logit cosine 0.99999999999997, the
  same argmax at every position, and nine seams recorded so a divergence localizes to a mechanism —
  hashed n-gram rows exact, the indexer's selected-token mask exact, and the n-gram features, PLE
  output, hyper-connection read and write shares, recurrence, mixture, sparse attention, and all five
  hidden states at 0.99999999999990 or better. The one defect the seams found was the indexer's block
  count: MLX's `/` on integers promotes to float, so `(position + 1) / ratio` let a query see the
  block it sits halfway through, which cost the last layer and nothing before it. A floor division
  fixed it.
  The released 180B is covered structurally instead — 1163 decoder parameters against the checkpoint's
  own safetensors headers, read by HTTP range request, 0 missing and 0 mismatched and 0 unaccounted —
  and the converse is asserted too, so the parts deliberately absent are named: 333 tensors are the
  vision tower, 31 the multi-token-prediction head, 3 the derived hash buffers, and 128 the n-gram
  table's shards, whose rows are checked to sum to the one table this module looks up. The tower's
  tensor names and geometry are the Qwen3-VL ViT's with `deepstack_visual_indexes` empty, so the
  encoder this package already runs is the likely fit for it; nothing here has been wired to it or
  measured against it, which is what makes the tower out of scope rather than pending.
  Scope: the text decoder, prefill only, like the hybrid and DeepSeek beside it. **Customization is
  ruled out**, on the constraint rather than the design: the smallest release of this architecture is
  the 180B, whose n-gram table alone is 51 billion parameters, and no size of it exists that a
  consumer machine holds — there is nothing here to fine-tune. An adapter trained elsewhere merges
  and loads through the ordinary factory.
  The indexer's scores sum per-head products rectified at zero, so at the oracle's two heads a quarter
  of the query/block scores are exactly zero and tie, and which tied block `torch.topk` keeps is not a
  rule. A second record widens the indexer to eight heads (`IK_QWEN4_INDEXER_HEADS=8 run_reference.py
  qwen4_exp`, `IK_PARITY_QWEN4_EXP_HEADS8`, run under `qwenimagevenv`, the environment that reproduces
  the first record byte for byte): the selection matches exactly and the logits read
  0.9999999999999756. The oracle's hyper-connection shares differ by up to 0.36 across the three
  streams and the streams sit 73% apart after the first layer, so the record discriminates a share
  written to the wrong stream, the defect DeepSeek's near-identical copies hid.
- `NFKMLXDeepSeek` — the DeepSeek V4 decoder (`DeepseekV4ForCausalLM`): Multi-head Latent Attention
  over a mixture of experts, a third architecture family beside the dense stack and the hybrid.
  Its arithmetic is measured — at a tiny all-sliding configuration against transformers' own
  plain-PyTorch implementation, which shipped after this port was written and is the third-party
  oracle DeepSeek's GPU-only inference code could not be (`run_reference.py deepseek_v4`,
  `IK_PARITY_DEEPSEEK_TINY`): every layer's hidden state 1.0000000000 and the logits 0.9999999999999908,
  with the oracle saving its weights in the release naming so the module loads them strictly. With
  every layer sliding and the sequence shorter than the window, the reference degenerates to exactly
  the dense-with-sink path this port computes, so the measurement covers the MLA projections, the
  per-head query norm, the trailing interleaved rope, the sink softmax, the output de-rotation, the
  grouped output projection, both routers, the clamped SwiGLU experts, and the hyper-connections. The
  compressor and indexer stay outside it (no compressed window closes at that length). The released
  weights still cannot run here — the measurement is of the implementation, not the checkpoint.
  The measurement immediately found a wrong class the structural check could not: the head's
  collapse (`NFKDeepSeekHyperHead`) predicts only read gates, so `hc_head_fn` is `[copies, copies ×
  hidden]` — and the module had built the full block connection there, `[(2 + copies) × copies, …]`.
  The structural check compares declared shapes against the release and the declaration was right, so
  it passed while the module was wrong; MLX's `update(parameters:)` then adopts a checkpoint's shapes
  wholesale, so a real load would have crashed in the forward, not at load.
  The hybrid's checkpoint is bf16, so a float module's shapes match it exactly; this one is
  **quantized**: attention is fp8 with 128×128 block scales and a routed expert is 4-bit packed two
  to an int8 byte, so `w1` is stored `[2048, 2048]` where the float weight is `[2048, 4096]`. The
  structural check therefore derives what each float parameter looks like stored, and because that
  derivation is an assumption, `testTheQuantizedLayoutIsWhatTheReleaseUses` asserts it against the
  observed dtypes and shapes instead of trusting it.
  Implemented: low-rank queries (`wq_a` → norm → `wq_b`), one shared latent key-value per position
  (`wkv` → norm) which is what keeps the cache small, a grouped low-rank output (`wo_a` applied per
  group of heads, then `wo_b`), the learned per-head `attn_sink`, rotary on the head's trailing
  channels, and the mixture of experts — square-root-softplus scoring, a bias that steers selection
  without entering the weights, renormalize-then-scale by `routed_scaling_factor`, the clamped SwiGLU,
  and one shared expert every token passes through. The first `num_hash_layers` route by a
  `tid2eid` table indexed by token ID rather than by the hidden state, which changes which parameters
  those layers carry, so the boundary is asserted.
  The compressor and the sparse indexer are implemented. `NFKDeepSeekCompressor` pools
  `compressRatio` consecutive positions into one: each contributes a value (`wkv`) and a score
  (`wgate`), the scores softmax across the window so its positions compete, and `ape` is a learned
  per-slot bias so a position's weight depends on where it sits as well as on what it holds. At ratio
  4 the projections are twice as wide and a second, overlapping window is pooled alongside — shifted
  back by one window so a boundary is covered from both sides, with the first window's absent
  predecessor filled with zero values and −inf scores. `NFKDeepSeekIndexer` runs its own compressor,
  projects the layer's low-rank query into an index space, scores every compressed position, combines
  the heads by a learned weight, and keeps the best `index_topk` — masking any window not yet complete
  at the querying position, which the reference marks −1. The reference's Hadamard rotation is
  applied to both the query and the compressed keys before fp4 quantization; being orthogonal and
  shared it cancels in the dot product, so float32 skips it, and bf16 applies it because the release
  rounds the rotated values (`nfkDeepSeekHadamardRotated`, below).
  **V4 is measured against its release's OWN code, and that found the decoder unfinished.** Every V4
  release ships `inference/model.py`, and it imports the same six kernel symbols V4.1's does, so the
  same CPU shim stands it up (`run_reference.py deepseek_v4_release`, `deepseek_v4_pro_release`, and
  `_bf16` and `_decode` variants). V4 Flash and V4 Pro share one `model.py`; Pro 0813 has its own,
  which moves `hc_head` onto the block and adds a draft stack. The tiny configuration reaches what
  the transformers comparison could not: ratio 4 pooling overlapping groups with its own indexer
  keeping 3 of 5 groups, ratio 8 standing in for 128, a token-table layer, and YaRN at an original
  length of 16 under a 20-token sequence. Measured, the attention sat at cosine 0.37 to 0.70 in every
  layer, and six defects were behind it:
  - A V4 compressed layer never attended to its compressed positions. The 0.3.1 source says so ("not
    implemented here"); the compressor and indexer ran and their output was dropped. V4.1's path
    attended to them only under its shared-cache layout. The attention now reaches them for any
    layer that owns a compressor, and a layer without an indexer takes every group complete at the
    querying position, as `get_compress_topk_idxs` lists.
  - A V4 layer attended to every earlier position. It received the stack's causal mask, where the
    release windows every layer; the window is now applied inside the attention for all versions.
  - The rotary multiplied its channels by YaRN's attention factor (0.1·ln 16 + 1 ≈ 1.28 on every
    release), and the de-rotation multiplied again. The release's `precompute_freqs_cis`
    interpolates the frequencies and builds unit-magnitude rotations. No oracle had YaRN on before
    this one, so V4.1 carried it too.
  - `configuration(fromHuggingFace:)` never set `normalizesQueryHeads` or
    `compressedLayersRotateAtCompressedBase`, so a V4.1 release read from its directory normalized its
    query heads and rotated its compressed layers at the wrong base, unlike the `v41Flash` preset the
    parity tests build from. Both releases rotate a compressed layer at `compress_rope_theta` with
    YaRN and a sliding one at `rope_theta` without it.
  - The presets disagreed with their releases: Pro carried Flash's routing scale (1.5 for 2.5),
    output groups (8 for 16), index top-k (512 for 1024) and layout, and no preset carried YaRN.
    `testEveryPresetIsItsReleasesConfiguration` now compares every stored property of each preset
    with its release's `config.json`, read the way a load reads it, by reflection, so a field added
    later is covered without naming it.
  - The overlapping compressor had no decode state. The release parks two windows at ratio 4, pools
    the previous window's first half with the current one's second half and then shifts; the port
    parked one, added `ape` across a whole chunk (which broadcast only for a chunk of 1 or of the
    ratio), and the indexer's own compressor kept no cache at all. It now parks under
    `NFKMLXDeepSeekCache.indexerState(_:)`, so a snapshot copies it with the rest.
  With those, V4 and V4 Pro 0813 read logit cosine 0.9999999999999812 and 0.9999999999999851 with
  every layer 1.000000000000 on the reference's own input, and decode five steps past an 11-token
  prefill with every token the reference's; the transformers comparison is unchanged at
  0.9999999999999908.
  **V4 Pro 0813's draft stack ships, measured against its own code.** 0813's `model.py` carries a
  DSpark stack of its own, and its checkpoint holds three stages of 384 experts each: the stages
  route over the decoder's experts, which is what a config without `dspark_n_routed_experts` means.
  Its `config.json` says `num_nextn_predict_layers: 1`, which contradicts both the checkpoint and the
  release's own `inference/config.json` (`n_mtp_layers: 3`), so `configuration(fromHuggingFace:)`
  counts the `mtp.<n>` stages in a `model.safetensors.index.json` beside the config where there is
  one. Four things differ from V4.1's stack, each a property: the main states are the target layers'
  OUTPUTS (`draftReadsTargetLayerOutputs`; V4.1 reads the stream entering them), the Markov tables
  are named `markov_w1` and `markov_w2` (`draftMarkovTableNames`), the last stage collapses its copies
  through its own learned `hc_head` as the decoder does, and the stages are V4 blocks (their own read
  weight per sub-block, V4's query-head norm). Against `run_reference.py deepseek_v4_pro_dspark`, from
  this port's own decoder: main states, each stage's attention and block, the draft logits and the
  confidence at 0.99999999999992 or better in float32, the drafted block the reference's; in bf16 the
  18 float32 draft parameters are the constructor's, and the main states, stage 0's attention, both
  stages' block outputs and the heads are bit-exact. Stage 1's attention seam differs in 9 of 192
  elements (cosine 0.99999986), traced to one element of `wo_a`'s output: torch's value there is the
  exactly rounded one (it equals a float64 sum rounded once), and MLX's float32 sum of 64 products
  lands one bf16 step away; `wo_b` spreads it, and the block's output absorbs it. Along the way the
  rotary tables became correctly rounded (`torch.polar` builds them so; the GPU's float32 `cos`
  and `sin` are not), which did not move this seam but removes a real source of the same kind. The
  stack reloads from 0813's own names and a greedy speculative run is the plain run token for token.
  **The stored "V4 Flash" files are not the live V4 Flash.** `~/.inferkit-validation/deepseek-v4-flash`
  holds a DSpark-shaped release (72317 tensors, three draft stages of 256 experts, `dspark_*` fields
  in its config), while `deepseek-ai/DeepSeek-V4-Flash` today serves 69187 tensors with one classic
  MTP stage (`e_proj`, `h_proj`), no `dspark_*` fields, and a `model.py` that builds that MTP block.
  No repository or commit reachable on 2026-09-23 matches the stored files. The `v4Flash` preset and
  the structural test follow the stored files; the live Flash's MTP block is not built.
  **In bf16 V4 is bit-exact, and five rounding points differ from V4.1's.** V4's `RMSNorm` builds a
  float32 weight, so every norm weight is held float32 (`holdsNormWeightsInFloat32`, 100 float32
  parameters in Flash's record, 104 in Pro's, 0 disagreeing). The per-head query norm is written as
  torch computes `q *= rsqrt(q.square().mean(-1) + eps)` on bf16: the mean summed in float32 and
  rounded once, `eps` added at float precision, and `rsqrt` as torch's bf16 kernel computes it, a
  square root rounded to bf16 and a reciprocal rounded again (100% of 100,000 values; a correctly
  rounded `rsqrt` agrees on 75%). The learned copy-collapse returns the stream's dtype, which the
  release's `hc_head` does with `y.to(dtype)`. The indexer rotates its query and keys through a
  Hadamard transform in bf16, applies its head-weight scale at float precision, and sums the heads
  in float32 before rounding, as torch sums a bf16 tensor. The oracle supplies `fast_hadamard_transform`
  itself, a CUDA library with no third-party transcription (transformers omits the rotation): the
  library's own `hadamard_transform_ref` with the kernel's float32 accumulation and one rounding.
  Measured, 0 elements differ at any layer of either release, the logits read 1.0 and
  0.9999999999999999, the index scores are identical at prefill and at every decode step, and a
  configuration read from a V4 release now computes in bf16 as V4.1's does.
  Hyper-Connections are implemented, and finding them corrected a real mistake here. The `hc_*`
  parameters are not a hash-clustering head, which is what this file previously called them: the
  residual stream is `hc_mult` (4) parallel copies of the hidden state, so every block works on
  `[batch, length, 4, hidden]`. A block predicts its mixing weights per position from the copies
  themselves — `hc_*_fn` projects the flattened, RMS-normalized copies into a read weight per copy, a
  write weight per copy, and a copy-to-copy matrix that is softmaxed and then **Sinkhorn**-normalized
  toward doubly stochastic so the copies do not collapse into one another. The copies are reduced to
  one stream before attention and before the feed-forward, and expanded back after each; `hc_head_*`
  collapses them at the top.
  The structural check missed it entirely. It compared every parameter the
  port declares against the release and reported zero problems, because it only asked "does what I
  declare exist" and never "does everything in the release exist here". A whole mechanism sat in the
  checkpoint with no counterpart in the code. `testEveryReleasedTensorIsDeclaredOrNamed` now asserts
  the converse — 34223 declared, 38094 named as deliberately unimplemented, **0 unaccounted**, which is
  the checkpoint's full 72317. The Qwen3.8 and Gemma 4 checks always had that assertion; this one did
  not, which is exactly where the gap opened.
  The quantization the release is stored in is decoded, and this part is measured.
  `NFKMLXDeepSeekQuantization` dequantizes both formats — fp8 `e4m3` with 128×128 block scales for
  attention and the shared experts, and `e2m1` 4-bit packed two to a byte with 32-value block scales
  along the last axis for a routed expert — and `NFKMLXDeepSeek.dequantized(_:shapes:)` turns a shard's
  arrays into the float parameters a module holds. The scales are `e8m0`, an exponent with no sign and
  no mantissa, so a scale is exactly a power of two. Both decodes are **exact** against a reference
  built from real checkpoint bytes fetched by HTTP range request (`run_reference.py deepseek_quant`):
  torch 2.13 is the first here with `float8_e4m3fn` and `float8_e8m0fnu` on the CPU, and since it has
  no CPU kernel for `float4_e2m1fn_x2` at all, the 4-bit side decodes through `ml_dtypes`, the same
  format from a different vendor. So the checkpoint's storage is measured even though its arithmetic
  cannot be.
  Two details are load-bearing and only one of them is measurable. The 4-bit blocks run along the
  last axis, which the checkpoint's own values corroborate: the reference quantizer clamps a block to
  ±6 and rounds its scale to the power of two that puts the block's largest magnitude in `(3, 6]`, so
  under the right grouping every block lands in that range and under a wrong one about 1% do not.
  The nibble order is not measurable — a byte's pair decodes to the same two values either way and
  both stay inside one block, so no statistic separates them — so it follows the format's own
  convention (low nibble first) and a test pins it against a hand-encoded byte, the same treatment the
  rotary convention gets.
  The draft stack is declared too, so `testEveryReleasedTensorIsDeclaredOrNamed` accounts for the
  stored index exactly with nothing left to name: 36599 declared, 35718 block scales each decoding a
  declared weight, **0 unaccounted**, the full 72317. Counting a scale as "unimplemented" was the
  weaker claim it used to make; it is now accounted for by the weight it decodes.
  The arithmetic measurement above post-dates the source audit below. When this port was written,
  DeepSeek's own `inference/model.py` was the only reference and it imports `sparse_attn` and the
  fp8/fp4 kernels from a GPU-only `tilelang` module — stubbing those would have made the oracle this
  port's own code, which proves nothing. The audit was therefore source-driven, applying the error
  classes the isolation harness exposed in the Qwen and Gemma decoders; the transformers oracle later
  confirmed its three findings numerically. Three were found and fixed by reading `model.py`:
  the rotary pairs adjacent channels (`view_as_complex`), where the other decoders here rotate
  halves — indistinguishable by shape, different in every value, so `NFKDeepSeekRotary` writes the
  convention out rather than selecting it with a flag; the attention output is DE-rotated on the
  way out, because the values share their latent with the keys, so the rotation has to be undone with
  the conjugate; and the learned per-head `attn_sink` was declared but never used — it is an extra
  logit that drains probability mass without contributing a value, which the fused attention call has
  nowhere to put, so the softmax is written out. Its normalization is the plain kind, which the port
  already had. Each is pinned by a test, since no measurement can catch them here. Verification enumerates the architecture
  **analytically** — 43 layers of 257 experts cannot be instantiated at float precision — and compares
  3975 parameters across the five layers whose headers were captured, with zero mismatches.
  DeepSeek V4 Pro (0813) gets the same treatment, and the enumeration generalized with no code
  change: 61 layers, hidden 7168, 384 experts, 128 heads, `q_lora_rank` 1536, `o_groups` 16 — 3540
  parameters compared across three captured layers with zero mismatches, and its 149782-tensor index
  accounted for exactly (75511 declared, 74271 block scales each decoding a declared weight, 0 named,
  0 unaccounted, the draft stack included). Pro adds a YaRN `rope_scaling`, which carries no parameters and so is
  invisible to a structural check — the reason a run of Pro without it would be silently wrong rather
  than a load failure. It is implemented now, through the shared `NFKMLXRoPEScaling`, and the config
  parser reads it. Sources:
  DeepSeek ships `inference/model.py` in the release, which is what this was written from.
  **V4.1 Flash** (`DeepseekV41ForCausalLM`, `deepseek_v41`, MIT, 763B) is the same architecture, and
  its parameters are enumerated and checked against its released headers
  (`NFKMLXDeepSeekConfiguration.v41Flash`, `IK_*_DEEPSEEK_V41`). Five differences show in the shapes.
  The compression ratios fall to 1 and 2, and only four layers own a compressor
  (`kv_source_layer_ids`) where V4 gave one to every compressed layer — the rest read that layer's
  compressed key-value, which is the point of the arrangement. A compressor at ratio 1 is a plain
  projection, so it ships no `wgate`, and the candidate source at layer 20 is exactly that layer;
  reading it the other way would declare a tensor the checkpoint does not have. The indexer derives
  its keys from the layer compressor's latent (`wk`, `k_norm`, on the four source layers) rather than
  compressing the hidden state a second time as V4's does, and it runs on eight layers rather than
  four. The learned collapse at the top is gone: there are no `hc_head_*` tensors, so the copies
  reduce by identity. And two layers carry an **n-gram memory** (`engram_layer_ids` [1, 14]) whose
  table holds 384 million fp8 rows of 256 channels each with a per-row block scale — 98 billion
  parameters a layer, which is where a quarter of the release's size lives. The hash is the same
  shape of idea as Qwen4-Exp's per-layer embedding: prime bucket counts per head, XOR-accumulated
  multiplied ids, a look-back that refuses to cross a boundary. It differs in what it hashes over —
  token ids first collapse through a NORMALIZED id space (`engram_compressed_vocab_size`, 99092),
  so " The", "the" and "THE" hash alike — and the gate is a sign-preserving square root of the
  normalized stream-against-key dot product, per hyper-connection copy.
  Measured: **48,496 declared parameters across all forty layers, 0 missing, 0 mismatched, 0
  unaccounted, 0 named as out of scope**, with 47,589 block scales decoding a declared weight. The
  two numbers sum to the 96,085 tensors the release's index holds. This is a complete capture rather
  than the shard subset V4 and V4 Pro were checked against, because `shapes.py` can now walk the
  release's 48 shards in about a minute (see `mlx-runtime-gotchas.md`). 763 billion parameters come
  to 510 GB on disk, because most of them are the four-bit-packed routed experts.
  **The oracle found a ninth difference that no structural check could.** V4 lets each
  hyper-connection collapse the copies with the read weight it predicts itself; V4.1 PIPELINES them,
  so attention reads with what the previous layer's feed-forward produced, the feed-forward reads
  with what this layer's attention produced, and the first block reads the first copy alone. The
  parameters are identical either way. Two more differ the same way: V4 normalizes each query head
  again after the up-projection and V4.1 rotates the projected heads directly, and a compressed
  layer builds its WHOLE rotary at `compress_rope_theta` with the extrapolation on — its queries,
  its window, its compressed positions and its indexer all read that one table, where an
  uncompressed layer builds one at `rope_theta` with the extrapolation off.
  **The decoder is built and measured end to end**: logit cosine **0.9999999999999821** at the
  oracle's size, against `run_reference.py deepseek_v41`. Every mechanism is measured on its own so
  that a divergence localizes: the hyper-connection coefficients (`pre`, `post` and the
  Sinkhorn-projected `comb`, each 0.999999999999997 or better), the compressor at BOTH ratios
  (0.9999999999999911 at ratio 2 and 0.9999999999999959 at ratio 1, with the layer's input matched
  first so a pooled latent that disagreed could not be confused with a block that handed it the
  wrong input), every attention layer including the four that read the shared compressed cache
  (0.99999999999996 or better), the mixture of experts (0.999999999999969), and the n-gram memory
  (1.0000000000000002). The candidate block mask and the indexer's chosen positions are compared as
  sets rather than as numbers, and both agree exactly: 0 of 256 block entries and 0 of 64 queries
  differ. V4 shares the block and measures 0.9999999999999908.
  **The residual mix summed over the wrong copy, and a nine-digit match hid it.** The release's
  `hc_post` gives copy j the sum over i of `comb[i, j]` times residual copy i. The port inserted the
  residual's new axis one place too far left, so it summed `comb[i, j]` times copy j: copy j scaled
  by `comb`'s column sum, which the Sinkhorn projection makes 1. The residual was never mixed. The
  oracle could not see it, because its copies never diverge: the stream starts as one copy
  repeated, every `post` sits near 1, and the four copies stayed within one bf16 unit of each other
  (a spread of 3e-4 at magnitude 0.18), where both sums give nearly the same number. The float32
  decoder read 0.9999999992 while every mechanism measured on its own read 0.99999999999996. That
  gap was the signal, and it went unread because 1e-9 looked like parity. bf16 exposed it: a 6e-4
  difference moves a bf16 rounding, so layers 1 to 5 disagreed on 10 to 60 of 1024 feed-forward
  inputs while every mechanism fed the reference's own input matched exactly, and feeding the
  reference's own mixing coefficients still left the same 60. The record now carries a probe
  (`probe.hc.*`) that hands the release's `hc_post` and `hc_pre` independently drawn copies (spread
  2.96), where the old sum is off by 3.0. `testTheHyperConnectionMixesDistinctCopiesAsTheReleaseDoes`
  holds both steps to it: exact in float32, and bit-identical in bf16. V4 and V4 Pro run the same
  block, and 0.3.0 and 0.3.1 shipped the defect. A seam test whose inputs cannot tell two
  orientations apart does not measure orientation. Hand it inputs whose symmetric part is not the
  whole of them.
  **The n-gram memory is the piece whose addressing no shape can check.** Every `(layer, n-gram
  size, head)` pair owns a prime-sized bucket range, drawn in order above `engram_vocab_size - 1`
  and never reused, so a layer's table is exactly as many rows as its primes sum to — which
  reproduces the released 384,006,168 and 384,016,682 exactly, and is what verifies the derivation
  without a checkpoint. The multipliers are drawn from `numpy.default_rng(10007 * layer_id)` and
  from no other stream, so `NFKDeepSeekNumPyGenerator` reproduces NumPy: `SeedSequence`'s entropy
  mixing, PCG64's seeding and XSL-RR output, and Lemire's bounded draw, which is what
  `Generator.integers` uses where the legacy `RandomState` uses a masked one and gives entirely
  different numbers. Ids collapse through a normalized space first (`engram_compressed_vocab_size`),
  so " The", "the" and "THE" hash alike; the release's own normalizer chain over its tokenizer
  gives exactly the 99,092 it states.
  **The DSpark draft stack is built and measured, loop included.** It is stored under `mtp.`: three
  stages, each a decoder layer that never compresses and routes over its own 128 experts.
  `NFKMLXDeepSeekDraftStack.propose(continuing:mainStates:through:)` proposes `dspark_block_size`
  tokens after a committed one, reading what the decoder already computed — the mean over the
  hyper-connection copies of the stream entering each `dspark_target_layer_ids` attention. Against
  `run_reference.py deepseek_v41_dspark`, reading THIS decoder's own states rather than the
  reference's recorded ones: `NFKMLXDeepSeekNet.draftStates(forTokens:)` reproduces what the
  reference hands the stack at 0.9999999999999925, and from those the draft attention measures
  0.9999999999999456 and 0.9999999999998992 at its two stages, the draft logits 0.999999999999992,
  the confidence 0.999999999999997, and the proposed block is exact token for token. The reference
  builds those states across a prefill and one decode step where a prefill-only port runs the whole
  committed sequence at once; the agreement is what says the two are the same arithmetic, every path
  into them being causal. The standalone heads are
  measured at inputs of their own as well, so a disagreement in the walk localizes to the walk:
  the main projection 0.999999999999972, the Markov embedding 0.9999999999999999 and its logit
  bias 0.9999999999999972, the confidence head 0.9999999999999908.
  Three things about the draft stack are easy to get wrong. Its expert count FALLS BACK to the
  decoder's where the release does not name its own, which is what `get_moe_config` does and what V4
  relies on; taking the absent key literally leaves V4's draft experts unaccounted. The release
  states `dspark_target_layer_ids` as the layers whose attention INPUT is read, not their output.
  And a drafted position attends to the whole drafted block INCLUDING the positions after it — the
  block is one parallel proposal, and the order among its tokens is imposed afterwards by the Markov
  head rather than by a causal mask. The reference splits the work over a prefill that seeds each
  stage's sliding-window ring and a step that drafts; the ring's contents are the most recent
  `window_size` main positions, which this port slices out of the main states instead of keeping a
  ring at all.
  **The collapsed token map is derived, and `makeNet` no longer refuses the release.**
  `NFKMLXDeepSeek.compressedTokens(fromTokenizerJSON:in:)` runs the reference's own normalizer chain
  over a release's `tokenizer.json` and reproduces its partition exactly: 129,280 ids collapse to
  99,092 buckets, with 0 ids sharing a bucket the reference keeps apart
  (`run_reference.py deepseek_v41_tokens --checkpoint <tokenizer.json>`). The size is checked against
  the `engram_compressed_vocab_size` the release states, because every hash multiplier is derived
  from that number: a collapse that drifts becomes a load-time error rather than a forward pass
  reading the wrong row of a table with 384 million of them.
  The record holds the reference's whole lookup rather than its size, and that is what made the
  derivation possible: two collapses can agree on how many buckets exist and still put the wrong ids
  together, and each of the four defects below passed a size check at some point.
  `StripAccents` drops SPACING and ENCLOSING marks as well as nonspacing ones, which is what
  collapses an Indic vowel sign onto the consonant beside it; reading it as nonspacing-only split 418
  groups. A token that is a fragment of a multi-byte character is keyed by the vocabulary's own
  SPELLING, and that spelling is meant to collide: byte `0xA1` spells `"¡"`, which the vocabulary
  also holds as a real token, so the two share a bucket and any private key for the fragment splits
  them. `String(data:encoding:.utf8)` strips a leading byte-order mark, which silently turns a token
  beginning U+FEFF into the token without it. `String.contains` matches grapheme clusters, so a
  replacement character followed by a combining mark hides the bare one the fragment branch tests
  for. And Swift compares and hashes strings under canonical equivalence, so two Arabic marks in
  either order are one key to a `[String: Int]`; the buckets are keyed by code points instead.
  **One cost is unexplained and left that way.** The derivation takes about ten seconds, and running
  it makes LATER tests in the same process slow — most of a minute for one that takes 0.054 s on its
  own. Skipping it returns them to normal, so it is the cause. Parsing the tokenizer once instead of
  three times and pooling the bridged temporaries each left it unchanged, so the mechanism is not
  simply how much it allocates. Against the full suite's 82 minutes it is under two per cent, which
  is why it is recorded rather than chased further.
  **The residual stream drifts where nothing is wrong, and the reason is worth knowing.** Every
  sublayer matches at 1e-13 and the logits at 1e-9, while the hyper-connected stream reads 1e-7 at
  the first n-gram layer and 3e-6 by the last. The engram's gate is
  `sigmoid(copysign(sqrt(clamp(|dot|, 1e-6)), dot))`, which is DISCONTINUOUS at `dot = 0`: a
  position whose dot product sits within float32 noise of zero lands on either side of the sign and
  its gate jumps by about 5e-4. The drift appears at the engram layers and nowhere else. It is a
  property of the reference's gate, not of the port.
  **The router's second bias is implemented, and a text-only parity run cannot see it.** A release
  with a vision tower carries `gate.bias_vl`, and a token inside an image span selects its experts
  with that bias instead of `gate.bias`. Every other measurement here routes through `gate.bias`, so
  a port that never built the second one matches all of them and still sends an image's tokens to
  the wrong experts. `run_reference.py deepseek_v41_vl_router` measures it on its own, at a
  configuration whose only reason for a vision tower is that the reference builds `bias_vl` nowhere
  else: the image bias reroutes 3 of 9 tokens, this port reroutes the same 3, and the experts they
  reach agree at 0.9999999999999827.
  **The fp8 block size is read, not assumed.** V4 blocks at 128 and V4.1 at 32, each stating it in
  `quantization_config.weight_block_size`. The dequantizer held 128 for every version, which decodes
  a V4.1 weight against a quarter of its scale grid silently; it now takes the release's own number,
  and a test holds that number to the scale shapes the release stores — 42 weights at 128 for V4, 25
  for V4 Pro, 355 at 32 for V4.1.
  **Customization is ruled out on the constraint, not the design.** The module exists now, but the
  release ships quantized — four-bit routed experts, fp8 attention and fp8 n-gram tables — so a
  gradient could not reach a weight without dequantizing 510 GB first, and no machine here holds
  that. An adapter trained elsewhere merges and loads through the ordinary factory.
  The storage is asserted rather than assumed, as V4's is, and V4.1 added a form that does not
  follow V4's rule: an attention weight carries one block scale per 32x32 block, while a routed
  expert and the n-gram table carry one per ROW per 32 columns. A dequantizer that took the square
  blocking for granted would read the wrong scale for every row of both. The n-gram table is the one
  place that matters at RUN time rather than at load: the reference dequantizes a row as it looks it
  up, so a port that holds the table dequantized has to fold the scale in when it loads.
  **The generation runtime is incremental, and the cache is the whole of it.** `NFKMLXDeepSeekCache`
  carries the five pieces of state a step cannot recompute: each layer's sliding window, the shared
  compressed key-value the four source layers publish, the index keys those same sources publish,
  the partial group a ratio-above-one compressor has pooled but not emitted, and the n-gram id
  history. Every one is indexed by ABSOLUTE position, which is what makes a decode step equal to the
  same position inside a longer prefill. `NFKMLXDeepSeekNet.generate(prompt:options:onToken:)` runs
  the prompt as one chunk and each token as a chunk of one; `NFKMLXDeepSeekBackend` (`@objc`) reads
  `NFKInputPrompt` / `NFKInputMessages` and honors temperature, top-p, max-tokens, seed, a JSON
  schema, and the release's own chat template. Three of the dense decoder's keys are deliberately
  absent, because the mechanisms they configure are ones this architecture supplies itself: a
  context window and a quantized key-value cache would be a second policy over state the model
  already bounds, and speculative verification at decode would mean rolling all five buffers back,
  which is unbuilt.
  Two measurements hold the runtime. `deepseek_v41_decode` records a prefill plus three steps with
  every buffer after each, and matches at every one (logits 0.99999999999998 a step, indexer scores within
  1e-9). Separately, generating through the cache produces the same tokens as running the whole
  growing sequence through the decoder each step, which is the property the cache exists to have.
  **Attend and carry are different things.** The cache keeps the last `slidingWindow` positions, but
  what a CHUNK may attend to is the ring plus the chunk itself: a query early in a multi-token prompt
  still needs keys the ring would have dropped. Trimming before returning removed a prompt's own
  first keys from its own first queries. The sliding rule then belongs to the mask, per query, on
  absolute positions.
  **One line of the reference is corrected, and correcting it halfway is worse than not at all.**
  `Indexer.forward` publishes `shared_attn.index_k` only inside `if self.owns_k and latent is not
  None`, while `_compress_kv` publishes its compressed key-value unconditionally. On a step where a
  ratio-2 compressor emits nothing, every ratio-2 layer therefore scores against whatever layer
  published last, at a different stride. The oracle corrects it, and the correction has to happen
  BEFORE the call: `Indexer.forward` reads `index_k` itself, so publishing on the way out repairs
  the followers and leaves the owner scoring its own step against stale keys. That half-correction
  read 3.5e-05 of score drift against a score scale of 2.5e-05 on exactly the no-emit step, and the
  kept positions came out shifted by one. Publishing first reads 5.5e-10.
  **These scores live at the noise floor**, every one of them between 1e-6 and 1e-5 at a test
  configuration, so a wrong score can still rank correctly by luck. Comparing which positions a
  layer KEEPS is therefore the weaker check; the test compares the scores.
  **A release directory loads through `NFKMLXDeepSeek.backend(directoryURL:)`** (`@objc`
  `deepSeekBackendWithDirectoryURL:error:`), held as `.automatic` plans it (the paging preset a
  residency comes to: `mlx-companion.md`, "Residency"), which reads `config.json`, `tokenizer.json`, derives
  the collapsed token map the n-gram memory addresses through, and decodes the weights a shard at a
  time. A key belongs to the decoder when `expectedParameters(for:)` declares it, which drops the
  vision tower and the aligner without a prefix list that would rot. The draft stack is the one
  exception and its prefix IS named: the enumeration covers `mtp.` because the structural check
  measures against the reference's whole module tree, while the decoder holds none of it, so the
  loader decoded a draft stack's worth of weights (about 50 GiB on V4.1) that `update(parameters:)`
  then dropped in silence. `NFKMLXNemotronH` and `NFKMLXQwen4Exp` both name the same prefix.
  The fit check
  is the architecture's own: the general one reads a directory's bytes and doubles them, which
  under-counts a block-quantized release by half, so this counts the parameters the configuration
  declares. V4.1 Flash decodes to **2.78 TiB** (2843.2 GiB) of float parameters, so it is refused on
  any machine here, and the refusal names what reaching it takes rather than implying a bigger
  machine would do. The enumeration still counts the draft stack, so the check is conservative by
  that 50 GiB; refusing a load that would have fit is the safe direction for a fit check.
  **The routed experts page.** `backend(directoryURL:paging:options:)`
  (`@objc` `deepSeekPagedBackendWithDirectoryURL:expertCacheBytes:error:`) holds each routed expert
  as the release stores it and decodes one when the router reaches it.
  `NFKMLXDeepSeekExpertStore` holds the bytes, and a paged `NFKDeepSeekMoE` holds no expert as a
  parameter, which is what makes the saving real instead of a second copy. The mixture groups a
  chunk's tokens by the expert they route to, so an expert decodes once for the chunk rather than
  once per token that reaches it, then applies it ONE TOKEN AT A TIME. The second half is
  deliberate: batching an expert's rows into one matrix multiply would be faster and would change
  the reduction that multiply performs, and keeping the resident path's arithmetic is what lets the
  test assert IDENTICAL logits rather than a cosine. What it costs is the contributions held until
  the accumulation reads them, `tokens x activatedExpertCount x hiddenSize` floats for the layer in
  hand. A bounded cache of decoded experts sits in front of the decode; its size is a policy choice
  and the hit rate it buys is UNMEASURED, because no machine here holds the release. At the oracle
  size the stored form is under a seventh of the decoded one, 156,672 bytes against 1,179,648.
  **Paging the experts is necessary and not sufficient, which is how the tables were found.**
  Holding the routed experts as the release stores them takes the decoder from 2843.2 GiB to
  **1087.1 GiB**, and a 512 GB machine is still short of that by more than a factor of two. What
  remained was not experts: the two n-gram tables are 384 million rows of 256 channels each, 732.4
  GiB of the 1087.1 GiB a load with only the experts paged still holds. The refusal names the
  largest parameter still held as floats for exactly this reason — the shortfall alone would have
  left a reader to work out which part is big.
  **The n-gram tables page by the row, which is what their scales are for.** Every other weight in
  this release carries one scale per SQUARE block; a table carries one per row per 32 channels, and
  that difference exists so a row can be decoded on its own. `NFKDeepSeekStoredTable` gathers the
  rows a lookup names out of the stored bytes and decodes only those, so the table costs a byte a
  channel instead of four and the decode is proportional to what is read rather than to what is
  held. A paged engram builds no float embedding at all; `NFKDeepSeekEngram.table` is absent and the
  stored table stands in its place. Nothing else here is paged: the attention weights and the shared
  expert are read by every token, so holding them stored would decode them as often as a resident
  load reads them.
  **The four figures, measured, and what they say about the machine.** V4.1 Flash declares 2843.2
  GiB of float parameters. Paging the routed experts alone gives 1087.1 GiB; paging the n-gram
  tables alone gives 2299.6 GiB; paging both gives **543.5 GiB**. A 512 GB Apple machine has 512
  GiB, because unified memory is counted in binary units, so the release does not fit fully paged
  either — it is over by 31.5 GiB before any activation, and `NFKMLXGPU.recommendedWorkingSetSize`
  is a fraction of physical memory rather than all of it. The enumeration also counts the draft
  stack, which the decoder never builds, so the figure a load would actually allocate is lower by
  that much; the fit check keeps the conservative number, because refusing a load that would have
  fit is the safe direction. Closing the remaining gap needs a third group held stored, and the
  candidates left are read by every token.
  **Mapping takes a held group to nothing held at all.** `NFKMLXDeepSeekPaging.mapsNgramTables`
  and `.mapsRoutedExperts` leave a group in the release and copy out the bytes a step reads, so the
  resident cost becomes the operating system's page cache. The measured progression on V4.1 Flash,
  as what the decoder allocates: 2843.2 GiB resident, 543.5 with both groups held stored (490.5
  once the draft stack it never builds is taken off), 301.7 with the tables mapped, and **32.7 GiB
  with every group mapped**. That last figure is the whole decoder minus its experts and tables,
  which is what a machine has to find room for once nothing else is held.
  **The mapping never becomes an `MLXArray`, and that is the point.** `MLXArray(rawPointer:…)`
  exists and would wrap the mapping, but a gather run as an MLX operation takes the whole tensor as
  its source and a source that large would be made resident to run it. The rows are copied out with
  `memcpy` into a few kilobytes and MLX only ever sees those, which is also why a mapped decode is
  BIT-IDENTICAL to a held one: the same bytes reach the same dequantizer.
  **The two groups map differently because their shards differ.** A mapped table means never
  reading its shard — a released table is 98 GB and `loadCheckpoint` cannot leave one tensor behind
  — so the loader refuses a shard that mixes a table with other declared parameters, which no real
  release produces because a tensor that size cannot share one. A mapped EXPERT still reads its
  shard, because the attention weights beside it are what the decoder is built from; expert shards
  are capped so a loader can stream them, so the peak is one shard and what changes is what is
  KEPT. A caller that maps the experts should RAISE `expertCacheBytes`: the cache now stands in
  front of a read rather than a decode.
  **Speculative decoding ships, and the cache is why it was hard.**
  `generate(prompt:draft:options:report:onToken:)` has the release's own draft stack propose
  `dsparkBlockSize` tokens, scores the block in one pass, and keeps the leading proposals that match
  this decoder's own argmax; the output is the same sequence plain decoding produces, token for
  token. Rejecting means putting the cache back, and this cache cannot be TRIMMED: the sliding
  window is a ring, so appending a block evicted the oldest positions and dropping the tail does not
  bring them back. `NFKMLXDeepSeekCache.Snapshot` captures all five buffers plus a sixth the draft
  stack needs — the target layers' states at every committed position, which a cached run has to
  accumulate because a chunk sees only its own. A round that rejects costs one extra forward over
  the accepted prefix, still fewer passes than stepping through it whenever two or more proposals
  are kept.
  **One buffer had to be copied, and finding out why took the direct test.** Every buffer a step
  carries is REPLACED on an append except the compressor's parked group, which is written a slot at
  a time straight into the array: `MLXArray` is a class whose subscript setter writes through. A
  snapshot of that reference was a view of the thing it was meant to preserve. The speculative test
  PASSED while this was broken — random draft weights accept nothing, so every round restored and
  committed a single token, and a tiny configuration's argmax survived the corruption. Only
  comparing a restored cache against one that never saw the rejected block caught it. The copy is
  evaluated inside the snapshot, because a lazy one would read that same buffer later and find it
  already overwritten.
  **The release's own activation rounding is an opt-in mode, default off.**
  `NFKMLXDeepSeekConfiguration.quantizesActivations` (Swift factory `quantizesActivations:`, `@objc`
  `NFKMLXDeepSeekLoadOptions.quantizesActivations` through `deepSeekBackendWithDirectoryURL:options:error:`)
  rounds where the release's `inference/model.py` rounds, in place so everything downstream reads
  the rounded value: the sliding-window key-value to fp8; the compressed latent to fp4 in groups of
  16 with an E4M3 scale; the indexer's keys and queries to fp4 in groups of 32; the draft stack's
  main-state and block key-values to fp8; the n-gram rows to bf16; and the input of every GEMM
  whose weight the release stores narrow, to fp8 in groups of 32. Every fp8 and indexer scale
  is the next power of two of the block maximum TIMES the reciprocal of the format's range; the
  compressed latent's alone DIVIDES by 6 and rounds the quotient to e4m3, so its values divide by a
  scale that is not a power of two. Off, the port is the unquantized model the default oracle holds
  it to; on, it is held to a second record, `deepseek_v41_quantized` and
  `deepseek_v41_dspark_quantized`, taken from the same code with the kernel shim's quantizers made
  real. In bf16, the default, it is bit-identical to that code (below); with `computesInFloat32` it
  is not.
  **The release rounds every narrow GEMM's input.** Its `linear` rounds the input of every fp8 or
  fp4 weight to fp8 in blocks of `fp8_block_size` before the GEMM: every `Linear` the release
  builds without an explicit dtype (fp8), and the routed experts (fp4). `wo_a` is applied through an
  einsum on its dequantized weight, so its input is not rounded. The oracle's
  `_deepseek_v41_serve_gemms` decides which weights are narrow from a probe model built with fp8 and
  fp4, rounds those weights, and wraps `reference.linear`; the port's `served` rounds the same
  inputs. A routed expert applies its routing weight BEFORE `w2`, as the reference does, because
  `w2`'s input is what is rounded, and scaling after the rounding is a different number.
  **A tie is the only disagreement, and the test tells it apart from a defect.** Measured, the
  decoder's logits agree at **0.9999945936** with the round trips and **0.99994450** for the same
  weights without them. Fed the reference's own input, every block agrees at 0.999999999999994 or
  better except layer 3, whose attention reads 0.9958: one query of 16 keeps a different compressed
  position, and the reference's own scores at that query's top-k boundary are EXACTLY equal (both
  0.0, from the relu), so the two top-k implementations break a genuine tie differently. The test re-derives
  every differing selection from the recorded scores (`seam.score.L`) and fails on any difference
  that is not an exact tie at the boundary. The draft stack's attention agrees at 1.0 against
  0.9991 and 0.9979 without the round trips, and its logits at 0.9999999999999946 against
  0.99999535, with the drafted block identical. The comparison WITHOUT the flag is what shows the
  mode is doing the work rather than both matching by accident.
  **MLX has no fp8 or fp4 type, so the rounding is spacing arithmetic.** The binade comes from the
  exponent bits through `view(dtype: .int32)`, never from `log2`, and MLX's `round` is
  `metal::rint`, which is ties-to-even. An e4m3 normal is (8+m)·2^(e−3) and an e2m1 normal
  (2+m)·2^(e−1), so an integer multiple of the spacing has the parity of its mantissa and
  ties-to-even on the integer IS ties-to-even on the format. `testTheGridRoundingIsTheFormatsOwn`
  pins it against torch's float8 and ml_dtypes' float4 on the ties that decide the mode.
  **V4.1's `fp8_block_size = 32` is one global for weight blocks AND activation blocks.** A test
  geometry that kept V4's 128 against 32-wide heads trapped on the first round trip: its weights
  were floats, so the block had never mattered until the activations were rounded by it. A
  quantizing configuration whose widths do not divide into blocks is refused at `makeNet`
  (`unroundableWidth(in:)`); the trap was a test building the decoder directly, past that check,
  and it took every other test's result with it.
  **Verification is greedy and says so.** Above temperature zero the run falls back to plain
  decoding: the rejection scheme that keeps a sampled run distributed as the target alone would
  needs the draft's own probability at each proposed token, and the Markov walk biases every
  position by the token chosen before it, so that probability is not the one the returned logits
  carry.
  **A V4.1 load computes in bf16 by default, and matches the release's own code bit for bit.**
  The release declares `"dtype": "bfloat16"` and its `model.py` sets `torch.set_default_dtype(
  torch.bfloat16)` unconditionally, so `configuration(fromHuggingFace:)` reads the declared dtype
  into `NFKMLXDeepSeekConfiguration.computesInBFloat16` and the `v41Flash` preset carries it too.
  Float32 is the opt-out (Swift factory `computesInFloat32:`, `@objc`
  `NFKMLXDeepSeekLoadOptions.computesInFloat32`). A V4 release declares bf16 as well and, measured
  against its own code, computes in it too (below). A bf16 V4.1 decoder holds parameters bf16 except the 51
  that the reference's constructor makes float32 under `set_dtype(bfloat16)`: the hyper-connection
  coefficients, the attention sink, both router biases, the head, and a pooling compressor's `wkv`
  and `wgate` (not its norm); the draft stack adds its confidence projection and its Markov head
  (18 in all), and the image tower its norms (7). `heldInFloat32` is the rule, and three tests
  compare it with the `dtype::` entries each bf16 record writes from the reference's own
  `state_dict`: 0 disagree. The Markov head was the one the rule missed: the release builds it as a
  `ParallelHead`, whose weight is float32 and whose input is widened, so its bias is float32 too. Activations
  are bf16 between the places the reference computes in float32: the norms, the rotary, the
  attention's scores and softmax, the hyper-connection mixing, a pooling compressor, the router, the
  experts' activation and the mixture's sum, the n-gram gate, and the head. Each upcasts, computes,
  and casts back to the dtype it was handed, so every added cast is an identity in float32 and the
  float32 decoder is unchanged. What the V4.1 Flash decoder holds goes 1396.4 GiB resident, 475.5
  held stored, and 17.7 GiB fully mapped, against 32.7 in float32. The fit figures and the draft
  stack's share are costed by one rule (`loadedBytes`), which a bf16 fully mapped decoder first
  measured at -8.8 GiB when the draft stack alone still counted four bytes a parameter. Against `deepseek_v41_bf16_plain` (the release's code built under
  `set_dtype(bfloat16)`) every layer's stream is identical, 0 elements differing at all 7, and the
  logits read 0.9999999999999998, where the float32 port reads 0.99998715 against the same record.
  With `quantizesActivations` as well, against `deepseek_v41_bf16`: 0 differing and logits 1.0,
  where the float32 port reads 0.99995566 and picks a different token at 1 of 16 positions.
  **Every other path the default reaches is held the same way.** Each float32 oracle mode gained a
  bf16 twin built through one helper (`_deepseek_v41_bf16`), whose float32 record reproduces byte
  for byte after the switch: `deepseek_v41_decode_bf16`, `deepseek_v41_dspark_bf16`,
  `deepseek_v41_vision_bf16` and `deepseek_v41_vl_router_bf16`. Decode: every bf16 buffer (window,
  compressed cache, index keys) is identical at prefill and at each of three steps, and every token
  is the reference's. A pooling compressor's parked group is float32 in the release, so it is held
  to a relative gap (1.9e-7), and only its written slots are compared: at prefill an odd prompt
  leaves one slot unwritten, with a score of negative infinity and contents that never reach the
  pool. Draft stack: the main states, both stages' attention and the proposed block are identical
  once the copy average sums in float32 and rounds once, which is what torch's `mean` over a bf16
  tensor does (694 of 2112 differed before). Image router: the choices and the mixture identical.
  **The image tower's attention belongs to the backend, so the record pins the backend.**
  `vision.py` calls `F.scaled_dot_product_attention`. In bf16 on this machine's CPU, torch runs its
  flash kernel, which exponentiates through `fexp_u20` (a cubic polynomial on NEON, lanes of 8, a
  scalar `exp` for the last `N mod 4`) and rounds the softmax numerators to bf16 before the value
  product; a CUDA kernel differs again. Emulating that kernel exactly reproduced torch to 0 of 8960
  elements, which confirmed the difference is the backend's arithmetic and not the model's. The bf16
  record is therefore taken under `sdpa_kernel(SDPBackend.MATH)`, torch's own definition: widened
  to float32, the scale split as its square root on queries and keys, one rounding. The port's bf16
  attention does exactly that and still runs MLX's fused kernel, in float32. The record keeps the
  default backend's result beside it: the two torch backends differ in 11 of 384 aligned elements.
  The rest of the tower needed three rounding points: its norms (float32 weights, one rounding),
  its rotary (float32 tables, one rounding, where the port had cast the tables to bf16), and `silu` /
  `gelu`, which MLX composes from narrow operations that each round where torch rounds once. With
  those, features and aligned tokens are identical. The tower computes in the dtype its parameters
  hold, and `loadImageStack` sets that dtype from the decoder's configuration.
  **MLX's fused RMSNorm rounds twice in bf16.** Its kernel computes `w[i] * T(x * inv)`: it
  normalizes in float32, rounds to the input's dtype, then multiplies by the weight in that dtype
  and rounds again. The reference promotes the weight and rounds once,
  `(weight * x.float()).to(dtype)`. In float32 the two are the same arithmetic, so no float32
  measurement could see it; in bf16 they disagree on most elements by one step.
  `NFKDeepSeekRMSNorm` runs the fused kernel on float32 operands and casts back, which is the
  reference's single rounding. The signature that found it: the bf16 port sat FARTHER from the bf16
  reference than the float32 port did, by about √2 in 1 − cosine. Two independent roundings of equal
  size where the reference has one gives √2. torch's CPU bf16 GEMM was cleared first: it accumulates
  in float32 and rounds once, bit-identical to a float32 GEMM rounded afterwards.
  **A prompt prefills in chunks.** `prefill(_:embeddings:images:cache:chunkSize:)` feeds the prompt
  in slices through the same cache, which bounds the peak by the chunk rather than by the prompt;
  `NFKMLXGenerationParameterKey.prefillChunkSize` reaches it from a request. It was scoped as
  unbuildable because the reference has no multi-token-step form to measure it against, and it is
  measured against this port's own single pass instead, which is what the cached-generation check
  already does. Exact describes the mathematics and not the last bit: a chunk of five queries and
  one of thirteen reach the same answer through differently shaped matrix multiplies, and the
  engram's gate is discontinuous at a zero dot product, so agreement is held to 2e-3 with the chosen
  token held exactly. The state a boundary could lose is a ratio-2 compressor's parked group, which
  is why the test runs chunk sizes that divide neither the prompt nor the ratio.
  **A chunk below the compression ratio is a different computation, not a smaller one.** A chunk
  that finishes no compressor group emits no compressed position at all, and its queries then attend
  over a compressed cache that a single pass would already have filled. Measured on the oracle
  configuration, whose largest ratio is 2: chunk sizes 2 through 13 agree with a single pass to
  1e-6, and a chunk of ONE reads 0.31, diverging at the stream entering layer 3 — the first layer
  that owns a compressor is layer 2. Splitting a group across a boundary is fine, which is why chunk
  sizes 3, 5 and 12 all pass while ending in a one-token chunk; what fails is a FIRST chunk too
  short to fill a group. `prefill` therefore raises the chunk size to
  `minimumPrefillChunk`, the largest ratio among the layers that own a compressor, because a chunk
  size is a bound on memory and raising it is the reading that keeps the answer.
  **A picture reaches the decoder as embeddings plus a mask, and both are needed.**
  `hiddenStates` takes `embeddings:` in place of what `embed` would look up, and the token ids stay
  the release's image placeholder at those positions: the n-gram memory hashes over IDS and the
  router selects an image token's experts with `gate.bias_vl` from the MASK, neither of which an
  embedding carries. `NFKMLXDeepSeekImageStack` holds the tower, the aligner, the span's learned
  delimiters and the preprocessor, and `NFKMLXDeepSeekBackend` reads `NFKInputImage` where the
  release carries a tower. Only the prompt carries pictures, because every produced token embeds its
  own id and marks nothing; the reference requires a picture to lie in the first chunk because that
  is where it runs its tower, while this splices the spans into the whole prompt first and a chunk
  takes its slice, so the restriction does not arise and the equality held is against this port's
  own single pass.
  **The preprocessor was the unmeasured half of the image path.** `deepseek_v41_vision` starts from
  PATCHES, so the step that produces them — the step that decides the grid every later shape follows
  from — had nothing holding it. `deepseek_v41_image` measures it, and two details decide whether it
  agrees. PIL resamples an 8-bit picture in TWO passes and rounds to 8 bits BETWEEN them, in fixed
  point at 22 fractional bits; a float resample of the same coefficients, including the one
  `NFKMLXRFDetr` already carries, is a different picture by about one part in 255. And
  `ImageOps.contain` rounds with Python's `round`, which is round-half-to-even, where Swift's
  `rounded()` is half-away-from-zero: over a sweep of sizes at the release's own settings, 1,958
  land exactly on a half. Both were found by running the port's arithmetic in Python against the
  reference's before any of it was written in Swift — 208,208 sizes agree on the grid and the
  resample is byte-exact — and the resample runs on the CPU in integers, which is exact and avoids
  float64 on the GPU.
  **Two defects the loader found, both wrong only in the forward pass.** A companion scale is named
  for the weight it decodes, and `dequantized(_:shapes:)` derived that name by replacing `.weight`
  with `.scale`. For any parameter whose key contains no `.weight` — `hc_attn_fn`, the engram's
  `q_weight`, a router `bias` — the replacement changed nothing, the tensor found ITSELF as its
  scale, and it was decoded through the fp8 table against its own bytes. A load covering every
  parameter reported nothing. Separately, a hyper-connection's own `scale` is a parameter whose name
  ends the way a block scale's does; a release spells it `hc_attn_scale` and never collides, but a
  checkpoint written in the module's own layout spells it `hc_attn.scale`, and the decode skips
  every `.scale` as a companion. Both are told apart by what the module DECLARES, not by the suffix.
- `NFKMLXMamba` / `NFKMLXMamba2Net` / `NFKMLXMambaBackend` (`@objc`) — the Mamba-2 decoder
  (`Mamba2ForCausalLM`, Mistral's Codestral-Mamba), the toolkit's first state-space model. It is not
  the dense stack `NFKMLXLanguageNet` runs and not the gated delta-rule recurrence
  `NFKMLXHybridLanguage` runs: every layer replaces attention with a **selective state-space scan**
  (SSD), linear in sequence length, carrying a fixed-size `[heads, head_dim, state]` state rather than
  a growing key-value cache. The mixer is a fused input projection, a depthwise causal convolution over
  the concatenated `x`/`B`/`C`, the scan, a gated RMS normalization, and an output projection; `B` and
  `C` are per-group and broadcast to the heads. One selective-scan serves the whole SSM class, so
  Granite 4.0-H and Nemotron Nano 2 build their hybrid Mamba-attention layers on this mixer. Prefill
  only, like the hybrid decoder. **Reference parity** against transformers' own
  `Mamba2ForCausalLM` at a tiny random configuration (`run_reference.py mamba2`): logit cosine 1.0, and
  every hidden state exact (embedding 0.9999983, block 0 0.9999963, block 1 0.9999994, the final-normed
  state 0.9999999). **Released Codestral-Mamba-7B**: the checkpoint's 579 tensors all match by name and
  shape (`testEveryParameterMatchesTheReleasedCheckpoint`, 0 missing, 0 mismatched, 0 unaccounted), and
  at bfloat16 (7B does not fit float32 on a 32 GB machine, so both sides run bf16 as the Gemma E4B
  parity does, `run_reference.py mamba2_real`) the prompt logits cosine 0.9999146 with the greedy
  continuation matching 12/12 tokens. Four facts are load-bearing. **The module tree mirrors the
  checkpoint's nesting with REAL submodules** (`backbone.embeddings`, `backbone.layers.N`,
  `backbone.norm_f`); a dotted `@ModuleInfo` key such as `"backbone.embeddings"` flattens to the right
  NAME so a coverage check passes, yet MLX splits a parameter key on `.`, so `NFKMLXWeights.apply`'s
  unflattened update cannot route the weight into it and the net runs on random values (the embedding
  seam localizes it at once). The **gated normalization applies the gate before the norm**
  (`rmsnorm(y · silu(gate))`), matching HF's `MambaRMSNormGated` whatever `norm_before_gate` says.
  transformers' `output_hidden_states` **applies `norm_f` to its LAST entry** (the opposite of the
  ModernBERT convention), so the final recorded seam is post-norm. The released `config.json` writes
  `time_step_limit` as `[0.0, Infinity]`, and Foundation's `JSONSerialization` rejects the bare
  `Infinity` Python's `json` accepts, so `configuration(fromDirectory:)` replaces the non-finite
  literals with large finite values before parsing (the upper step limit never binds a softplus
  output). The tokenizer is `NFKMLXMistralTokenizer`, a metaspace byte-fallback BPE read from the
  release's `tokenizer.json`, token-exact against the `tokenizers` fast library; it differs from the
  Gemma reader in the leading-`▁` prepend on encode (`prepend_scheme: "first"`) and the leading-space
  strip on decode. `NFKMLXMamba.backend(directoryURL:)` (`@objc mambaBackendWithDirectoryURL:error:`)
  builds a prefill-only text-generation backend that prepends `<s>` and stops at `</s>`. **Customization
  is ruled out**, on the constraint the dense and hybrid decoders name: the smallest release of this
  architecture is 7B, which leaves no room on a consumer machine to fine-tune, so there is nothing here
  to train. An adapter trained elsewhere merges through `NFKMLXLoRA` onto the `in_proj`/`out_proj`
  Linear layers.
  The release sets `residual_in_fp32`, so transformers keeps the residual stream float32 between
  blocks at bf16 and rounds each block's norm input to bf16; `NFKMLXMamba2Configuration.residualInFloat32`
  models it, and the head rounds the float32 stream to its own type. The mixer, shared with Granite
  and Nemotron-H, sums its convolution in float32 and rounds once, forms `softplus(dt + dt_bias)` in
  the weights' type, and hands the float32 scan output to the gated norm, which multiplies by its
  weight in float32 and rounds once at `out_proj`, as the reference does. The first four blocks of the
  release, cut by `Tools/validation-assets/truncate.py` (1.4 GB), match transformers at float32 in every
  state, the worst at 0.9999999999993715 and the logits at 0.9999999999989264; the 7B whole never fit
  float32 here. At bf16 each block on the reference's own input reads at most 0.0011 of the
  reference's bf16-versus-float32 distance.
- `NFKMLXGraniteHybrid` / `NFKMLXGraniteHybridNet` / `NFKMLXGraniteBackend` (`@objc`) — the Granite 4.0-H
  decoder (`GraniteMoeHybridForCausalLM`, IBM, Apache-2.0), the first hybrid Mamba-attention decoder. Most
  layers are the Mamba-2 **selective-scan mixer reused verbatim** from `NFKMLXMamba2Mixer`, and every
  sixth is grouped-query attention with **no positional embedding** (NoPE) whose scale is Granite's
  `attention_multiplier` (1/128) rather than `1/√d`. The feed-forward is a gated-linear shared MLP
  (`input_linear` → chunk into gate and lift → `silu(gate) · lift` → `output_linear`); the MoE sizes
  (h-tiny, h-small) add a routed mixture of experts beside it, whose summed output forms the
  feed-forward. Four **scalar multipliers** scale the embedding (×12), each of the two residual adds
  (×0.22), the attention logits, and the output logits (÷6). The routed experts store their weights as
  the released `[experts, out, in]` stacks (`block_sparse_moe.input_linear` the fused gate+lift,
  `output_linear` the down projection, `router.layer` the scorer) and run through one gathered matrix
  multiplication over the chosen experts (`NFKLMSwitchLinear`, shared with the language-model MoE); the
  router's softmax is taken over the top-`k` raw logits. Prefill only, like the pure-Mamba and hybrid
  decoders. **Reference parity** against transformers' own `GraniteMoeHybridForCausalLM`. Tiny dense
  config (`run_reference.py granite_hybrid`, the four multipliers set to non-default values so a port that
  hard-codes them diverges): logit cosine 0.99999577, every hidden state ≥ 0.9999953 (embedding
  0.99999994, the final-normed state 0.99999565). Tiny MoE config (`run_reference.py granite_hybrid_moe`,
  8 experts routed 2 at a time): logit cosine 0.99999547, every hidden state ≥ 0.99999577. **Released
  granite-4.0-h-1b** (the dense hybrid, a single `model.safetensors`, tied word embeddings so no
  `lm_head`): the checkpoint's 466 tensors all match by name and shape (0 missing, 0 mismatched, 0
  unaccounted), and at float32 (1B fits) the worst block seam is 0.99999964, the final-normed state 1.0,
  the logit cosine 1.0, and the greedy continuation 12/12 tokens. Two facts are load-bearing. **The
  module tree mirrors the checkpoint with a real `model` submodule** (`NFKMLXGraniteModel` holding
  `embed_tokens`, `layers`, `norm`); a dotted `@ModuleInfo(key: "model.embed_tokens")` flattens to the
  right name yet MLX splits the key on `.`, so the weight never routes and the net runs on random values
  (the embedding seam localizes it, the same lesson as the Mamba and SigLIP2 ports). The released
  `config.json` writes `time_step_limit` with a bare `Infinity`, which `configuration(fromDirectory:)`
  sanitizes to a finite value before `JSONSerialization` parses it, as the Mamba loader does. The
  tokenizer is Granite's byte-level BPE (the GPT-2 family), read from the release's `tokenizer.json`
  through the shared release-tokenizer reader (`NFKMLXLanguage.releaseTokenizer`), token-exact against the
  `tokenizers` fast library; the release ships no `bos_token`, so the backend prepends none, matching the
  reference's raw-id generation. `NFKMLXGraniteHybrid.backend(directoryURL:)` (`@objc
  graniteBackendWithDirectoryURL:error:`) builds a prefill-only text-generation backend that stops at the
  end-of-sequence marker. **Customization ships** (the first on-device language-decoder fine-tune):
  `NFKMLXGraniteHybrid.fineTune` adapts the attention query and value projections with LoRA, the Mamba
  layers and everything else frozen, over `NFKMLXGraniteObjective` (causal language-model teacher
  forcing, the reference's `labels=` loss, measured within 1e-3: 4.854839 vs 4.8548384 on identical
  logits). The round trip goes through `network(weightsURL:configuration:)` and its single-file loader.
  Because Granite gives only a minority of layers attention, the adapters cover a sparse subset of the
  layer array, which `NFKMLXLoRA` reaches through a per-owner update fallback.
  The `.h1B` preset placed attention every sixth layer; the release places it at 5, 15, 25 and 35
  (`layer_types`), which the preset and the configuration's default now carry. At bf16 (the backend's
  default precision) the residual multiplier 0.22 multiplies in float32, as torch multiplies by a
  Python float; rounding it to bf16 first (0.2197) flipped half of every attention layer's elements. With
  that, eager attention, the single-rounding `silu`, and the mixer's placements, every layer of the
  released 1B on the reference's own bf16 input reads at most 0.04 of the reference's
  bf16-versus-float32 distance (previously up to 1.6). End to end, layer 5, the first attention layer,
  still reads four times that distance from float32 while its isolated run is exact: it amplifies the
  difference it receives, and the stream returns within the floor by the last layer.
- `NFKMLXNemotronH` / `NFKMLXNemotronHNet` / `NFKMLXNemotronBackend` (`@objc`) — the Nemotron-H decoder
  (`NemotronHForCausalLM`, NVIDIA, NVIDIA Open Model License) behind Nemotron Nano 2, the second hybrid
  Mamba-attention decoder. Its 56 layers interleave three mixers, one per block from a
  `hybrid_override_pattern` string: the Mamba-2 **selective-scan mixer reused verbatim** from
  `NFKMLXMamba2Mixer` (27 layers), a **ReLU-squared dense feed-forward** (`down_proj(relu(up_proj(x))²)`,
  25 layers), and grouped-query attention with **no positional embedding** (NoPE) at the standard `1/√d`
  scale (4 layers). Unlike Granite there are **no scalar multipliers**, each block carries a **single
  pre-norm** with a plain residual add (`residual + mixer(norm(x))`), the feed-forward is its own block
  rather than a per-block companion, and the output projection is untied (its own `lm_head`). The one
  numeric departure from the Codestral/Granite mixer: Nemotron's gated output norm is **grouped by
  `n_groups`** (HF `Zamba2RMSNormGated`, `group_size = intermediate / n_groups`), and the discretization
  step clamps its lower bound at `time_step_min`; `NFKMambaRMSNormGated` gained a `groups` parameter
  (`1` = the ungrouped Codestral/Granite path, byte-identical). Prefill only, like the pure-Mamba and
  Granite decoders. **Reference parity** against transformers' own `NemotronHForCausalLM` (transformers
  ≥ 5, the musicvenv oracle). Tiny config (`run_reference.py nemotron_h`, a Mamba/MLP/attention stack
  with `n_groups` 2 so the grouped gated norm is exercised): logit cosine 1.0, every hidden state ≥
  0.99999994 (embedding 1.0, the final-normed state 1.0; no multipliers to amplify a seam). **Released
  Nemotron-Nano-9B-v2** (56 layers = 27 Mamba + 25 ReLU-squared MLP + 4 NoPE attention): the checkpoint's
  341 tensors all match by name and shape (0 missing, 0 mismatched, 0 unaccounted). The released numeric
  run (`run_reference.py nemotron_h_real`, bf16 since the 9B does not fit float32 on a 32 GB machine) is
  available when disk permits the ~17.8 GB download; the tiny numeric and the full structural check carry
  the shipped parity otherwise. Two facts are load-bearing. **The released checkpoint uses the original
  `backbone.*` naming** while the module tree follows the transformers-integrated `model.*` naming (which
  the tiny oracle records), so the directory loader remaps `backbone.` → `model.` and drops any `mtp.*`
  multi-token-prediction tensors, as transformers does; the module tree mirrors the checkpoint with a
  real `model` submodule (a dotted `@ModuleInfo` key would flatten to the right name yet never route the
  weight — the Mamba/SigLIP2 lesson). The tokenizer is Nemotron's byte-level BPE (Split + ByteLevel
  pre-tokenizer), read from the release's `tokenizer.json` through the shared release-tokenizer reader
  (`NFKMLXLanguage.releaseTokenizer`), token-exact against the `tokenizers` fast library (end-of-sequence
  id 12); the release sets `add_bos_token` false, so the backend prepends none, matching the reference's
  raw-id generation. `NFKMLXNemotronH.backend(directoryURL:)` (`@objc nemotronBackendWithDirectoryURL:error:`)
  builds a prefill-only text-generation backend that stops at the end-of-sequence marker. **Customization
  ships**: `NFKMLXNemotronH.fineTune` adapts the attention query and value projections with LoRA (the
  blocks Nemotron gives attention, a sparse subset of the layer array reached through `NFKMLXLoRA`'s
  per-owner fallback), the Mamba layers, feed-forwards, and everything else frozen, over
  `NFKMLXNemotronObjective` (causal language-model teacher forcing, the reference's `labels=` loss,
  measured within 1e-3: 4.851059 vs 4.8510590 on identical logits). The round trip goes through
  `network(weightsURL:configuration:)` and its single-file loader.
  The released 9B, cut to its first fifteen layers by `truncate.py` so the cut ends on its first
  attention layer (6.1 GB), matches transformers at float32 in every state, the worst being the logits
  at 0.9999999999942345. At bf16 its attention runs the eager placement Granite's does, the Mamba
  layers run the shared mixer's, and every layer on the reference's own bf16 input reads at most
  0.0053 of the reference's bf16-versus-float32 distance. The release sets `residual_in_fp32` false,
  so its residual stays in the weights' type.
- `NFKMLXChatTemplateRenderer` (`NFKMLXChatTemplate.swift` / `…Engine.swift` / `…Expr.swift`) — a
  native Jinja renderer for the release's `chat_template`, so the language backend reproduces an
  instruct model's trained input instead of the hand-coded ChatML approximation. Rendering the template
  wrong silently changes the model's input, the same failure class as the `qwen2` pre-tokenization
  defect, so the faithful path is to render the release's own template. A compact interpreter for the
  subset chat templates use: text with `{{ }}` output and `{% %}` control (for / if / elif / else /
  set), the whitespace model transformers compiles a template with (`trim_blocks` + `lstrip_blocks` +
  the explicit `{%-`/`-%}` markers), and the expression language — attribute/index access, slicing
  (`messages[::-1]`), `namespace`, the `loop` variable, `is` tests, string methods, and the
  `tojson`/`trim` filters. Pure Foundation below no runtime at all (no MLX), so it and its tests run
  under `swift test`. Reference parity against transformers' own `apply_chat_template` over six
  cases (`Tools/reference-parity/generate_chat_templates.py`, config key `IK_CHAT_TEMPLATE_REF`): the
  Qwen3 template (namespaces, reversed slicing, `is` tests, the tool-call and tool-role branches),
  Llama-3 (`bos_token` + `| trim` precedence), and Gemma (`%`, `!=` on booleans, `set role`, the
  `raise_exception` guards), each rendered byte-for-byte. **Two whitespace traps, both measured:**
  `lstrip_blocks` strips a block tag's line indentation only when the tag begins a source line (a
  trailing whitespace run after content on the same line stays — the first draft stripped it
  unconditionally, which dropped a real space); and `| trim` binds tighter than `+`, so
  `a + b | trim + c` trims only `b`. Wired into the backend as `NFKMLXChatTemplate.jinja(template:
  bosToken:eosToken:)`; from ObjC a `chatTemplate` request parameter carrying Jinja delimiters
  (`{%`/`{{`) is rendered the same way (the associated-value enum case is Swift-only, per the parity
  rule, and the string parameter is the bridge). **One documented divergence:** `tojson` emits an
  object's keys sorted, where transformers emits insertion order — Foundation dictionaries do not
  preserve it, so a faithful whole-object serialization needs an ordered-map pipeline; it affects a
  tool schema's key order, not a plain or multi-turn chat. The `NFKJinja*` types (value, namespace,
  parser, evaluator) are internal.
