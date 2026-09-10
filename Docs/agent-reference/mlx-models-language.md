<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX models: language models

The dense decoder and its generation runtime, the hybrid decoder, DeepSeek, rotary scaling, chat templates.

- `NFKMLXRoPEScaling` — the rotary frequency scaling a release declares (`rope_scaling`), shared by the
  dense decoder and DeepSeek. At reference parity against `transformers`' own `ROPE_INIT_FUNCTIONS`
  for `linear` and `yarn` across five configurations (worst relative frequency difference < 1e-5),
  driven by `run_reference.py rope_scaling` — which needs no weights, since the scaling is a function
  of the rotary geometry and the config alone. A kind this does not implement (`dynamic`, `llama3`,
  `longrope`) is refused rather than approximated: all three appear in released configs, all compute
  different frequencies, and loading one under the wrong rotary runs and is wrong.
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
  Gemma 4, the hybrid, and DeepSeek do not use this cache: they take no cache argument at all and
  run prefill-only, so the window reaches the dense decoder alone. Sampling is greedy at temperature 0, otherwise
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
  Qwen3-Embedding-4B and -8B (398 each). Both are **sharded**: every release above 0.6B splits its weights across files with a
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
  `NFKMLXJSONSchemaConstraint` (`NFKMLXJSONSchemaConstraint.swift`, 0.4.0) is the schema grammar:
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
- `NFKMLXDeepSeek` — the DeepSeek V4 decoder (`DeepseekV4ForCausalLM`): Multi-head Latent Attention
  over a mixture of experts, a third architecture family beside the dense stack and the hybrid.
  Its arithmetic is measured — at a tiny all-sliding configuration against transformers' own
  plain-PyTorch implementation, which shipped after this port was written and is the third-party
  oracle DeepSeek's GPU-only inference code could not be (`run_reference.py deepseek_v4`,
  `IK_PARITY_DEEPSEEK_TINY`): every layer's hidden state ≥ 0.9999995 and the logits 0.9999999999,
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
  deliberately omitted: it is applied to both the query and the compressed keys before fp4
  quantization, and being orthogonal and shared it cancels in the dot product — it spreads information
  for quantization rather than changing the score, so omitting it and the quantization together gives
  the unquantized ranking the reference approximates. Prefill only; the incremental decode path keeps
  rolling state buffers a single forward pass never enters.
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
  Still not implemented, and named by `testEveryReleasedTensorIsDeclaredOrNamed` rather than merely
  absent: the multi-token-prediction and DSpark speculative-decoding stack, 4705 tensors. It serves
  speculative decoding, which needs a generation loop this port does not have, and no oracle here can
  run it, so it would be unmeasurable code serving an absent path. That test now accounts for the
  release exactly — 34223 declared, 33389 block scales each decoding a declared weight, 4705 named as
  unimplemented, **0 unaccounted**, which is the checkpoint's full 72317. Counting a scale as
  "unimplemented" was the weaker claim it used to make; it is now accounted for by the weight it
  decodes.
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
  accounted for exactly (71983 declared, 70790 block scales each decoding a declared weight, 7009
  MTP/DSpark, 0 unaccounted). Pro adds a YaRN `rope_scaling`, which carries no parameters and so is
  invisible to a structural check — the reason a run of Pro without it would be silently wrong rather
  than a load failure. It is implemented now, through the shared `NFKMLXRoPEScaling`, and the config
  parser reads it. Sources:
  DeepSeek ships `inference/model.py` in the release, which is what this was written from.
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
