<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX models: the Gemma families

Gemma 2, Gemma 3, Gemma 3n, and Gemma 4 (text, vision, audio, fusion).

- `NFKMLXGemma3` / `NFKMLXGemma3Net` / `NFKMLXGemma3Model` / `NFKMLXGemma3Backend` — the Gemma 3 line,
  end to end (`gemma3_text` for the 270M and 1B, the multimodal `gemma3` for the 4B and up), at
  reference parity on the released weights against transformers' own Gemma 3 for every size and
  every stage, on the first numeric run. The decoder is the Gemma 3 block the EmbeddingGemma encoder
  already ran, now shared (`NFKGemma3Block`/`NFKGemma3Attention`/`NFKGemma3Norm` in
  `NFKMLXGemma3.swift`; the encoder is the same blocks under a bidirectional mask): `(1 + w)` RMS norms,
  the sandwich block, five sliding-window layers to one full layer (`layer_types`, or derived from
  `sliding_window_pattern`), the local rotary base 10000 on the sliding layers and the global 1e6 on the
  full ones, the 4B's `rope_scaling {linear, factor 8}` applied to the full layers only (`MLXFast.RoPE`
  with `scale: 1/factor`), per-head QK norm before the rotary, `query_pre_attn_scalar^-0.5`, GeGLU
  `gelu_tanh`, a tied head, both soft-caps supported (none released; an attention soft-cap runs the
  softmax explicitly). Generation runs through a hybrid cache (`NFKMLXGemma3Cache`: an unbounded
  `NFKMLXKeyValueCache` for the full layers and one bounded to the window for the sliding ones, the
  reference's hybrid cache), with the masks built per kind from absolute positions
  (`NFKMLXGemma3Masks`: a full layer causal, a sliding layer causal and `q − k < window` against the
  `min(offset, window − 1)` retained positions; a single cached step needs no mask). Measured: 270M
  logit cosine 0.99999999999368 and 1B 0.99999999999821 (every hidden state exact, argmax 6/6, the
  greedy continuation through the cache 12/12 token for token with the reference's own cached
  decode); a tiny record (`run_reference.py gemma3_tiny`, 4-position window over sliding/sliding/full,
  scaling 8, soft-caps 50/30) pins the cached step-by-step decode against a teacher-forced pass (worst
  step 1 − cosine 6.7e-8) and the continuation 6/6. **The multimodal 4B**: `NFKMLXGemma3VisionNet` is
  SigLIP so400m at 896×896 (27 layers, 1152 wide, patch 14 → 4096 patches) built from the shared
  SigLIP encoder with the SigLIP-2 row-major position embedding; `NFKMLXGemma3MultimodalProjector`
  average-pools the 64×64 patch grid in 4×4 cells to 256 soft tokens (a reshape-mean, row-major), a
  Gemma `(1 + w)` norm, then `x · W` with `mm_input_projection_weight` `[1152, 2560]` (a matmul, not a
  Linear). The prompt is the release's Jinja `chat_template.jinja` through `NFKMLXChatTemplateRenderer`
  (token-exact against `apply_chat_template`), the image spelled `<start_of_image>` in the last user
  turn and expanded in the text before tokenizing to `\n\n<start_of_image>` + 256 ×
  `<image_soft_token>` + `<end_of_image>\n\n` — the processor's own order, and load-bearing: `user\n` +
  `\n\n` tokenizes as one id (109), so expanding after tokenizing reads a different sentence; the whole
  processor prompt is reproduced id for id. An image's soft tokens attend to each other **bidirectionally**
  (the reference's `token_type_ids` blockwise rule: full layers `causal OR same-block`, sliding layers
  `window AND (causal OR same-block)`; `blockIds(for:)` numbers each soft-token run), the decode steps
  are plain causal as the reference's are. 4B numbers: text-only logit cosine 0.99999999999735 with every one of the 35 hidden states at
  1.0000000000 and the cached continuation 12/12; `gemma3_vision_real` patch embeddings
  0.99999999999953, layer 0 0.99999999999904, tower output 0.99999999875, the 256 projected soft tokens
  0.99999999971; `gemma3_conditional_real` the processor's 276-id prompt reproduced exactly, every
  layer ≥ 0.99999995, **argmax 275/275** over the fused sequence, last-16 logits 0.99999999565, and the
  greedy continuation token for token; on the real validation photograph through the CoreGraphics
  processor the 4B answers "The main subject of the image is a **puppy**." `NFKMLXGemma3Backend`
  reads `NFKInputPrompt` / `NFKInputMessages` (+ `NFKInputImage` on the 4B: `CGImage`, `CVPixelBuffer`,
  or texture through the image bridge, resized by CoreGraphics — the documented approximation of PIL
  bilinear), honors temperature / top-p / max-tokens / seed, streams each token through a submitted
  job's `partialResult` and cancels between tokens; `NFKMLXGemmaLanguage.backend(directoryURL:)` routes
  a Gemma 3 config here. ObjC: `[NFKMLXGemma3 backendWithDirectoryURL:error:]`,
  `gemma3WithDirectoryURL:error:` + `answerForImage:question:error:` / `answerForQuestion:error:`.
  Four facts were load-bearing. (1) The released 4B is in the transformers **4.x key layout**
  (`language_model.model.*`, `vision_tower.vision_model.*`, `multi_modal_projector.*`, no `model.`
  prefix) while a 5.x-written one nests them under `model.`; `decoderName(of:)` / `visionName(of:)` /
  `projectorName(of:)` accept both and drop any `lm_head.weight`. (2) transformers 5 **flattened
  `SiglipVisionModel`** (no `vision_model` child), so the oracle loads the tower's state dict without
  that prefix. (3) The processor's `apply_chat_template(tokenize=False)` text already spells `<bos>`, so
  the oracle tokenizes it with `add_special_tokens=False` — the reference's own double-BOS quirk is not
  reproduced. (4) EmbeddingGemma's bidirectional window was wrong and unmeasured: the release states
  `sliding_window: 512` and the reference turns it into the exclusive bound `sliding_window // 2 + 1 = 257`
  on `|q − k|`; the encoder had used 512, invisible on the ~20-token parity query. Now
  `NFKMLXGemma3EncoderConfiguration.geometry` applies the rule, and `run_reference.py
  gemma3_bidirectional_tiny` (span 6 → bound 4 over 12 tokens) measures it: last hidden 0.99999999999967.
  `NFKMLXGemmaTokenizer` now matches `added_tokens` as literals before the merge (a `<…>` scan, HF's
  own rule), so a rendered template or a 256-soft-token run encodes to ids; `decode(_:skipSpecial:)`
  drops the markers. Oracles: `gemma3` (a release directory; the chat ids and six tokenizer probes
  ride along), `gemma3_tiny`, `gemma3_bidirectional_tiny`, `gemma3_vision_real` (the tower + projector
  loaded selectively, the plate through the release's own image processor), `gemma3_conditional_real`
  (the full model; the record keeps the argmax at every position and the logits of the last 16 —
  the whole `[276, 262208]` matrix is 290 MB), all under the gemma interpreter (which gained Pillow +
  torchvision). Weights: `unsloth/gemma-3-{270m,1b,4b}-it`, ungated mirrors of the gated `google/`
  releases (536 MB / 2 GB / 8.6 GB bf16; the 4B is ~17 GB at float32, which fits beside nothing else).
  Not ported: pan-and-scan (off in every release). The 12B / 27B are the same architecture at 24 / 54 GB
  bf16 and are held to their released headers by shape through the 4B's configuration reader —
  decoder, vision tower, and projector together, 1065 / 1247 tensors, 0 missing, 0 mismatched,
  0 unaccounted. Gemma 3n is `gemma3n`, a separate family, and is refused here — `NFKMLXGemma3n` runs it.
- `NFKMLXGemma3n` / `NFKMLXGemma3nNet` / `NFKMLXGemma3nAudioNet` / `NFKMLXGemma3nVisionNet` — Gemma 3n,
  tri-modal and end to end, at reference parity on the released E2B weights for every stage, each on
  its first numeric run. A distinct architecture from Gemma 3 and Gemma 4, sharing the family name and
  almost nothing else; four mechanisms none of the others carry, and each changes the forward pass.
  - **AltUp** (Alternating Updates): the residual stream is `altup_num_inputs` (4) parallel copies, so a
    layer works on `[copies, batch, length, hidden]`. A learned per-token map `predict`s every copy from
    the active one before the block runs and `correct`s them from its output after. The prediction
    coefficients are reshaped `[n, n]` and transposed before the multiply; the correction coefficients
    take a `+ 1` so an untrained map is the identity; and the router's input scale is the hidden size's
    Reciprocal, not its inverse square root. Closest relative here is DeepSeek's hyper-connections.
  - **LAuReL** (Learned Augmented Residual Layer): a rank-64 detour beside the attention residual,
    normalized and added, with the sum divided by `sqrt(2)`.
  - **Per-layer embeddings**: a second 262144-row embedding gives every layer its own 256-wide slice,
    gated into that layer's output and added to the inactive copies only. The per-layer vocabulary is
    Smaller than the token vocabulary (262144 against 262400) — the ids past it are the vision and audio
    tokens, which carry no per-layer embedding and read row zero.
  - **Activation sparsity**: the first ten layers zero everything in the feed-forward's gate below
    `mean + Phi⁻¹(0.95)·sd` of that token's own gate, the deviation being the population one. There is no
    `erfinv` in Foundation, so `NFKGemma3nStatistics.standardNormalQuantile` is Acklam's approximation
    refined once by Halley's method against `erfc`.
  Two smaller differences are equally load-bearing: attention runs at scale 1.0, not
  `1/sqrt(headDim)` — the query normalization stands in for it — and the values carry their own
  normalization with no weight (`with_scale: false`), so the checkpoint holds no tensor for it. The norm
  is `x_norm · w`, the plain scale, which is Gemma 4's convention rather than Gemma 3's `(1 + w)`.
  **Key-value sharing**: the last `num_kv_shared_layers` (10 of E2B's 30) compute no keys or values at
  all and reuse the last non-shared layer OF THEIR own kind — a sliding layer reuses a sliding
  layer's, a full layer a full layer's — so the module declares no `k_proj`/`v_proj`/`k_norm`/`v_norm`
  for them and the released checkpoint carries none. Because a donor hands its full-length keys to the
  layers below it, `NFKMLXGemma3nCache` keeps every layer's keys whole and enforces the window by the
  Mask; a donor trimmed to its own window would hand a shorter history than the reference does.
  **Measured**: tiny (every mechanism, both attention kinds, a shared tail of each kind) every layer
  1.0000000000, logit cosine 0.9999999999999682, cached greedy continuation 6/6 with worst step
  1 − cosine 5.0e-14; released E2B every one of the 30 hidden states 1.0000000000, logit cosine
  0.9999999999943566, argmax 6/6, cached continuation 11/11 token for token.
  **The audio encoder** (`NFKMLXGemma3nAudioNet`) is a Universal Speech Model Conformer, a different
  network from the Gemma 4 Conformer rather than a configuration of it: two strided 2-D convolutions
  under a **cumulative group normalization** (a frame is normalized by every frame up to and including
  itself, and the variance takes each frame's deviations from THAT frame's cumulative mean before
  summing over time — not the running variance it resembles), then 12 blocks of feed-forward /
  chunked attention / causal depthwise convolution / feed-forward. The time axis is padded on the right
  only (`kernel − 1`, JAX's reverse-causal), the frequency axis by one each side. The attention is
  chunked with a Transformer-XL relative-position shift, its queries scaled by `1/sqrt(headDim)` divided
  by `softplus(0)` times a learned per-dimension softplus, and the activation clamp
  (`gradient_clipping`) runs at inference, six times a block. Two bugs, both found by seam
  isolation rather than guessed at: the block mask's bounds are relative to the query's row, not to
  its position in the context (`k >= q` and `k <= q + past + future`) — the two coincide when the right
  context is zero, which the release sets, so only the tiny configuration exposed it; and the validity
  mask is never skipped, because a block's context is zero-padded at both ends and the reference marks
  those frames invalid by padding the mask itself with false. Running with no mask at all left the first
  and last blocks attending to zeros (0.9987) while every weight was right. **Measured**: released E2B
  front end 0.9999999999999402, first block 0.9999999999998025, encoded 0.9999999999999257; tiny (which
  is where a non-zero right context is exercised) 0.9999999999999707.
  **The vision tower** (`NFKMLXGemma3nVisionNet`) is **MobileNetV5-300M**, a convolutional encoder rather
  than the SigLIP transformer every other vision model here carries, and it reaches the release through
  `timm` rather than transformers. Four stages of edge residuals, universal inverted residuals, and
  multi-query attention over the feature map (many query heads sharing one key and one value head,
  with no positional embedding of any kind) feed a fusion adapter that joins the last two stages.
  Five facts are load-bearing: padding is TensorFlow's `SAME`, asymmetric at stride two (3×3 pads
  (0, 1), 5×5 pads (1, 2) — symmetric padding gives the same output size and a shifted picture);
  there is **no BatchNorm anywhere**, every normalization being an RMS norm over the channel axis
  with a weight and no running statistics (the checkpoint's `bn` names are legacy); the activation is
  the **tanh-approximate** GELU; the inverted residual's first depthwise convolution runs before the
  expansion and carries no activation, with the stride on the second where the block downsamples; and
  the fusion concatenates the coarse stage after the fine one, nearest-upsampled, an order the
  `[3840, 1920, 1, 1]` weight it feeds cannot reveal. Measured on the released weights, first numeric
  run: stem 0.9999999999996717, stages 0.9999999999955526 / 0.9999999999982407 / 0.9999999999571081 /
  0.999999999990077, fused grid 0.9999999999981896.
  **The fusion** splices in two stages, unlike the other multimodal models here: a placeholder id is
  first embedded hard through a small per-modality table indexed by an offset into the token vocabulary,
  and the tower's soft tokens then overwrite those positions. Doing only the hard pass gives a model
  that runs and ignores the picture. The vision grid is scaled by `sqrt(2048)` before its embedder reads
  it. A clip shorter than 188 soft tokens is padded with the audio modality's last id.
  Measured end to end on the released E2B: the prompt this port builds reproduces the release
  processor's 272 ids exactly, argmax **272/272** over the fused sequence, last-16 logit cosine
  0.9999999999785188, and the cached greedy continuation reproduces the reference's caption token for
  token. The prompt is built as text and tokenized once, which is load-bearing and is the same trap
  Gemma 3 hit here: the `\n` closing `user` and the `\n\n` opening the image run merge into one id
  (109), so a prompt assembled from separately encoded pieces reads a different sentence to the model.
  The audio front end (`NFKMLXGemma3nAudioFeatures`) pairs HTK's mel scale with no area
  normalization (`norm=None`, where every other filterbank in this package is Slaney-normalized —
  adding it shifts each band's log by a constant and scored −0.149), cuts frames one sample longer than
  the window so HTK's pre-emphasis has a predecessor for every sample it keeps, and runs the transform
  at twice the window's length (`fft_overdrive`). Measured 0.999999999999857.
  `NFKMLXGemma3nBackend` reads `NFKInputPrompt` / `NFKInputMessages` (+ `NFKInputImage` and
  `NFKInputAudio`), honors temperature / top-p / max-tokens / seed, and streams through a submitted
  job's `partialResult`. ObjC: `[NFKMLXGemma3n backendWithDirectoryURL:error:]`,
  `gemma3nWithDirectoryURL:error:` + `answerForImage:question:error:` / `answerForQuestion:error:`.
  The image processor resizes to 768×768 and scales to `0...1` and **nothing more** — the release's
  preprocessor file states an `image_mean` and an `image_std` and then sets `do_normalize` false, so a
  `-1...1` frame is a plausible-looking mistake. Weights: `unsloth/gemma-3n-E2B-it` (10 GB, an ungated
  mirror of the gated `google/`). E4B is measured too, at the precision it ships in: 35 layers,
  fifteen of them sharing keys and values, read by the same configuration reader, and 16 GB of bf16
  that doubles past this machine at float32, so both sides run bf16 (`IK_GEMMA_DTYPE=bfloat16`,
  `.checkpoint` here) — logit cosine 0.99989 with the argmax matching at 5 of 6 positions, the one flip
  at the prompt's flattest position where the reference's own margin is half a logit. Whether that is
  rounding or a defect was measured, not argued: the E2B, exact at float32, recorded the same way at
  bf16 reads the same 0.99989 (`IK_PARITY_GEMMA3N_E2B_BF16`), so that is the floor the E4B is held to
  (`testGemma3nE4BMatchesTheReferenceLogits`: cosine above 0.999, at most one flip). Its decoder is
  also held to the released headers by shape: 806 tensors consumed, 870 named as dropped (the towers
  and the k/v projections the release still ships for its sharing layers), 0 unaccounted. Not ported:
  the MatFormer nesting that slices E2B out of E4B, which is a checkpoint operation rather than a
  forward pass.
- `NFKMLXGemmaLanguage` — the Gemma 4 text decoder (`gemma4_text`), a fourth architecture family, at
  reference parity against transformers' own implementation on the released E2B weights (logit
  cosine 0.9999999999994, and every one of the 36 hidden states exact layer by layer). Measuring it
  required installing Python 3.12 beside the system 3.9, because Gemma 4 is in no transformers that
  runs on 3.9; `oracle_environments` records that interpreter.
  This is the model that proved a structural check is not a numeric one. Its 600 parameters matched
  the release by name and shape while the forward scored **0.0044**, and four corrections read from the
  reference moved it only to 0.48 — two of them making it worse. What resolved it was the **per-layer
  isolation harness** (`testGemma4LayerByLayerAgainstTheReference`): the oracle records the state
  entering the stack and the state each layer produced, so the first divergence is located rather than
  guessed. It put the fault at layer 0's `input_layernorm` while that layer's input was exact, and a
  sub-step probe inside layer 0 narrowed it further. Three real defects came out of it:
  Gemma 4 normalizes with a plain scale, `x · w` — the `x · (1 + w)` convention is **Gemma 3's**,
  and assuming the family inherited it is what broke the port. The feed-forward uses Gemma's own
  activation (`gelu_pytorch_tanh`), not the SwiGLU's silu, which surfaced only once attention was
  exact. And the full-attention layers use the `proportional` rotary: frequencies computed over the
  Whole head width with the first `partial_rotary_factor` of the pairs real and the rest **zeroed**,
  which is not the same as rotating a contiguous leading slice — with rotate-half a pair is
  `(i, i + width/2)`, so a quarter-turned 512-wide head turns channels 0…63 and 256…319. That one left
  every sliding layer exact and every full layer subtly wrong, which is exactly what the harness showed
  at layer 4.
  Its distinguishing feature is **per-layer input embeddings**: beside the ordinary token embedding, a
  second much wider one (`embed_tokens_per_layer`, 262144 × 8960 = 35 layers × 256) gives every layer
  its own slice, gated into the residual after the feed-forward. Normalization is Gemma's `x · (1 + w)`
  rather than `x · w`, so a Gemma checkpoint in a plain RMSNorm produces near-zero activations and
  looks like a broken model instead of a convention mismatch. Logits are soft-capped through `tanh`.
  Two things the config does not say, both read from the checkpoint. A full-attention layer runs
  `global_head_dim` 512 where a sliding one runs `head_dim` 256, so seven of E2B's 35 layers have
  doubled attention widths — using `head_dim` throughout mismatched 42 tensors. And the layers that
  share keys and values run a doubled feed-forward: E2B's first fifteen are 6144 wide and its last
  twenty — exactly `num_kv_shared_layers` — are 12288, which no config field states. Both were found by
  the structural check rather than by reading, which is the argument for running it before trusting a
  port.
  Scope: the text decoder. The release is tri-modal, carrying a vision tower (659 tensors) and an audio
  Conformer (752); a test names them so they are known rather than overlooked. The configuration guard
  accepts `gemma4_text` and `gemma4` and rejects everything else: `gemma4_unified_text` (the 12B) is a
  Different decoder architecture (`Gemma4Unified*` classes), so it is refused rather than loaded into
  this stack and made to produce fluent nonsense. The 26B-A4B mixture (`enable_moe_block`) is
  implemented — see the mixture entry below; the dense sizes carry the expert fields nulled and set
  the flag false, so the flag distinguishes them.
  The Gemma 4 mixture of experts (`NFKGemmaRouter` / `NFKGemmaExperts`, the 26B-A4B family) runs a
  routed branch beside every layer's dense feed-forward and sums the two — not a dense-FFN swap. The
  block computes `post_ff_norm(post_ff_norm_1(mlp) + post_ff_norm_2(experts(pre_ff_norm_2(residual))))`,
  where the router reads the pre-feed-forward `residual` (not the normed copy) and the experts read a
  separately-normed copy of it. The router is a scale-free RMS norm, a learned per-channel `scale`
  times `hidden^-0.5`, a projection to the experts, a softmax, the top-k, a renormalization, and a
  learned `per_expert_scale` on the kept weights; the experts are a fused `gate_up_proj` `[E, 2·inter,
  hidden]` and a `down_proj` `[E, hidden, inter]` dispatched through `gatherMM`, and they apply the
  routing weights themselves (the reference's index-add), so the block sums rather than re-weights.
  Reference parity against transformers' own `Gemma4ForCausalLM` at a tiny two-layer configuration
  (`run_reference.py gemma4_moe`, `IK_PARITY_GEMMA4_MOE`, the gemma oracle): every hidden state exact
  layer by layer, logit cosine 0.9999999999996, on the first numeric run after the geometry was
  corrected. **Two geometry facts were load-bearing, both invisible on the E-series:** the per-layer
  input embedding is a fixed 262144 rows (`vocab_size_per_layer_input`), not the token vocabulary — the
  two coincide on the E-series because its token vocabulary is 262144, so the net had been reading the
  token vocabulary and it only mattered at a tiny test; and a full-attention layer runs a fixed 512-wide
  head (its own `head_dim`) where the sliding layers run theirs, so setting the global head width to the
  sliding one crashed the full layer's projection reshape. The mixture is read from `config.json`
  through `NFKMLXGemmaLanguage.configuration(fromHuggingFace:)` (the same entry the dense sizes use),
  which turns on the routed branch from `enable_moe_block`. The Gemma 4 decoders run through
  `NFKMLXGemmaBackend` (`NFKMLXGemmaLanguage.backend(directoryURL:)` / `@objc
  gemmaBackendWithDirectoryURL:error:`), which reads a release directory and dispatches on its config's
  model type; internally it builds through `configuration(fromHuggingFace:)` plus `makeNet` /
  `loadWeights` (the path the parity tests use). See the `NFKMLXGemmaBackend` entry below.
- `NFKMLXGemma4UnifiedNet` (`NFKMLXGemma4Unified.swift`) — the **12B `gemma4_unified_text` decoder**, a
  Different architecture from the E-series: no per-layer input embeddings and no mixture, only the
  sandwich block with a per-layer scalar. Its attention is the same one the E-series runs — learned
  query/key norms, a scale-free value norm, attention at scale 1, per-layer head widths (a full layer
  runs 512), and the proportional rotary on the full layers — so `NFKGemmaAttention` and
  `NFKGemmaFeedForward` are reused directly and only the block and model are new (this is why the port
  matched on the first numeric run rather than after a hunt). `NFKMLXGemmaLanguage.unifiedConfiguration(fromHuggingFace:)`
  reads a `gemma4_unified_text` config (rejecting the E-series and everything else); `makeUnifiedNet` /
  `loadUnifiedWeights`. Embeddings scale by `√hidden`, logits are tied, no softcap. Reference parity
  against transformers' own `Gemma4UnifiedForCausalLM` at a tiny sliding/sliding/full configuration
  (`run_reference.py gemma4_unified`, `IK_PARITY_GEMMA4_UNIFIED`, the gemma oracle): every layer exact
  (cosine 1.0000000000), logit cosine 0.9999999999995, on the first run.
- `NFKMLXGemma4VisionNet` (`NFKMLXGemma4Vision.swift`) — the Gemma 4 vision encoder, the image
  tower of the tri-modal release. It embeds flattened patches through one linear projection, adds a
  learned 2-D position embedding (an x-table and a y-table summed per patch, padding zeroed), and runs
  the sandwich block bidirectionally. The attention keeps the learned query/key norms, the scale-free
  value norm, and scale 1, and applies a **2-D rope** beside the learned position embedding — the head
  is split per spatial axis (`headDim/2` channels each) and each half is rotated (rotate-half) by its
  coordinate at rope base **100**. The rope was missing at first and the tiny random-weight test could
  not see it — at `head_dim` 8 with small positions it moved the tiny encoder by 2e-8 (0.9999999850,
  which read as ordinary imprecision), but on the released weights (`head_dim` 64, 16 layers) it
  compounded to a cosine of 0.77; the real-weight test is what caught it. The projections are
  `Gemma4ClippableLinear` (`NFKGemmaClippableLinear`) — a bias-free linear under a `.linear` key with
  optional input/output clamps. The release trains them with finite clamp bounds (`use_clipped_linears`,
  a quantization-aware-training artifact — ±12, ±2.4, …), which are load-bearing at inference; the tiny
  oracle used `use_clipped_linears=False`, so this too only surfaced on real weights. Both fixes are
  Measured load-bearing on the released weights (`testGemma4VisionRopeAndClampsAreLoadBearingOnTheReleasedWeights`):
  running the real encoder with an identity rope (cos 1, sin 0) drops it to 0.768, and with no clamp
  bounds to 0.837, against the 0.9999999999898 the pair reaches — each isolated by leaving the other in
  place. `makeVisionNet` /
  config `NFKMLXGemma4VisionConfiguration` (`ropeTheta`, `useClippedLinears`). Reference parity at a
  tiny configuration (`run_reference.py gemma4_vision`, `IK_PARITY_GEMMA4_VISION`, encoder 0.9999999999)
  And on the released E2B weights (`gemma4_vision_real`, `IK_PARITY_GEMMA4_VISION_REAL`: encoder
  0.9999999999898, pooled 0.9999999999975, projected 0.9999999999948 — the vision tower, pooler, and
  embedder loaded selectively from the tri-modal checkpoint). The pooler is implemented too
  (`softTokens(_:positionIds:)`): a position-based average pool that folds the patches falling into each
  `k × k` grid cell (with `k` read from the input patch count over the output token count) and scales by
  `√hidden`, producing the soft tokens a language model reads. The optional standardization is
  implemented too: `standardize` creates `std_bias`/`std_scale` (`@ParameterInfo` optionals, absent by
  default) and applies `(pooled − std_bias) · std_scale` after the pooler, as `Gemma4VisionModel` does. No
  released Gemma 4 enables it (E2B and E4B both `standardize: false`), so the `.tiny` configuration and
  the `gemma4_vision` oracle set it on to exercise the path (pooled cosine 0.99999999999999). The image
  processor is `NFKMLXGemma4ImageProcessor` below.
- `NFKMLXGemma4AudioNet` (`NFKMLXGemma4Audio.swift`) — the Gemma 4 audio Conformer, the most complex
  tower. A **2-D convolutional subsampler** (two stride-2 3×3 convolutions, a channel LayerNorm with no
  bias, a ReLU, then a linear projection of the flattened frequency-and-channel features — the frequency
  dim is tied to `subsampling_conv_channels[0]`) feeds **Conformer layers**: a macaron feed-forward, a
  **blocked relative-position attention**, a **light depthwise convolution**, a second macaron
  feed-forward, and sandwich norms, then an output projection. Built in two isolated phases, each at
  parity against transformers' own `Gemma4AudioModel` (`run_reference.py gemma4_audio`,
  `IK_PARITY_GEMMA4_AUDIO`): the subsampler (`makeAudioSubSample`, cosine 0.9999999999999845) and the
  Conformer fed the reference's post-subsample hidden, position encoding, and mask (`conformer(...)`,
  cosine 0.9999999999999974, the blocked-attention seam alone 0.9999999999999688).
  **The blocked attention is a Transformer-XL relative-position attention:** queries group into
  non-overlapping blocks of `chunk_size`, each block attends over a `contextSize = chunk + (left-1) +
  right` window extracted by a padded gather (MLX has no `unfold`), the content score adds a
  relative-position score built from `relative_k_proj` and reshaped through the appendix-B `relativeShift`
  (pad, reshape, drop, reshape), the logits are `tanh`-softcapped, the mask fills the invalid positions,
  and the queries carry a per-dimension **softplus** scale beside `q_scale`/`k_scale`. The light conv is
  a GLU, a **causal** depthwise convolution (left-padded so a frame sees no future), and a pointwise
  projection. The convolution kernels load transposed from PyTorch's `[out,in,kH,kW]` / `[ch,1,k]` to
  MLX's layouts. **The parity trap was in the oracle, not the port:** the shared `_randomized` helper
  perturbs the clippable-linear clamp buffers (`input_min`/`max`), which the released model ships at
  ±inf (identity) — leaving them randomized clamped the reference's activations aggressively and the
  first feed-forward scored 0.35; resetting them to ±inf in the oracle (the port models no clamp)
  restored parity. The full tower runs end to end (`callAsFunction(_ features:)`): it subsamples,
  builds its own sliding-window blocked mask (`blockedMask`), and runs the Conformer — reference parity
  against `Gemma4AudioModel` from the mel features (cosine 0.999999999999996). The mask construction was
  the one off-by-one: the window function admits `dist ∈ [0, leftWindow)` (strict), so a valid past
  distance is `< maxPast`, not `≤ maxPast` — the inclusive form scored 0.9987. Reference parity on the
  released E2B weights too (`gemma4_audio_real`, `IK_PARITY_GEMMA4_AUDIO_REAL`: encoded
  0.9999999999995, projected 0.9999999999994 — the audio tower and its embedder loaded from the tri-modal
  checkpoint, with the finite `use_clipped_linears` clamps that the tiny oracle left at ±inf). The audio
  configuration gained `useClippedLinears`, threaded to every projection, the same as the vision tower.
- `NFKMLXGemmaBackend` (`NFKMLXGemmaBackend.swift`) — the **text-generation backend** for the Gemma 4
  decoders, so a consumer runs them through the InferKit contract. `NFKMLXGemmaLanguage.backend(directoryURL:)`
  / `@objc gemmaBackendWithDirectoryURL:error:` reads a release's `config.json` and dispatches on its
  model type: the E-series and the 26B-A4B mixture through `configuration(fromHuggingFace:)` + `makeNet`,
  the 12B through `unifiedConfiguration` + `makeUnifiedNet`. **Gemma runs prefill-only** — the decoders
  carry no key-value cache — so generation re-runs the growing sequence each step (quadratic, fine for
  the short outputs an on-device assistant produces). The tokenizer is `NFKMLXGemmaTokenizer` (the core
  reader cannot produce Gemma's byte-fallback BPE), which gained `decode` and `id(forToken:)`; a raw
  prompt encodes after `<bos>`, and a message list builds Gemma's `<start_of_turn>role\n…<end_of_turn>`
  turns from the special-token IDS rather than encoding the markers as text. Generation is greedy at
  temperature 0, else temperature-sampled, stopping on `<eos>`/`<end_of_turn>`. Measured end to end
  on the released E2B (`testGemmaBackendGeneratesText`): "The capital of France is" → " Paris." The net
  itself is already at parity layer by layer, so the backend test exercises the tokenizer and the
  generation loop it adds. The networks and tokenizer cross the async job boundary through an
  `@unchecked Sendable` holder, as the core language backend does.
- `NFKMLXGemma4ImageProcessor` (`NFKMLXGemma4ImageProcessor.swift`) — the vision **image processor**:
  a `CGImage` to the flattened patches and `(x, y)` positions the tower reads. It resizes preserving
  aspect ratio to fit a patch budget (both sides a multiple of `poolingKernelSize · patchSize`),
  rescales to `0 … 1`, splits into patches flattened `(row, column, channel)`, and pads to the budget
  with `(-1, -1)` positions. The resize is CoreGraphics, not the reference's torchvision bicubic, so the
  patch pixels are a documented approximation (as SmolVLM's are); the resized dimensions, the patch
  layout, and the position ids are the reference's exactly, checked by `testGemma4ImageProcessorLayout`.
- `NFKMLXGemma4AudioFeatureExtractor` (`NFKMLXGemma4AudioFeatureExtractor.swift`) — the audio **mel front
  end**: raw 16 kHz audio to the log-mel features `[frames, 128]` the subsampler reads. Semicausal
  framing (prepend `frameLength / 2` zeros, unfold `frameLength + 1` and drop the last sample), a
  periodic Hann window, the magnitude (not power) of a 512-point real FFT, a 128-band HTK triangular mel
  filterbank (`log10`-mel, `norm=None`, distinct from the Slaney bank the Whisper front end uses), and
  `log(mel + 1e-3)`. Reference parity against transformers' own `Gemma4AudioFeatureExtractor`
  (`run_reference.py gemma4_mel`, cosine 0.9999999999928).
- `NFKMLXGemma4MultimodalEmbedder` / `NFKMLXGemma4Fusion` (`NFKMLXGemma4Fusion.swift`) — the **multimodal
  fusion**. The embedder projects a tower's soft tokens into the decoder's space (a scale-free RMS norm,
  then a bias-free linear to the text hidden size) at reference parity against
  `Gemma4MultimodalEmbedder` (`run_reference.py gemma4_embedder`, cosine 0.9999999999999927); `fuse`
  replaces the text embeddings at the placeholder positions with the projected soft tokens in order, the
  same `where`-over-a-gathered-index splice the SmolVLM fusion uses. The whole chain end to end
  (image/audio to an answer) additionally needs the released TRI-MODAL weights — the towers, the
  embedders, and the E-series decoder together — which the text-only E2B release does not carry, so the
  components are each verified at parity rather than the full run.
- `NFKMLXGemma4ConditionalGeneration` (`NFKMLXGemma4Fusion.swift`) — the full tri-modal chain wired end
  to end: an image and/or a waveform and a placeholder-carrying token sequence in, a generated
  continuation out. It runs the image through the processor and the vision tower to soft tokens (and the
  waveform through the mel front end and the audio tower), projects each through its embedder, splices
  them at the placeholder positions with `fusedEmbeddings`, and runs the E-series decoder prefill-only
  over the fused embeddings. The decoder gained an `embed` / `logits(fromEmbeddings:tokens:)` seam so the
  main stream is supplied pre-spliced while the per-layer input identity still reads the
  placeholder-padded token ids (the reference's split — the context projection reads the spliced
  embeddings, the identity reads the ids). A placeholder token embeds as the pad token and the soft
  token then replaces it. Reference parity on the released E2B weights, end to end
  (`testGemma4ConditionalGenerationOnTheReleasedWeights`, `run_reference.py gemma4_conditional_real`,
  `IK_PARITY_GEMMA4_CONDITIONAL_REAL`): an image over a 6×6 patch grid fills four image placeholder
  tokens, and the fused sequence's logits match transformers' own `Gemma4ForConditionalGeneration` at
  logit cosine 0.999999999959, argmax 8/8 (every position predicts the same token). The E2B release
  Is the full tri-modal `Gemma4ForConditionalGeneration` — its 10 GB checkpoint carries the vision
  tower, the audio Conformer, both embedders, and the decoder — so the numeric end-to-end run needed no
  separate weights; an earlier note wrongly called it text-only. The oracle and the Swift side each load
  the sub-towers selectively from that one checkpoint (`model.vision_tower.` / `model.embed_vision.` /
  `model.language_model.`), so neither has to hold the whole tri-modal graph at float32 at once.
- Nothing remains in the Gemma 4 family. The four architectures, both towers' full forwards (the
  optional vision standardization included), the input adapters, the fusion, and the
  conditional-generation chain are all at reference parity on the released weights; the text decoders
  generate through `NFKMLXGemmaBackend`.
  E4B is measured too, at the precision it ships in: its 16 GB of bf16 weights double past this
  machine's RAM at float32, so both sides run bf16 (`IK_GEMMA_DTYPE=bfloat16` for the oracle,
  `.checkpoint` here) — logit cosine 0.9998 with the same argmax at every position, and the strict
  load itself confirms the doubled feed-forward on its 18 kv-shared layers, since a wrong width fails
  loudly. The measurement surfaced a defect only bf16 could: the attention masks are built float32,
  and the fused attention refuses a mask that does not promote to a bf16 module's own type, so the
  mask now takes the queries' dtype — invisible at float32, which is why no float32 run ever raised
  it. The dense and hybrid decoders had the identical latent crash on any `.checkpoint` load and are
  fixed the same way, each pinned by a bf16-forward test.
- `NFKMLXGemma2Net` — the Gemma-2 text decoder (`Gemma2Model`), SANA's text encoder (its DiT
  cross-attends to Gemma-2's last hidden state). Gemma 2 is a distinct architecture from the Gemma 3 /
  Gemma 4 text models here: it keeps the `(1 + w)` RMS normalization and the sandwich block (a norm
  before and after each of attention and the feed-forward), but it has no query/key norm, it soft-CAPS
  the attention logits (`tanh(logit/cap)·cap`, cap 50), it uses a single rotary base (10000) with
  sliding-window attention on the even layers, and it scales the query by `query_pre_attn_scalar^-0.5`.
  The attention is computed explicitly (matmul + soft-cap + softmax, not the fused SDPA) because of the
  soft-cap. Module keys are the checkpoint's (`embed_tokens`, `layers.N.self_attn.{q,k,v,o}_proj`,
  `layers.N.mlp.{gate,up,down}_proj`, the four sandwich norms, `norm`), no transpose. Reference parity
  against transformers' Gemma2Model at a tiny configuration with a small sliding window (so the
  alternating sliding/full layers differ): last hidden cosine 0.9999999999998679 on the first numeric
  run (`run_reference.py gemma2`, the `llm` oracle env). The released sizes are presets
  (`.gemma2_9B`: 3584 / 42 layers / 16 heads / 8 kv / head 256 / 14336, `query_pre_attn_scalar` 256;
  `.gemma2_27B`: 4608 / 46 / 32 / 16 / 128 / 36864, scalar 144), each held to its released headers by
  shape (464 / 508 tensors, 0 missing, 0 mismatched, 0 unaccounted).
