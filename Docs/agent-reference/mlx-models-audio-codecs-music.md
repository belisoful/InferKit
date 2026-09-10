<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX models: neural audio codecs and music generation

- `NFKMLXMusic3` — MiniMax Music 3, the text+lyrics→music model, ported stage by stage with a
  measured parity record gating each stage. The full model is a hybrid: a Qwen3-8B autoregressive
  stage over 8 RVQ codebooks (one semantic of 16384 living inside the LM's own vocabulary, seven
  acoustic of 1024 filled per frame by a 4-layer depth decoder — there is no MusicGen delay pattern;
  the depth decoder is what replaces it), a condition encoder blending the 8 per-codebook hidden
  states (synthesis conditions on the hidden states, the codes only close the AR feedback loop), a
  36-layer flow-matching DiT over 8-second latent windows, and a DAC-style Snake vocoder. The oracle
  is diffusers' own implementation (>= 0.40.0 ships `MiniMaxMusic3Vocoder`,
  `MiniMaxMusic3RVQDepthDecoder`, `MiniMaxMusic3ConditionEncoder`, the DiT, and the modular
  pipeline), under its own `musicvenv` interpreter (`oracle_environments.music`) — a third-party
  reference, where the community MLX ports of this model have none. The `music_ar` and
  `music_tokenizer` oracles import the pipeline's own helper functions (`_sample_top_k`,
  `_generate_depth_codes`, `_embed_audio_frame`, `_clean_caption`, `_normalize_lyrics`), so the
  arithmetic compared against is the reference's, not a copy. All five networks are at measured
  parity, the prompt contract is token-exact, and the chained pipeline generates audio end to end.
  - **Prompt contract** (`NFKMusic3Prompt`): the caption/lyrics cleaners, the special-token
    template, the release byte-level BPE through the core `NFKTokenizer` (slow-format vocab/merges
    from `qwen_7B/qwen3-8B-tokenizer-music/`, special tokens from `added_tokens.json`), and the
    CFG-row substitution (interior tokens → `<|audio_cfg|>` 151654) — an exact token match against
    the reference tokenizer over the shared `MUSIC_PROMPTS` cases (markdown, `<|tag value|>`
    rewrites, structure tags, multi-byte text, whitespace forms), both rows. Reaching it required
    the core tokenizer's `qwen2` pretokenization (see `core-runtime-notes.md`, Tokenizers): under the GPT-2
    default the same prompt encodes to different, valid-looking ids, which no output would ever
    reveal. The cleaners are additionally pinned against the reference's own intermediate strings,
    so a cleaning bug reads as a string diff before it reads as a token mismatch.
  - **`NFKMLXMusicBackend`** (`NFKMLXMusic3.backend(directoryURL:)`, registry `minimax-music3`,
    in `registerAll`): `NFKInputPrompt` + `NFKInputLyrics` (a new core key) →
    stereo 44.1 kHz `NFKAudioAsset` under `NFKOutputAudio`; honors `NFKParameterDurationSeconds` /
    `Seed` / `Steps` / `GuidanceScale`. The stages load from the release directory per run and each
    is freed when its part is done (`clearCache` between): the bf16 LM (16 GiB) and the float32 DiT
    (9.7 GB) together exceed a 32 GB machine's working set and the pipeline is strictly sequential.
    There is no random-weights form — the factory takes the release directory, and `isReady`
    reports presence. Cancellation is honored between stages and per flow step; progress reports
    through the job.
  - End to end, measured on the real weights
    (`testTheMusicBackendGeneratesAClipEndToEnd`): a 2-second request produces 1.997 s of stereo
    audio at RMS 0.051 in 77 s wall clock, the whole 27 GB stack staged through. A sampled song
    cannot be compared to the reference bitwise (the random streams differ by construction) — the
    per-stage records are the numeric ground; the e2e asserts duration, rate, channels, and that
    the clip is signal rather than silence or clipping. `IK_MUSIC3_KEEP_CLIP` keeps the WAV for
    listening.
  - Quantized releases and residency
    (`NFKMLXMusic3.quantizeRelease(at:to:bits:transformerBits:groupSize:)`): writes a quantized
    copy of the release in the release's own layout, so `backend(directoryURL:)` takes it
    unchanged — 27 GB falls to **7.7 GiB**. The split default is measured, not assumed: at 4-bit
    the language model holds (first-step logits cosine **0.99933** with its input embedding packed
    too, **0.99952** with the embedding left bf16; prefill 0.9905 against the same full-precision
    parity records) while the DiT's velocity falls to **0.9775** — the flow field is the
    quantization-sensitive stage — so the DiT defaults to **8-bit**, where it measures **0.99990**
    (6-bit measures 0.99844, an order of magnitude worse for ~0.6 GB, so it is not the default;
    `testTheDiTQuantizationBitWidthSweep` is the record). The LM's `Linear` layers and its input
    embedding pack — the model is untied, so the embedding is separate from the packed `lm_head`, and
    at 1.6 GiB it is the stack's largest tensor (`includeEmbeddings: true`, reclaiming 1.10 GiB). The
    vocoder and condition encoder copy through unquantized; the LICENSE copies too — it travels with
    the weights. Whether the stages stay loaded between runs is decided from the weights
    (`keepsStagesResident`): stack bytes + a 4 GB reserve (activations + the CFG pair's KV cache)
    against the recommended working set — deliberately not live free memory, which a resident
    backend's own weights would count against and evict themselves. The quantized stack goes
    resident (measured: two consecutive 2-s generations at 34.5 s / 32.5 s with `resident true`);
    the full-precision stack stages per run, with each stage now scoped so the language
    model releases before the DiT loads (the original code's locals lived to function exit, so the
    claimed staging never actually happened — found while making residency real).
  - **Vocoder** (`NFKMusic3VocoderNet`, latents `[B, T, 128]` → stereo `[B, T·512, 2]`): waveform
    cosine 0.9999999999990, worst |difference| 8.3e-7, all 121 tensors accounted both directions,
    first numeric run. Stereo folds the latent's channel halves into the batch through one shared
    decoder, pinned weight-free by `testSwappingTheLatentHalvesSwapsTheStereoChannels`. The released
    file is already safetensors, so there is no offline converter: `loadVocoderWeights` fuses the
    weight-norm pairs itself (`g·v/‖v‖`, norm over every axis but the first, which covers the
    forward and transposed convolutions alike) and transposes layouts, all gated on
    `needsConvTranspose` so a fine-tuned save round-trips. The Snake α is stored `[1, C, 1]` for the
    reference's NCL and held `[1, 1, C]` for NLC; the loader transposes it under the same gate.
  - **RVQ depth decoder** (`NFKMusic3DepthDecoderNet`, 4 causal layers, a learned 16-position
    embedding rather than a rotary, 7 heads over 1024 codes each, an offset-packed
    `audio_embeddings` table of 7 × 1024): forward 0.999999999997, heads 0.999999999996, projection
    0.9999999999995, embedding 1.0 — the record covers all four parameter families because the
    pipeline reads them through different paths and a forward alone touches only the first. 47/47
    tensors accounted both directions; ships bf16, loads at float32 by default.
  - **Condition encoder** (`NFKMusic3ConditionEncoderNet`): 0.9999999999997. A learned softmax
    blend of the 8 per-codebook hidden states, a scalar gain, a 3-wide convolution, and PyTorch's
    exact nearest-neighbor resample onto the latent rate — `floor(i · frames/latents)` with the
    scale at Float precision; 13 frames land on `int(13 · 44100/24000 · 960/512)` = 44 latents.
  - **Flow-matching DiT** (`NFKMusic3DiTNet`, 36 layers, partial rotary over the leading 32 of each
    head's 64 channels, the trained Fourier timestep prepended as token 0 and stripped after the
    blocks, `ff_in` splitting into `value · silu(gate)`, input `[latent, zeros, condition]` on
    channels where the zeroed block is the reference's unfilled audio-prompt slot): velocities
    0.999999999994–0.999999999999 at three timesteps and the zero-condition unconditional branch.
    The release is float32 and sharded under the diffusers spelling, which
    `NFKMLXReleaseWeights.files` now also resolves (`diffusion_pytorch_model.safetensors[.index.json]`).
    Its 9.7 GB cannot sit in a structural test, so `testTheDiTReleaseIsAccountedBothDirections`
    enumerates: the tiny module's key template expanded to 36 layers equals the shard index's own
    key set exactly. `NFKMusic3FlowSchedule` matches diffusers' `FlowMatchEulerDiscreteScheduler`
    (`invert_sigmas`: σ = 1 − linspace(1, 1/N, N) with a terminal 1; the model consumes σ directly
    as its timestep; a step is `x + (σ_next − σ)·v`), measured against the scheduler configured from
    the release's own config. `NFKMusic3FlowMatcher` is the windowed loop — 200-frame windows at hop
    100, the overlap re-blended toward the previous window's carry at every Euler step
    (`(1 − (1 − 1e-6)σ)·noise + σ·previous`) and locked after it, crops 86/258 latents at the
    stitch — with the boundary lock pinned weight-free.
  - **Autoregressive stage** (`NFKMusic3AutoregressiveStage` over `NFKMLXLanguageNet`, whose
    embed / hiddenStates-from-embeddings / logits-from-hidden seams were opened for it): parity
    bf16 both sides, the Gemma E4B treatment, because the LM's geometry counts to 8,584,475,648
    parameters (measured by `NFKMLXModelSizing` before any load: 16.0 GiB at 16-bit, 32.0 GiB at
    float32, which does not fit this machine). Teacher-forced with the reference's own sampled
    codes so the comparison measures the networks rather than two random streams: prompt prefill
    0.99993, first-step logits 0.999995, guided band 0.99998 with 51/52 shared top-50 candidates
    and the same argmax, fused frame hiddens 0.99994. The CFG pair is a batch of 2 through one
    cache; a frame is one position (the 8 code embeddings sum, scaled by 8^-0.5); the depth
    interleave replaces any MusicGen delay pattern; the warm-up decode step past `<|audio_start|>`
    is not an emitted frame; and guidance is gated to the conditional branch's top-50 before
    sampling. `NFKMusic3Sampler` takes its top-k threshold by CPU sort — `MLX.top` is unsorted, and
    reading its last slot as the threshold silently turns sampling into argmax.
  Diffusers enforces the prompt and frame limits the community ports drop (a > 5000-token
  prompt raises; frames cap at 9000), and this port additionally enforces what neither does: prompt
  + frames must fit the LM's 10240-position budget (`NFKMusic3Contract.positionBudget`), rejected
  before any forward runs. The music LM's config is transformers-5.x-shaped, which
  `NFKMLXLanguage.configuration(fromHuggingFace:)` now reads: `layer_types` listing only
  `full_attention` is dense (only a mixed stack is rejected), and `rope_theta` nests under
  `rope_parameters`.
  The weights are not permissively licensed (MiniMax-Music3 Community License: UI attribution in
  commercial products, separate authorization above USD 20M revenue, safeguard obligations for
  hosted generation) — recorded in `Docs/companions.md`, the manifest's MUSIC3 entry, and the LICENSE
  fetched beside the weights.
- `NFKMLXDAC` (`@objc`) — the Descript Audio Codec, the toolkit's first neural audio codec and the class a
  codec-token speech-LLM generates into. Three parts: a convolutional **encoder** (a wide first conv,
  then downsampling stages of three dilated residual units + Snake + a strided conv, doubling the width
  and halving the resolution), a **residual vector quantizer** (`NFKDACResidualVectorQuantize`: a stack
  of quantizers, each projecting the latent to the codebook width through `in_proj`, matching each frame
  to its nearest L2-normalized codebook entry, projecting the raw chosen entry back through `out_proj`,
  and coding the residual the previous ones left), and a **decoder** (the mirror, upsampling through
  transposed convs). The Snake activation, the dilated residual unit, and the decoder's upsample block
  are the shared Music 3 vocoder blocks (`NFKMusic3Snake`/`NFKMusic3ResidualUnit`/`NFKMusic3VocoderBlock`);
  the strided encoder and the RVQ are what the codec adds. `NFKMLXDAC.encode(_:)` returns the codebook
  tokens `[[Int]]` (codebook × frame) — the codec's product — and `decode(_:)` reconstructs; the
  `NFKMLXDACBackend` runs the round trip (audio → codes → audio under `NFKOutputAudio`). `+register`
  under `dac`. The released convolutions are weight-normalized, so the loader **fuses `g·v/‖v‖`** (reusing
  `NFKMLXMusic3.fusedWeightNorm`) and transposes; `remapReferenceKey` translates the reference's nested
  `nn.Sequential` names (`encoder.block.N.block.M.block.K`, `decoder.model.N.block.M`,
  `quantizer.quantizers.N`). The nearest-neighbor search compares normalized vectors (maximizing the dot
  product of unit vectors is minimizing Euclidean distance); the reconstruction uses the raw codebook
  entry, as the reference does. Reference parity against `descript-audio-codec` on the released 44.1
  kHz model (`run_reference.py dac`, llm oracle env, needs `descript-audio-codec`), on the first numeric
  run: codebook tokens matching exactly (783/783 over 9 codebooks × 87 frames) and the decoder
  reconstructing the reference's waveform from its codes at cosine 0.99999999999986. `NFKMLXDACConfiguration`
  carries the released `.dac44kHz`/`.dac24kHz`/`.dac16kHz` geometries (all four encoder rates, differing
  in the rates, codebook count, and sample rate); the native torch reader loads the released `.pth`
  directly (its `state_dict` is unwrapped, the weight norm fused). Converter `Tools/dac-to-safetensors`.
- `NFKMLXSNAC` (`@objc`) — SNAC, the toolkit's second neural audio codec and the first multi-SCALE one.
  Its codebooks code at different temporal rates: the residual is average-pooled before a coarse codebook
  quantizes it and repeat-interleaved back afterward, so codebook 0 emits one token per `vqStrides[0]`
  frames, codebook 1 per `vqStrides[1]`, and so on (`[4, 2, 1]` for the 24 kHz model). This is the 24 kHz
  Speech model — the codec the common speech codec-token LLMs use — with depthwise-separable convolutions,
  a decoder **noise block**, three codebooks, and no bottleneck attention (the release's
  `attn_window_size` is null). Structure like DAC with three SNAC-specific pieces: the depthwise blocks
  (`NFKSNACResidualUnit`'s dilated conv is grouped over every channel), the `NFKSNACNoiseBlock` (a learned
  per-position scale times a fresh Gaussian, added in the decoder), and the multi-scale RVQ
  (`NFKSNACResidualVectorQuantize`: `avgPool` before `in_proj`, `repeatInterleave` after `out_proj`). The
  Snake activation and the transposed-conv upsample are the shared Music 3 blocks. `NFKMLXSNAC.encode(_:)`
  returns the per-codebook token streams `[[Int]]` at their own rates (the coarser emit fewer),
  `decode(_:deterministic:)` reconstructs, and `NFKMLXSNACBackend` runs the round trip under
  `NFKOutputAudio`. `+register` under `snac`. The noise block is non-deterministic (`torch.randn`), so
  a decode is reproducible only with `deterministic: true` (which skips it); parity is measured that way,
  the noise's expected contribution being zero. The release weight-normalizes through torch's
  parametrization API (`parametrizations.weight.original0` = g, `original1` = v), where DAC used the older
  `weight_g`/`weight_v`; `NFKMLXSNAC.fusedWeightNorm` fuses that form. `remapReferenceKey` translates the
  nested `encoder.block.N.block.M.block.K` / `decoder.model.N.block.M` (0 snake, 1 transposed conv, 2 the
  noise block, 3..5 residual units) / `quantizer.quantizers.N` names. Reference parity against the
  `snac` package on the released 24 kHz model (`run_reference.py snac`, llm oracle env, needs `snac`), on
  the first numeric run: per-codebook tokens matching exactly (42/42 over the three multi-scale codebooks)
  and the decoder reconstructing at cosine 0.9999999999998. Native torch reader loads the release
  directly. Converter `Tools/snac-to-safetensors`.
  The two music models are at parity too (`.snac32kHz` / `.snac44kHz`, `NFKMLXSNACVariant.music32kHz`
  / `.music44kHz`, registered as `snac-32khz` / `snac-44khz`): encoder 64 wide at rates `[2, 3, 8, 8]`, a
  1536-wide decoder, four codebooks at strides `[8, 4, 2, 1]`, and — what the speech model omits — a
  windowed local attention at the bottleneck of both encoder and decoder (`NFKSNACLocalAttention`,
  `attentionWindow` 32: a LayerNorm, bias-free `to_qkv` / `to_out`, `headDim = min(64, dim)`, rotate-half
  rotary over the positions within the window, fused attention per window), which shifts every later
  `Sequential` slot by one in the remap and raises the padding multiple to `hop · lcm(stride₀, window)`.
  Codes 60/60 exact on both, reconstruction 0.99999999999978 / 0.99999999999984.
