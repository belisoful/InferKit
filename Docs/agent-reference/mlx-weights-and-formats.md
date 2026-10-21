<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX weights, checkpoints, and formats

Runtime quantization, the release reader, the native GGUF and PyTorch checkpoint readers.

- `NFKMLXQuantization` — runtime MLX quantization, the package's first path for running a model in
  MLX-quantized form (`NFKMLXDeepSeekQuantization` only decodes a stored format). `quantize(module:
  bits:groupSize:includeEmbeddings:)` packs `Linear` layers whose input width divides the group size
  into affine `QuantizedLinear` at 2, 3, 4, 5, 6, or 8 bits (everything else computes as built; already-quantized
  layers are excluded, which matters because `QuantizedLinear` subclasses `Linear` and satisfies a
  type test silently). Each layer is replaced through its owner (`NFKMLXModuleReplacement.place`),
  so a skipped layer inside a module array stays; the call throws. `includeEmbeddings` (default off) also packs `Embedding` layers into
  `QuantizedEmbedding`; it is off by default because a tied model reuses its input embedding as the
  logit head, so quantizing it quantizes the head too — a per-model cost. That cost is measured and
  small (`testTheTiedEmbeddingQuantizationCostAgainstTheRecord`, opt-in `IK_QWEN_EMB_PROBE=1`): on
  the tied Qwen3 0.6B and 1.7B, packing the embedding at the same width as the Linear layers moves the
  logit cosine by ~1e-5 at 8-bit and ~0.006 at 4-bit — the Linear bit width dominates, not the tied
  head (8-bit Linear scores 0.9962 / 0.9994, 4-bit only 0.9222 / 0.9488, so these small models want
  8-bit regardless). So `includeEmbeddings: true` is safe for a tied model quantized at 8-bit; the
  default stays off because the wrong-width case (4-bit) is where packing the head costs most, and a
  caller should choose it deliberately. The music LM opts in
  (`quantizeRelease` passes `includeEmbeddings: true`): that LM is untied, its `lm_head` is a separate
  packed `Linear`, and the input embedding is its largest tensor (200000×4096, 1.6 GiB) — quantizing
  it at 4-bit measured logits cosine 0.99933 against 0.99952 for the bf16 embedding and reclaims
  1.10 GiB. The checkpoint contract closes the packed-uint32 hazard: `NFKMLXWeights.save` detects
  quantized leaves and records `inferkit.quantization` = "bits:groupSize" in the metadata;
  `loadCheckpoint` reads it back, and a loader calls `NFKMLXQuantization.matchStructure(of:on:)`
  Before applying, so the packed arrays land on matching structure — without that, a packed weight
  loaded into a plain `Linear` adopts the wrong shape and dtype with no error. The metadata records
  one bits/groupSize, not which layer kinds were packed, so `matchStructure` reads whether the
  embedding was quantized from the checkpoint itself — a packed embedding weight is `uint32` where an
  unquantized one is a float — which keeps a file saved before embeddings were quantizable loadable. A
  quantized checkpoint loads at its stored dtypes whatever precision the caller requests (the packing
  is uint32 regardless, and the scales keep the precision the quantization was computed at). Wired
  through the language-model loaders (single-file releases route through `loadCheckpoint`; a
  quantized module saves as one file, so quantized-sharded does not arise) and the music loaders;
  round-tripped exactly by `testAQuantizedCheckpointRoundTripsThroughTheLoaders`, embedding-packed
  case included.
  `quantize(module:…)` leaves the quantized arrays lazy over the weights they replace, because
  `matchStructure` calls it on a freshly built module where evaluating would materialize random
  weights. A caller that quantizes a LOADED module and runs it calls `eval(module)` then
  `NFKMLXGPU.clearCache()` first. Measured 2026-10-09: a bf16 TranslateGemma 4B quantized to 4 bits
  without them kept its 7.8 GB of bf16 weights until the first forward pass, which freed them into
  the buffer cache (about 7 GB in `NFKMLXGPU.cacheMemory`), and the process peaked in the 24.6 GB
  bucket. Clearing without the eval frees nothing. Shipped callers: the drafted Gemma 3 factory
  (`NFKMLXGemma3.model(directoryURL:draftDirectoryURL:…)`) evaluates and clears;
  `NFKMLXMusic3.quantizeRelease` evaluates through `NFKMLXWeights.save`, drops the module, and clears.
- mlx_lm quantized releases (`mlx-community/*-4bit`, `-8bit`, mixed recipes) load through the language
  loader, single-file or sharded. `config.json` states the quantization (`quantization`, or a
  `quantization_config` with no `quant_method`); `NFKMLXQuantization.releaseQuantization(configJSON:)`
  reads it, returns MXFP4 for gpt-oss's `quant_method: mxfp4`, and refuses every other `quant_method`
  (GPTQ, AWQ, bitsandbytes, FP8, compressed tensors) by name. Which layers are packed, and at what
  geometry, comes from the tensors, as mlx_lm decides it on load:
  - a layer whose `scales` the release carries is packed; the group size is the input width over the
    scales' last dimension, and the bits are the words' last dimension × 32 over the input width
    (`releaseLayerQuantization`), so a per-layer override (gpt-oss's 8-bit attention beside MXFP4
    experts, an 8-bit router) needs no reading of the config;
  - `biases` present → affine; absent with byte scales → MXFP4 (4 bits / 32), MXFP8 (8 / 32), or NVFP4
    (4 / 16);
  - `matchStructure(toRelease:on:)` shapes a freshly built module from the header alone, before any
    weight is read, and the load runs at `.checkpoint` precision whatever the request, so the scales
    keep the dtype mlx_lm stored;
  - mlx_lm's stacked `switch_mlp` is the module's `experts`, and an older Mixtral's per-expert
    `w1`/`w2`/`w3` `weight`/`scales`/`biases` stack into it; a paged load pages the packed experts;
  - mlx_lm's gpt-oss layout stores the expert gate and up projections apart where the module holds
    them fused, and is refused.
  `NFKMLXReleaseQuantizationTests` round-trips dense, sharded, stacked-mixture (resident and paged),
  and per-expert releases exactly. It holds `mlx-community/Qwen3-0.6B-4bit` and
  `Llama-3.2-1B-Instruct-4bit` to two references: mlx_lm itself at the stored precision
  (`Tools/reference-parity/mlx_lm_logits.py`, the `mlxlm` interpreter), and, with the stored floats
  widened to float32, transformers on the same weights dequantized (`run_reference.py
  mlx_quantized_release`). Widened, both match to about 1e-11, so every packed layer reads exactly. As
  stored, Qwen3-0.6B's bfloat16 scales and norms move its logits to cosine 0.99727 from float32 where
  torch's bfloat16 run of the same weights reaches 0.99989. Llama 3.2 reads near float32 as stored
  (0.9999988), because its `llama3` rotary table is float32 and promotes the activations after the first
  rotation. mlx_lm's own run sits at 0.99752 from float32 on Qwen3 (0.95991 at position 0, the
  attention-sink token), so float32 cannot judge the stored precision. An mlx_lm release therefore
  runs with mlx_lm's placement (USER 2026-10-20): the loader sets
  `NFKMLXLanguageConfiguration.halfPrecisionPlacement` to `.mlx` from the release's mlx_lm
  `quantization` block, and the rotary, attention, and gated activation take MLX's fused rope, fused
  attention in `causal` mode, and `silu(gate) · up` compiled shapeless as mlx_lm's `swiglu` is. A
  llama3 rotary table is computed as mlx_lm's `Llama3RoPE` computes it. A transformers release keeps
  `.torch` (`NFKReferenceRounding`). The seam localization that chose this (`mlx_lm_seams.py`, whose
  rebuilt forward equals `model(tokens)` bit for bit, against the torch placement) found every
  projection, norm (`q_norm`/`k_norm` included), residual, and the head bit-exact, and only the rotary,
  attention, and SiLU apart: Qwen3's attention seam at 0.99974, its logits at 0.99949 from mlx_lm,
  Llama's seams near 1e-7. mlx_lm's GPU prefill orders (`orders.incremental`, `chunk2`, `chunk3`)
  reproduce its Qwen3 run exactly, so the gap was placement, not order. The bar is exact: every seam
  and the logits equal mlx_lm's. mlx_lm's CPU stream is no reference for it; it reduces sequentially in
  float32 and sits at 0.9534 from its own GPU run on Qwen3. A mixture's routing (softmax, top-k, and
  the weighted combine) and its experts' activation keep the torch placement; no mlx-community mixture
  release is in the store to measure them against.
- `NFKMLXReleaseWeights` — one reader for a downloaded release's weights, single-file or sharded
  (`model.safetensors.index.json`, each shard read once), with a remap closure whose nil skips a
  tensor. The dense, hybrid, and Gemma loaders all read through it; before it each had its own copy,
  and Gemma's copy had no sharded path — a capability gap consolidation removed as a side effect.
  The per-family differences stay in the loaders where they belong: the tied-`lm_head` drop, the
  hybrid's `model.language_model.` remap and depthwise-conv transpose, Gemma's tower skip.
  `arrays(inDirectory:precision:remap:)` returns lazy reads; `materializedArrays(…transform:)` reads
  and converts in groups of about 256 MB and returns them evaluated, which lowers a large load's
  footprint and leaves no file read for a GPU command buffer to wait on. Every loader whose tensors
  convert one at a time reads through it (2026-10-08). A caller that builds new arrays from most of
  what it reads stays lazy: an eager read would hold every original beside its result, which doubles
  the peak of per-expert stacking (`stackingExperts`) and adds about 13 GB to Phi-4's float32 LoRA
  fold. The lazy readers that remain:
  - the per-expert tensors of a language release (`NFKMLXLanguage.isPerExpert`), read beside the
    materialized rest;
  - Phi-4-multimodal's decoder, whose LoRA fold spans tensors;
  - FLUX.2's latent codec, which takes two `bn.` statistics from the autoencoder's file;
  - Sa2VA-Qwen's SAM 3 probe, which reads one key's presence and no tensor.
  A remap that drops what a loader never applies keeps an eager read from reading it: the SmolVLM and
  Sa2VA-LLaVA loaders filter by prefix before the read.
- A single file follows the same split (2026-10-08). `NFKMLXWeights.materializedCheckpoint(url:reading:)`
  evaluates the arrays `reading` keeps before it returns, and the 108 loaders that apply a whole
  checkpoint read through it. `loadCheckpoint(url:)` stays lazy at 43 loader call sites, by kind:
  - probes that read a key or a shape and then load the file again (`classCount` in YOLO and RT-DETR,
    the `lm_head` probes in Qwen3-VL and SmolVLM, Conv-TasNet's configuration, Basic Pitch's
    normalization, the denoiser's width);
  - one part of a file that other loaders read too (DeepSeek's image and draft stacks, Chatterbox's
    tokenizer, the SAM 2 component loaders, Florence-2's vision half, Depth Anything's encoder-only
    network) or that drops a large unused part (AdaIN's and the VGG perceptual nets' classifiers,
    BiSeNet's aux heads, FRCRN's and MODNet's duplicate copies, Silero's 8 kHz branch);
  - fusions across tensors: weight norm (DAC, SNAC, HiFi-GAN, BigVGAN, Kokoro, Chatterbox HiFT,
    MossFormer2 SR, Music 3's vocoder, Resemble Enhance), DDColor's spectral norm, the Granite Speech
    adapter fold, and the SD text encoder's and SegFormer's joined projections;
  - DeepSeek's decoder, whose paging maps expert bytes it never reads, and the Sa2VA probes that open
    a file only to learn its layout;
  - a tensor kept as data and read where it is used (Kokoro's voicepack, Stable Diffusion's stored text
    context, Chatterbox's built-in conditionals).
  A `reading` predicate keeps an eager read to the part a loader applies: the single-file language
  loader skips per-expert tensors (stacked lazily) and a paged load's experts, SAM 3's tower loaders
  read their tower out of the 3.44 GB file, and V-JEPA 2 skips its pretraining predictor. A raw
  PyTorch checkpoint is in memory once read, whichever reader opens it.
- Two tests hold the split (2026-10-09). `NFKMLXLazyReadSitesTests` reads the sources and counts each
  file's lazy `arrays(inDirectory:precision:remap:)` and `loadCheckpoint(url:)` calls against a list
  that gives each file's reason; a new lazy read fails it until the loader reads first or the list
  names why it stays lazy, and a stale entry fails it too. `NFKMLXLoadFootprintTests` loads synthetic
  2 GiB releases through each reader and holds the kernel's interval footprint peak, reset before the
  load (`proc_reset_footprint_interval`), to a budget above the result. Measured: a conversion holds
  one 256 MiB group, a stored-type or single-file read holds nothing, and a lazy widening, the
  control the widening test asserts exceeds its budget, holds 1216 MiB. A footprint delta needs a
  settled start: buffers MLX releases leave `phys_footprint` over the following moments, and a
  start read before they leave undercounts the load.
- `NFKMLXGGUF` / `NFKMLXGGUFFormat` — the **native GGUF reader**, the sequel to the native PyTorch
  checkpoint reader, and the format most quantized language models are distributed in. Same contract:
  pure Foundation below the MLX materialization (parsing and dequantization run under `swift test`;
  `NFKMLXGGUFFormat` is the Foundation layer, `NFKMLXGGUF` the `@objc` MLX face), and a type it does not
  implement is refused per-tensor, not per-file — an unknown GGML type leaves the tensor listed (so a
  consumer sees the whole model) and only reading it throws. The container is a header of typed key/value
  metadata (`readValue` covers the 13 GGUF value types, including nested arrays) and a tensor table
  (name, dims, GGML type, offset), then the tensor data aligned to `general.alignment` (default 32).
  GGUF stores the fastest-varying dimension first, so a tensor's row-major (MLX) shape is the
  Reverse of its stored dims — a Linear weight stored `ne=[in, out]` is shape `[out, in]`. Dequantizers:
  `F32`, `F16`, `Q4_0`, `Q5_0`, `Q8_0`, `Q4_K`, `Q6_K` — the k-quants (`Q4_K`: a 256-value super-block of
  eight 32-value sub-blocks, a block `d`/`dmin` scaling per-sub-block 6-bit scales/mins unpacked from 12
  packed bytes, each value `d·sc·q − dmin·min`; `Q6_K`: sixteen 16-value sub-blocks, a 6-bit quant from a
  4-bit low and 2-bit high part centered at 32, scaled by `d` and a per-sub-block int8 scale) plus the
  legacy 32-value blocks. The scalar per-block port matches the vectorized `gguf` reference exactly
  (both compute the one canonical dequantization). **Bit-exact** against the `gguf` package on the
  released SmolLM2-135M Q4_K_M (`run_reference.py gguf`, the `llm` oracle + the `gguf` package): the
  first tensor of each of F32/Q8_0/Q5_0/Q4_K/Q6_K dequantizes to **worst |difference| 0.0** across 262144
  values. A real `Q4_K_M` file mixes Q5_0 (the bulk here), Q4_K, Q6_K, Q8_0, and F32, so supporting Q5_0
  is what makes the file readable rather than mostly-refused. A stored `MLXArray` constant is not held on
  the format struct; the dequant returns `[Float]`, materialized to an `MLXArray` only in the face.
  Wired into the language-model loader (`NFKMLXGGUFLanguage.swift`): a GGUF release generates text
  end to end. `NFKMLXLanguage.configuration(fromGGUF:)` maps the metadata (`<arch>.block_count`,
  `<arch>.embedding_length`, …) onto an `NFKMLXLanguageConfiguration`, reading two structural facts
  from the tensors because the metadata carries no flag for them — a model is tied when it ships no
  `output.weight`, and it normalizes queries and keys when it ships `blk.0.attn_q_norm.weight` (Qwen3
  does, Qwen2/Llama do not). `loadWeights(into:fromGGUF:)` remaps the llama.cpp names
  (`blk.N.attn_q.weight` → `model.layers.N.self_attn.q_proj.weight`, `token_embd`/`output_norm`/`output`)
  and `ggufTokenizer` rebuilds the embedded byte-level BPE (the already-encoded tokens and merges written
  to a temp `vocab.json`/`merges.txt` for the core reader, specials read from `token_type` 3/4). Only
  the dense `llama`/`qwen2`/`qwen3` families are accepted; another architecture throws. Factory
  `backend(ggufURL:)` / `@objc backendWithGGUFURL:error:`.
  **The Q/K permute is load-bearing:** llama.cpp permutes the query and
  key projections during conversion so its interleaved rotary reads adjacent channels, where this
  decoder rotates split halves — loading the raw weights runs mostly-right and subtly wrong (logit
  cosine 0.95, the model saying "Paris" but diverging after). `unpermuteRotary` undoes it per head
  (reshape `[heads, headDim/2, 2, …]`, swap the split-half axes, reshape back — transformers'
  `_reverse_permute_weights`), with the query using the head count and the key the KV-head count. With
  it, reference parity against transformers loading the same GGUF (`run_reference.py gguf_lm`,
  `IK_PARITY_GGUF_LM`, the llm oracle now carrying `accelerate`): logit cosine 0.9999999999971 with the
  same argmax at every prompt position, the first greedy token identical, and the rebuilt tokenizer
  reproducing the reference's ids. A full greedy continuation is not asserted exact — two dequantization
  implementations flip an occasional near-tie on quantized weights — so the check is teacher-forced
  agreement (8/10, near-ties excepted) with the logit cosine as the tight bound. Measured on the
  released SmolLM2-135M-Instruct Q4_K_M.
- `NFKMLXTorchCheckpoint` / `NFKMLXTorchFormat` — the native PyTorch checkpoint reader: a consumer's
  raw `.pth`/`.pt`/`.ckpt`/`.th`/HF `.bin` loads with no Python toolchain. `NFKMLXWeights.loadCheckpoint`
  sniffs a file's leading bytes (never the extension — an HF torch `.bin` and a safetensors `.bin` are
  told apart by content), so every `weightsURL:` factory accepts a raw checkpoint wherever it accepts a
  converted safetensors, reported as `needsConvTranspose: true`. Three layers, all pure Foundation
  below the MLX materialization, so the parsing tests run under `swift test`:
  `NFKMLXZipArchive` (central-directory ZIP with zip64 and deflate; a stored entry's contents are a
  zero-copy slice of the memory-mapped file), `NFKMLXPickle` (a restricted pickle machine, protocols
  2–5: no global ever executes — `collections.OrderedDict` is the only one the machine itself
  interprets, and every other construction becomes an inert opaque node that flattening drops), and
  `NFKMLXTorchFormat` (both containers: the modern zip and the pre-1.6 five-pickle stream, whose
  storages arrive after the pickles and whose persistent tuples carry a trailing view entry).
  Training wrappers unwrap in the converters' own precedence (`state_dict`, `model_state_dict`,
  `params_ema`, `params`, `model`, `generator`, `state`) before the root is flattened — a Lightning
  checkpoint keeps optimizer tensors beside its state_dict, so root-first sweeps those in.
  Whisper's releases store their Linear weights as transposed fp16 views, found by the first real
  parity run after the plan assumed state dicts are contiguous: `bytes(for:)` gathers a strided
  tensor to row-major, held to torch's own materialization by comparing the raw `whisper_tiny.pt`
  against its converted safetensors tensor for tensor. The byte oracle throughout is the offline
  converters' own output (the raw file `IK_RAW_<KEY>` names; fetch.py writes it under `<validation root>/raw/`).
  `NFKMLXTorchCheckpoint` is the public `@objc` face: inspect `tensorNames`/`infoForTensor:`, read a
  tensor's bytes, or convert on device with `writeSafetensorsToURL:` (a hand-rolled pure-Swift
  safetensors writer — no Metal needed — whose output carries no `inferkit.layout` metadata, which is
  the PyTorch-layout marker; float64 narrows to float32 as the converters do). A release directory read
  at `.float32` (`NFKMLXReleaseWeights.arrays`) widens its floating tensors and keeps every integer
  one as stored, as `NFKMLXWeights.converted` does: gpt-oss's MXFP4 blocks are packed bytes, and a byte
  cast to float is a different value. Refused with errors
  naming the offline converter: TorchScript archives (CLIP), an opaque module tree (YOLO), `.nemo`
  big-endian saves, sparse/quantized storages. There are no deferred models — YOLO, VAD, and CLIP
  all load. Three walks/unwraps, all non-executing (no class constructed, no serialized `code/`
  interpreted): (1) a checkpoint that pickled a live `nn.Module` tree (YOLO's ultralytics
  DetectionModel) is walked through the standard `_parameters`/`_buffers`/`_modules` state — `walkModule`,
  reproducing `nn.Module.state_dict()` exactly (parameters + persistent buffers, recurse `_modules`,
  skip a None param / non-persistent buffer / plain-attribute tensor), matched against the real
  yolov8n's 498-key state dict; (2) a **TorchScript archive** (CLIP) is walked through its
  attribute-keyed scripted-module state — `walkScriptedModule`, where each object's `BUILD` state is
  a flat dict of `name → tensor | submodule | scalar` rather than the eager layout, and the leaf
  tensors are the same `_rebuild_tensor_v2` records. The earlier "TorchScript needs its `code/` IR"
  claim was wrong: the probe showed `data.pkl` carries every attribute name (`visual.conv1.weight`,
  `transformer.resblocks.0.attn.in_proj_weight`), matched against the real ViT-B/32's 302-key
  state dict; (3) a `.nemo` PAX/ustar tar is unwrapped to the checkpoint inside it (`readTar`), the
  member named `model_weights.ckpt` first, because an archive can carry more than one checkpoint
  (Canary-1B-v2 holds `timestamps_asr_model_weights.ckpt` AHEAD of its weights, so "the first
  checkpoint member" read the wrong model); (4) a file that saved one bare tensor (`torch.save(tensor)`,
  a Kokoro voicepack) comes back under the single key `tensor`, and the released `af_heart.pt` reads
  identical to the converted `.safetensors` element for element. The
  scripted walk is scoped to archives carrying `constants.pkl` (the TorchScript marker), so the eager
  path is untouched; its int config attributes (`input_resolution`) are surfaced and ignored by the
  loaders' coverage the way `num_batches_tracked` is.
  The sniff tells a safetensors file from a pickle by more than its first two bytes: a safetensors file
  opens with its little-endian header length, and a length of 640, 896, 1152, or 1408 bytes (plus any
  multiple of 65,536) opens `80 02` … `80 05`, which is a protocol 2–5 pickle's opening. Such a file was
  read as a pickle and failed to load (found through a Cosmos Tokenizer record whose header was 1152
  bytes). A safetensors file carries its header's `{` at offset 8, where a torch pickle holds its magic
  number or a frame length, and the sniff now checks it
  (`testASafetensorsHeaderLengthThatReadsAsAPickleProtocolIsStillSafetensors`).
  A TorchScript release can also carry its module's derived constants at the storage precision. The
  Cosmos Tokenizer releases keep their Haar wavelet taps as a persistent bfloat16 buffer (0.70703125 for
  1/√2). A loader reads the parameters and computes such constants itself; a float32 oracle that loads
  the stored copy measures its own rounding (4e-4 from the first convolution).
  Every converter's rename/transform is ported into its model's Swift loader, so all non-excluded
  models load a raw checkpoint end to end, each verified by an `NFKMLXTorchParityTests` equivalence
  test: the raw file and the converted file must land identical parameters through the model's own
  `loadWeights` (u2net's legacy `rebnconvN` index rename, colorizer's Sequential table + ConvT
  permute, hifigan's weight-norm fusion — held to 1e-6, the one tolerance, because two float32
  evaluations of `g·v/‖v‖` differ in the last ulp — nafnet/raft/rife's renames, lama's `generator.`
  and fastspeech2's `model.` strips, and pose, whose raw and converted files carry identical key
  names differing only in deconv axis order, which is why `Checkpoint.isNativeTorch` exists).
  Conv-TasNet and the denoiser needed no change — their shape-keyed 3-D branches already read the
  raw layout — and that is verified, not assumed. RAFT found the package's newest MLX hazard:
  its reference reuses each block's `norm3` inside `downsample`, the rename collides the two names
  deliberately, and duplicate keys crash `ModuleParameters.unflattened` with a stack overflow —
  dedupe through a dictionary first (see `mlx-runtime-gotchas.md` and `Docs/mlx-runtime-hazards.md`). The
  nafnet/rife/lama/modnet raw checkpoints are in the validation manifest, which grew two acquisition
  routes to serve them: `gdrive` (a Google Drive id, MODNet) and `extract` (a member path inside a
  zip at `url`, LaMa's Lightning `best.ckpt`); the rest download from `url` as before. Their
  equivalence tests read the `IK_RAW_*` keys `fetch.py` stamps and skip when absent, like every other
  parity test.

## Release download layer: progress, cancellation, status, and removal

`NFKMLXReleaseDownload` (`NFKMLXMarian.swift`) and its `components:` overload
(`NFKMLXReleaseComponents.swift`) fetch a release file by file through `NFKHFHub`; the core's notes
("Download plans, progress, status, and removal" in `core-runtime-notes.md`) cover the object they
report to. Added 2026-10-15, InferKit 0.4.0.

- No factory has a progress variant. Every fetch already reports to the core's thread-local
  `NFKDownload.current`, so the layer only carries the download where a factory changes thread:
  `NFKMLXReleaseDownload.asyncValue` (generic) captures the current download on the calling thread,
  begins an activity, makes it current on the background queue around `build`, and ends the activity
  with `build`'s error. `NFKMLXReleaseDownload.async` is its form for a factory that delivers a backend,
  and wraps the backend so it can reload. The 39 asynchronous peers that ran their own
  `DispatchQueue.global(...).async { do { completionHandler(try …, nil) } catch { … } }` now call one of the two;
  `NFKMLXDownload.backend` (the single-file factories and `NFKMLXHub`) brackets its own activity. A
  new asynchronous peer goes through one of the two, or it drops the download.
- The download finishes when the factory delivers its result, building included, so a download
  covering one factory reads completed exactly when the backend arrives.
- Sizes up front: under a current download, when a required file or every weights candidate is
  missing from the cache, `directory(…)` lists the repository once (`planDownload(ofRepo:revision:paths: nil)`)
  and declares the files it will fetch (`expectedPaths`: the required, the optional the listing holds,
  the first weights candidate it holds). The shards a `*.index.json` names are declared when the index
  is read, before their bytes, so `expectedByteCount` grows once by the shard bytes. A listing that
  fails is ignored and the fetches report sizes as they start.
- A cached release loads with no network, because offline use depends on it (2026-10-15). Two
  lookups used to reach the server on every cached build, and both are gone:
  - Weights: `cachedWeights` picks the first candidate whole in the cache (a shard index counts only
    with every shard it names), in candidate order, and `fetchWeights` uses it with no request. A
    sharded release used to ask for `model.safetensors` first on every build. With nothing cached the
    candidates are tried in order as before. Parakeet and VoiceRestore, which kept their own candidate
    loops, go through `firstServed`.
  - Optional files: `fetchOptional` collects the optional files the server refused with HTTP 404
    (`NFKDownloadErrorNotFound`, `isAbsent`) and the completion record stores them
    (`recordCompleteDownload(ofRepo:revision:paths:absentPaths:)`). A later build skips a recorded file
    that is not cached. A network failure is not an absence and is never recorded, so a build that ran
    offline asks again once the network returns. `forgetRecordedAbsentPaths(ofRepo:revision:)` is the
    forced refresh; another revision is another snapshot with its own record.
  - A release cached before the record existed, or one whose refusals were forgotten, asks once more
    and records them.
- Cancellation: a cancelled fetch of an optional file or a weights candidate looks like an absent file
  to the `try?` loops, so `checkCancelled()` runs after each optional fetch and on each weights
  failure, and throws `NFKDownloadError(.cancelled)`.
- Completion record: after the last file, `directory(…)` records every path it fetched or found
  (`recordCompleteDownload(ofRepo:revision:paths:)`), so `downloadStatus(repo:revision:paths: nil)`
  reads complete and a removal with nil paths removes the release. A release removed that way keeps
  any file a complete catalog variant at the same repository lists.
- Objective-C: `+[NFKMLXHub hubWithCacheDirectoryURL:]` returns the hub the factories use (the
  default cache when nil), and
  `+[NFKMLXHub backendForModel:variant:cacheDirectoryURL:progressHandler:completionHandler:]` downloads a
  catalog variant through the core's `downloadModelSource:` and builds the backend registered under the
  variant's (else the model's) registry name from the downloaded location. It fails before any transfer
  for a missing variant, a variant without a source, or an unregistered name; the returned download is
  then already failed.
- Tests: `NFKMLXReleaseDownloadProgressTests` serves a release from `NFKStubReleaseProtocol` (a
  `URLProtocol` at `http://release.test`) and evaluates no MLX: progress that only grows and reaches the
  release's bytes with one listing request, a cached release (sharded, with a refused optional file,
  and in component folders) that makes no request, a forgotten refusal asked for again, a cached
  release built through `NFKFailingTransportHub` (every request fails) with no attempt, a cancel
  part-way through
  a shard through the asynchronous helper, the helper carrying the download, and the catalog call
  building a registered `NFKPassthroughBackend`. The `NFKOfflineHub` release tests never make a download
  current, so they never list.
