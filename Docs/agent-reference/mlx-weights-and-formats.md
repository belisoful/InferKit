<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX weights, checkpoints, and formats

Runtime quantization, the release reader, the native GGUF and PyTorch checkpoint readers.

- `NFKMLXQuantization` — runtime MLX quantization, the package's first path for running a model in
  MLX-quantized form (`NFKMLXDeepSeekQuantization` only decodes a stored format). `quantize(module:
  bits:groupSize:includeEmbeddings:)` packs `Linear` layers whose input width divides the group size
  into affine 4- or 8-bit `QuantizedLinear` (everything else computes as built; already-quantized
  layers are excluded, which matters because `QuantizedLinear` subclasses `Linear` and satisfies a
  type test silently). `includeEmbeddings` (default off) also packs `Embedding` layers into
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
- `NFKMLXReleaseWeights` — one reader for a downloaded release's weights, single-file or sharded
  (`model.safetensors.index.json`, each shard read once), with a remap closure whose nil skips a
  tensor. The dense, hybrid, and Gemma loaders all read through it; before it each had its own copy,
  and Gemma's copy had no sharded path — a capability gap consolidation removed as a side effect.
  The per-family differences stay in the loaders where they belong: the tied-`lm_head` drop, the
  hybrid's `model.language_model.` remap and depthwise-conv transpose, Gemma's tower skip.
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
  converters' own output (raw in `~/.inferkit-validation/raw/`, `IK_RAW_<KEY>` written by fetch.py).
  `NFKMLXTorchCheckpoint` is the public `@objc` face: inspect `tensorNames`/`infoForTensor:`, read a
  tensor's bytes, or convert on device with `writeSafetensorsToURL:` (a hand-rolled pure-Swift
  safetensors writer — no Metal needed — whose output carries no `inferkit.layout` metadata, which is
  the PyTorch-layout marker; float64 narrows to float32 as the converters do). Refused with errors
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
  state dict; (3) a `.nemo` PAX/ustar tar is unwrapped to the checkpoint inside it (`readTar`). The
  scripted walk is scoped to archives carrying `constants.pkl` (the TorchScript marker), so the eager
  path is untouched; its int config attributes (`input_resolution`) are surfaced and ignored by the
  loaders' coverage the way `num_batches_tracked` is.
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
