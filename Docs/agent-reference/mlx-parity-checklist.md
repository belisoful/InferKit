<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# Completing an InferKitMLX model to parity (the documentation checklist)

A model is not done when its parity test passes — it is done when every listing is updated too. A
partial update leaves the model missing from some indexes, which is the failure this checklist exists
to prevent. Update all of these, in the modality's existing section, mirroring the sibling rows:

- `Docs/agent-reference/mlx-models-<class>.md` — a full per-model entry in the model list of the
  file for the model's class (image restoration, detection, speech restoration, …), and the model's
  registered name in the `registerAll` prose list in `mlx-companion.md`. Use the actual measured
  cosines, not the `> 0.999` test threshold. `CLAUDE.md` / `AGENTS.md` carry no per-model entries.
- `README.md` — the model's name in the modality bullet of the model list.
- `Docs/companions.md` — the full model gallery (README links here as "the full model gallery").
- `Docs/model-index.md` — the index row (entry class, network, configuration, registered name, the
  Swift + Objective-C copy-and-paste, base backend).
- `Docs/model-parity.md` — the parity row with the real recorded numbers (capture them by temporarily
  printing the measured cosines in the parity test, then revert the prints).
- `Docs/examples.md` — the modality's gallery snippet.
- `Docs/inference-guide.md` — the Roadmap, if the model was a roadmap item (mark it shipped).
- `InferKitMLX/Sources/InferKitMLX/InferKitMLX.docc/InferKitMLX.md` — the gallery Topics list.
- `InferKitMLX/Sources/InferKitMLX/InferKitMLX.docc/ModelGallery.md` — the gallery table row.
- `InferKitMLX/Sources/InferKitMLX/InferKitMLX.docc/ModelIndex.md` — the index table row and the
  per-section copy-and-paste code block.
- `InferKitMLX/Examples/MLXModelGalleryExamples.swift` — the live per-model gallery example (a build +
  representative forward). This is a compiled test, so run it (`InferKitMLXExamples` scheme).
- `Tools/validation-assets/manifest.json` — the checkpoint/record/config entry (and an
  `oracle_environments` note if the model needs a new interpreter or extra packages).
- `~/.inferkit-validation.json` — the model's `IK_VAL_*` / `IK_PARITY_*` keys, as absolute paths into
  the local validation store, so the full check exercises the model by default rather than only when
  those keys are set in the environment. The test class must read them through
  `NFKMLXValidationConfig.environment` (the process environment merged with that JSON), not
  `ProcessInfo.processInfo.environment` directly, or the JSON keys never reach it. This file is a
  machine's local config, not a tracked repository file, so its "update" is provisioning rather than a
  commit — but skipping it leaves the model's parity test silently skipped on a plain run (green with
  nothing behind it), which is exactly the gap a full models test exists to catch.

One line of code belongs with the listings, because nothing else catches it: a model whose backend
reads a request key beyond what its base backend reads declares that key at the construction site
(`forwardInputKeys` / `forwardParameterKeys` on `NFKMLXModuleBackend` and `NFKMLXMattingBackend`,
`encodedInputKeys` / `encodedParameterKeys` on `NFKMLXDiffusionBackend`). The backend's
`supportedInputKeys` and `supportedParameterKeys` answer what the engine acts on, and a missing
declaration makes that answer wrong. See "Declared keys" in `mlx-companion.md`.

Not a listing, so not required per model: `InferKitMLX/ObjCExamples/MLXObjCExample.m` is a curated
illustrative set, not an exhaustive gallery.

The code/wiring that accompanies the docs (the model file, `NFKMLXReferenceModels.registerAll`
registration, the `run_reference.py` oracle mode, and the parity test) is covered by the shipped-model
pattern; a converter under `Tools/<model>-to-safetensors/` is optional because the native `.pth`/`.pt`
reader loads most released checkpoints directly.

## Customization is part of parity

A parity cosine is the inference half of done. The other half is the model's customization path, and a
model is not at parity until that half is shipped or ruled out in writing. The levels, the minimum
shipped set, and the tests that prove it are defined in `mlx-training.md` ("Customization is part of
parity"). This checklist adds the listings that half touches:

- `Docs/agent-reference/mlx-models-<class>.md` — the model's entry states its customization level
  (probe, head retarget, zero-reference, LoRA, full), the objective's reference source, and the
  measured objective parity; or it names the constraint that makes the model offline-only or
  untrainable here.
- `Docs/model-parity.md` — a row in "Training objectives and the checkpoint path" for the objective:
  its `run_reference.py` mode and the measured agreement on identical tensors.
- `Docs/examples.md` — the recipe snippet under "Customizing a model on a consumer's own data",
  mirrored by a compiled example in `InferKitMLX/Examples/MLXExamples.swift`.
- `InferKitMLX/Sources/InferKitMLX/InferKitMLX.docc/InferKitMLX.md` — the network, objective, and
  recipe symbols in the "Customizing a model" Topics list.
- `Docs/model-index.md` and the DocC `ModelIndex.md` — the construction cell gains the
  `network(weightsURL:)` line, the way the HT Demucs row carries its fine-tuned form.
- `Tools/validation-assets/manifest.json` — the objective's record in `training_records` (its
  `IK_PARITY_*` key, the file under `~/.inferkit-validation`, the oracle environment, and the
  `run_reference.py` command that regenerates it), and its mode in that environment's `modes`. A
  record is an oracle output, so `fetch.py` does not provision it; the key reaches
  `~/.inferkit-validation.json` through the IO Coordinator.

## Every optional behavior needs a measured case that exercises it

A parity cosine covers the configuration it was measured at. A flag that the releases set and that no
measured configuration sets stays unmeasured at any cosine.

Two ways this hides, both found in shipped code:

- **The configuration axis.** A tiny oracle builds the reference from the port's own parameters. A
  field the port does not read is a field the reference never receives, so both sides take the same
  default and agree. Gemma 4's `attention_k_eq_v` held a 0.999999999999 tiny cosine on the two sizes
  that set it while the decoder ignored the flag outright. The sizes measured on released weights were
  the ones that leave it off.
- **The boundary axis.** A comparison over the body of a signal passes while its edge diverges. A beat
  decoder that trims by a checkpoint-tuned threshold emits one extra beat at the end of the track with
  every earlier beat matching.

Two requirements on the measured set follow:

- A flag, variant, or optional branch that a release sets → one measured case sets it, at a value
  differing from the default, so a field the port drops builds the wrong thing rather than the right
  thing by coincidence.
- A sequence, spectrogram, or track → one comparison includes the first and the last element.

## A GPU-only reference is usually still an oracle

Before recording that a model has no third-party oracle, measure how much of its reference a FLOAT
configuration actually reaches. A release whose code imports CUDA kernels at module scope reads as
un-runnable and often is not, because the kernels stand in for storage rather than for arithmetic.
DeepSeek V4.1 is the worked case: `inference/model.py` imports six symbols from a tilelang `kernel`
module, and at bf16 only two of them carry model math. Its `linear()` falls back to `F.linear` for an
unquantized weight, so the two quantized GEMMs are unreachable; the two activation quantizers are a
fused quantize-then-dequantize when called in place, so skipping them yields the UNQUANTIZED model,
which is what a float port should be held to anyway. Standing a substitute module in `sys.modules`
under the imported name runs the release's own code on the CPU, and the real kernel file is never
imported, so its toolchain need not exist. The same technique already served StoRM (a fused CUDA op
behind a `sys.modules` shim) and MossFormer2 (basicsr shimmed rather than installed).

Three rules keep such a shim honest, and each is a line of code rather than a promise:

- A substitute that stands in for STORAGE raises, so "the float path does not route through it" is
  asserted rather than assumed.
- A substitute that stands in for a lossy round trip is neutral, and the entry says the oracle is
  therefore the unquantized model. Where the round trip is worth reproducing too, the substitute
  gains a switch that makes it REAL, transcribed from the kernel it stands for and checked against
  an implementation independent of the port, and the model carries a second record rather than
  moving its first: DeepSeek V4.1's `deepseek_v41_quantized` is the worked case, and its unquantized
  record reproduces byte for byte after the switch was added.
- A substitute that carries ARITHMETIC is transcribed from a source that is itself verified, never
  written from the reference's intent. For DeepSeek V4.1 both such pieces came from transformers'
  plain-PyTorch `deepseek_v4`, which this package had already measured those mechanisms against.

Such an oracle is worth building before the runtime exists, not after. Comparing a port's analytic
parameter enumeration against the module tree the reference itself builds — at a configuration whose
layer pattern differs from the release's in every respect the enumeration's rules turn on — checks
the RULES, where a release's headers only check one instance of them. That comparison caught a rule
tied to the wrong thing (a router bias that belongs to a vision tower, not to a model version) on its
first run.

## An end-to-end figure below its mechanisms' figures is a finding

When every mechanism measured on its own agrees at 1e-14 and the composed model agrees at 1e-9, the
difference lives in something no seam isolates. DeepSeek V4.1 read 0.9999999992 end to end from its first
measurement while each mechanism read 0.99999999999996: the hyper-connection residual mix summed over the wrong
copy index, and the oracle's four copies stayed within one bf16 unit of each other, so both
orientations gave nearly the same number. Two rules follow.

- A seam test's inputs must be able to tell the candidate answers apart. A mixing step fed
  near-identical copies measures nothing about which axis it reduces. The oracle records a probe with independently drawn inputs from a private
  generator, which leaves the rest of the record byte-identical (`probe.hc.*` in the DeepSeek
  records).
- A bf16 mode is held to the reference run in bf16, bit for bit, by counting the elements that
  differ at each layer, not by cosine. A bf16 port FARTHER from the bf16 reference than the float32
  port is, by about √2 in 1 − cosine, is rounding twice where the reference rounds once. MLX's fused
  `rmsNorm` is one such place: it multiplies by the weight after rounding to the input's dtype.
  Rounding-point bugs and orientation bugs both surface this way, and neither shows in float32.
- A mechanism the source says is not implemented is refused, not run. DeepSeek V4's attention said
  in its own comment that compressed positions were "not implemented here", while the loader built
  the model and generated from it, silently window-only. A release-code oracle at a configuration
  that reaches every mechanism finds this; a comparison that degenerates past a mechanism cannot.
- A preset is compared with its release's `config.json` field by field, by reflection over every
  stored property, not only by the shapes it implies. DeepSeek's presets differed from their
  releases in routing scale, output groups and YaRN while every shape check passed.
  `NFKMLXPresetReleaseTests` holds every language, image and video preset that has a reader to its
  release this way. Its first run found four more: `.mistralSmall3` normalized queries and keys,
  Gemma 4 `.e2b` made every layer sliding, `.twelveB` was the E-series shape, and Granite `.h1B`
  placed attention every sixth layer where the release places it at 5, 15, 25 and 35.
- An operation the reference delegates to a backend (`F.scaled_dot_product_attention`) has no
  single bf16 answer: torch's CPU flash kernel, its MATH backend and a CUDA kernel each round
  differently. Prove which backend the reference ran by emulating it exactly, then record under
  `sdpa_kernel(SDPBackend.MATH)`, torch's own definition, and keep the default backend's result
  beside it as a measured gap (DeepSeek V4.1's image tower).

### Half precision against the half-precision reference

`NFKMLXBFloat16ParityTests` holds each released decoder at bf16 to its reference built at bf16 with
eager attention, and uses the reference's own float32 run on the same input as the floor. The
records come from `run_reference.py hf_layer_probe` (`IK_PROBE_DTYPE=bfloat16`, `IK_PROBE_LAYERS`),
which also records every submodule's input and output inside the probed layers, cloned at capture,
and the q/k/v, probabilities and output of the eager attention function. Two checks apply:

- Each block runs alone on the reference's own bf16 input. What remains once every rounding sits
  where the reference places it is GEMM summation order (Metal against torch's CPU kernels) and the
  last float32 bit of a transcendental (Metal's `tanh` against Sleef's), under a quarter of the floor
  in `1 - cosine`. A misplaced rounding reads at the floor or above it.
- End to end, this side's bf16 sits no farther from float32 than twice the reference's bf16 does.
  The two bf16 streams diverge chaotically past the first flipped element, so an end-to-end element
  count carries no signal.

When a block fails the first check, `hf_layer_probe`'s pieces locate it: run each Swift submodule on
the recorded input and count the differing elements. `NFKMLXReferenceRounding` holds the placements
transformers uses. Each takes the fused or plain path on a float32 input, so a float32 forward stays
byte-identical.

- A norm computes in float32 and rounds once (`gemmaNorm`, and `scaledNorm` with `pow(·, -0.5)` for
  Gemma 3n and 4). Llama-style norms round twice, and MLX's fused `RMSNorm` already matches them.
- `gelu(approximate="tanh")`, `silu` and `softplus` widen and round once. MLX composes them from
  ops that each round.
- The rotary tables are built in float32 and rounded to the input's type, then
  `x · cos + rotate_half(x) · sin` rounds at each op. `MLXFast.RoPE` rounds once.
- Eager attention rounds the scores, the scaled scores, and the float32 softmax, then the weighted
  sum. The fused attention keeps its softmax in float32.
- A tensor times a Python float multiplies in float32 (`scaled`, `divided`). MLX rounds a `Float`
  literal to bf16 first, which changes the product when the factor is not exact in bf16 (0.22,
  `1536^-0.5`, `2^-0.5`, `144^-0.5`).
- A float32 array mixed into a bf16 expression promotes the rest of the layer to float32. Gemma 3n's
  AltUp floor `MLXArray(Float(1e-5))` and the M-RoPE tables did this; both now take the stream's type.
  So did Voxtral's computed Whisper sinusoids, FLUX's rotary tables, and every diffusion transformer's
  sinusoidal timestep projection. The signature is this side sitting closer to float32 than the
  reference does.
- `NFKLayerNorm`, `NFKGroupNorm`, `NFKBatchNorm`, `NFKConv1d` and `NFKConv2d` compute in float32 and
  round once, as torch does. MLXNN's `BatchNorm` adds `eps` to the half-precision variance and takes
  its `rsqrt` there, so widening the input alone leaves it wrong; `NFKBatchNorm` forms the scale and
  shift from the stored statistics in float32.
- Three attention forms, by what the reference calls. transformers' eager attention is `attention`.
  torch's MATH `scaled_dot_product_attention` is `mathAttention`: float32 throughout, the scale split
  as its square root on the queries and on the keys, one rounding (Granite Speech's Conformer). torch's
  default CPU backend for a half-precision input is its flash kernel, which diffusers and the MiniMax
  Music 3 depth decoder reach: `flashAttention` forms the scores, the row max and the exponentials in
  float32, rounds the exponentials to the operands' type for the product with the values, divides by
  their float32 sum, and rounds once. That form is one key block, up to 512 keys.
- A Swish written as `x * torch.sigmoid(x)` rounds the sigmoid and then the product (`swish`, Phi-4's
  Conformer). `F.silu` rounds once.
- diffusers rotates queries and keys in float32 and rounds once (FLUX, FLUX.2, Z-Image, Wan). Wan's
  block also keeps its modulation, its norms and its gated residual sums in float32 and rounds each
  step once. FLUX and FLUX.2 cast the timestep and guidance to the latents' type before the factor of
  1000, and every sinusoidal projection takes the embedder's type before its MLP.
- A diffusers pipeline runs its latents and conditioning in the transformer's type and takes a
  flow-matching Euler step as `round(float32(sample) + round(dt · velocity))` (`eulerStep`). Wan's
  pipeline keeps float32 latents and casts only the transformer's input.
- MLX's `exp`, `sin` and `cos` differ from torch's in the last float32 bit on either device. A
  sinusoidal timestep projection carries that into a few bf16 roundings of the timestep embedding:
  Wan's velocity reads 0.24 of the floor from it with every block piece exact.

Some blocks amplify a one-ulp GEMM-order difference into a large fraction of the floor, so a whole
block run on the reference's input cannot discriminate there. Granite Speech's Conformer spreads one
flipped element across a frame through its pointwise convolutions, its decoder amplifies about tenfold
per layer from layer 1, and Phi-4's last Conformer layer does it through its final LayerNorm. For
those the isolated check runs on each sub-piece, and ours-against-float32 is the end-to-end check.

Two oracle traps build a reference that the release's own load does not:

- `model.to(torch.bfloat16)` rounds a non-persistent buffer that `from_pretrained` leaves as the
  constructor built it (Qwen-Image's timestep frequencies, a rotary's `inv_freq`): Qwen-Image's tiny
  record read 41 times the floor from it. `_randomized` under `IK_DIT_DTYPE` restores those buffers.
  A module in `_keep_in_fp32_modules_strict` stays float32 under a bf16 load (Voxtral's
  `embed_positions`).
- A release that ships a float32 LoRA which the port folds at load is held to a reference with the
  adapter folded the same way (`IK_GRANITE_SPEECH_DTYPE=bfloat16-folded`). transformers applies the
  adapter as its own bf16 branch, and that difference alone moves Granite Speech's logits by 1.2 times
  the floor.

A cut too large for a float32 load here takes its float32 record from a bf16 load with each module
widened to float32 only while it runs (`IK_PROBE_STREAM_F32=1`). A bf16 load builds the buffers the
constructor computes (Gemma's `sqrt(hidden)` embedding scale, a rotary table) at bf16, and widening
cannot undo that rounding, so those buffers come from a float32 construction of the same model whose
parameters live on the meta device. Holding torch's default type at float32 through `from_pretrained`
instead builds every parameter at float32, which is the full float32 load the mode exists to avoid.
The streamed record must match a true float32 load of a smaller cut to 1e-9 before it stands in.

A mixture-of-experts layer measured at bf16 carries two differences a dense layer does not. Each is
measured on the reference's own layer, not assumed:

- A router score is a bf16 projection, so experts can tie at the top-`k` boundary. `torch.topk` breaks
  such a tie with no fixed rule, so a token whose `k`-th and `(k + 1)`-th scores lie within one bf16
  step is left out of the isolated measurement, with the count named on the row.
- Its eight or so expert matmuls let accumulation order alone reach 0.27 of the floor (Gemma 4
  26B-A4B), so a mixture layer's isolated bar is 0.5 of the floor rather than 0.25.

The tiny diffusers configurations the float32 parity tests use serve the same check at bf16:
`IK_DIT_DTYPE=bfloat16` runs `_randomized`'s model at bf16 with every floating input rounded, and
records every submodule as `probe.<name>.in` / `.out`; `bfloat16-weights` runs float32 arithmetic on
the same rounded weights and inputs for the floor.

A release too large for float32 on this machine is cut to its first N layers by
`Tools/validation-assets/truncate.py <repo> <out> N`. It fetches each kept tensor by HTTP range
request into compact shards, since the Swift release reader loads every tensor in a shard file, and
writes a reduced `config.json` and index. Keep the index's `metadata` block, or transformers raises a
`KeyError`. The cut is a valid small release that both sides load unchanged, at float32 and at bf16.

A release too large to run numerically still takes the structural check (`NFKMLXReleasedSizesTests`
against a `shapes.json` inventory), which catches a wrong shape and a tensor the checkpoint does not
carry. A size with neither numeric nor structural coverage is unverified.

## Entry house style (keep the whole consistent)

These files accreted across many sessions and drifted into several voices for the same thing. A new
entry matches its neighbors in the file it lands in, not the last entry a different session happened to
write. The "Documentation Style (enforced)" section of `CLAUDE.md` governs the prose (no em-dash dramatic asides,
no antithesis, no rule-of-three, one fact per sentence, present tense, American English); it applies to
these entries too, and where an older entry breaks it, the break is not a precedent to copy. The
per-file shape:

- `Docs/agent-reference/mlx-models-*.md` model list — one bullet, `` `NFKMLXFoo`` `` (`` `(@objc)` `` only when the
  class is), then ` — `, then a lowercase noun phrase naming what it is and its reference
  (`the X (`ReferenceClass`, Vendor)`). State the architecture, the load-bearing facts, and the measured
  parity with the ACTUAL cosines. Say "at reference parity" in running prose — one casing, lowercase, no
  bold on the phrase itself; reserve bold for a specific load-bearing noun, not for emphasis. "at parity"
  is only for a short back-reference to a result already stated (a second size, another variant). Depth
  is proportional to the model's novelty, not to how recently it was added; a configuration-only variant
  is a sentence, not a section.
- **README.md** — the model's name in the modality bullet, nothing more.
- **Docs/companions.md** — a prose gallery bullet. Describe the model and its variants/presets and say
  "at reference parity against <reference>"; do NOT quote a tiny-config cosine here (those live in
  model-parity.md). A released-weight cosine may appear when it is the headline result, matching the
  neighbors that do.
- Docs/model-index.md and the DocC `ModelIndex.md` — one table row (entry class, network,
  configuration, registered name or "Swift API", the construction line, base backend). In the DocC file,
  the section also carries a copy-and-paste code block that is a per-FAMILY cheat-sheet — one
  construction line per model family, not a mirror of every table symbol — so every family in the table
  has at least one line there, and a code line never names a family the table omits. The drift to stop is
  a family present in one and absent from the other (a new family's table row with no cheat-sheet line,
  as SD3/FLUX/ControlNet were), not a per-symbol mismatch.
- Docs/model-parity.md and the DocC `ModelGallery.md` — the parity row (and gallery row) with the
  REAL recorded cosines at full precision, not rounded or a `> 0.999` threshold. Multiple seams are
  `seam A x; seam B y; final z`.
- **DocC `InferKitMLX.md`** — the symbol(s) in the modality's Topics list.
- **`MLXModelGalleryExamples.swift`** — a build-plus-forward example, for a REGISTERED `@objc` model.
  A generative pipeline (SD3/FLUX/Z-Image/…) is not registered and is absent here by design, so a new
  one that is also unregistered stays absent — consistently, not by omission.

When an edit touches a file, leave that file MORE consistent than you found it: if the neighbors already
share a shape, conform to it; if a genuinely better shape is warranted, do not introduce a third — raise
it so the whole file moves together rather than one more divergent entry accruing.
