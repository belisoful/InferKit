# Runtime hazards on Apple Silicon

A catalogue of places where MLX, mlx-swift, Metal, or Core ML **return a wrong answer quietly**
rather than failing. Every entry is either measured here or found by reading the code, and each says
which. They are collected publicly because a hazard that produces no error is one every project has
to rediscover independently, and several of these cost days elsewhere before they were understood.

Measured against **mlx-swift 0.31.6**, macOS 15+, Apple Silicon. The probes live in
`InferKitMLX/Tests/InferKitMLXTests/NFKMLXRuntimeHazardTests.swift` and run under
`xcodebuild test -scheme InferKitMLXTests`. A future mlx-swift bump re-runs them, so a change in
behavior is reported rather than rediscovered.

Corrections and additions are welcome — particularly a reproduction of anything in "Reported
elsewhere, not reproduced here".

---

## Confirmed

### A quantized layer's `weight.dtype` is the storage type, not the compute type

`QuantizedLinear.weight` holds values packed into `uint32`. Reading `weight.dtype` to decide what to
cast activations to therefore truncates them to integers — at every bit width, identically, with no
error and no shape change.

**Measured:** `weight.dtype` is `.uint32` while `scales.dtype` is `.float32`.

```swift
let quantized = QuantizedLinear(Linear(64, 32, bias: false), groupSize: 32, bits: 4)
quantized.weight.dtype    // .uint32  — packed storage
quantized.scales.dtype    // .float32 — what the layer computes in
```

**Rule:** read `scales.dtype`. Never `weight.dtype`.

*Probe: `testAQuantizedLayerStoresItsWeightInAPackedIntegerType`. Reported by
`PipeNetwork/minimax-h3-mlx`, confirmed here.*

### `QuantizedLinear` is a subclass of `Linear`, so type-based selection catches it silently

Anything shaped `layer as? Linear` matches a quantized layer without announcing it. A LoRA
implementation that wraps the match then builds a low-rank detour around packed integers: it
constructs cleanly, trains, and adapts nothing.

**Found by reading**, and it was a live defect in this package until it was guarded.
`NFKMLXLoRA.apply(to:)` now rejects a quantized layer by name rather than adapting it.

**Rule:** every `as? Linear` in a model-surgery path needs an explicit `is QuantizedLinear` check.

### Lazy dequantization holds every source and intermediate at once

`mx.dequantize`, and any decode built from MLX ops, produces a lazy graph. An unevaluated graph pins
its inputs *and* its intermediates. Decoding a whole checkpoint and deferring to one final `eval`
therefore holds every decode live simultaneously — and for block-scaled formats the expanded scale
array alone is the weight's full size, so the intermediates dominate the peak, not the results.

**Rule:** evaluate each entry as it is produced. `NFKMLXDeepSeek.dequantized(_:shapes:)` does.

*Probe: `testDequantizingEvaluatesEachEntryRatherThanDeferringThemAll`. Reported by `marzukia/qMLX`,
confirmed here.*

### Metal flushes subnormal floats to zero; the CPU stream does not

A quantity computed by squaring small numbers — a gradient norm, a variance, a cosine over tiny
vectors — can come back as exactly zero on the GPU and as a small positive number on the CPU.

**Measured:** `MLXArray([1e-21]).square()` evaluates to `0.0` on the GPU and `1e-42` on the CPU
stream, on the same machine in the same process.

This one was found the direct way: a test computing a gradient norm by scaling to the *original*
magnitude read zero and looked like a bug in the code under test.

**Rule:** when reducing for numerical stability, scale toward a magnitude the type holds comfortably,
never away from one. A norm computed as `‖g/max|g|‖ · max|g|` is safe; the same expression with an
arbitrary large reference is not.

*Probe: `testSubnormalsFlushToZeroOnTheGPUButNotTheCPU`. Found here.*

### A finite gradient set can have a non-finite norm

Every entry of a gradient can be a perfectly ordinary `Float` while the sum of their squares
overflows: `3e20` is finite, `9e40` is not. A norm computed directly then comes back infinite,
`maxNorm / infinity` is zero, and the whole update is scaled to nothing. The optimizer's moment
estimates are then built from zeros, and subsequent steps can produce non-finite *parameters* — while
the **loss stays finite the entire time**, so a divergence guard watching the loss never fires. The
run reports a plausible loss curve while the model is destroyed.

**Rule:** zero non-finite entries first, then compute the norm relative to the largest magnitude
present. `NFKMLXTrainer.bounded(_:maxNorm:)` does both.

*Probe: `testAGradientWhoseSquaresOverflowIsStillScaledToTheNorm`. Reported by
`Acelogic/Retrieval-based-Voice-Conversion-MLX`, whose `docs/TRAINING_STABILITY_FIXES.md` records
hitting it in a real run; confirmed here.*

### Merging a LoRA delta into a quantized base discards the training

A rank-r detour's contribution to any one weight is small by construction. Requantizing `W + Δ` snaps
every contribution below half a quantization step back to where it started. The output file is the
right size, loads without complaint, and holds the original model.

**Rule:** merge at full precision, then quantize — in that order. `NFKMLXLoRA.merge(into:)` refuses a
quantized model rather than writing one.

*Reported by `darrenoakey/engram`. Adopted here as a rule; the rounding itself is not separately
measured.*

---

### Duplicate keys crash `ModuleParameters.unflattened` with a stack overflow

`update(parameters: ModuleParameters.unflattened(pairs))` takes a LIST of key–value pairs, and two
entries with the same key crash the process — `NestedItem.unflattenedRecurse` recurses until the
stack guard page (SIGSEGV, `KERN_PROTECTION_FAILURE`), not a thrown error. A Python `dict`
deduplicates the same collision silently, which is why a remap ported from a converter can carry the
hazard invisibly: RAFT's reference reuses each block's `norm3` inside its `downsample` Sequential,
so a raw checkpoint lists one tensor under two names and a rename that collides them (deliberately —
they are the same tensor) hands `unflattened` a duplicate.

**Rule:** build remapped parameters into a `[String: MLXArray]` before applying, so a colliding
rename resolves to one entry (`NFKMLXRAFT.loadWeights`, `NFKMLXMODNet.loadWeights`).

*Found by loading the raw `raft_things.pth` through the native checkpoint reader; no executable
probe, because the failure is a process kill that would truncate the suite it ran in.*

## Confirmed previously, and still true

These were found while building this package and are recorded in `CLAUDE.md`. They are repeated here
because they are not specific to InferKit.

### Never pass `padding:` to an mlx-swift pooling layer

`Pool.callAsFunction` builds its pad widths as `[0, 0] + padding + [0, 0]` — two entries too many. A
four-axis input takes the first four, so a 2-D pool pads **width and channels** instead of height and
width. Nothing is raised; the output shape is silently wrong, and the failure surfaces later as a
channel mismatch reported against an innocent layer. Every `Pool` subclass shares the initializer.
Pooling with no padding is unaffected.

### Never assign to a `@ParameterInfo` or `@ModuleInfo` property

`attention.sink = newValue` aborts the process with "please call update() on the array rather than
setting it" — a fatal error, not a thrown one. In a test run that kills the process and **silently
truncates the reported test count**: a suite of ten reported six and still said "0 failures". Mutate
through `update(parameters: ModuleParameters.unflattened([...]))`.

### Never give a `@ModuleInfo` a numeric key

MLX's `update(parameters:)` parses a numeric key as an **array index**, so a checkpoint unflattened
against `@ModuleInfo(key: "0")` arrives as a list where the module tree has a child module, and the
update aborts the process. Use semantic keys and translate the reference's positions in a remap. A
genuine `[Module]` array property is fine — that is what numeric keys are for.

### `update(parameters:)` adopts a checkpoint's shapes and dtypes wholesale

It does not validate them against the module. Two consequences:

- A wrong architectural assumption **loads cleanly and fails later, or not at all**. A model whose
  attention should be cross-attention but is built as self-attention takes the checkpoint's key and
  value widths without complaint.
- A bfloat16 release turns a float32 module into a bfloat16 one. Measured on the SD 2.1 text tower,
  that cost three orders of magnitude of accuracy (0.9999956 against 0.9999999999841).

**Rule:** assert declared shapes against the release before loading, and make precision a choice
rather than a consequence.

### MLX's buffer cache is not returned between models

A process that loads several models in sequence accumulates cache, and the accumulation starves the
next large forward — which surfaces as a Metal **command-buffer timeout**: a process kill that
truncates the run with "0 failures" reported. Measured here: the same test passes in 25 s alone and
dies mid-suite.

**Rule:** set a standing cache cap rather than remembering to clear at every boundary.
`NFKMLXGPU.applyStandingLimits()` does.

### MLX needs `default.metallib` even for CPU-only work

The first stream request initializes the scheduler, which constructs the Metal device, so even
`mlx_default_cpu_stream_new` throws without it. Measured: a probe pinned to `Device(cpu, 0)` aborts
with the library absent and runs fine with it present. This is why MLX array evaluation aborts under
a plain `swift test` and works under `xcodebuild`.

### A scoped device selection is task-local and does not cross a dispatch

`withDefaultDevice(_:_:)` is inherited by a synchronous call on the calling thread, and **not** by a
block dispatched asynchronously inside the scope, nor by a fresh `Thread` — both report the global
default. So it wraps a synchronous inference and not a background-queue one.

---

## Reported elsewhere, probed here, **not** reproduced

These were reported against mlx-swift 0.31.6 by other projects, each with the same signature:
*prefill exact, decode around 0.99 cosine*. Reduced to their smallest form and probed on 0.31.6, none
reproduces. That is not evidence the reports were wrong — a reduction can miss the shape that
triggers a bug, and a patch release may have fixed it. It is evidence that **this package's usage
pattern is not affected**, which is the question a port actually needs answered.

If you can reproduce any of these, the probe file is the place to add the case.

| Reported | Probed as | Result |
|---|---|---|
| `scaledDotProductAttention` ignores the cached KV tail at query length 1 | one query against caches of 1–129 positions with no mask, against an explicit softmax reference; float32 and bfloat16 | agrees to < 1e-5 (float32) at every length |
| `MLXFast.RoPE` disagrees at `T == 1` | each position rotated alone at its own offset, against the same row of a full-sequence pass; both `MLXFast.RoPE` and `MLXNN.RoPE` | agrees to < 1e-5 at every position |
| Subscript assignment into an `MLXArray` held by an **optional property** does not persist (copy-on-write) | write through an optional property, a non-optional property, and an array element | all three persist |
| `eval` on a thread MLX has not seen faults with "no Stream in current thread" | full forward plus `eval` on a fresh `Thread` | completes normally |

*Reported by `mikolaj92/minimax-music3-swift` (first three) and `darrenoakey/engram` (the fourth).*

The cross-thread probe is **off by default**: if it reproduces it aborts the process rather than
failing, which truncates the run. Enable it with `"IK_PROBE_CROSS_THREAD": "1"` in
`~/.inferkit-validation.json`.

---

## Not MLX, but adjacent

### Core ML's `MLComputeUnits` is a request, not a guarantee

Core ML places an operation the Neural Engine cannot run somewhere else and reports nothing about
having done so. A model asked for the Neural Engine can run entirely on the CPU and behave exactly as
if it had not — only slower and warmer.

**Measured here, and it is not hypothetical.** A GPT-2 converted by this repository's own
`Tools/inferkit-convert` — the shape `NFKCoreMLLanguageBackend` runs — places **0 of 448 operations
on the Neural Engine**. Every one goes to the GPU, under `MLComputeUnits.ALL`. The conversion also
emits `MILCompilerForANE error: ... ANECCompile() FAILED` to the console, which is the only warning
anyone gets, and it appears in the middle of ordinary converter output.

The Neural Engine is reachable on the same machine, so this is a property of the model rather than of
the hardware: a plain attention block at sequence 64 places **100%** of its operations there.

| Model | Neural Engine share |
|---|---|
| Attention block, sequence 64, `(B, S, C)` with `nn.Linear` | 18/18 placed (100%) |
| The same block in the `(B, C, 1, S)` 1×1-convolution layout | 137/137 placed (100%), from 40 operations to 313 |
| A single-token step, stateless | 0% — CPU |
| A single-token step carrying a Core ML state | 0% — CPU |
| GPT-2 through `Tools/inferkit-convert` | 0/448 — GPU |

Two things follow. **The transformer layout guidance did not buy anything here**: the ordinary layout
was already fully placed, and rewriting it into the 4-D 1×1-convolution form left placement unchanged
while multiplying the operation count eightfold. And **whatever moves a real language model off the
Neural Engine is still unidentified.**

The two single-token rows do NOT narrow it, and reading them as evidence would be a mistake: both
landed on the CPU with four and eight placed operations, which is Core ML declining to dispatch a
trivial graph rather than a statement about Neural Engine eligibility. They are inconclusive.

### The cause: a single-token forward is not placed on the Neural Engine, and one function taints a package

Isolated by adding one property at a time to a GPT-2-shaped model that IS fully placed
(`Tools/ane-placement/add_one_property.py`). Same twelve layers, same weights throughout:

| Variant | Neural Engine |
|---|---|
| 12 layers, 768 wide, **sequence 64** | 265/265 placed (100%) |
| the same model at **sequence 1** | 0/265 — all GPU |
| sequence 64 + an embedding gather | 265 on the Neural Engine, 4 on the CPU (98.5%) |
| sequence 1 + a Core ML state | 0% |
| sequence 1 + embedding + state | 0% |
| **sequence 64 and sequence 1 in one multifunction package** | **0%** |

Two facts, and the second is the one that bites:

1. **A single-token forward goes to the GPU.** Sequence length is the whole difference — not the
   stateful cache, which was the obvious suspect and is innocent, and not the embedding gather, which
   costs four CPU operations.
2. **A multifunction package takes one placement decision.** The last row is the same seq-64 function
   that scores 100% alone, packaged with a seq-1 function, and it loses the Neural Engine entirely.

That explains the converted language model exactly. `Tools/inferkit-convert` emits `decode` at one
token and `prefill` at 64 in one package; the prefill half is ANE-eligible on its own and is dragged
onto the GPU by the decode half it ships with.

**Whether it is worth fixing** — measured on the same models, milliseconds per call:

| Shape | `ALL` | `cpuAndGPU` | `cpuAndNeuralEngine` | `cpuOnly` |
|---|---|---|---|---|
| prefill, sequence 64 | 3.82 | 4.98 | **3.77** | 7.92 |
| decode, sequence 1 | 3.18 | 3.76 | 3.47 | 4.88 |

So the Neural Engine is about **1.3× the GPU on prefill** and nothing on decode. Splitting the package
into two models would buy that much time-to-first-token and no more. It is a real but modest win for a
converter change and a second file to ship.

One caution about the two tables: a compute plan reports the **preferred** device, which is a planning
artifact rather than an execution trace. They do not line up perfectly — the seq-1 model is planned
entirely onto the GPU yet runs fastest under `ALL` rather than under `cpuAndGPU`. Treat placement as
"where Core ML intends to run this" and timing as the thing that decides.

Measured on an M1 Max, macOS 26, coremltools 9.0, fp16, `minimum_deployment_target=iOS18`.

**Rule:** measure the placement, and do not assume a layout rewrite fixes it. `NFKComputePlan` reads
placement per operation from a compiled model without running it, and names the operators that fell
off. It needs macOS 14.4 / iOS 17.4; `powermetrics --samplers ane_power` is the runtime cross-check
where that is unavailable, though it needs elevated privileges.

### coremltools' own compute-plan binding reports nothing

`MLComputePlan.get_compute_device_usage_for_mlprogram_operation` returns `None` for every operation in
coremltools 9.0 on macOS 26, including for models the Objective-C `MLComputePlan` reports on in the
same session. A Python-side placement check therefore reads as "nothing is on the Neural Engine"
whatever the truth is — the same wrong answer for two very different reasons.

Take placement from the Objective-C API.

### `MLComputeUnitsCPUOnly` is zero

So a synthesized `MLComputeUnits` property defaults to CPU-only, and a backend that forgets to
initialize it quietly moves every model off the accelerators — which looks exactly like a Neural
Engine that does not work.

### `MPSGraphExecutable.run` takes feeds in `executable.feedTensors` order

Not the order they were declared at compile time. Also: an **empty `name:` string breaks MPSGraph**,
and `hasUnifiedMemory` gates whether results need `synchronizeResults`.

*From `madebyollin/maple-diffusion`; not reproduced here, since this package has no MPSGraph path.*

---

## Testing hazards

Two ways a test can lie about the code it covers, both hit here.

### `ObjectIdentifier` is unique only among **live** objects

Counting distinct objects by `ObjectIdentifier` undercounts when the objects are short-lived: a
released object hands its address to its successor. Keep a strong reference to each object you have
counted, or count something else.

### A `partialResult` holds the *last* non-nil value

`NFKInferenceJob.reportProgress:partialResult:` ignores a nil partial rather than clearing the
previous one. So "the job has a partial result" is not the same as "this step produced one", and a
test that checks presence rather than change will count every progress report.

---

## Contributing

Add a probe alongside the claim. A hazard entry without an executable reduction is a rumor, and this
file exists because rumors about these cost more to re-investigate than to pin down once.
