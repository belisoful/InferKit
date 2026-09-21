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

### `freeze()` does not stop a `BatchNorm` from training

`Module.train(_:)` visits every module in the tree and sets the flag unconditionally.
`freeze(recursive:keys:)` marks parameters as taking no gradient. The two are independent, and
`BatchNorm.callAsFunction` branches on the flag alone: when it is set it normalizes with the batch's
own mean and variance and folds them into `running_mean` / `running_var`.

A frozen backbone under a head-only fine-tune therefore does two wrong things at once. It computes a
different function from the one it computes at inference, because a batch of one or two examples is
not the released statistics. And it overwrites those statistics, which a checkpoint write then
persists, so the damage outlives the run. Nothing reports it: the loss falls, the head learns, and
the model is quietly worse.

**Rule:** after `train(true)`, return every subtree that holds parameters and has none trainable to
evaluation mode. A subtree with no parameters at all follows its parent, or a dropout inside the
group that trains would stop dropping. `NFKMLXTrainer` does this for every run.

*Probes: `testAFrozenNormalizationKeepsItsReleasedStatistics` and
`testAFrozenNormalizationNormalizesWithItsReleasedStatistics` fail without the rule;
`testAnUnfrozenNormalizationStillUpdatesItsStatistics` fails if it over-applies.*

### A seed makes the weights reproducible, not the run

`NFKMLXRandom.seed` fixes what a net starts from. Build the same net twice from the same seed and its
parameters are bit-identical. Measured over 30 builds on an M1 Max (macOS 26.6.2, mlx-swift 0.31.6),
the sum of every parameter was the same value on both devices every time.

Nothing after the weights is fixed. A forward pass is not reproducible on the CPU: over 180 builds
from one seed, the loss of the first forward took four distinct values between 0.51985323 and
0.52548116. A `BatchNorm` over a batch of one divides by that batch's own standard deviation, which
turns an accumulation difference near 1e-07 into a difference near 5e-03 in the loss. A forward pass
on the GPU is reproducible while no backward pass has run in the process, at 10 readings of one value
in evaluation mode and 10 in training mode.

A backward pass on the GPU is a separate matter, and it is wrong rather than noisy.

**The defect is in mlx core, and mlx core 0.32.0 fixes it.** The same graph transcribed into Python
reproduces the fault against `mlx` directly, which places it below the Swift bindings. The Python CPU
reference is 1.425528234774301e-07, matching the Swift reading. GPU readings matching the CPU with
the cache untouched, measured on this machine:

| mlx core | Readings matching the CPU |
| --- | --- |
| 0.31.1 | 0 of 25 |
| 0.31.2 | 2 of 25 |
| 0.32.0 | 25 of 25 |
| 0.32.2 | 60 of 60 |

mlx-swift 0.31.6 vendors mlx core 0.31.1, which is why this package still carries the workaround.
0.31.6 is also the newest mlx-swift tag. mlx-swift `main` vendors core 0.32.2.
`InferKitMLX/Package.swift` requires mlx-swift `from: "0.31.6"`, so a 0.32.x tag is taken up when one
is published. Which of the 146 commits between 0.31.2 and 0.32.0 carries the fix is not identified. A
lone grouped strided convolution returns the correct gradient on 0.31.1, so the composed graph is
still what the fault needs.

**The fix is confirmed in Swift, not only in Python.** Pointing `InferKitMLX/Package.swift` at
mlx-swift `main`, which vendors core 0.32.2, and running the watch alone in a fresh process reports
the fault not observed at 20 of 20. The same command against the shipped 0.31.6 pin reports it still
present at 0 of 20. Same machine, same test, same graph, with the vendored core as the only
difference. `NFKMLXBufferCacheGradientTests` passes on both, because it asserts the CPU's accuracy
and the mitigations rather than the fault.

**Retire the workaround when mlx-swift ships a release vendoring core 0.32.0 or later.** Run
`swift test --filter NFKMLXUpstreamWatchTests` alone in a fresh process. Measure a training loop with
the cache on as well, which is the condition the default answers to: at core 0.31.1 the loss rose in
8 of 60 six-step runs with the cache left alone, and that figure is unmeasured on 0.32.x. When both
report the fault is not observed, ``NFKMLXTrainingCachePolicy/unchanged`` becomes the trainer
default and this section becomes history. Everything below records the defect as it behaves on core 0.31.1.

**The CPU is the accurate device, and this was arbitrated rather than assumed.** On the smallest graph
that shows the fault, a MobileNetV3-style stem and four inverted residuals at 32x32x3 under a
mean-of-squares loss, central finite differences along the CPU gradient's own direction give ratios
of 0.968, 0.9996, and 1.000 at steps of 1e-2, 1e-3, and 1e-4. A ratio converging on 1 as the step
shrinks is what a correct gradient does. The norm is 1.42553e-07, and the CPU returns it on 25 of 25
calls.

**The GPU is accurate on its first backward pass in a process and wrong after it.** The first call
returns 1.42912e-07, which the same finite-difference probe confirms at a ratio of 0.995. Of 25 calls
in one process with the buffer cache left alone, 1 matched. The other 24 took 11 distinct values
around 0.127 and 0.377, wrong by a factor near a million, with no infinity and no not-a-number among
them.

**MLX's Metal buffer cache is involved, and the mechanism is not established.** Holding the cache
limit at zero gives 25 correct calls out of 25. Returning the cache to the system immediately before
each backward pass gives 25 out of 25. A cache that is already empty gives the correct answer as
well, which is why a reading taken after a training run can look healthy.

What the fault is not, each measured rather than assumed:

- It is not a race at the backward pass boundary. A `Stream.gpu.synchronize()` before each call gives
  1 correct out of 25, which is what doing nothing gives.
- It is not the gradient reading leftover values. Filling the cache with buffers set to 0.0, 1.0,
  1e3, and 1e6 leaves the wrong answer unchanged at about 0.127309 in every case. A gradient reading
  those bytes would move with them. Whether that fill reaches the buffers the backward reuses is not
  established, so this narrows the explanation rather than closing it.
- It is not random. The wrong answer is about 0.127309 on run after run, which is a definite wrong
  value rather than garbage.

Two things unrelated to the cache also make it correct, and both point at how much of the lazy graph
is evaluated. Materializing the parameters before calling `valueAndGrad` gives 12 correct out of 12.
Evaluating the loss value together with the gradient, rather than evaluating the gradient alone,
gives 12 out of 12. Neither helps a training loop: over 20 six-step runs with the cache left on, the
loss rose 2 times evaluating the value afterward and 5 times evaluating it alongside the model. Only
the cache limit fixes the loop.

This reaches a real fine-tune. Over 60 six-step runs of the matting net's tiny configuration on the
GPU, the loss ended above where it started 8 times with the cache left alone, 4 times with the cache
returned before each step, and 0 times with the limit held at zero. The median final loss over those
runs is 0.496, 0.440, and 0.322 against a first loss of 0.520.

Returning the cache before each step is a partial measure, because the step refills the cache before
its own backward pass runs. The gradient cannot be corrected between the forward and the backward,
because `valueAndGrad` runs both in one call and MLX builds the graph lazily, so a clear inside the
loss closure runs while the graph is being built and before anything executes.

**Rule:** hold the buffer cache at zero for the duration of a GPU training run, which
``NFKMLXTrainingCachePolicy/disabledOnGPU`` does and ``NFKMLXTrainer`` uses by default. Code that
calls `valueAndGrad` directly does the same, or accepts that only its first gradient in the process
is trustworthy.

The cost is throughput, and the cache is not holding down the model's footprint. Measured over 10
steps of a 40-layer 256-channel stack, 23.6M parameters at 2776.2 MB of active memory: the run takes
2.39 to 2.79 seconds with the cache, and 3.02 seconds without it, which is 15% to 26%. Peak active
memory is 2776.2 MB either way, while the cache holds a further 4.58 GB that the run no longer holds
when the limit is zero. On a small model the setting is faster rather than slower, at 3.45 seconds
against 4.82 over 40 steps of the matting net.

This is specific to what a graph does. Fourteen other graphs were reproducible on both devices and
agreed with each other to five or six digits: smooth stacks up to sixteen deep, global-pool gating,
bilinear resampling, skip concatenation, and stacks built on `relu`, `hardswish`, and `hardsigmoid`.
Single layers are exact on both devices. Within the matting net the disagreement appears at the
fourth inverted residual, where the CPU reads 1.43e-07 and the GPU reads 0.373; through the third the
two devices agree to three digits.

One reading is unexplained. The first six-step run in a fresh GPU process reported an anomalous first
loss 19 times in 20 with the cache limit at zero, against 7 in 20 with the cache returned each step
and 4 in 20 with it left alone. Later runs in the same process do not show it. No mechanism for this
is established.

*Probes: `NFKMLXBufferCacheGradientTests` holds the reference norm and the agreement under the
policy. `NFKMLXTrainingDeterminismTests` holds what a seed does and does not fix.
`NFKMLXGradientDeterminismTests` holds the layer-by-layer agreement. `NFKMLXUpstreamWatchTests`
reports whether MLX still has the defect, and does not fail while it does.*

### Training-mode work on the CPU kills the process

A training-mode forward pass on the CPU ends the process outright, at a measured rate near one run in
ten. There is no exception to catch. The suite prints "0 failures" for a run that died part way
through, so an exit code is the only honest reading of a test run.

Two stacks appear. The CPU one faults on unmapped memory inside MLX's own convolution, on MLX's
scheduler thread:

```
SIGSEGV KERN_INVALID_ADDRESS / SIGBUS KERN_PROTECTION_FAILURE
  mlx::core::slow_conv_2D<float>
  mlx::core::scheduler::StreamThread::thread_fn()
```

The GPU one throws from a Metal completion handler, where nothing can catch it:

```
SIGABRT  [METAL] Command buffer execution failed: Invalid Input
  mlx::core::gpu::check_error(MTL::CommandBuffer*)
  thread com.Metal.CompletionQueueDispatch
```

What triggers it is training mode, not the backward pass. Measured over 20 processes each: 60
evaluation-mode forward passes crashed 0 times, 60 training-mode forward passes with no backward pass
anywhere crashed 2 times, and a twelve-step training run crashed between 1 and 4 times depending on
the cache policy. Pooled over every run today, a CPU training run crashed 5 times in 62 with the
cache untouched, 11 times in 114 with the cache returned each step, and 9 times in 50 with the limit
at zero. The same work on the GPU crashed 0 times in 122 processes under every policy.

The cache limit belongs in that list because zero is the setting that returns pages to the system,
which is the operation the faulting stack implicates. This is why
``NFKMLXTrainingCachePolicy/disabledOnGPU`` leaves a CPU run's cache alone.

`slow_conv_2D` is reached for a reason worth knowing, because it decides which models are exposed.
`conv_2D_cpu` uses an explicit-GEMM convolution only when every dilation is 1 and the group count is
1, and calls `slow_conv_2D` otherwise. A depthwise convolution has a group count equal to its channel
count, so every MobileNetV3 inverted residual takes the faulting path on the CPU, as does any dilated
convolution. Measured on the CPU, a depthwise convolution costs 6.98 times a dense convolution of the
same shape, which is the two paths showing themselves.

**Rule:** do not pin a training run to the CPU. `NFKMLXDevice.perform(on: .cpu)` around a fine-tune is
the way into this, and a fine-tune has no reason to be there: the GPU is correct under the trainer's
default policy and does not crash. Evaluation-mode inference on the CPU is unaffected.

*Watch: `NFKMLXUpstreamWatchTests` times a depthwise convolution against a dense one and reports when
MLX routes them the same way. It does not run the faulting path, because a crash truncates the suite
rather than reporting anything.*

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

These were found while building this package and are recorded in `Docs/agent-reference/mlx-runtime-gotchas.md`. They are repeated here
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
