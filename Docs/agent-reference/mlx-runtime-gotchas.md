<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX runtime gotchas

Hazards measured in this package against mlx-swift; the public catalogue is `Docs/mlx-runtime-hazards.md`.

- MLX runtime: MLX needs `default.metallib` whether or not anything runs on the GPU — the first
  stream request initializes the scheduler, which constructs the Metal device, so even
  `mlx_default_cpu_stream_new` throws without it (measured: a probe pinned to `Device(cpu, 0)` aborts
  with the library absent and runs convolutions and a full backend inference with it present). It is
  built and bundled by the **Xcode build system**
  (`mlx-swift_Cmlx.bundle/Contents/Resources/`) but a plain CLI `swift build`/`swift test` does not —
  so MLX array evaluation aborts under `swift test`. Run MLX-eval tests via
  `xcodebuild test -scheme InferKitMLX -destination 'platform=macOS' -skipPackagePluginValidation`
  (the `-skip…` flag gets past the unrelated `CudaBuild` plugin's validation). The matting round-trip
  tests auto-detect this: they skip when the test bundle is under `.build` and run under xcodebuild.
- Runtime hazards are catalogued publicly in `Docs/mlx-runtime-hazards.md`, with an executable
  probe for each in `NFKMLXRuntimeHazardTests`. Four hazards reported elsewhere against mlx-swift
  0.31.6 — the fused attention dropping the cached tail at query length 1, `MLXFast.RoPE` disagreeing
  at `T == 1`, subscript-set through an optional property not persisting, and `eval` faulting on a
  fresh thread — were reduced and probed here and **none reproduces**, in float32 or bfloat16, at any
  cache length from 1 to 129. That is not proof the reports were wrong; it is proof this package's
  usage pattern is unaffected, which is the question a port needs answered. Re-run the probes after an
  mlx-swift bump.
- A quantized layer's `weight.dtype` is the packed storage type, `uint32`, not what the layer
  computes in. Aligning activations to it truncates them to integers, identically at every bit width,
  with no error. Read `scales.dtype`. And `QuantizedLinear` is a subclass of `Linear`, so
  `layer as? Linear` matches one without announcing it, which is a live hazard for any model surgery
  selecting by type. `NFKMLXLoRA.apply(to:)` rejects a quantized layer explicitly for that reason.
- Metal flushes subnormal floats to zero; the CPU stream keeps them. Measured:
  `MLXArray([1e-21]).square()` is `0.0` on the GPU and `1e-42` on the CPU, same machine, same process.
  Anything that squares small numbers — a norm, a variance, a cosine over tiny vectors — can read
  exactly zero. Reduce toward a magnitude the type holds comfortably, never away from one.
- A lazy decode pins its sources and intermediates. `NFKMLXDeepSeek.dequantized` evaluates each
  entry as it is produced; returning lazy graphs would hold the whole shard's decode live at once, and
  for a block-scaled format the expanded scale array alone is the weight's full size.
- Never pass `padding:` to an mlx-swift pooling layer. `Pool.callAsFunction` (mlx-swift 0.31.6)
  builds its pad widths as `[0, 0] + padding + [0, 0]`, two entries too many: a four-axis input gets the
  first four, so a 2-D pool pads width and channels instead of height and width. It raises nothing —
  the output shape is silently wrong, and the failure surfaces later as a channel mismatch reported
  against an innocent layer. Every `Pool` subclass (1-D/2-D/3-D, max and average) shares that
  initializer. Pool through `NFKMLXResample.maxPooled` / `.averagePooled`, which border the input
  explicitly and then window at padding zero. Pooling with no padding is unaffected.
- A size too large to run here is held to the module by shape, and the convention is fixed.
  `Tools/validation-assets/shapes.py <repo> <dir>` fetches a release's `config.json` and every
  tensor's shape from its safetensors headers by HTTP range request (no weights; about a megabyte for
  a 54 GB release) into `~/.inferkit-validation/shapes/<name>/{shapes.json,config.json}`, and
  `IK_SHAPES_ROOT` in `~/.inferkit-validation.json` names that directory. `NFKMLXReleasedSizesTests`
  reads a release through `shapes(name)`, builds the module from the release's own config, and
  `assertStructure` compares `net.parameters().flattened()` — converted to the release's names and
  layouts (a 4-D convolution back to `[out, in, kH, kW]`, a transposed one to `[in, out, kH, kW]`) —
  against the inventory in both directions, reporting consumed / missing / mismatched / named-dropped /
  unaccounted, with a closure naming what the loader deliberately drops. Every structural row in
  `Docs/model-parity.md` reads 0 missing, 0 mismatched, 0 unaccounted; the manifest's `shapes` section
  lists the 27 repositories captured. A structural pass is not a numeric one (the DeepSeek and Gemma 4
  lessons above), so a family's arithmetic rests on the size that is measured, and a checkpoint header
  captured by hand rather than by `shapes.py` is a record nobody can regenerate.
- A test process that loads many models back to back must clear MLX's cache between them. The
  GPU cache survives from test to test, and the accumulation starves the largest float32 forward
  (Gemma E2B, ~20 GB) into a Metal command-buffer timeout — a process kill that truncates the run
  with "0 failures" reported. Measured: the same test passes in 25 s alone and dies mid-suite.
  `NFKMLXReferenceParityTests` clears in `tearDown` (`NFKMLXGPU.clearCache()`).
- Never assign to a `@ParameterInfo` or `@ModuleInfo` property. `attention.sink = newValue` aborts
  the process with "please call update() on the array rather than setting it" — not a thrown error, a
  fatal one, which in a test run kills the process and silently truncates the reported test count (a
  suite of 10 reported 6 and still said "0 failures"). Mutate through
  `update(parameters: ModuleParameters.unflattened([...]))` instead. Same family as the numeric-key
  trap below.
- Never hand `ModuleParameters.unflattened` two entries with the same key. It recurses to a
  stack overflow — a SIGSEGV process kill, not an error. A Python dict deduplicates the same
  collision silently, so a remap ported from a converter carries the hazard invisibly; a rename that
  deliberately collides two aliases of one shared tensor (RAFT's `norm3`/`downsample.1`) must build
  into a `[String: MLXArray]` first.
- Never give a `@ModuleInfo` a numeric key (`@ModuleInfo(key: "0")` to mirror a reference
  `nn.Sequential` position). MLX's `update(parameters:)` parses a numeric key as an **array index**, so
  the unflattened checkpoint arrives as a list where the module tree has a child module, and the update
  aborts the process (`try!` over `UpdateError.incompatibleItems` inside MLXNN). Use semantic keys
  (`conv`, `bn`) and translate the reference's positions in the model's `remapReferenceKey` — RVM's
  covers every form. A genuine `[Module]` array property is fine; that is what numeric keys are for.
- Padding modes: MLX offers `.constant` and `.edge` only. Reflection padding is
  `NFKMLXResample.reflectPadded` — a border approximation is not cosmetic in a network that normalizes
  over whole feature maps (it cost style transfer a 0.049 mean pixel error).
- Download-and-build coverage (`NFKMLXDownloadTests`): the hub/factory happy path is tested hermetically
  with no network — a real safetensors is written to the exact cache location the hub resolves, so
  `NFKHFHub` cache-hits and the sync + async factories build from it (proving the download → registry →
  factory → `loadWeights` chain and byte-exact weight flow). A separate **live download** test through
  `NFKMLXHub` is gated behind `INFERKIT_LIVE_MODEL` / `_REPO` / `_WEIGHTS_PATH` (optional `_REVISION`) and
  skips unless they are set, so CI stays green while the real Hugging Face path stays runnable on demand.
  Per-model correctness against real trained weights (the validation sweep) still needs a converted
  checkpoint fed to that live test.

- **`freeze()` does not stop a `BatchNorm` from training.** `train(_:)` sets the flag on every module
  in the tree and freezing marks parameters as taking no gradient; the two are independent, and
  `BatchNorm` branches on the flag alone. A frozen backbone under a head-only fine-tune therefore
  normalizes with the training batch's statistics instead of the released ones, and folds the batch
  into `running_mean` / `running_var`, which a checkpoint write persists. `NFKMLXTrainer` returns every
  wholly frozen subtree to evaluation mode after `train(true)`; a subtree with no parameters at all
  follows its parent, so a dropout in the trainable group still drops. Consumer-facing write-up and
  probes: `Docs/mlx-runtime-hazards.md`.

- **A seed fixes the weights; it does not fix the gradients.** `NFKMLXRandom.seed` makes a net's
  initialization exactly reproducible, and a test that reads a training loss looks deterministic
  because of it. It is not. Measured 2026-09-19 on an M1 Max (macOS 26.6.2, Xcode 27), seeding
  `20_260_904` and building `NFKMLXRVMNet(.tiny)`: the parameter sum is bit-identical across runs,
  while six SGD steps from those identical weights land somewhere different on every run. Over 12 seeded GPU runs the final loss ranged 0.13 to 0.54 against that
  first loss of 0.52, and the run ended higher than it started **5 times out of 12**. The same test
  passed three times out of three in isolation, which is what made it read as order-dependent when
  it is simply a coin flip.
  - Most of the nondeterminism enters at the backward pass. The loss recorded at step 0, taken before
    that step's update, read `0.5203694` on all 24 CPU runs and on 11 of the 12 GPU runs, and step 1
    onward moves every time. **Superseded in part:** the reading of three identical training-mode
    forwards was three samples. At 180 builds the CPU forward takes four distinct values between
    0.51985323 and 0.52548116, so the forward is not exact either. See "The buffer cache corrupts a
    backward pass" below.
  - It is not GPU-only. The CPU device is not bitwise reproducible either, but the same six steps
    there land between 0.171 and 0.208 over 24 runs against the same first loss, so the margin is
    roughly eight times the spread. On the GPU the six-step margin is smaller than the spread, and
    lengthening the run does not buy enough: at 12, 20, and 30 steps the worst final loss was 0.502,
    0.472, and 0.438, all inside the band the six-step runs already covered.
  - **Superseded:** that test was pinned to the CPU with `NFKMLXDevice.perform(on: .cpu)`, and is not
    any more. A CPU training run kills its process about one time in ten, and the GPU is both correct
    and crash-free under the trainer's default cache policy. See "Training-mode work on the CPU kills
    the process" below.
  - The other loss-descent assertions in the package have margins that clear this noise by an order
    of magnitude and are left alone: `NFKMLXTrainerTests` asks for a tenfold fall over 100 steps, and
    `NFKMLXZeroDCETrainingTests` measured 2.57 → 0.13 over 60 steps with a spread under 0.01.
    `NFKMLXTrainingDeterminismTests` pins both halves of the finding.
  - The rule this leaves: a single-digit-step training run is noise on this runtime. Assert on
    something the noise cannot reach, such as parameters moving or a long run's fall on a pinned
    device. Never assert on the shape of a short loss curve.

- **A wrong backward pass on the GPU, fixed in mlx core 0.32.0 (2026-09-19, resolved 2026-09-20).**
  Measured on an M1 Max (macOS 26.6.2, Xcode 27, mlx-swift 0.31.6).
  - **The fault is in mlx core and the current release does not have it.** The graph was transcribed
    into Python `mlx.nn` and reproduces against `mlx` directly, so it sits below the Swift bindings.
    The Python CPU reference is 1.425528234774301e-07 against the Swift 1.42553e-07, which is how the
    transcription is known to be the same graph. GPU readings matching the CPU with the cache
    untouched: 0 of 25 on core 0.31.1, 2 of 25 on 0.31.2, 25 of 25 on 0.32.0, and 60 of 60 on 0.32.2.
    `clear_cache()` and `set_cache_limit(0)` each give 25 of 25 on 0.31.1, as they do in Swift.
    Python's `synchronize()` gives 25 of 25 where Swift measures 1 of 25, which is unexplained and
    does not change the conclusion.
  - **The fixing commit is not identified.** 146 commits separate 0.31.2 from 0.32.0. `Fix conv2
    gradients in grouped strided case on Metal (#3800)` matches by title and lands 2026-07-07, after
    the mlx-swift 0.31.6 tag of 2026-07-02. It is not confirmed: a lone grouped strided convolution
    returns the correct gradient on 0.31.1, at 8 of 8 for group counts 1 and 16 against strides 1 and
    2. The composed graph is still required, which agrees with the kernel survey below.
  - **Nothing is filed upstream for this.** The defect is fixed in the current release, so there is
    no report to make. The drafted report is withdrawn.
  - **Confirmed in Swift against mlx-swift `main` (2026-09-20).** Pinning the package to mlx-swift
    `main`, which vendors core 0.32.2, and running the watch alone in a fresh process reports the
    fault not observed at 20 of 20, against 0 of 20 on the shipped 0.31.6 pin under the same
    command. The build takes no source change, and `NFKMLXBufferCacheGradientTests` passes on both
    runtimes. What is still unmeasured on 0.32.x is a training loop with the cache left on, which is
    the condition the trainer default answers to. Measure that before flipping it.
  - **The exit condition.** mlx-swift's newest tag is 0.31.6, vendoring core 0.31.1. mlx-swift `main`
    vendors core 0.32.2. `InferKitMLX/Package.swift` requires `from: "0.31.6"`, so a 0.32.x tag is
    taken up when one is published. On that bump, run `swift test --filter NFKMLXUpstreamWatchTests`
    alone in a fresh process, and make ``NFKMLXTrainingCachePolicy/unchanged`` the trainer default
    once the watch reports the fault is not observed. The bullets below record the defect as it
    behaves on core 0.31.1.
  - **The CPU is the accurate device, arbitrated rather than assumed.** On the smallest graph that
    shows the fault, central finite differences along the CPU gradient's own direction give ratios of
    0.968, 0.9996, and 1.000 at steps of 1e-2, 1e-3, and 1e-4. The norm is 1.42553e-07, returned on
    25 of 25 calls. Finite differences are built from forward passes alone, which is what makes them
    able to arbitrate between two devices that disagree about a gradient.
  - **The GPU is accurate on its first backward in a process.** It returns 1.42912e-07, confirmed at
    ratio 0.995. Of 25 calls in one process with the cache left alone, 1 matched; the other 24 took
    11 distinct values near 0.127 and 0.377.
  - **Two cache mitigations work and a synchronize does not.** Holding `cacheLimit` at zero gives 25
    of 25. Calling `clearCache()` immediately before each backward gives 25 of 25. A
    `Stream.gpu.synchronize()` before each backward gives 1 of 25, which is what doing nothing gives,
    so this is not a race at the backward boundary. An already-empty cache is also correct, which is
    why a reading taken after a training run can look healthy.
  - **What it is not, each measured.** It is not the gradient reading leftover values: filling the
    cache with buffers set to 0.0, 1.0, 1e3, and 1e6 leaves the wrong answer unchanged at about
    0.127309 in every case, and a gradient reading those bytes would move with them. Whether that
    fill reaches the buffers the backward reuses is not established, so this narrows the explanation
    without closing it. It is not random either: 0.127309 recurs run after run, which is a definite
    wrong value rather than garbage. An earlier draft of this entry asserted a recycled buffer that
    is not fully initialized; that is the hypothesis the fill was meant to confirm, and it did not.
  - **How much of the lazy graph is evaluated also decides it.** Materializing the parameters before
    calling `valueAndGrad` gives 12 of 12 correct. Evaluating the loss value together with the
    gradient, rather than the gradient alone, gives 12 of 12. Neither helps a training loop: over 20
    six-step runs with the cache on, the loss rose 2 times evaluating the value afterward and 5 times
    evaluating it alongside the model. The isolated probe and the training loop therefore do not
    respond to the same lever, and only the cache limit fixes the loop.
  - **Clearing per step is not enough inside a training loop,** because the step refills the cache
    before its own backward runs. Over 60 six-step GPU runs the loss ended above where it started 8
    times with the cache left alone, 4 times clearing per step, and 0 times with the limit at zero.
    Median final loss 0.496, 0.440, 0.322 against a first loss of 0.520. There is no way to clear
    between the forward and the backward: `valueAndGrad` runs both in one call, and MLX builds the
    graph lazily, so a clear inside the loss closure runs before anything executes.
  - **`NFKMLXTrainer` defaults to ``NFKMLXTrainingCachePolicy/disabledOnGPU``,** which holds the limit
    at zero for the run and restores it in a `defer`. The limit is process-wide, so a concurrent
    inference on another thread allocates without a cache until the run returns, and two concurrent
    trainers race on the restore. A CPU run is left alone deliberately; see the CPU crash entry below.
  - **Cost is throughput, and the cache is not holding the footprint down.** Over 10 steps of a
    40-layer 256-channel stack (23.6M parameters, 2776.2 MB active): 2.39 to 2.79 seconds with the
    cache, 3.02 without, so 15% to 26%. Peak active memory is 2776.2 MB either way, and the cache
    holds a further 4.58 GB. On the tiny matting net the setting is faster, 3.45 seconds against 4.82
    over 40 steps.
  - **It is specific to the graph.** Fourteen synthetic graphs are reproducible on both devices and
    agree to five or six digits, including smooth stacks to depth sixteen, global-pool gating,
    bilinear resampling, skip concatenation, and stacks of `relu`, `hardswish`, and `hardsigmoid`.
    Every single layer is exact. Within the matting net the disagreement appears at the **fourth**
    inverted residual: through the third the devices agree to three digits, at the fourth the CPU
    reads 1.43e-07 and the GPU 0.373.
  - **A seed fixes the weights and nothing after them.** The parameter sum is identical over 30 builds
    on both devices. The first forward's loss took four distinct values over 180 CPU builds, between
    0.51985323 and 0.52548116, because a `BatchNorm` over a batch of one divides by that batch's own
    standard deviation and turns a 1e-07 accumulation difference into a 5e-03 difference in the loss.
    A GPU forward is exact while no backward has run in the process.
  - **Unexplained.** The first six-step run in a fresh GPU process reported an anomalous first loss 19
    times in 20 with the limit at zero, 7 in 20 clearing per step, and 4 in 20 with the cache left
    alone. Later runs in the same process do not show it. Recorded rather than explained.
  - **How the first pass at this went wrong, because the trap is cheap to repeat.** An earlier reading
    concluded from six samples per condition that clearing per step was a complete fix, and that
    every forward pass was exact. Both fell over at 25 and 60 samples. A mitigation for a fault that
    appears in 24 of 25 calls looks total at n=6 whatever it actually does. Size the sample to the
    claim before writing the claim down.

- **Training-mode work on the CPU kills the process (2026-09-19).** A CPU training-mode forward ends
  the process at a rate near one run in ten. There is no exception to catch, and the suite prints
  "0 failures" for a run that died part way through, so read the exit code.
  - **Two stacks.** The CPU one faults on unmapped memory in MLX's convolution on MLX's own scheduler
    thread: `mlx::core::slow_conv_2D<float>` under `mlx::core::scheduler::StreamThread::thread_fn()`,
    as SIGSEGV `KERN_INVALID_ADDRESS` or SIGBUS `KERN_PROTECTION_FAILURE`. The GPU one throws from a
    completion handler where nothing can catch it: `mlx::core::gpu::check_error(MTL::CommandBuffer*)`
    on thread `com.Metal.CompletionQueueDispatch`, reported as
    `[METAL] Command buffer execution failed: Invalid Input`.
  - **Training mode is the trigger, not the backward.** Over 20 processes each: 60 evaluation-mode
    forwards crashed 0 times, 60 training-mode forwards with no backward crashed 2 times. Pooled over
    every run: a CPU training run crashed 5 in 62 with the cache untouched, 11 in 114 clearing per
    step, and 9 in 50 with the limit at zero. The same work on the GPU crashed 0 times in 122
    processes under every policy.
  - **Zero is the aggressive setting,** which is why the trainer leaves a CPU run's cache alone.
    Returning pages to the system is the operation the faulting stack implicates.
  - **Which models are exposed, and why.** `conv_2D_cpu` uses `explicit_gemm_conv_ND_cpu` only when
    every `wt_dilation` and `in_dilation` is 1 and the group count is 1, and calls
    `dispatch_slow_conv_2D` otherwise. A depthwise convolution has a group count equal to its channel
    count, so every MobileNetV3 inverted residual takes the faulting path on the CPU, as does any
    dilated convolution. Measured on the CPU, a depthwise convolution costs 6.98 times a dense one of
    the same shape.
  - **Both defects are watched, not asserted.** `NFKMLXUpstreamWatchTests` reports whether a later GPU
    backward still disagrees with the CPU, and times a depthwise convolution against a dense one to
    report whether MLX still routes them differently. Both pass either way, because a red suite for a
    defect no change here can fix teaches a maintainer to ignore the suite. The backward watch is
    deterministic only in a fresh process, at 0 of 20 on three runs out of three; after a training
    test has run it gave 12, 16, and 20 of 20, because the trainer leaves the cache empty behind it.
    A clean reading therefore prints `not observed in this process` rather than any claim of a fix,
    and the authoritative check is `swift test --filter NFKMLXUpstreamWatchTests` on its own.

- **Whether to own a convolution until MLX fixes it (2026-09-19, analysis, not adopted).** Replacing
  `slow_conv_2D` is reachable without touching C++: a depthwise convolution is a strided gather into
  patches followed by a broadcast multiply and a sum over the window, and a dilated convolution is a
  dense convolution over interleaved sub-grids. Both are MLX ops, both avoid the faulting path, and a
  CPU-only switch on `NFKMLXDevice.currentType` would confine the change to the device that needs it.
  - **The case for.** The faulting path is also 6.98 times the cost of the GEMM path, so a replacement
    plausibly makes CPU depthwise inference several times faster. That argument stands on its own and
    survives an upstream fix.
  - **The case against, which is the reason it is not adopted.** The exposed population is small and
    already has a better route. Evaluation-mode CPU inference does not crash, measured at 0 in 1200
    forwards, so the only consumers at risk are those pinning a *training* run to the CPU, and the
    documented answer for them is to train on the GPU, which is both correct and crash-free. Against
    that, the cost is a convolution reimplementation validated against every shipped model that uses
    a depthwise or dilated convolution, at the package's own standard of measured reference parity,
    plus an im2col materialization of roughly nine times the activation for a 3x3 kernel.
  - **What would change the decision.** A consumer requirement for CPU-only inference of MobileNet
    class models, where the 6.98x would be the point rather than the crash. If it is taken up, it is
    a performance change with a correctness side effect, gated on its own parity run, and removed
    when `UPSTREAM WATCH cpu grouped convolution` reports `APPEARS FIXED`.

- **Upstream reports: one withdrawn, one without a reproduction (2026-09-19, revised 2026-09-20).**
  The gradient report is withdrawn, because mlx core 0.32.0 already fixes what it describes. The
  convolution crash has no script upstream can run: the same workload in Python, with training mode,
  depthwise and dilated convolutions, a backward pass and an optimizer step, 12 steps at 256x256 on
  the CPU stream, killed 0 of 40 processes on core 0.31.1 and 0 of 40 on 0.32.2. Against the Swift
  rate near 5 in 62, 0 of 40 has a probability near 0.036, which is evidence against a
  Python-reachable defect rather than noise. It does not say where the fault is. It does say that a
  report belongs to mlx-swift rather than mlx, and that it waits on a standalone Swift reproduction.
  The text below is kept for that report.
  - **Withdrawn, kept for the record.** Build `NFKRVMBackbone(NFKMLXRVMConfiguration.tiny)`,
    call `train(false)`, feed `[1, 32, 32, 3]`, take the stem and the first four blocks, and use
    `(out * out).mean()` as the loss. Call `valueAndGrad` 25 times in one process on the GPU. The
    first norm is 1.42912e-07 and 24 of the remainder take 11 distinct values near 0.127 and 0.377.
    `MLX.Memory.cacheLimit = 0` or `MLX.Memory.clearCache()` before each call makes all 25 return the
    first value; `Stream.gpu.synchronize()` does not. Materializing the parameters before the call,
    or evaluating the returned loss value alongside the gradient rather than the gradient alone, also
    makes all of them correct, so how much of the lazy graph is evaluated is part of it. Filling the
    cache with buffers set to 0.0, 1.0, 1e3, or 1e6 leaves the wrong value unchanged at 0.127309, so
    the gradient is not reading those bytes. The CPU returns 1.42553e-07 every time and
    central finite differences confirm it at ratio 1.000. `NFKMLXBufferCacheGradientTests` holds it in
    runnable form. Note against [ml-explore/mlx#3689](https://github.com/ml-explore/mlx/issues/3689)
    that this reproduction uses stock `MLXNN` through `valueAndGrad` with no custom extensions and no
    aliased snapshot buffers, which is the ground on which
    [PR #3688](https://github.com/ml-explore/mlx/pull/3688) was closed.
  - **`slow_conv_2D` faults on unmapped memory.** Run a training-mode forward of a MobileNetV3-style
    net with depthwise convolutions on the CPU stream, repeatedly, in one process. About one process
    in ten dies in `slow_conv_2D<float>` on `scheduler::StreamThread::thread_fn`, as SIGSEGV
    `KERN_INVALID_ADDRESS` or SIGBUS `KERN_PROTECTION_FAILURE`. Evaluation mode does not reproduce it
    at 1200 forwards. Setting `cacheLimit = 0` raises the rate, which points at buffers being returned
    to the system while the CPU stream still reads them.

- **The composed backward is what moves, not any one kernel (2026-09-19).** Following the entry
  above, the gradients were compared directly rather than through a training curve, on an M1 Max
  (macOS 26.6.2, Xcode 27, mlx-swift 0.31.6). What the comparison settles, and what it does not:
  - **No single kernel is at fault.** A plain convolution, a depthwise convolution, a grouped
    convolution, a `BatchNorm` in training mode, and a `Linear`, each on its own, give a bitwise
    identical gradient on repeat on both devices, and the two devices agree to five or six digits.
    So do a squeeze-excitation block, a whole inverted residual with one, a recurrent gate from a nil
    state, and stacks of up to sixteen convolution-and-normalization pairs. `NFKMLXGradientDeterminismTests`
    keeps that set as a probe.
  - **The forward is far steadier than the backward, and the same 66 of 1024 output pixels sit inside
    the clamp every time.** **Superseded in part:** "exact" came from three readings. A GPU forward is
    exact while no backward has run in the process, and a CPU forward takes four distinct values over
    180 builds. The backward still moves by a factor of a million, which the forward never approaches.
  - **The composed backward is not.** For the same net at the same seeded weights, the CPU gradient
    norm repeats as 2.9022650575922 to fourteen digits, while the GPU returns 7.17, 57.6, 295.5,
    385.2, 67.0, 28.2, 327.5, and 497.1 across runs and processes. In evaluation mode the CPU repeats
    0.14247507032592 while the GPU ranges 0.1425 to 8.25. The values hold no infinities and no
    not-a-numbers; the GPU simply reports a different, plausible-looking gradient each time.
  - **The loss this was found on is also ill-conditioned, which is a separate fact.** The matting
    net's alpha is a clamp rather than a sigmoid, and at random initialization 958 of 1024 pixels sit
    on the floor, where they contribute exactly nothing. The whole gradient comes from the remaining
    66. That makes the surface kinked, which is why a finite-difference probe cannot arbitrate here:
    stepping along the CPU's own gradient direction implies a true norm of at least 0.98, and
    stepping along a GPU run's implies 0.37. Two contradictory bounds mean the function is not
    differentiable at that point, not that one device is right.
  - **Answered by the entry above.** The cause is the buffer cache and the CPU is the accurate device.
    Reclaiming the cache makes a lone backward accurate, and holding the cache limit at zero is what a
    training loop needs. The clamp reading below stands as a separate fact about that test's
    conditioning, and is not what made the gradients move.
