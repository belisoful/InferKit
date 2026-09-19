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
  - The nondeterminism enters at the backward pass, not the forward. Three training-mode forwards on
    the same weights score identically, and the loss recorded at step 0, which is taken before that
    step's update, read `0.5203694` on all 24 CPU runs and on 11 of the 12 GPU runs. Step 1 onward
    moves every time.
  - It is not GPU-only. The CPU device is not bitwise reproducible either, but the same six steps
    there land between 0.171 and 0.208 over 24 runs against the same first loss, so the margin is
    roughly eight times the spread. On the GPU the six-step margin is smaller than the spread, and
    lengthening the run does not buy enough: at 12, 20, and 30 steps the worst final loss was 0.502,
    0.472, and 0.438, all inside the band the six-step runs already covered.
  - So a training test that reads a loss is pinned to the CPU with `NFKMLXDevice.perform(on: .cpu)`,
    which is what `NFKMLXRVMTests.testAFineTuneMovesTheSqueezeExciteAndHardswishBlocks` now does.
    Pinning keeps the assertion exactly as strict rather than widening the threshold. The test's own
    question is whether gradients reach the squeeze-excitation gates through the hardswish blocks,
    which is about the autodiff graph, and the CPU answers that as well as Metal does.
  - The other loss-descent assertions in the package have margins that clear this noise by an order
    of magnitude and are left alone: `NFKMLXTrainerTests` asks for a tenfold fall over 100 steps, and
    `NFKMLXZeroDCETrainingTests` measured 2.57 → 0.13 over 60 steps with a spread under 0.01.
    `NFKMLXTrainingDeterminismTests` pins both halves of the finding.
  - The rule this leaves: a single-digit-step training run is noise on this runtime. Assert on
    something the noise cannot reach, such as parameters moving or a long run's fall on a pinned
    device. Never assert on the shape of a short loss curve.
