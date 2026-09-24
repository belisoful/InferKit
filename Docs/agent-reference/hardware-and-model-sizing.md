<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# Sizing a model against the machine

`NFKHardwareProfile` (core, sysctl + Metal) reports what the machine is and what is free. Three
ceilings, and they are not interchangeable: `physicalMemory` is what is installed,
`recommendedWorkingSetSize` is what Metal expects to stay resident and is what a model should be sized
against, and `maximumBufferLength` bounds a single allocation however much of the budget is unspent.
On an M1 Max those read 32 GB, 25 GB, and 18.7 GB. `availableMemory` is live — free, inactive and
purgeable pages on macOS; `os_proc_available_memory` on iOS and tvOS, where the process's own
allowance is the ceiling that actually applies. Every reading degrades to zero or an empty string
rather than throwing, so an unrecognized machine still reports.

`NFKMLXModelSizing` (companion) turns that into a decision. `parameterCount(of:)` counts a dense
decoder from its geometry alone — counted rather than built, because the point is to answer before
allocating anything and a 27B model cannot be instantiated to be measured — and
`testTheParameterCountMatchesABuiltModule` checks the arithmetic against a module that is built,
across five geometries, which is what keeps the count from being a plausible guess. The cache is
counted at the KEY-value head count, not the query count; grouped-query attention is what makes it
affordable and the wrong reading overstates it twofold on Qwen3.

`fit(of:tokens:precision:budget:)` returns `fits` / `fitsWithinWindow(n)` / `tooLarge(shortfall:)`, and
`options(for:requesting:...)` hands back `NFKMLXGenerationOptions` with `contextWindow` **derived**
rather than guessed — a hand-picked window is a guess about a machine the author was not using. A model
whose weights alone do not fit throws there, because failing at the load is a process kill rather than
an error.

The bandwidth is measured, not tabulated. No sysctl reports memory bandwidth, and a per-chip table
would be numbers copied from somewhere rather than a property of the machine running the code. The
probe reads a large array and times it, which is what a decode step does to the weights. It has to
be big enough: swept on an M1 Max (400 GB/s specified) it reads 40 GB/s at 16 MB, 126 at 64 MB, 158
at 256 MB, 274 at 512 MB, and settles near 330 from 1 GB — below half a gigabyte the launch overhead
and the caches are most of what is timed. The size therefore comes from the working-set budget rather
than a constant. `decodeCeiling` divides bandwidth by the bytes a token reads;
`achievedFraction` inverts it, and a rate above the ceiling means the model is not reading every
parameter, which is what a sparse model doing its job looks like.

The cache is process-wide, which makes test order matter: a suite that measures a small probe first
would have every later reading report that. `resetMeasuredBandwidth()` is why the reporting test
starts by clearing it.

## Neural accelerators, and how to measure them

Apple silicon carries matrix units ("NAX" in MLX) from the M5 on. `NFKHardwareProfile` reports
`graphicsGeneration` and `hasNeuralAccelerators`, which are MLX's own gate rather than a reading of
our own, so a claim here and a kernel there cannot disagree. The gate, from
`mlx/backend/metal/device.cpp`: the OS is 26.2 or newer, and the generation parsed from
`MTLDevice.architecture.name` is at least 17, or 18 when the name's last character is `p`, a phone
GPU.

**The parse is positional, not by prefix.** MLX takes the two characters before the last one and
treats a non-digit as zero, so `applegpu_g13s` is 13 and a device name such as `Apple M1 Max` is 0.
Reading "the integer after g" instead would agree on today's names and diverge on a name with a
longer suffix. `graphicsGenerationForArchitecture:` and `architectureHasNeuralAccelerators:` are
class methods taking a name, so the M5 and M6 answers are pinned by tests on hardware that has
neither, and only the performance question is left for the machine.

Measured on the development machine, 2026-09-21, as the baseline an M5 reading sits beside:

| Reading | Apple M1 Max, macOS 26.6.2 |
|---|---|
| `graphicsArchitecture` | `applegpu_g13s` |
| `graphicsGeneration` | 13 |
| `hasNeuralAccelerators` | NO |
| `MTLGPUFamilyApple7` / `Apple8` / `Apple9` | YES / NO / NO |
| `MTLGPUFamilyApple10` / `Apple11` | NO / NO |
| `MTLGPUFamilyMetal4` | YES |

The OS half of the gate already passes here, so on an M5 running this same OS the hardware alone
decides. Which Metal family an M5 reports, and whether `Apple10` or `Apple11` is the one that tracks
the accelerators, is the open question: both read NO here, so this machine cannot answer it.

### The procedure, when an M5 is in hand

1. Read the table above on the M5 and record it beside this one. That alone answers the family
   question and confirms the gate agrees with the hardware.
2. Check what MLX believes. `mlx-c` does not expose `is_nax_available()`, so the reading is indirect:
   set `MLX_METAL_GPU_ARCH=applegpu_g13s` and MLX parses that string instead of the device's, which
   forces its gate to fail on a machine whose hardware passes. A run with the variable and a run
   without it is an A/B on one machine and one build. It is a close proxy rather than a pure toggle:
   the same string also feeds MLX's per-chip buffer limits, so the spoofed run differs in more than
   the accelerators. Say so when recording the numbers.
3. Measure what the pinned core actually accelerates, which is matmul and attention, not everything:
   `steel_matmul` fused and split-K, quantized matmul that is transposed with `K % 64 == 0`, and
   scaled-dot-product attention at any head dimension except 80. A microbenchmark of those three
   needs no weights and no downloads, which is what makes it the first thing to run.
4. Then measure the toolkit's own matmul-bound work: prefill tokens per second on a language model,
   a vision tower forward, one diffusion step, and a restoration model. These are where the
   accelerators pay for a consumer.
5. Check the prediction this sizing model rests on: decode is bandwidth-bound, so `decodeCeiling` and
   `achievedFraction` should mean what they mean today, and decode tokens per second should not move
   with the accelerators. If it does move, the sizing model is wrong about that machine and this file
   is what needs rewriting.
6. Note the OS and the pin. On an OS 27 runtime the NAX gemm and gather kernels are JIT-compiled
   with the system's Metal compiler, and the address-space fix for Metal 4.1 is in mlx core 0.32.1.
   The companion pins a main revision vendoring core 0.32.2, so that fix is present; a tagged
   `mlx-swift` carrying core 0.32 did not exist as of 2026-09-21. Record the mlx core version the
   run used, read from `Source/Cmlx/mlx/mlx/version.h` in the checkout, rather than the mlx-swift
   version, because the two have diverged.

Record the numbers in this file with the chip, the OS build, and the mlx-swift version they were
taken on. A figure without those three is not reproducible.
