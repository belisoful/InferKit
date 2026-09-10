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
