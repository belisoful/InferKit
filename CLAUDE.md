# InferKit Agent Guidelines

This file and `AGENTS.md` are one document under two names: edit `CLAUDE.md`, then `cp CLAUDE.md AGENTS.md`.
It carries the rules and the entry points only. The accumulated maintainer notes (per-model entries,
measured parity numbers, runtime hazards, provider probes, packaging recipes) live in
`Docs/agent-reference/`, one file per subject, indexed by `Docs/agent-reference/README.md`. Read the
file for the subject you are working on before changing that area, and write new notes there, never
here. An agent that learns something durable about a subject adds it to that subject's file.

## Project Overview

**InferKit** is a small, cross-platform inference toolkit for Objective-C. It provides a swappable
backend protocol, request/result value types, an async job handle, the shipped backends (mock,
in-process Core ML, an on-device Core ML language-model runner, OpenAI-compatible chat and
transcription clients, a submit-poll-fetch base, and runtime discovery — plus the companion-package
MLX and Foundation Models backends), a texture-tensor conversion, tokenizers, and a Hugging Face
model-download layer. It has no host-framework dependency, so any Metal/Apple app (macOS, iOS, tvOS)
can use it. The class prefix is `NFK`.

The package is source-distributed through both Swift Package Manager and CocoaPods. Two optional
companion packages build on the core without raising its platform floor or adding dependencies to
it: `InferKitMLX/` (MLX-backed inference, plus on-device fine-tuning of the models it ships, on Apple
Silicon) and `InferKitFoundationModels/` (a bridge to Apple's on-device system language model).

## Build & Test Commands

```bash
# Core (host platform, macOS)
swift build
swift test

# Cross-platform compile checks (the core supports macOS 11 / iOS 14 / tvOS 14)
xcodebuild build -scheme InferKit -destination 'generic/platform=iOS'
xcodebuild build -workspace InferKit.xcworkspace -scheme InferKit -sdk appletvos26.5 -arch arm64

# MLX companion (Apple Silicon, macOS 14 / iOS 17). Place MLX's Metal library first or the
# MLX-dependent tests skip.
cd InferKitMLX && swift build --build-tests && ../Tools/mlx-metallib.sh && swift test

# Foundation Models companion (macOS 26 / iOS 26)
cd InferKitFoundationModels && swift build && swift test

# CocoaPods spec (fast, no build)
pod lib lint InferKit.podspec --quick
```

### Full Check (required before commit)

Code is commit-ready only when every step passes. `.github/workflows/ci.yml` runs the hosted-runner
subset; the MLX test schemes stay local because they need Metal and multi-gigabyte checkpoints.
The reasons behind each step, and how to find the offender when a leg breaks, are in
`Docs/agent-reference/build-and-verification.md`.

1. `swift build` + `swift test` on the host: **0 warnings**, all tests green. Grep the log for
   `warning:` without a `tail`; a tail has hidden warnings before.
2. `xcodebuild build` for `generic/platform=iOS`, and for tvOS through the SDK via the workspace
   (the command above). A tvOS destination does not resolve on this machine; naming the SDK builds it.
3. `xcodebuild analyze -scheme InferKit -derivedDataPath <FRESH DIR>`: **0 analyzer issues**. The
   analyzer is cached, so a reused derived-data path silently reports nothing.
4. `InferKitMLX/`: when a change touches the companion, run `Tools/mlx-metallib.sh` then `swift test`,
   or the three xcodebuild schemes `InferKitMLXTests`, `InferKitMLXExamples`, `InferKitMLXObjCExamples`
   (`-destination 'platform=macOS' -skipPackagePluginValidation`). Without the Metal library
   `swift test` must still exit 0 with the MLX tests skipped; a crash mid-run prints "0 failures" and
   is a truncated run, not a green one. Read the exit code.
5. `InferKitFoundationModels/`: `swift build` + `swift test` when a change touches that companion
   (all 24 tests across its three test targets).

## Distribution

Source through SwiftPM (`Package.swift`, public API in `Sources/InferKit/include/`) and CocoaPods
(`InferKit.podspec`, globbing the same files). New public headers go in
`Sources/InferKit/include/InferKit/` and the umbrella `InferKit.h`; the globs pick up new files. The
version lives in three places that move together on release: `s.version` in the podspec,
`NFKInferKit.version`, and the `vX.Y.Z` tag. `Tools/` ships in no distribution. Details, the
XCFramework builds, and the consumer linking recipes: `Docs/agent-reference/distribution-and-packaging.md`
and `Docs/installation.md`.

## Project Structure

```
Sources/InferKit/                # Core ObjC: include/InferKit/ (public headers + umbrella), NFK*.m
Tests/InferKitTests/             # XCTest for the core
Examples/, SwiftExamples/        # Compiled examples mirroring Docs/examples.md (ObjC and Swift)
Docs/                            # Consumer docs; Docs/agent-reference/ holds the maintainer notes
InferKit.xcworkspace             # Core + both companions in one Xcode window (rules in the full tree)
InferKitMLX/                     # MLX companion package (own Package.swift, tests, examples, DocC)
InferKitFoundationModels/        # Foundation Models companion package
Tools/                           # Converters, validation assets, reference-parity oracles, xcframework,
                                 #   DocC build, mlx-metallib.sh; ships in no distribution
Package.swift, InferKit.podspec  # The two distribution manifests
```

The full tree with per-directory notes is `Docs/agent-reference/project-structure.md`.

## Core subsystems

The core's notes are split by subject under `Docs/agent-reference/`: `core-runtime-notes.md`
(value-type accessors, tokenizers, grammar-constrained sampling, dynamic backend discovery),
`remote-providers.md` (every preset, the transport, catalogs, runners, streaming, tools, media),
`coreml-compute-plan.md` (where Core ML places a model, measured), and
`hardware-and-model-sizing.md` (`NFKHardwareProfile` and model sizing). Two rules from them apply
everywhere: image, mask, and video keys stay on `outputForKey:` / `inputForKey:` (no typed accessors,
their representation is the backend's choice), and `MLComputeUnitsCPUOnly` is zero, so a compute-units
property is always initialized explicitly.

## InferKitMLX (companion package)

A separate SwiftPM package (Apple Silicon, macOS 14 / iOS 17) depending on the core and `mlx-swift`
only. Backends adopt `NFKInferenceBackend` from Swift; every shipped model is ported into `MLXNN` and
measured at reference parity against its published implementation. Before touching the package read
`Docs/agent-reference/mlx-companion.md` (backend seams, `@objc` factories, registry, hub, runtime
wrappers) and `mlx-runtime-gotchas.md` (process-killing mlx-swift hazards); before touching a model
read its class file `mlx-models-<class>.md`; before a weights or format change read
`mlx-weights-and-formats.md`; before a training change read `mlx-training.md`.

Standing rules for the package:

- A model ships at measured reference parity, recorded with the actual cosines. A structural
  (shape) match is not a numeric one. Localize a mismatch with per-seam records rather than predicting
  where it is.
- A model is done only when every listing is updated: follow
  `Docs/agent-reference/mlx-parity-checklist.md`, which names each file and the house style its entry
  keeps, including the `~/.inferkit-validation.json` keys that make the parity test run by default.
- Customization is part of parity. A model whose customization path can be implemented is not at
  parity until that path ships end to end: a public network builder, a freezing policy, an objective
  measured against the reference training code, a `fineTune` recipe, and a train → save → reload
  round trip through the model's own factory, all reachable by a consumer without `@testable`. Where
  the path cannot be implemented (offline-only, or untrainable), the model's entry names why. A
  numeric parity figure alone is the inference half of done. The levels, the minimum shipped set, and
  the tests that prove it: `Docs/agent-reference/mlx-training.md` ("Customization is part of parity").
- Every `Task.detached` passes `priority: .userInitiated`; every test that reaches MLX calls its class's
  `requireMLXRuntime()` first; a test that loads many models clears the cache in `tearDown`.

## InferKitFoundationModels (companion package)

A separate SwiftPM package (macOS 26 / iOS 26) depending only on the core. `NFKFoundationModelsBackend`
wraps `LanguageModelSession` (prompt and messages, tools, structured output, streaming), and
`NFKFoundationModelsProvider` activates it through `NFKDynamicBackend` when the package is linked.
Generation tests skip where the model is unavailable. Notes: `Docs/agent-reference/foundation-models-companion.md`.

## Code Conventions

- Objective-C header/implementation pattern (`.h`/`.m`); public headers under `include/InferKit/`.
- Class prefix `NFK`; match existing naming in the file you edit.
- `if` statements always use a block (`{}`), never a single-line body.
- Explicit `nullable` / `nonnull` annotations on public API.
- Uniform Access Principle / self-encapsulation: read and write state through accessors, not direct
  ivar access.
- Extract Method → Predicate/Guard Clause (Fowler) is preferred over nested conditionals.
- **ARC.** The package compiles under ARC. The `NARC_` macros in `NFK_ARC.h` are ARC no-ops kept so
  the migrated sources compile unchanged; do not introduce new `NARC_` calls in new code — write
  plain ARC. `NARC_RELEASE` used to expand to `obj = nil`, which made the static analyzer flag every
  `dealloc` that cleared an ivar backing a `nonnull` property; it is now a true no-op. That is safe
  because every remaining use is either a local (ARC releases it at scope end) or is immediately
  reassigned. An error helper taking `NSError **` returns `BOOL` (always `NO`) rather than `void`,
  which is what the analyzer's Cocoa convention expects; callers ignore the result.
- Backward compatibility — the toolkit is pre-1.0. Until the first tagged release, prefer the
  correct design over source compatibility. Once shipped, point releases stay backward compatible.
- Document the introducing version on new public methods and classes.
- Backends are the extension seam. A consumer brings a heavier runtime (MLX, a C/Rust engine) by
  adopting `NFKInferenceBackend`. The core ships only backends with zero third-party dependencies and
  no license entanglement (pure Apple frameworks).
- Inference by contract is synchronous and multi-second; a caller runs it off the main/render thread
  and prefers `submitInferenceJobForRequest:` (or the `NFKInferenceSubmit` wrapper) for progress and
  cancellation. The asynchronous path runs at user-initiated quality of service: the core's
  queues use `QOS_CLASS_USER_INITIATED`, and every `Task.detached` in InferKitMLX passes
  `priority: .userInitiated`. An unprioritized `Task.detached` inherits the default, which Apple
  Silicon schedules on the efficiency cores — a decode loop there is several times slower for no
  stated reason.
- Two Objective-C test traps that recur: an `@[ ]` or `@{ }` literal inside an `XCTAssert…` macro
  argument splits the macro on its comma (parenthesize the argument, or hoist it into a local), and
  adjacent string-literal concatenation as a direct element of a collection literal raises
  `-Wobjc-string-concatenation` (parenthesize the element).

### Objective-C parity (InferKitMLX)

Objective-C parity is required for new and changed code, not optional. Every public Swift API a
consumer would use must be reachable from Objective-C unless it *cannot* bridge; adding or changing a
model, backend, option, or subsystem includes checking its ObjC surface in the same change. What
legitimately stays Swift-only is a fixed list — anything whose signature takes or returns an
`MLXArray`, a Swift closure, generics, a `Module`/`Linear`/`Optimizer`, a scheduler, or a Swift-only
`struct`/enum-with-associated-values (the `*Configuration` structs, `NFKMLXModelFit`); everything else
bridges. Concretely, when you add or touch one of these, the parity obligations are:

- **A shipped model** gets the full `@objc` factory set, and every factory honors the variant: the
  local `+backendWith[Variant:]weightsURL:error:`, the download `+backendWith[Variant:]repo:weightsPath:revision:cacheDirectoryURL:error:`,
  its async `…completionHandler:` peer, and `register()`. A variant enum that some factories ignore
  is a bug — the download and async peers must take the same `@objc` variant the local one does
  (the audit found Whisper's enum unused and NAFNet/YOLO/SwinIR download factories dropping it).
- A generation or inference option that has no core `NFKParameter*` key gets an `@objc` request-
  parameter key (see `NFKMLXGenerationParameterKey`) read in `runInference`, so ObjC configures it the
  way it sets `NFKParameterTemperature`. The Swift options struct stays; the request key is the bridge.
- A config-free machine or runtime property (a `Double`/`Int`/`String` reading, no configuration
  or `MLXArray`) goes on an `@objc` `NSObject` wrapper — `NFKMLXGPU`/`NFKMLXRandom`/`NFKMLXDevice` —
  not on a caseless-enum namespace, which ObjC cannot reach.
- A value type a backend hands back or a consumer inspects (a detection, a landmark set) is an
  `@objc` class of the shape the core's `NFKKeypoint`/`NFKDetection` use, never a Swift `struct`.
- **A closure backend** stays Swift-only to construct, but must have an ObjC use-path: at least one
  instance registered by name through `NFKMLXReferenceModels.registerAll` so `NFKMLXModelRegistry.backendNamed:`
  reaches its kind. A closure-backend type with no registered instance is an ObjC dead end.

A model that loads a whole release directory takes a `backendWith[Model:]directoryURL:error:` factory
instead of the repo/weightsPath set (the directory reads its own `config.json`), which is how an
ObjC caller reaches a model whose geometry lives in a Swift-only configuration.

## Code Comments

The bar for a comment is high. Code should explain itself through clear naming and structure;
comments are reserved for what the code cannot express.

- Use HeaderDoc `/*! ... */` blocks for public types, methods, and properties. Put multi-sentence
  rationale in the method's `@discussion`.
- Write an inline `//` comment only when it documents something non-obvious the code cannot state on
  its own: a subtle invariant (e.g. "must run on the lock"), an external constraint or platform quirk
  (e.g. an SPM header-search-path requirement), or a deliberate omission a maintainer might otherwise
  "fix".
- Do not narrate what the code obviously does, restate the method name, or leave historical /
  "fix:" / "previously the code did X" justifications. The diff and commit message carry that.
- The same bar applies to test code: the test method name describes intent; add a comment only for a
  non-obvious setup or invariant.

## Documentation Style (enforced)

HeaderDoc blocks and comments are technical documentation written with direct technical statements.
Language and National Variety: English — American.
Qualities of the writing: clear, thorough, easy to comprehend, not verbose (brevity), timeless,
integrated, wholistic.
Tense: Present — tuned for ease of comprehension.

_Banned constructions_:
- Antithesis / "not merely X — it Ys": no "does not just X, it Ys", "is not a Y, it's a Z",
  "rather than X, it Ys". State what it does, once.
- **Em-dash dramatic asides** used for emphasis or reveal ("— and that's the point"). Use a period
  or plain clause.
- **Editorializing / filler.**
- **Rule-of-three rhetorical lists** and build-up sentences. One fact per sentence.

Prefer subject–verb–object declaratives, and bullet lists of `condition → result` where appropriate.
Documentation informs and describes; it is not persuasive writing. The same rules govern the files
under `Docs/agent-reference/`.

## Documentation (DocC)

The core's DocC catalog is `Sources/InferKit/InferKit.docc/`; each companion carries its own. Build
with `Tools/docc/build.sh` (`--preview`, `--companion <name>`, `--all`). The core needs the script
because no Xcode or plugin path extracts a symbol graph for a pure-Objective-C target. Notes:
`Docs/agent-reference/documentation-docc.md`.

## Examples

`Docs/examples.md` is the examples reference. Its core (weight-free) snippets are mirrored by the
compiled `InferKitExamples` test target (`Examples/NFKExamples.m`), which `swift build` / `swift test`
compiles and runs so the documented code cannot silently drift. **Any change to an example may require
a change to the other:** update the `Examples/` method and the matching `Docs/examples.md` snippet
together. The `Examples/` target is SwiftPM-only (outside `Sources/`, so the podspec globs skip it).
Every package carries its examples in both languages, so a documented snippet cannot rot in
either: core `Examples/` (ObjC) + `SwiftExamples/` (Swift, which also pins what the ObjC importer
renames — `runInference(for:)`, throwing instead of `NSError **`); `InferKitMLX/Examples` (Swift) +
`InferKitMLX/ObjCExamples`; `InferKitFoundationModels/Examples` (Swift) +
`InferKitFoundationModels/ObjCExamples`.
`InferKitMLX/Examples/MLXModelGalleryExamples.swift` is the live example of every shipped MLX model:
it builds each through its public `@objc` factory and runs a representative forward per modality (mirrors
the "Model gallery" section of `Docs/examples.md`). Exhaustive per-model forwards stay in the individual
`NFKMLX*Tests`.

## Adding New Source Files

1. Add `.h` to `Sources/InferKit/include/InferKit/` and `.m` to `Sources/InferKit/` (SwiftPM and the
   podspec globs pick them up automatically).
2. Add a corresponding test file to `Tests/InferKitTests/`.
3. Add the public header to the umbrella `InferKit.h`.
4. If the file links a new system framework, add it to `Package.swift` `linkedFramework` and the
   podspec `frameworks`.

## Reference files

`Docs/agent-reference/README.md` indexes every note file. By subject:

- Repository: `build-and-verification.md`, `project-structure.md`, `distribution-and-packaging.md`,
  `documentation-docc.md`, `mlx-parity-checklist.md`.
- Core: `core-runtime-notes.md`, `remote-providers.md`, `coreml-compute-plan.md`,
  `hardware-and-model-sizing.md`.
- InferKitMLX: `mlx-companion.md`, `mlx-runtime-gotchas.md`, `mlx-weights-and-formats.md`,
  `mlx-training.md`, and the model classes `mlx-models-diffusion.md`, `mlx-models-dit-generation.md`,
  `mlx-models-image-restoration.md`, `mlx-models-depth-segmentation-matting.md`,
  `mlx-models-detection-pose.md`, `mlx-models-video.md`, `mlx-models-embeddings-retrieval.md`,
  `mlx-models-language.md`, `mlx-models-gemma.md`, `mlx-models-vision-language.md`,
  `mlx-models-speech-recognition.md`, `mlx-models-text-to-speech.md`,
  `mlx-models-source-separation.md`, `mlx-models-speech-restoration.md`,
  `mlx-models-audio-codecs-music.md`.
- InferKitFoundationModels: `foundation-models-companion.md`.

Consumer-facing documents stay in `Docs/` (`inference-guide.md`, `examples.md`, `installation.md`,
`coreml-llm.md`, `companions.md`, `model-index.md`, `model-parity.md`, `mlx-runtime-hazards.md`) and
are updated through the parity checklist, not duplicated into the reference files.

## Safeguards (Anti-Patterns)

Required without exception:

- **NEVER** run `git clone/mv/restore/rm/add/branch/commit/merge/rebase/reset/pull/push/fetch`
  without developer approval first.
- **NEVER** run `rm` on any path without developer approval first.
- **NEVER** erase or overwrite files for the task of unit testing — the changes being tested must be
  preserved.
- **NEVER** delete a file or folder until its associated task is completely finished.
