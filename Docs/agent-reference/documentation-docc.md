<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# Documentation (DocC)

The core ships a DocC catalog at `Sources/InferKit/InferKit.docc/` (landing page `InferKit.md`, concept
articles, per-symbol extension `.md` files, and `Resources/*.svg` diagrams). Symbol pages come from the
`/*! … */` HeaderDoc in the public headers; the articles and extensions add concepts, curated topics, and the
diagrams. The catalog sits under `Sources/` without disturbing `swift build`, `swift test`, or the
podspec (which globs only `.h`/`.m`).

Build it with `Tools/docc/build.sh` (output → `.docc-build/`, gitignored) or `Tools/docc/build.sh --preview`.
**Why a script:** DocC needs a symbol graph, and neither `xcodebuild docbuild` nor the swift-docc-plugin
extracts one for a pure-Objective-C SwiftPM library target (both emit an empty archive). The script runs
`clang -extract-api -x objective-c-header` over **all** public headers as inputs (which emits symbols
only for the input files, excluding the SDK — passing the umbrella alone yields nothing) and feeds the
result to `docc convert` with the catalog. Diagrams are self-contained light-card SVGs (legible on both
light and dark pages); each is audited by rendering.

The three Swift companions carry their own catalogs, documented through the swift-docc-plugin (which does
extract a symbol graph for a Swift target) rather than the clang recipe:

- `InferKitFoundationModels/Sources/InferKitFoundationModels/InferKitFoundationModels.docc/` — landing,
  the `ToolsAndStructuredOutput` article, `tool-calling.svg`, and per-class example pages.
- `InferKitMLX/Sources/InferKitMLX/InferKitMLX.docc/` — landing (gallery Topics grouped by modality),
  the `ModelGallery` / `BringYourOwnBackends` / `DiffusionAndSchedulers` / `WeightsAndConversion`
  articles, four diagrams (`model-gallery`, `backend-families`, `diffusion-loop`, `weights-pipeline`),
  and headline per-class example pages.
- `InferKitAppleSwift/Sources/InferKitAppleSwift/InferKitAppleSwift.docc/` — the landing page.

Each companion adds swift-docc-plugin as a dev-only dependency.

The companions build with SwiftPM's default build system; `--build-system native`, which the script
passed from 2026-09-24, is deprecated. InferKitFoundationModels and InferKitAppleSwift build through
the plugin. InferKitMLX does not: the plugin's symbol-graph step runs `clang -extract-api` over every
dependency, and on mlx-swift's C++ `Cmlx` target the first standard header fails (`'functional' file
not found`). In a `.build` that already held a build the step did not run at all (no ExtractAPI in
the log, no `Cmlx.symbolgraphs`) and the plugin succeeded, so only an empty scratch path tests this.
`swift package dump-symbol-graph` fails the same way. So the script builds
InferKitMLX, runs `swift-symbolgraph-extract` on the InferKitMLX module alone (the C targets it
imports, `Cmlx` and `_NumericsShims`, reached through their `include/module.modulemap`, and the core
through SwiftPM's generated `InferKit.modulemap`), and runs `docc convert --emit-lmdb-index`.
Measured on Xcode 27.0 (Swift 6.4), 2026-10-07: all 8259 pages are content-identical to the plugin's
native-build archive, `navigator.index` is byte-identical, and only `data.mdb` and
`availability.index` differ, as they do between two runs of the same convert. Graphs from the
compiler's `-emit-symbol-graph` were not used: they print `Error?` for `(any Error)?`, leave nested
types unqualified, and show an inherited member's summary as "Inherited from", which changed 928
pages.

Build a companion with `Tools/docc/build.sh --companion <name>`. `Tools/docc/build.sh --all` builds the
core and all three companions. Only symbol links to the companion's
own types resolve when it builds alone, so the catalogs reference core types (`NFKInferenceBackend`, the
`NFKInput*`/`NFKOutput*` keys) in code font, not as ``doc``/symbol links, to stay warning-free. `plan(for:)`
and other internal helpers reachable only via `@testable import` are not documented — the pages show the
public path instead.
