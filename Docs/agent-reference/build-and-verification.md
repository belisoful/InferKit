<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# Build, test, and the Full Check

The complete build and verification notes for the core and both companions.

```bash
# Build and test the core (host platform, macOS)
swift build
swift test

# Cross-platform build check (the core supports macOS 11 / iOS 14 / tvOS 14)
xcodebuild build -scheme InferKit -destination 'generic/platform=iOS'
# tvOS goes through the SDK, not a destination — see the Full Check below.
xcodebuild build -workspace InferKit.xcworkspace -scheme InferKit -sdk appletvos26.5 -arch arm64

# The MLX companion (Apple Silicon, macOS 14 / iOS 17) is a separate package. SwiftPM cannot
# compile Metal shaders, so place MLX's Metal library beside the test binary first or the
# MLX-dependent tests skip.
cd InferKitMLX && swift build --build-tests && ../Tools/mlx-metallib.sh && swift test

# The Foundation Models companion (macOS 26 / iOS 26) is a separate package
cd InferKitFoundationModels && swift build && swift test

# Validate the CocoaPods spec (fast, no build)
pod lib lint InferKit.podspec --quick
```

## Full Check (required before commit)

Code is commit-ready only when every check below passes.

`.github/workflows/ci.yml` runs the hosted-runner subset on every push and pull request: the core's
build (zero warnings) + tests, the iOS and tvOS compile legs, the analyzer at a fresh derived-data
path, the podspec lint, and compile checks for both companions. The MLX test schemes stay a local
step: they evaluate real MLX arrays (Metal) and read multi-gigabyte checkpoints from
`~/.inferkit-validation`, which a runner does not have — there they would skip, proving nothing.

1. `swift build` + `swift test` on the host — **0 warnings**, all tests green.
2. `xcodebuild build` for a `generic/platform=iOS` destination (cross-platform compile), and for tvOS
   through the SDK: `-workspace InferKit.xcworkspace -scheme InferKit -sdk appletvos26.5 -arch arm64`.
   A tvOS destination does not resolve here and that is not the same as tvOS being unbuildable.
   Xcode derives destinations from installed platform *support*, which a machine without the tvOS
   platform lacks even when the tvOS SDK is present — and the SDK is what a compile needs. Naming the
   SDK builds the core for tvOS. This check recorded that leg as unverified for as long as it used the
   destination form. A bare package rejects `-sdk` (it demands `-destination`), so the invocation goes
   through the workspace.
2b. `xcodebuild analyze -scheme InferKit -derivedDataPath <FRESH DIR>` — **0 analyzer issues**. Use a
   fresh derived-data path: the analyzer is cached, and reusing one silently reports nothing.
3. `InferKitMLX/` `swift build` + `swift test` when a change touches the MLX companion. SwiftPM
   cannot compile Metal shaders, so a plain build carries no `default.metallib` and MLX aborts the
   Process at the first array it has to evaluate, with "0 failures" still printed for the classes
   that ran before it, so a crash there is a truncated run rather than a red one. **Run
   `Tools/mlx-metallib.sh` first**: it compiles mlx-swift's nine kernels with `xcrun metal` (the same
   sources Xcode's build system compiles) and places `mlx.metallib` beside the SwiftPM test binary,
   which is the first place MLX's loader looks, before any bundle. `swift test` then runs every test
   in all three test targets (measured 2026-09-09: 1192 run, 6 skipped — all opt-in probes — 0
   failures, 20 minutes with the real-weight parity tests; the file survives a relink).
   Re-run the script after `swift package clean` or an mlx-swift bump. The package has **three**
   test targets, each with its own shared scheme in `.swiftpm/xcode/xcshareddata/xcschemes/`, and
   xcodebuild covers them without the script because Xcode compiles the shaders into
   `mlx-swift_Cmlx.bundle` inside each test bundle. Run all three (`-destination 'platform=macOS'
   -skipPackagePluginValidation` throughout):

   ```
   xcodebuild test -scheme InferKitMLXTests        …    # the model and API suite
   xcodebuild test -scheme InferKitMLXExamples     …    #  61 — the Swift documented snippets
   xcodebuild test -scheme InferKitMLXObjCExamples …    #  36 — the Objective-C ones
   ```

   Without the library, `swift test` must still exit 0, with the MLX-dependent tests reported as
   skipped (2026-09-08: 1192 run, 891 skipped, 301 pure-Foundation tests actually executed —
   pickle, zip, GGUF parsing, chat templates, flow schedulers, configuration readers, remaps). The
   guard is `NFKMLXGPU.metalLibraryURL`, which mirrors the loader's own search (colocated
   `mlx.metallib`, the `mlx-swift_Cmlx.bundle` beside the main bundle, in any loaded bundle, or as a
   framework, then `Resources/default.metallib`) and is nil exactly when the first evaluation would
   abort. It is deliberately not a check on the build directory: a `swift test` with the library
   placed runs everything, and an xcodebuild run never skips. The convention that keeps the leg
   green: every test method that reaches MLX — constructing a module (a `Linear` is enough), seeding
   (`NFKMLXRandom.seed`), clearing the cache (`NFKMLXGPU.clearCache`), reading a memory counter or
   the device, `loadArrays` on a record, `eval` — calls the class's private `requireMLXRuntime()`
   (an `XCTSkipIf` on that URL being nil) first; a shared helper that constructs a net
   (`tinyModel()`) carries the call so every caller skips; `setUp` seeds and `tearDown` cache clears
   are wrapped in the same check, because they run for a skipped method too; and a class whose every
   test needs MLX (`NFKMLXReferenceParityTests`) skips once in `setUpWithError`. A skip on a missing
   validation key is not a runtime guard — `testCLIPTokenizerMatchesTheReferenceIds` had one and
   still crashed on `loadArrays`. To find the offender when this leg breaks, run each class alone
   (`swift test --filter '\.NFKMLXFooTests/'`, ~6 s each) and take the last `started` case with no
   `passed`/`skipped` line; XCTest's lines are unbuffered while `print` output flushes at the abort,
   so the printed lines beside the crash belong to earlier tests.
   `-scheme InferKitMLX` is the library scheme and runs only the first testable, which is the
   collapse the core's workspace note above describes: it executed the `InferKitMLXTests` methods
   and silently ran neither examples target, so the 78 example tests went unclaimed while the command
   reported success. The examples targets are what keeps a documented snippet from rotting, which is
   exactly what a silent skip defeats. The per-target schemes exist to make that impossible; keep one
   testable in each. Only MLX is forced onto xcodebuild — `swift test` runs every test target a
   package declares, so the core (204) and `InferKitFoundationModels` (24) are covered by step 1 and
   step 4 whatever Xcode does with their schemes.
4. `InferKitFoundationModels/` `swift build` + `swift test` when a change touches that companion. That
   covers all 24 tests across its three test targets. Through Xcode it collapses the same way MLX does
   — the generated `InferKitFoundationModels` scheme runs 15 and skips the 9 in the two examples
   targets — so it carries the same per-target shared schemes
   (`InferKitFoundationModelsTests` / `…Examples` / `…ObjCExamples`, 15 / 5 / 4). Nothing here needs
   them, since this package evaluates under `swift test`; they exist so an Xcode run cannot quietly
   cover less than the command line does.
