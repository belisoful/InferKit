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
xcodebuild build -workspace InferKit.xcworkspace -scheme InferKit -sdk appletvos -arch arm64

# The MLX companion (Apple Silicon, macOS 14 / iOS 17) is a separate package. SwiftPM cannot
# compile Metal shaders, so place MLX's Metal library beside the test binary first or the
# MLX-dependent tests skip.
cd InferKitMLX && swift build --build-tests && ../Tools/mlx-metallib.sh && swift test

# The Foundation Models companion (macOS 26 / iOS 26) is a separate package
cd InferKitFoundationModels && swift build && swift test

# Validate the CocoaPods spec (fast, no build)
pod lib lint InferKit.podspec --quick
```

The tvOS SDK is named without a version, so the command resolves to the SDK the installed Xcode ships
(`appletvos` is `appletvos27.0` under Xcode 27). A versioned name fails once a different Xcode is
installed.

The build and test commands need Xcode's toolchain. When `xcode-select -p` names
`/Library/Developer/CommandLineTools`, `swift build` fails with `Could not initialize build system` and
`Unknown error parsing property list`, and `xcodebuild` refuses to run. Exporting
`DEVELOPER_DIR` with the installed Xcode's `Contents/Developer` path points the commands at Xcode
without changing the system setting. Measured 2026-09-15 with Xcode 27.0 installed.

Xcode 27 installs its Metal compiler as a separate component. Until
`xcodebuild -downloadComponent MetalToolchain` has run, `xcodebuild -showComponent MetalToolchain`
reports `uninstalled` and any `metal` invocation fails with `missing Metal Toolchain`. SwiftPM under
Xcode 27 compiles mlx-swift's Metal kernels during `swift build` (a `CompileMetalFile` step), so
`InferKitMLX` does not build at all without the component, including the leg where MLX tests are
expected to skip. With the component installed, the same build places `default.metallib` inside
`mlx-swift_Cmlx.bundle` in each of the three test bundles (`.build/out/Products/Debug/<target>.xctest`,
which `.build/debug` links to), so a plain `swift test` runs the MLX-dependent tests without
`Tools/mlx-metallib.sh`: `NFKMLXTrainerTests` ran all 21 tests, none skipped. The script recognizes
that layout and exits without placing anything; for a toolchain that builds the single
`InferKitMLXPackageTests.xctest`, it still compiles the kernels and places `mlx.metallib`. Measured
2026-09-15 with Xcode 27.0 (27A266a).

Xcode 27's Swift 6.4 optimizer aborts the compile of
`NFKMLXJSONSchemaConstraint.startValue(_:node:byte:)`. Its CopyPropagation pass fails its own
ownership verification with `Found over consume?!`, and the frontend ends with `fatal error
encountered during compilation`. It reproduces with a plain `swift build -c release` as well as
through `Tools/xcframework/build-mlx.sh`, so it reaches a consumer who builds the package in Release,
not only the release assets. Debug builds are unaffected, which is why the whole test suite passed the
same day. The function carries `@_optimize(none)` until the compiler is fixed. After an Xcode bump,
remove the attribute and run `swift build -c release --package-path InferKitMLX`: a clean build means
the workaround can go. Measured 2026-09-15, Xcode 27.0 (27A266a), swiftlang-6.4.0.34.1.

## Full Check (required before commit)

Code is commit-ready only when every check below passes.

`.github/workflows/ci.yml` runs the hosted-runner subset on every push and pull request: the core's
build (zero warnings) + tests, the iOS and tvOS compile legs, the analyzer at a fresh derived-data
path, the podspec lint, and compile checks for the companions. `InferKitFoundationModels` and
`InferKitAppleSwift` each build and test behind an SDK guard: both need the macOS 26 SDK, so a runner
on an older image reports what it skipped instead of failing on toolchain availability. The MLX test schemes stay a local
step: they evaluate real MLX arrays (Metal) and read multi-gigabyte checkpoints from
`~/.inferkit-validation`, which a runner does not have — there they would skip, proving nothing.

1. `swift build` + `swift test` on the host — **0 warnings**, all tests green.
2. `xcodebuild build` for a `generic/platform=iOS` destination (cross-platform compile), and for tvOS
   through the SDK: `-workspace InferKit.xcworkspace -scheme InferKit -sdk appletvos -arch arm64`.
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
   that ran before it, so a crash there is a truncated run rather than a red one. The summary line and
   the exit code disagree for a second reason too, and this one is silent: a test can fail while
   every `Executed N tests` line in the log says "0 failures", because those lines are per test
   SUITE and xctest prints one per suite that passed. **Read `swift test`'s own exit code, and read
   it without a pipe.** `swift test | grep … ; echo $?` reports the status of the LAST stage of the
   pipeline, which is the grep, and is 0 whatever the tests did. Redirect instead:

   ```bash
   swift test > /tmp/suite.log 2>&1; code=$?; echo "EXIT $code"
   grep -E "^Test Case .* failed" /tmp/suite.log
   ```

   A run reported here as green on the strength of a piped summary is not evidence. This mistake
   produced two such reports on 2026-09-21, both of which had to be withdrawn and re-run. **Run
   `Tools/mlx-metallib.sh` first**: it compiles mlx-swift's nine kernels with `xcrun metal` (the same
   sources Xcode's build system compiles) and places `mlx.metallib` beside the SwiftPM test binary,
   which is the first place MLX's loader looks, before any bundle. `swift test` then runs every test
   in all three test targets (the file survives a relink).

   **How long the suite takes, and when to re-record it.** Measured 2026-09-22 on an M1 Max against
   mlx core 0.32.2: 1423 tests, 0 failures, **28 minutes** (1664 s). The figure covers test EXECUTION
   and excludes the build. A spread of several minutes is ordinary and comes from thermal state,
   whether the checkpoints are warm in the file cache, and whether another session holds the GPU.
   A later run the same day over 1434 tests took 1779 s with the core suite and an oracle running
   beside it, which is the upper bound the rule below describes rather than a new figure.

   This replaces the 35 minutes recorded the day before over 1366 tests, and it is a REDUCTION of 26%
   against 57 more tests, which is the machine rather than the suite: that earlier figure was taken
   while a second session was running its own released-weight tests on the same box, and this one was
   taken with the Large Model Coordination lock held (`Tools/lmc/lmc.py`, below). A figure measured
   beside another session's multi-gigabyte loads is an upper bound. Take the lock before recording.

   The 35-minute figure itself replaced 27 minutes over 1355 tests, when SAM 3's three tests (a
   3.44 GB checkpoint read twice) and SAM 2's coverage check (five released checkpoints in a row)
   arrived together.

   Re-record it when either holds:

   - a run differs from the figure above by more than **10%** (about 3 minutes at this size), which
     is outside the observed spread → record the new figure with the date, the machine, the mlx core
     version, and the test count.
   - a change adds or removes a REAL-WEIGHT parity test → record deliberately, without waiting for
     drift, because that is a known cause with a known owner.

   **Do not build while the suite runs.** SwiftPM serializes on a lock over `.build`, so a
   `swift build` started during a `swift test` sits at zero CPU until the tests finish, and the
   measurement it shares the machine with is no longer clean. A run measured alongside other work is
   an upper bound, not a figure to record.

   Time does not track the test count. The 2026-09-09 figure was 20 minutes for 1192 tests: the count
   rose about 14% to 1355 and the time rose about 35%, because one test that loads a multi-gigabyte
   checkpoint costs more than a hundred tiny ones. A count without a time says little about either.
   Re-run the script after `swift package clean` or an mlx-swift bump. The package has **three**
   test targets, each with its own shared scheme in `.swiftpm/xcode/xcshareddata/xcschemes/`, and
   xcodebuild covers them without the script because Xcode compiles the shaders into
   `mlx-swift_Cmlx.bundle` inside each test bundle. Run all three (`-destination 'platform=macOS'
   -skipPackagePluginValidation` throughout):

   ```
   xcodebuild test -scheme InferKitMLXTests        …    # the model and API suite
   xcodebuild test -scheme InferKitMLXExamples     …    #  61 — the Swift documented snippets
   xcodebuild test -scheme InferKitMLXObjCExamples …    #  48 — the Objective-C ones
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

5. `Tools/doc-snippets/check-objc.py` when a change touches an Objective-C code block in `README.md`,
   `Docs/examples.md`, or `Docs/inference-guide.md`, or any public API a block names: **0 failed**.
   It compiles every ```` ```objc ```` block with `clang -fsyntax-only -Werror` against the core's
   headers and the `-Swift.h` header each companion's build generates, and reports an error at the
   Markdown line that carries it. It runs `swift build --target <companion>` for each companion
   first, so the headers match the Swift source; with the companions already built the whole run
   takes about 20 seconds. The compiled examples targets mirror a chosen subset of the snippets and
   run them; this check covers every block and runs none. CI runs it in the InferKitMLX job, after
   that job's macOS build, and skips it on a runner whose macOS SDK is older than 26.

   A block is a fragment, so the variables its prose supplies are declared once in
   `Tools/doc-snippets/snippet-context.h`. A new snippet that uses a new free variable adds it there
   with the type the document gives it. When a name means different types in different blocks, the
   block that differs carries a directive on the line above its fence:
   `<!-- objc-check: given CGImageRef frame = NULL; -->` for extra declarations,
   `<!-- objc-check: continues -->` for a block that compiles in the scope of the block before it,
   and `<!-- objc-check: skip (reason) -->` for a block that cannot compile. The first run on
   2026-09-23 found fourteen defects across the 100 blocks, none of which any compiled mirror
   covered: seven calls to an `-initWithInputs:` that does not exist, `NFKInferenceBackend *` used
   as a class type twice, a Vision factory under a selector it does not have, a block assigning a
   captured variable without `__block`, a call to a method no type declares, and two blocks that
   were fragments of a statement.

## Large Model Coordination (LMC)

Several sessions share one Mac, and two multi-gigabyte test runs at once thrash memory: each
becomes an upper bound and neither is a figure to record. `Tools/lmc/lmc.py` is the lock. A
session that wants a large-model test asks the LMC for the lock, runs only once it holds it, and
releases as the first thing it does when the run ends. Every session coordinates every
large-model run through it; a run started around it is the 2026-09-22 failure mode, where one
session's seven full-suite restarts held the GPU for two hours against a slot file nobody
reaped.

```bash
Tools/lmc/lmc.py run --test full-check -- xcodebuild test -scheme InferKitMLXTests …   # acquire, run, release
Tools/lmc/lmc.py acquire --test deepseek-v41-released        # or: hold it yourself …
Tools/lmc/lmc.py release --outcome passed                    # … and release it first thing after
Tools/lmc/lmc.py status                                      # holder, queue, outcomes waiting to be read
```

- A test is a name the session chooses. A request names one or more tests and holds the lock for
  all of them. The working tree (git toplevel of the cwd, or `--root`) is part of the request.
- Requests for the same tests from the same tree combine into one run, whether that run is still
  queued or already in progress. The first requester runs; the others ride. The runner's `passed`
  or `failed` satisfies every rider, and `wait` reports the outcome with the runner's log path.
  A session whose edits landed after a run started passes `--fresh` to queue its own run.
- A run that is `stopped` mid-way (a signal, a holder whose process is gone, `--outcome stopped`)
  satisfies nobody: every rider's `wait` exits 12, and each of those sessions re-requests.
- `full-check` always carries priority 100, the lowest, whatever `--priority` says. Other requests
  default to 50 and are granted lowest number first, then earliest.
- The holder is the session process (`CLAUDE_PID`); `run` also records the child it spawned. A
  holder or child that is gone is reaped as `stopped` at the next command, so a stale hold cannot
  outlive the process that took it. A queued request whose session is gone is dropped.
- `wait` and `acquire` take `--timeout`; the harness caps a foreground command at ten minutes, so a
  wait that returns 13 keeps its request and is called again with the same id. A run longer than
  that goes through `run` in the background.
- `run --quiet-seconds 180` waits, after the grant, for `memory_pressure` free ≥ 70% and no process
  named `xctest`, `swift-test`, or `swift-build` (`pgrep -x`, never command text) for that long,
  which is the quiet gate the full suite ran behind before the tool existed.
- The tool writes the legacy `~/.claude/inferkit-test-slot-mlx` while the lock is held and treats
  a live holder of that file as holding the lock, so a script that still reads the file stays
  coordinated through the transition. State and per-run logs are in `~/.claude/inferkit-lmc/`;
  `history` tails the event log. `Tools/lmc/test_lmc.py` is the tool's test suite.

## The oracle interpreters break when Xcode is renamed

Six of the oracle environments under `~/.inferkit-validation/` were created from Xcode's bundled
`python3`, so their `bin/python3` is a symbolic link into `/Applications/Xcode.app`. Installing Xcode
27 beside the old one leaves that path gone (`/Applications/Xcode 27.app/…` now), and every oracle in
`da3venv`, `dfnvenv`, `llmvenv`, `ltxvenv`, `sdvenv`, and `vrvenv` fails with "no such file or
directory" before it reaches a model. The repair is one link per environment, to the same 3.9.6
interpreter Apple ships at a stable path:

**That relink does not work, and it fails silently.** Pointing `bin/python3` at `/usr/bin/python3`
leaves an ABSOLUTE symlink out of the environment, and CPython 3.9 resolves `pyvenv.cfg` relative to
the symlink's TARGET, so venv detection fails and `sys.prefix` lands in Xcode's framework. The
environment's own `site-packages` never reaches `sys.path`. Nothing errors: imports fall through to
the USER site-packages, so all six reported the same `torch 2.8.0, transformers 4.33.3` while
`llmvenv` actually holds transformers 4.57.6 and `ltxvenv` holds a diffusers the run could not see. A
reading taken that way measures whatever is installed globally, not the pinned environment. Repairing
the stale `home =` line makes no difference; the symlink is the cause.

The six were rebuilt on 2026-09-21 with Homebrew's `python3.12`, whose path is stable. The pins come
from the old environment's own `dist-info` directory names, and the install takes `--no-deps`:

    /opt/homebrew/bin/python3.12 -m venv <env>
    <env>/bin/pip install --no-deps -r <pins>

`--no-deps` is load-bearing. Pip's resolver REFUSES the full pinned closure even though that exact
set installed fine incrementally, and a retry without pins silently upgrades the environment: it took
`llmvenv` to transformers 5.17.0, a major version, which would have changed its oracles.

`dfnvenv` cannot move: `DeepFilterLib` has no 3.12 wheel and does not build from source, so it stays
on 3.9 and needs `PYTHONPATH=~/.inferkit-validation/dfnvenv/lib/python3.9/site-packages` on every
invocation until a wheel exists or `python@3.11` is installed.

The root cause is worth stating plainly: `/usr/bin/python3` itself resolves INTO the Xcode bundle
(`sys.base_prefix` is `/Applications/Xcode 27.app/...`), so any environment built from the system
interpreter is hostage to the next rename. The environments built from Homebrew's `python@3.12`
(`chatterboxvenv`, `gemmavenv`, `musicvenv`, `nemovenv`, `reenhancevenv`, `rfdetrvenv`,
`wananimatevenv`) were never affected. A new environment takes the Homebrew interpreter for that
reason.

## The asset store spans two volumes

The vision, vision-language, and image/video-generation assets live on an external volume at
`/Volumes/WindowsBoot/InferKit/validation`, in the same relative layout they had under
`~/.inferkit-validation` (moved 2026-09-22, 514 files, 119 GiB). The language, audio, and music
assets, the oracle environments, `records/`, `shapes/`, and the remainder of `raw/` and `converted/`
stay on the home volume. `~/.inferkit-validation.json` is what resolves either one, so a key is the
only thing a test reads; nothing derives a path from the store root.

Relocating any part of the store has one invariant: **no key in `~/.inferkit-validation.json` points
at a path that does not exist.** Check that over every key, not over the set of files moved. A key
whose value is a DIRECTORY is missed by a rewrite that matches moved file paths, because the
directory is never itself a moved file. Nine RF-DETR and RT-DETR keys (`IK_VAL_RFDETR*`,
`IK_VAL_RTDETR*`) name a directory inside `raw/` and were left pointing at the emptied location for
exactly that reason.

A relocation is proven only by running the affected tests AFTER the originals are gone. A test whose
asset key is missing calls `XCTSkip`, and a skipped test does not fail its suite, so `swift test`
exits 0 either way. A run made while the originals are still in place passes whether or not the keys
were rewritten, and reports nothing. The verification order is: copy, verify each file by content,
rewrite the keys, run the affected tests, delete the originals, then run the affected tests again.
That last run is the only one that distinguishes a working index from a stale one.

Freed space does not return at once. APFS local snapshots hold the deleted blocks until macOS thins
them, and `tmutil isexcluded` does not change that: exclusion governs what reaches a backup
destination, not what a volume-level snapshot captures.

`~/.inferkit-validation.json` is shared by every session on the machine, and each one writes it by
reading the whole file, changing its own keys and writing the whole file back. A session that holds
its copy while another adds a key then writes that key away: on 2026-09-22 a key set at 22:59 was
gone after another session's write at 23:26, and its parity test skipped rather than failed, so
nothing reported the loss. Read the file immediately before changing it and write it straight after,
never from a copy held across other work. A test that begins skipping after passing is this until
shown otherwise; check its key before its code.

`flux-schnell-release` is a directory of symbolic links into `~/.cache/huggingface/hub`, so `du`
reports it at 10 MB while it resolves to 31 GB. Moving it relocates the links and frees nothing. Any
asset directory is worth a `du -shL` before it is counted or moved.
