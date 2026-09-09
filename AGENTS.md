# InferKit Agent Guidelines

## Project Overview

**InferKit** is a small, cross-platform inference toolkit for Objective-C. It provides a swappable
backend protocol, request/result value types, an async job handle, the shipped backends (mock,
in-process Core ML, an on-device Core ML language-model runner, OpenAI-compatible chat and
transcription clients, a submit-poll-fetch base, and runtime discovery — plus the companion-package
MLX and Foundation Models backends), a texture-tensor conversion, tokenizers, and a Hugging Face
model-download layer. It has no
host-framework dependency, so any Metal/Apple app (macOS, iOS, tvOS) can use it. The class prefix is
`NFK`.

The package is source-distributed through both Swift Package Manager and CocoaPods. Two optional
companion packages build on the core without raising its platform floor or adding dependencies to
it: `InferKitMLX/` (MLX-backed inference, plus on-device fine-tuning of the models it ships, on Apple
Silicon) and `InferKitFoundationModels/` (a bridge to Apple's on-device system language model).

## Build & Test Commands

```bash
# Build and test the core (host platform, macOS)
swift build
swift test

# Cross-platform build check (the core supports macOS 11 / iOS 14 / tvOS 14)
xcodebuild build -scheme InferKit -destination 'generic/platform=iOS'
# tvOS goes through the SDK, not a destination — see the Full Check note.
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

### Full Check (required before commit)

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

## Documentation (DocC)

The core ships a DocC catalog at `Sources/InferKit/InferKit.docc/` (landing page `InferKit.md`, concept
articles, per-symbol extension `.md` files, and `Resources/*.svg` diagrams). Symbol pages come from the
`///` HeaderDoc in the public headers; the articles and extensions add concepts, curated topics, and the
diagrams. The catalog sits under `Sources/` without disturbing `swift build`, `swift test`, or the
podspec (which globs only `.h`/`.m`).

Build it with `Tools/docc/build.sh` (output → `.docc-build/`, gitignored) or `Tools/docc/build.sh --preview`.
**Why a script:** DocC needs a symbol graph, and neither `xcodebuild docbuild` nor the swift-docc-plugin
extracts one for a pure-Objective-C SwiftPM library target (both emit an empty archive). The script runs
`clang -extract-api -x objective-c-header` over **all** public headers as inputs (which emits symbols
only for the input files, excluding the SDK — passing the umbrella alone yields nothing) and feeds the
result to `docc convert` with the catalog. Diagrams are self-contained light-card SVGs (legible on both
light and dark pages); each is audited by rendering.

The two Swift companions carry their own catalogs, documented through the swift-docc-plugin (which does
extract a symbol graph for a Swift target) rather than the clang recipe:

- `InferKitFoundationModels/Sources/InferKitFoundationModels/InferKitFoundationModels.docc/` — landing,
  the `ToolsAndStructuredOutput` article, `tool-calling.svg`, and per-class example pages.
- `InferKitMLX/Sources/InferKitMLX/InferKitMLX.docc/` — landing (gallery Topics grouped by modality),
  the `ModelGallery` / `BringYourOwnBackends` / `DiffusionAndSchedulers` / `WeightsAndConversion`
  articles, four diagrams (`model-gallery`, `backend-families`, `diffusion-loop`, `weights-pipeline`),
  and headline per-class example pages. Each companion adds swift-docc-plugin as a dev-only dependency.

Build a companion with `Tools/docc/build.sh --companion <InferKitFoundationModels|InferKitMLX>`, or the
whole set (core + both companions) with `Tools/docc/build.sh --all`. Only symbol links to the companion's
own types resolve when it builds alone, so the catalogs reference core types (`NFKInferenceBackend`, the
`NFKInput*`/`NFKOutput*` keys) in code font, not as ``doc``/symbol links, to stay warning-free. `plan(for:)`
and other internal helpers reachable only via `@testable import` are not documented — the pages show the
public path instead.

## Distribution

InferKit ships as source through two channels that reference the same files:

- **Swift Package Manager** — `Package.swift`. The target's public API is `Sources/InferKit/include/`,
  so `#import <InferKit/NFKFoo.h>` resolves the same way against SwiftPM and the built framework.
  Quoted sibling imports (`#import "NFKFoo.h"`) resolve through the target's `headerSearchPath`
  entries (`include/InferKit` and `.`).
- **CocoaPods** — `InferKit.podspec`, a source pod. `source_files` compiles the same `.h`/`.m`;
  `public_header_files` marks `include/InferKit/*.h` as the public surface. The MLX companion is not
  a pod (MLX distributes through SwiftPM only), though it does package as an XCFramework — see
  "Packaging InferKitMLX as an XCFramework".

Keep the two in sync: a new source file is picked up by the SwiftPM glob automatically; the podspec
globs the same paths, so no per-file edit is needed there either. New public headers go in
`Sources/InferKit/include/InferKit/` and the umbrella `InferKit.h`.

`Tools/` ships in no distribution — it is developer and validation tooling, not the library. The
SwiftPM targets name `Sources/InferKit` / `InferKitMLX/Sources` as their paths, the podspec globs only
`Sources/InferKit/**/*.{h,m}`, and the XCFramework release assets are built binaries; none of them
compile or carry anything under `Tools/`. A consumer resolving the SwiftPM package clones the whole
repository (so `Tools/` lands on disk), but nothing there is built into the products or the release.
The `~30 Tools/*-to-safetensors/convert.py` scripts are therefore not required to consume the
library: `NFKMLXTorchFormat` reads a raw `.pth`/`.pt`/`.ckpt`/`.th`/HF `.bin` natively (see the
native-checkpoint-reader entry), so a consumer needs no Python. The converters stay for two reasons
that are not consumer-facing — they are the **byte oracle** `NFKMLXTorchParityTests` holds the native
reader to, and the **offline path** to a portable, pickle-free safetensors — and the Python
reference-parity tooling (`Tools/reference-parity`, `Tools/validation-assets`) is the ground truth
every Swift port is measured against, which is irreducibly Python (torch / transformers / diffusers).
`Tools/README.md` records this boundary.

The version lives in three places that must move together on release: `s.version` in
`InferKit.podspec`, the string `NFKInferKit.version` returns (`Sources/InferKit/NFKInferKit.m`), and
the `vX.Y.Z` git tag. A test asserts the shape of `NFKInferKit.version`, not its value, so a stale
constant does not fail CI — bumping it is a release step. SwiftPM takes its version from the tag, so
`Package.swift` carries none.

## Project Structure

```
./
├── Sources/InferKit/
│   ├── include/InferKit/            # Public headers (the API surface)
│   │   ├── InferKit.h               # Umbrella (imports every public header)
│   │   ├── NFKInferKit.h            # Library info (NFKInferKit.version)
│   │   ├── NFKInferenceBackend.h    # Swappable-engine protocol + NFKInferenceSubmit
│   │   ├── NFKInferenceRequest.h    # Immutable request (inputs + parameters + outputModality)
│   │   ├── NFKInferenceResult.h     # Immutable result (outputs)
│   │   ├── NFKInferenceJob.h        # Thread-safe async job handle
│   │   ├── NFKDynamicBackend.h      # Runtime backend discovery (activate an engine only if it's linked)
│   │   ├── NFKPassthroughBackend.h  # The mock; keeps builds/tests green with no weights
│   │   ├── NFKCoreMLBackend.h       # In-process Core ML (image + tensor I/O)
│   │   ├── NFKCoreMLLanguageBackend.h  # On-device causal language model through Core ML (macOS 15 / iOS 18)
│   │   ├── NFKRemoteBackend.h       # OpenAI-compatible chat client
│   │   ├── NFKRemoteTranscriptionBackend.h  # OpenAI-compatible audio→text (Whisper) client
│   │   ├── NFKAsyncGenerationBackend.h  # Submit-poll-fetch base for generation services
│   │   ├── NFKComputePlan.h         # Where Core ML plans to run each operation (ANE / GPU / CPU)
│   │   ├── NFKHardwareProfile.h     # What the machine is and how much of it is left
│   │   ├── NFKTensorConversion.h    # RGBA-interleaved ↔ planar CHW/HWC float tensors
│   │   ├── NFKMLMultiArray.h        # Interleaved ↔ MLMultiArray bridge
│   │   ├── NFKHFHub.h               # Hugging Face resolve/download/checksum/cache
│   │   ├── NFKTokenizer.h           # Text ↔ token ids (BPE / CLIP / WordPiece / Unigram, from tokenizer files)
│   │   ├── NFKVideoAsset.h          # Video clip value type
│   │   ├── NFKAudioAsset.h          # Audio clip value type
│   │   ├── NFKDetection.h           # Detected-object value type (label + confidence + normalized box)
│   │   ├── NFKKeypoint.h            # Pose-landmark value type (name + normalized position + confidence)
│   │   ├── NFKClassification.h      # Predicted-class value type (label + index + confidence)
│   │   ├── NFKAudioSegment.h        # Time-span value type (start/end seconds + label + confidence)
│   │   ├── NFKModality.h            # Text/Image/Video/Audio enum
│   │   ├── NFKInferenceKeys.h       # Shared input/parameter/output key vocabulary
│   │   └── NFKErrors.h              # NFKInferenceErrorDomain + codes
│   ├── NFK*.m                       # Implementations
│   └── NFK_ARC.h                    # Private ARC/MRC shim (NARC_ macros)
├── Tests/InferKitTests/             # XCTest (NFK*Tests.m)
├── Examples/                        # Compiled ObjC examples mirroring Docs/examples.md
├── SwiftExamples/                   # The same examples in Swift — pins the imported API shape
├── Docs/                            # README links out to these: inference-guide, examples,
│                                    #   installation, coreml-llm, companions
├── InferKit.xcworkspace             # Opens the core + both companions in one Xcode window. Each
│                                    #   FileRef must name a package DIRECTORY (`group:.`), not its
│                                    #   Package.swift, or Xcode treats that package as a dependency
│                                    #   and gives it no schemes. Un-ignored in .gitignore. Xcode does
│                                    #   not autocreate a scheme for `InferKitTests`, so that one is
│                                    #   shared in xcshareddata/xcschemes — ONE testable per scheme,
│                                    #   because several in one scheme silently collapse to the first.
│                                    #   A hand-written scheme needs all FIVE actions (Build, Test,
│                                    #   Launch, Profile, Analyze, Archive); omitting LaunchAction
│                                    #   makes Xcode refuse to build it — "not configured for running".
│                                    #   The core is referenced as `self:` (what Xcode's own generated
│                                    #   package.xcworkspace uses), so it is labelled by package name.
│                                    #   `group:.` shows literally "."; a named `<Group>` wrapper shows
│                                    #   a folder CONTAINING "."; `group:../InferKit` labels it but
│                                    #   resolves against the PARENT, breaking in any checkout not
│                                    #   named exactly InferKit (a fork, or a downloaded ZIP).
│                                    #   The core CANNOT move into a subdirectory to become a peer:
│                                    #   SwiftPM's only URL forms are `package(url:version:)` and
│                                    #   `package(url:range:)` — no subpath — so a package must sit at
│                                    #   the repository root to be consumable by URL at all.
├── InferKitMLX/                     # Optional MLX companion package (own Package.swift + tests)
├── InferKitFoundationModels/        # Optional Foundation Models companion (own Package.swift + tests)
├── Tools/inferkit-convert/          # Offline Python converter: HF causal-LM -> Core ML model dir
├── Tools/ane-placement/            # Paired Core ML models, to MEASURE what lands on the ANE
├── Tools/realesrgan-to-safetensors/ # Offline: Real-ESRGAN .pth -> safetensors for NFKMLXRealESRGAN
├── Tools/depth-anything-to-safetensors/ # Offline: Depth Anything V2 .pth -> safetensors (self-validating)
├── Tools/lama-to-safetensors/       # Offline: LaMa .ckpt generator -> safetensors for NFKMLXLaMa
├── Tools/u2net-to-safetensors/      # Offline: U²-Net .pth -> safetensors (renames rebnconvN keys)
├── Tools/sam-to-safetensors/        # Offline: SAM .pth -> safetensors (--list-keys; the remap lives in Swift)
├── Tools/sam2-to-safetensors/       # Offline: SAM 2 .pt -> safetensors (unwraps the `model` key; names pass through)
├── Tools/nafnet-to-safetensors/     # Offline: NAFNet .pth -> safetensors (renames sca.1/ups/middle keys)
├── Tools/rife-to-safetensors/       # Offline: RIFE HDv3 flownet.pkl -> safetensors (renames nested keys)
├── Tools/raft-to-safetensors/       # Offline: RAFT .pth -> safetensors (renames update_block/flow_head keys)
├── Tools/whisper-to-safetensors/    # Offline: OpenAI Whisper .pt -> safetensors (names already match)
├── Tools/hifigan-to-safetensors/    # Offline: HiFi-GAN g_* or espnet-paired -> safetensors (fuses weight norm)
├── Tools/fastspeech2-to-safetensors/# Offline: FastSpeech2Conformer .bin -> safetensors (names pass through)
├── Tools/espeak/install.sh          # Installs system espeak-ng (GPLv3, not bundled) for the espeak phonemizer
├── Tools/demucs-to-safetensors/     # Offline: Demucs checkpoint -> safetensors (--list-keys; the remap lives in Swift)
├── Tools/style-transfer-to-safetensors/ # Offline: TransformerNet .pth -> safetensors (drops IN running-stats; names match)
├── Tools/clip-to-safetensors/       # Offline: OpenAI CLIP .pt -> safetensors (JIT/state-dict; --list-keys; names match)
├── Tools/rvm-to-safetensors/        # Offline: Robust Video Matting .pth -> safetensors (names pass through; the positional remap lives in Swift)
├── Tools/codeformer-to-safetensors/ # Offline: CodeFormer .pth -> safetensors (names pass through; the fuse-dict/Sequential remap lives in Swift)
├── Tools/zero-dce-to-safetensors/   # Offline: Zero-DCE DCE-Net .pth -> safetensors (names match)
├── Tools/modnet-to-safetensors/     # Offline: MODNet .ckpt -> safetensors (--list-keys; the backbone remap lives in Swift)
├── Tools/yolo-to-safetensors/       # Offline: Ultralytics YOLO .pt -> safetensors (needs `ultralytics` to unpickle; the model.N remap lives in Swift)
├── Tools/segformer-to-safetensors/  # Offline: HF SegFormer .bin -> safetensors (--list-keys; the encoder remap lives in Swift)
├── Tools/swinir-to-safetensors/     # Offline: SwinIR .pth -> safetensors (--list-keys; drops rel-pos-index; the block remap lives in Swift)
├── Tools/colorizer-to-safetensors/  # Offline: eccv16 .pth -> safetensors (complete Sequential-index rename + ConvT axis swap; self-validating)
├── Tools/pose-to-safetensors/       # Offline: SimpleBaseline (mmpose ResNet-50) .pth -> safetensors (ConvT axis swap; stubs mmengine to unpickle)
├── Tools/deeplab-to-safetensors/    # Offline: torchvision DeepLabV3 .pth -> safetensors (--list-keys; the ResNet/ASPP remap lives in Swift)
├── Tools/conv-tasnet-to-safetensors/# Offline: Asteroid Conv-TasNet .pth -> safetensors (--list-keys; decoder axis+width fix; the separator remap lives in Swift)
├── Tools/denoiser-to-safetensors/   # Offline: facebookresearch/denoiser .th -> safetensors (--list-keys; shares the Demucs loader and its remap)
├── Tools/vad-to-safetensors/        # Offline: NeMo MarbleNet VAD -> safetensors (--list-keys; the separable-block remap lives in Swift)
├── Tools/audio-tagger-to-safetensors/ # Offline: PANNs Cnn14 .pth -> safetensors (carries the mel filterbank the model loads)
├── Tools/bisenet-to-safetensors/    # Offline: BiSeNet .pth -> safetensors (--list-keys; the two-path remap lives in Swift)
├── Tools/video-sr-to-safetensors/   # Offline: BasicVSR .pth -> safetensors (names pass through; the generator/Sequential remap lives in Swift)
├── Tools/mpsenet-to-safetensors/    # Offline: MP-SENet g_best .pth -> safetensors (names pass through; the GRU fold + Sequential remap live in Swift)
├── Tools/gtcrn-to-safetensors/      # Offline: GTCRN .tar/.pth -> safetensors (names pass through; the GRU fold + conv transpose live in Swift)
├── Tools/sgmse-to-safetensors/      # Offline: SGMSE+ Lightning .ckpt -> EMA safetensors (torch_ema applies EMA via model.eval(), then dumps dnn.state_dict())
├── Tools/storm-to-safetensors/      # Offline: StoRM Lightning .ckpt -> EMA safetensors (both nets under denoiser_net./score_net.)
├── Tools/build-all.sh               # Builds (and optionally tests) all three packages in one command
├── Tools/xcframework/build.sh       # Core -> a 3-slice universal static XCFramework. `swift build`
│                                    #   emits objects + a module, never a binary; `xcodebuild archive`
│                                    #   on a package scheme emits ONE merged .o, which `xcrun libtool
│                                    #   -static` turns into the .a that -create-xcframework wants.
│                                    #   (Apple's libtool — GNU's shadows it and rejects -static.)
├── Tools/xcframework/build-mlx.sh   # InferKitMLX -> a static AND a dynamic xcframework (arm64, three
│                                    #   slices) from one archive each, each carrying its Metal library.
│                                    #   See "Packaging InferKitMLX as an XCFramework" below.
├── Tools/xcframework/verify-mlx.sh  # Links a consumer against each artifact and RUNS a model, because
│                                    #   a binary that links can still fail to find its metallib.
├── Tools/validation-assets/         # Manifest + fetch.py: every real checkpoint the parity/triage suites load, and the reference sources the oracles import, into a durable ~/.inferkit-validation
│                                    #   shapes.py: a release's config + every tensor's shape by HTTP range request (no weights), for the structural checks
├── Tools/reference-parity/          # Offline: runs a model's (or a training objective's) reference implementation, records input + result for numeric comparison
├── Package.swift                    # SwiftPM manifest
├── InferKit.podspec                 # CocoaPods source spec
├── AGENTS.md / CLAUDE.md            # One document, two names: edit CLAUDE.md and copy it to AGENTS.md.
│                                    #   They drifted apart between v0.2.0 and 0.3.0 (AGENTS.md took
│                                    #   abbreviated edits) and were reconciled on 2026-09-01.
├── LICENSE                          # MIT
└── README.md
```

## Remote providers

`NFKRemoteProvider` carries the endpoint and protocol for the services a consumer is likely to call, so
pointing at one is a name rather than a hand-typed URL. Every endpoint was verified to exist when the
preset was added (a 401 or 405 without credentials is what confirms the path).

- **OpenAI-compatible** — one wire format, so `NFKRemoteBackend` serves them all: `openai`, `xai`
  (Grok), `gemini` (Google's OpenAI-compatible layer), `groq`, `mistral`, `deepseek`, `together`,
  `openrouter`, and the local servers `ollama`, `lmstudio`, `llamacpp`, `vllm`.
- **`anthropic`** is the exception and has its own backend, `NFKAnthropicBackend`. Four differences a
  URL swap cannot cover: the key is an `x-api-key` header rather than a Bearer token, an
  `anthropic-version` header is required, `max_tokens` is required rather than optional, and a system
  prompt is a top-level field rather than a message with a role. The reply is a list of typed blocks,
  so the text blocks are joined. A caller writes the same `NFKInputMessages` either way — a leading
  system turn is lifted into the top-level field.

No preset carries a default model name. Model identifiers change faster than a release does, and a
stale default fails at the first call with a message about the model rather than about the default.
The provider's own list is the source instead: `modelsWithAPIKey:error:` (and a completion-handler
form at user-initiated QoS) returns `NFKRemoteModel`s — identifier, display name, and where the
provider publishes them `ownedBy` / `createdAt` / `contextLength`, with the entry kept under `raw`.
Every preset answers the same `data[].id` envelope, hosted and local alike, so one parser serves all
thirteen; only the credential headers differ, and Anthropic paginates (`has_more` / `last_id` →
`after_id`, page size raised to 1000 from its default 20), which the catalog follows to the end. The
list is deliberately not filtered to chat models: the envelope carries no capability field, so any
filter would be a name heuristic that breaks on the next release. `NFKRemoteModelCatalog` is the
object under the convenience (timeout, session, `isReachableWithError:`, and the overridable
`sendRequest:` seam the tests stub). Readiness is not the same question:
`NFKAnthropicBackend` reports not-ready without a model because the API requires one, while an
OpenAI-compatible backend is ready with an endpoint alone, which **llama.cpp depends on** — its server
answers for whatever model it has loaded.

A preset carries one address, `baseURL`, and derives the rest (`endpointURL` = base +
`/chat/completions`, or `/messages` for Anthropic; `modelsURL` = base + `/models`; `URLForPath:` for
anything else, joining with exactly one slash). The bases differ per provider in ways a caller should
not have to know (Gemini's is `/v1beta/openai`, Groq's `/openai/v1`, OpenRouter's `/api/v1`), and the
derivation reproduces the literals the presets carried before, asserted by
`testEveryOperationURLIsDerivedFromTheBase`. `providerWithBaseURL:` re-points a preset — Ollama on
another port, or on a LAN machine — keeping its identity, protocol, and key requirement.

`NFKRemoteTransport` is the shared plumbing the chat, Anthropic, transcription, and catalog classes
used to each carry a copy of: the semaphore-blocked send, `authorizeRequest:apiKey:style:` (Bearer,
or `x-api-key` + `anthropic-version`), and `errorForResponse:data:` (a non-2xx status becomes
`kNFKError_InferenceBackendFailure` carrying the body, which is where a provider explains a rejected key
or an unknown model). Each class keeps its own `sendRequest:response:error:` override seam delegating
there, so a test stubs one class without touching the others. A failure that produced no response
at all is `kNFKError_RemoteUnreachable` with the URL-loading error under `NSUnderlyingErrorKey` — a
runner that is not running is a different answer from one that answered with an error or an empty
list, and an app shows "start Ollama" on that code. It is measured against a refused connection on the
discard port (`testTheTransportReportsARefusedConnectionAsUnreachable`), not stubbed.

`NFKRemoteEmbeddingBackend` is the embeddings counterpart (`POST /embeddings`, every preset but
Anthropic): `NFKInputPrompt` or joined `NFKInputMessages` → `NFKOutputEmbedding`, the key the MLX
embedders answer with, so a consumer's search code is engine-agnostic. `embeddingsForTexts:error:`
batches, ordered by the provider's `index` rather than by arrival (measured: a stub returning
index 1 before index 0 comes back in text order). `backendForProvider:apiKey:modelName:` answers nil
for Anthropic — it imports to Swift as the failable initializer `NFKRemoteEmbeddingBackend(for:apiKey:modelName:)`,
which the Swift example pins.

The local runners' native APIs are a second surface, and they are built (`NFKLocalModelRunner`,
`-[NFKRemoteProvider localRunner]` — a property, so Swift reads `.localRunner`; as a method it imported
as a function value and `runner?.x` failed to compile). The OpenAI endpoints say nothing about what is
installed, loaded, how large, or how to get one; the protocol's required set reads (`isRunning`,
`installedModelsWithError:`, `loadedModelsWithError:`, `detailsForModel:error:`) and its `@optional`
set changes the machine (`versionWithError:`, `pullModel:` → `NFKInferenceJob`, `deleteModel:error:`),
so an adapter adopts only what its runner has and a caller checks `respondsToSelector:` before
offering a button. `NFKOllamaRunner` adopts everything; `NFKLMStudioRunner` the reading set over
`/api/v0/models` (entries carry `state: loaded`, `type`, `quantization`, `max_context_length`; LM
Studio was not running here, so its shapes are its documented v0 rest API, stub-tested only);
llama.cpp and vLLM answer nil (nothing beyond the OpenAI surface to adapt — `/health` is what
`isReachableWithError:` already answers via `/v1/models`). The native base is the provider's base
minus `/v1` (`NFKLocalRunnerNativeBase`, private `NFKLocalRunnerSupport.h`), so a re-based preset
keeps its runner.
Measured against a live Ollama 0.33.2, read-only plus a pull/delete of a nonexistent name
(`testALiveOllamaAnswersItsNativeAPI`, gated on `INFERKIT_LIVE_LOCAL_MODEL`): `/api/tags` Already
carries `details.context_length`, `details.quantization_level`, and `capabilities` per model, so a
picker fills in one call with no `/api/show` per model; `/api/show` carries no id and keys the context
length by architecture inside `model_info` (`gptoss.context_length`), which the adapter lifts to where
`NFKRemoteModel` reads it; and a failing `/api/pull` answers HTTP 200 with `{"error":…}` as a line
Inside the NDJSON stream, so the pull job reads every line and treats an error line as failure and
`status: success` as completion — trusting the status code would report a failed pull as success. The
stream seam is `streamRequest:lineHandler:completionHandler:cancellation:` (overridable; the stub feeds
staged lines), distinct from `sendRequest:` because a streamed body cannot go through a blocking send.
`NFKRemoteModel` gained `sizeBytes` / `quantization` / `capabilities` and takes its id from `model` or
`name` where a list carries no `id` (Ollama's). A colon is legal in a URL path segment and is how
Ollama spells a tag (`llama3.2:latest`), but Foundation's `URLPathAllowedCharacterSet` encodes it to
`%3A`; `modelWithIdentifier:` and the LM Studio detail add `:` to the allowed set, pinned by a test.
**Two hazards from this round:** an `@[ a, b ]` literal inside an `XCTAssert…` macro argument splits
the macro on its comma unless the whole expression is parenthesized; and adjacent string-literal
concatenation as a direct element of an `@[ ]`/`@{ }` literal raises `-Wobjc-string-concatenation`
(parenthesize the element) — four of those had shipped in the catalog tests a round earlier because a
`tail` on the test log hid them. Grep the log for `warning:` without a tail.

The remaining modalities have remote backends, so every on-device direction has a hosted
counterpart on the same key (`NFKRemoteSpeechBackend` → `NFKOutputAudio` as an `NFKAudioAsset`, WAV
by default to match `NFKMLXSpeechBackend`; `NFKRemoteImageBackend` → 32BGRA `CVPixelBuffer` under
`NFKOutputImage`, choosing generations / edits / inpaint from `NFKInputImage` + `NFKInputMask` exactly
as `NFKMLXBackend` chooses; vision rides through the existing chat backends — `NFKRemoteBackend` attaches
`NFKInputImage` to the last user turn as an inline `image_url` content part, `NFKAnthropicBackend` as a
base64 `image` block before the text). Which presets serve which path was measured by probe, and the
probe needs a control: a `401` on a host that walls every path (DeepSeek answers 401 to
`/v1/nonesuch` too) proves nothing, so each host was also sent a nonsense path — served means the real
path answers 401/422/validation while the nonsense path 404s. Speech: openai, groq, together, xai,
mistral, openrouter. Generations: openai, together, xai, openrouter. Edits: openai, xai. Gemini's
OpenAI layer and all four local runners serve none; DeepSeek is undeterminable. Recorded on each
factory's `@discussion`. Vision is measured live (`testALocalVisionModelSeesTheImage`, gated on
`INFERKIT_LIVE_VISION_MODEL`): Ollama `qwen3.5:27b` given a flat blue square answers "blue" through the
content-parts shape, 23 s with the model load. `NFKImageCoding` is the public codec (ImageIO, now linked
by the core in `Package.swift` And the podspec): CGImage / 32BGRA-32RGBA CVPixelBuffer / BGRA8-RGBA8
MTLTexture → PNG or data URL; ImageIO-readable bytes → 32BGRA pixel buffer. Its private `CGImage…`
helpers must carry `CF_RETURNS_RETAINED` in a class extension — the public method promises +1 and the
analyzer reported five RetainCount issues when the helpers it delegates to did not. Importer names the
Swift examples pin: the class factories are failable initializers
(`NFKRemoteSpeechBackend(for:apiKey:modelName:voice:)`, `NFKRemoteImageBackend(for:apiKey:modelName:)`),
`NFKImageCoding.pngData(forImage:)` keeps `forImage:` (the parameter is not the type name), and a
`CF_RETURNS_RETAINED` CF return imports managed (no `takeRetainedValue()`). Two more ObjC traps from the
round: `inline` is a C keyword and not a variable name; and the `@{ a, b }`-inside-an-XCTAssert-macro
comma split struck twice more — hoist the request into a local before the macro, every time.

The remote chat path streams, cancels, calls tools, and returns structured output — the last
asymmetries with the on-device engines are closed. `submitInferenceJobForRequest:` on
`NFKRemoteBackend` / `NFKAnthropicBackend` sends `stream: true` and parses SSE through
`NFKRemoteTransport.streamRequest:session:lineHandler:completionHandler:` (the line-delimited primitive
the Ollama pull now shares; a failing status's body is collected whole and handed to the completion,
since a provider explains a rejected request in JSON, not a stream) and `SSEDataForLine:`. Each backend
keeps an overridable `streamRequest:lineHandler:completionHandler:` seam returning the cancel block,
which becomes `job.cancellationHandler` — the generic `NFKInferenceSubmit` wrapper never wired
cancellation, so a cancelled remote job used to run to the end on the server; now it reaches the
streamed form (`respondsToSelector:`) and the task is cancelled. OpenAI deltas: `choices[0].delta.content`
appends; `delta.tool_calls[]` assemble by index (id/name in the first delta, `function.arguments`
fragments after); `data: [DONE]` finishes; a stream closing without `[DONE]` still delivers what it
delivered. Anthropic events: `content_block_start` opens a block by index, `text_delta` /
`input_json_delta` append, `message_stop` finishes, an `error` event fails. **Tools:**
`NFKParameterTools` (`{name, description, parameters}`) → OpenAI `{type: function, function}` /
Anthropic `{name, description, input_schema}`; replies → `NFKOutputToolCalls` = `{id, name, arguments
(parsed), argumentsJSON}` (`result.toolCalls`). The key is spelled `"tools"`, the wire field's own
name, so an entry already in wire shape (has `type`, or `input_schema`) passes through unwrapped —
otherwise a caller who folded OpenAI tools by name gets double-wrapped. **Schema:**
`NFKParameterJSONSchema` → `response_format: {type: json_schema, json_schema: {name: response, schema}}`
(no `strict`, which demands `additionalProperties: false` throughout) / Anthropic has no response
format, so it is a forced tool `structured_output` (`tool_choice: {type: tool, name}`) whose `input` is
`NFKOutputStructured` and is not listed as a tool call. JSON is promoted to `structured` only when JSON
was asked for (schema, or a folded `response_format` of type `json_object`/`json_schema`) — JSON-looking
text is not guessed at. `NFKInputImages` attaches further images after `NFKInputImage`. **Retry:** the
transport's blocking send retries 429/502/503/504 after `Retry-After` (seconds; an HTTP-date falls to
the schedule) or `0.5·2^attempt`, `retryAttempts` (2) more times, never past `maximumRetryDelay` (8 s)
— a longer Retry-After ends the retries rather than waiting; a refused connection is not retried
(unreachable is an answer). Tested through an `NSURLProtocol` registered on a session configuration,
which drives a real `NSURLSession` with no network — the retry schedule, the ragged-chunk line splitter
(a CR is dropped, an unterminated last line arrives), and the whole-body collection on a 401.
Measured live on Ollama `qwen3.5:27b`: streaming delivers the reply in more than one partial with
the last partial equal to the final text (`testALocalRunnerStreamsTokenByToken`,
`INFERKIT_LIVE_LOCAL_MODEL`), and a declared `get_weather` tool is called with `{"city": "Paris"}`
(`testALocalModelCallsTheTool`, `INFERKIT_LIVE_TOOL_MODEL`). Swift importer: `submitInferenceJob(for:)`,
`result.toolCalls`, `NFKRemoteTransport.retryAttempts` as a class property. The two stubs' stream seams
deliver lines synchronously, so a test that wants to read a partial holds the stream open
(`holdOpen`) and asserts `.running`; a `[DONE]` inside the staged lines finishes the job before the
call returns.

The chat backends take audio, documents, and video in, and speak out; three more services close
the surface. `NFKRemoteAttachments` (private, `NFKRemoteMediaSupport.h`) gathers a request's media
Once — images + `NFKInputImages` + the frames sampled from `NFKInputVideo` as PNG, `NFKInputAudio` as
bytes + a format from the file extension, `NFKInputDocument(s)` as `{data, filename}` — and each backend
writes its own wire shape: OpenAI `image_url` / `input_audio` / `file` parts, Anthropic `image` /
`document` blocks (the Messages API takes no audio, so `NFKAnthropicBackend` Refuses `NFKInputAudio`
and `NFKParameterAudioOutput` with `kNFKError_InferenceUnsupported` rather than dropping them).
`NFKParameterAudioOutput` → `modalities: [text, audio]` + `audio: {voice, format}` (format defaults to
wav, the container `NFKMLXSpeechBackend` writes); the reply's `message.audio.data` (base64) is written
through `NFKRemoteWriteMediaFile` → `NFKOutputAudio`, and `message.audio.transcript` stands in for the
null `content`; streamed `delta.audio.data` chunks are concatenated AS BASE64 then decoded once.
`NFKVideoSampling` (public; core now links AVFoundation — Package.swift and the podspec) samples
`count` frames at `(i + 0.5)/count` of the duration **plus one millisecond**, because a clip whose
frames divide the count evenly lands every midpoint on a frame edge, with a half-frame seek tolerance
from the track's `nominalFrameRate` (loaded through `loadValuesAsynchronouslyForKeys:@[@"tracks"]` —
`loadTracksWithMediaType:` is macOS 12 / iOS 15, above the core's floor, and trips
`-Wunguarded-availability-new`). **Measured on this machine:** sampling a clip
right after Ollama's 27B model had occupied the GPU fails with `AVFoundationErrorDomain -11821`
"Cannot Decode", underlying OSStatus **-12911 `kVTVideoDecoderMalfunctionErr`**, after a ~4-minute
timeout — a broken hardware decode session — and the next session works; the sampler recreates its
generator once on `AVErrorDecodeFailed`, which turns that run from a failure into a slow pass
(`testFewerSamplesAreSpacedEvenlyThroughTheClip` at 245 s in the live ordering, 0.1 s otherwise). Two
boundary theories preceded that finding and were wrong; the sampler's own error, once the test
printed it, was what settled it. The test clip writer (`NFKTestClip`, AVAssetWriter, 64×64 H.264 at
2 fps) must hold every pixel buffer until `finishWriting` — the pool hands a released buffer straight
back and the encoder may still be reading it — and H.264 chroma subsampling bleeds ~0.27 into a pure
primary's other channels, so colour assertions carry a 0.35 tolerance. Video → text is measured
live: `qwen3.5:27b` names red and blue from four frames of a red–green–blue–white clip
(`testALocalVisionModelDescribesASampledClip`, `INFERKIT_LIVE_VISION_MODEL`).
`NFKRemoteTranscriptionBackend` gained `emitsTimestamps` (`response_format=verbose_json` unless the
caller set one; `segments[]` → `NFKAudioSegment` with `exp(avg_logprob)` as confidence, matching the
on-device Whisper backend's `NFKOutputSegments`) and `translates` (the path's last component swapped
to `translations`). `NFKRemoteVideoBackend` is the first shipped `NFKAsyncGenerationBackend`
subclass (OpenAI `/v1/videos`, verified by probe; no other preset serves one): JSON submit, or
multipart with `input_reference` when `NFKInputImage` is present — the base gained the
`submitRequestForRequest:` hook for that and `failureReasonFromStatusResponse:` so the service's
`error.message` reaches the job — percentage `progress` → fraction, poll every 5 s, then get
`/videos/{id}/content` → `.mp4` `NFKVideoAsset`. `NFKRemoteReranker` (`/rerank`, together +
openrouter verified; results arrive in relevance order and are put back in the documents' order) and
`NFKRemoteModerationBackend` (`/moderations`, openai + mistral; `category_scores` →
`NFKClassification`s most confident first, the verdict under `NFKOutputStructured`). Unverified live:
audio in/out, PDFs, video generation, rerank, moderation — all need paid keys; their envelopes are
stub-tested and their paths probe-verified.

**Deliberately absent.** Midjourney has no official public API (its API host does not resolve), so
shipping a preset would imply one exists. `opencode.ai` answers `Not Found` on its API path — it is a
coding agent that calls other providers rather than an inference service. Codex is OpenAI's coding
agent, not a separate endpoint; it is the `openai` preset.

`NFKRemoteProviderTests` stubs the transport to assert Anthropic's request shape without a network, and
carries one live test gated on `INFERKIT_LIVE_LOCAL_MODEL` that runs against a local server when one is
listening.

## Measuring where Core ML runs

`MLComputeUnits` is a request. Core ML places an operation the Neural Engine cannot run somewhere
else and reports nothing about having done so, so a model asked for the Neural Engine can run
entirely on the CPU and behave exactly as if it had not. `NFKComputePlan` answers that: it reads a
compiled model's plan per operation, without running it, and reports the counts per device plus
`operatorNamesOffNeuralEngine` — which is the list to work from when a conversion is being tuned,
since one unsupported operator in the middle of a network splits it and costs more than its own
share of the time.

It needs macOS 14.4 / iOS 17.4 / tvOS 17.4, which is where Core ML began publishing the information.
`isAvailable` reports whether the OS can answer, and an older system fails with
`kNFKError_InferenceUnsupported` rather than returning an empty plan, because zero operations on the
Neural Engine and "cannot tell" are different answers. `powermetrics --samplers ane_power` is the
runtime cross-check where the API is unavailable; it needs elevated privileges, which is why it is
documentation rather than code here.

What it says about this repository's own Core ML language model is bad news, and it is measured.
A GPT-2 converted by `Tools/inferkit-convert` places 0 of 448 operations on the Neural Engine —
all of them on the GPU, under `MLComputeUnits.ALL` — and the conversion emits `ANECCompile() FAILED`
into the middle of its ordinary output, which is the only warning there is. The Neural Engine is
reachable on the same machine: a plain attention block at sequence 64 places 100% of its operations
there. The transformer layout guidance did not help. The ordinary `(B, S, C)` + `nn.Linear` form
was already fully placed, and rewriting it into the 4-D `(B, C, 1, S)` 1×1-convolution form left
placement unchanged while taking the operation count from 40 to 313 — so the converter's ANE-friendly
rewrite is work with no measured benefit, and it is deliberately not done. What actually moves a
language model off the Neural Engine is still unidentified, and the single-token comparison in
`Tools/ane-placement/` does not settle it: both of those models landed on the CPU with four and eight
placed operations, which is Core ML declining to dispatch a trivial graph rather than evidence about
Neural Engine eligibility. That experiment is inconclusive, not negative.
The cause is now isolated, by adding one property at a time to a model that is fully placed
(`Tools/ane-placement/add_one_property.py`). Two facts, over the same twelve-layer 768-wide model:
a single-token forward is not placed on the Neural Engine — sequence 64 scores 100% and sequence 1
scores 0%, with the stateful cache innocent and the embedding gather costing four CPU operations — and
a multifunction package takes one placement decision, so the seq-64 prefill function that scores
100% alone drops to 0% when packaged with the seq-1 decode function. That is exactly what the
converter emits, and exactly why the whole thing runs on the GPU.
`ANECCompile() FAILED` was a red herring: it appeared once during conversion and does not reproduce.

Whether to act on it is a smaller question than it looks. Timed on the same models, prefill takes
3.77 ms on the Neural Engine against 4.98 ms on the GPU, and decode is unchanged — so splitting the
package into two models buys about 1.3× on time-to-first-token and nothing per token. Note also that a
compute plan reports the preferred device, not an execution trace: the seq-1 model is planned entirely
onto the GPU and still runs fastest under `ALL`. Placement is where Core ML intends to run something;
timing is what decides.

`coremltools` cannot be used for this. Its own `MLComputePlan` binding returns `None` for every
operation in 9.0 on macOS 26, including for models the Objective-C API reports on in the same session,
so a Python-side check reads as "nothing is on the Neural Engine" whatever the truth is.

`NFKCoreMLBackend` gained `computeUnits` to go with it (`NFKCoreMLLanguageBackend` already had one).
**`MLComputeUnitsCPUOnly` is zero**, so the property is initialized explicitly — an unset one would
quietly move every model to the CPU.

## Sizing a model against the machine

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

## Value-type convenience accessors

`NFKInferenceResult` and `NFKInferenceRequest` expose typed convenience accessors over their
dictionaries for the keys with a single natural type: `result.text` / `result.structured` /
`result.embedding` (NSArray<NSNumber *> for `NFKOutputEmbedding`) / `result.detections`
(NSArray<NFKDetection *> for `NFKOutputDetections`) / `result.pose`
(NSArray<NFKKeypoint *> for `NFKOutputPose`) / `result.classifications`
(NSArray<NFKClassification *> for `NFKOutputClassifications`) / `result.segments`
(NSArray<NFKAudioSegment *> for `NFKOutputSegments`) and
`request.prompt` / `request.negativePrompt` / `request.messages`. Each is a read-only computed getter
that type-checks and returns nil on a mismatch (no crashing cast). Image / mask / video keys stay on
`outputForKey:` / `inputForKey:` because their representation is chosen by the backend or caller
(CVPixelBuffer, texture, CGImage) — do not add typed accessors for those.

## Tokenizers

`NFKTokenizer` is a class cluster: `tokenizerForManifest:directory:error:` reads the manifest's
`tokenizer.type` and returns the subclass named there. The concrete subclasses are private; the
factory is the public entry, so a new type needs no header change.

- `bpe-bytelevel` → `NFKByteLevelBPETokenizer`, the GPT-2 / Qwen / o200k scheme. The
  pre-tokenization pattern is selectable (`"pretokenizer": "gpt2"` (default), `"qwen2"`, or `"o200k"`
  — OpenAI's o200k_base / o200k_harmony, which gpt-oss ships: digits in runs of at most three and words
  split wherever their case pattern turns over — in the manifest spec), and
  the choice is load-bearing: a merge cannot cross a pretoken boundary, so a Qwen vocabulary encoded
  under the GPT-2 pattern produces different, valid-looking ids for the same text ("-pop" is one
  Qwen2 pretoken because a letter run may absorb one leading punctuation character; digits split
  singly; a punctuation run absorbs trailing newlines; whitespace runs ending in a newline hold
  together). Found by the MiniMax Music 3 prompt parity record, which mis-tokenized under the
  default; the Qwen3 text path (`NFKMLXLanguage.backend(directoryURL:)`) now names `qwen2` too. An
  unknown pretokenization name is refused rather than silently defaulted.
- `clip` → `NFKCLIPTokenizer`, which the CLIP image-text model and every Stable Diffusion text encoder
  take. It **subclasses** the byte-level tokenizer through four hooks — `normalizedText:`,
  `pretokenizationPattern`, `symbolsForWord:`, `finalizedText:` — because CLIP shares byte-level BPE
  and differs in exactly those places: text lowercases and its whitespace collapses; the pattern takes
  a run of letters, **one** digit, or a run of other non-space characters, with no leading space (so
  "2024" is four tokens); a word's last character carries `</w>`, which the vocabulary distinguishes;
  and decoding turns `</w>` back into a space.
- `unigram` / `sentencepiece` → `NFKUnigramTokenizer`; `wordpiece` → `NFKWordPieceTokenizer`.

`bytesForTokenId:` (0.3.0) returns the bytes one id contributes to decoded text — a fragment of a
multi-byte character comes back as that fragment, a word-piece word start carries its space, a
special token its literal — which is what a byte-level grammar reasons over; `NFKMLXVocabulary`
reads the whole table once per tokenizer.

`encode:` returns the ids for the text alone. A model input's start and end markers and its padding
are the model's geometry, not the tokenizer's, so they are added where the context length is known —
`NFKMLXSDPromptTokenizer` for the diffusion path.

## Grammar-constrained sampling in the core

`NFKTokenConstraint` (`Sources/InferKit/NFKTokenConstraint.m`, 0.4.0) ports the MLX companion's
byte-level engine into the core so `NFKCoreMLLanguageBackend` can constrain its own sampler:
`NFKTokenVocabulary` holds every id's bytes (from `NFKTokenizer.bytesForTokenId:` at the model's logit
width, or explicit `NSData`s), `NFKJSONConstraint` is JSON syntax with an `NFKJSONRoot` and the same
8-byte whitespace cap the MLX grammar carries, `NFKChoiceConstraint` a fixed set, and a
`NFKTokenConstraintCursor` masks a `float *` logit buffer in place (`-inf` for the inadmissible; the end
token admitted only at a complete document, or when nothing else is admissible so the run stops rather
than emitting a refused token). The grammar state is a fixed-size C struct (`NFKConstraintState`,
64 nesting levels; deeper is refused) so a token's bytes are tried on a copy — no per-state mask cache,
which at the Core ML backend's vocabularies is a few milliseconds a step. The backend builds the
constraint lazily on the first sample (the logit width is only known then) from the core keys
`NFKParameterOutputFormat` (`"json"` / `"json-object"` / `"json-array"`) and `NFKParameterChoices`, masks
Before temperature and nucleus, feeds each emitted token back to the cursor, and returns the parsed
reply under `NFKOutputStructured` when JSON was asked for. The MLX backend honors the same two core keys
as aliases of its own, so a request is engine-agnostic. The schema grammar (`NFKMLXJSONSchemaConstraint`)
stays MLX-only. Two ObjC test traps struck while writing its tests: an `@[ ]` literal inside an
`XCTAssert` macro argument splits the macro (parenthesize the argument), and `NSSet` has
`isSubsetOfSet:` but no `isSupersetOfSet:`.

## Dynamic backend discovery

`NFKDynamicBackend` (core, Foundation-only) activates an optional engine only when its classes are
linked into the consumer's build, with no build dependency on that engine. InferKit ships only
zero-dependency backends; a heavier engine (Stable Diffusion, a Core ML/MLX model, a C/Rust runtime) is
brought by the consumer and discovered at runtime.

- A consumer adds a small class conforming to `NFKDynamicBackendProvider` (one method,
  `+makeInferenceBackend`) that builds a backend around their engine.
- InferKit resolves it by name through `NSClassFromString`, so it never references the engine's
  symbols. When the engine is not linked, the class is absent and resolution returns nil — the feature
  is simply unavailable, with no link error and no crash.
- Resolve by provider class name (`+backendForProviderClassName:error:`) or by capability: a consumer
  registers provider class names under a capability string (`+registerProviderClassName:forCapability:`),
  and `+backendForCapability:error:` activates the first present one (most-recently-registered first).
- Each built-in capability has a default provider class name (a `capability → class name` map in the
  core), tried last so a registered override wins:
  - `NFKCapabilityStableDiffusion` (`"stable-diffusion"`) → `NFKStableDiffusionProvider` — **InferKitMLX
    ships it** (wraps `NFKMLXBackend`), so linking InferKitMLX makes `stableDiffusionBackend()` work.
  - `NFKCapabilityTranscription` (`"transcription"`) → `NFKMLXWhisperProvider` — **InferKitMLX ships it**
    (wraps `NFKMLXWhisper`); a consumer's native engine (whisper.cpp) registers to override.
  - `NFKCapabilityTextGeneration` (`"text-generation"`) → `NFKFoundationModelsProvider` —
    **InferKitFoundationModels ships it** (wraps `NFKFoundationModelsBackend`), so linking that package
    activates on-device LLM.
  - `NFKCapabilityControlNet` (`"controlnet"`) → `NFKControlNetProvider` — no shipped default; a consumer
    brings a ControlNet/SD engine and adopts that name or registers their own.
  Providers build lazily (construction is cheap; weights/pipeline initialize on first use, off the
  render thread). This is how the **existing** SD / Whisper / Foundation Models implementations activate
  in the core only when the companion is linked, with no build dependency.

## Packaging InferKitMLX as an XCFramework

`Tools/xcframework/build-mlx.sh` emits **two** artifacts from one archive per slice (arm64; macOS,
iOS device, iOS simulator):

- `InferKitMLX.xcframework` — static: `libInferKitMLX.a` (every target's object merged with `libtool`,
  the core included), headers, a modulemap, and the `mlx-swift_Cmlx.bundle` that slice's consumer
  ships. 47 MB a slice, 140 MB in all.
- `InferKitMLXDynamic.xcframework` — dynamic: `InferKitMLX.framework`, Metal library inside. A slice is
  a 15.5 MB binary, 3.6 MB of shaders, and 12 MB of Swift module interfaces an Objective-C consumer
  never reads; 96 MB in all.
- `CoreHeaders/` (the core's headers beside a modulemap) for Objective-C consumers of the dynamic one.

Either way a consumer's binary grows by about 14 MB plus the shaders.

What made it possible. MLX needs `default.metallib`, which SwiftPM delivers as a
`mlx-swift_Cmlx.bundle` resource, and a bare library carries no resources. MLX's loader
(`mlx/backend/metal/device.cpp`) tries four places, and `current_binary_dir()` is `dladdr` on MLX's
own code, so it names whichever mach-O image MLX was linked into:

1. `<binary dir>/mlx.metallib` — the static route, beside the consumer's binary.
2. `<binary dir>/Resources/mlx.metallib`.
3. `mlx-swift_Cmlx.bundle` in the main or any loaded bundle — the static route inside an app bundle,
   and the layout Copy Bundle Resources produces.
4. `<binary dir>/Resources/default.metallib` — commented in the reference as "if SwiftPM wrapped as a
   dynamic framework", which is exactly the dynamic route.

A dynamic framework satisfies (4) by itself. A static consumer satisfies (3) from the bundle inside
their slice. All three paths are measured by `verify-mlx.sh`, which links a consumer and **runs a
model**: a binary that links can still fail at the first array evaluation, which is the failure this
packaging exists to avoid.

The static xcframework carries its own shaders, and that is delivery only. An xcframework is a
build-time container that never ships, and Xcode copies nothing out of a static one, so the consumer
still places the bundle in their own product — measured, not assumed: a `binaryTarget` consuming a
slice with the bundle inside builds and then throws `Failed to load the default metallib`, and copying
the bundle beside the binary makes the same probe pass. What co-location buys is that the shaders
cannot arrive separately from the library, at 1.7 MB compressed against the two forms an earlier layout
shipped in a second asset. SwiftPM accepts the extra file in a slice without complaint. The earlier
separation was also justified here by a claim that a `binaryTarget` zip may hold nothing beside the
xcframework; that claim was never tested — SwiftPM requires `https` for a `binaryTarget` URL, so it
cannot be checked against a local server — and the question is now moot, because the zip holds exactly
one `.xcframework` either way.

Neither variant is smaller. A consumer using one model links to 13.8 MB static against 14.2 MB
dynamic — dead-stripping buys about 3%, because `Cmlx.o` is one merged object and MLX's runtime is
densely interconnected. Of the ~18 MB a slice weighs, MLX's C++ is 72% of the code and its shaders are
3.6 MB; InferKitMLX and the core together are about 320 KB. So the choice is deployment mechanics:

- Static needs no embedding and no code signing, and vends both modules from one plain modulemap.
- Dynamic is one self-contained drop-in and is shared between several consumers, but clang refuses a
  non-framework module inside a framework, so its `InferKit` module needs a modulemap passed with
  `-fmodule-map-file`. An Objective-C consumer is better served by the static variant.

Linking recipes and the release matrix live in `Docs/installation.md`, one per consumer shape, each
verified by building and running it: Xcode static and dynamic, SwiftPM `binaryTarget` for both variants
(SwiftPM wires the static one's headers and modulemap automatically; the dynamic one still needs
`CoreHeaders` passed through `-fmodule-map-file`), plug-in bundles, and a consumer's own static library.
The artifacts are not committed — this repository is source-distributed, so a consumer resolving it
clones its history. Three compressed release assets, one per variant carrying every slice: core 0.7 MB,
static 28 MB, dynamic 19 MB. The core's `build.sh` cleans only its own artifact — both scripts share
`.xcframework-build/`, and an `rm -rf` of the whole directory once discarded a twenty-minute MLX build
beside a twenty-second core one.

`--variant static|dynamic|both`, `--slices macos,ios,iossim`, and `--no-swift-interfaces` trim the
output. `--slices` also trims the build: each slice compiles those 289 translation units again, so it
is the difference between about seven minutes and twenty. `--no-swift-interfaces` applies only to the
dynamic framework — the static library never carried interfaces, and is Objective-C-only by
construction. `verify-mlx.sh` reports what it skipped rather than failing when a variant or the
macOS slice is absent; an iOS binary cannot be executed on the host.

Nothing in `Package.swift` changes for distribution. One `xcodebuild archive` of the ordinary
`InferKitMLX` product yields each target's merged object; `libtool` makes the static library from them
and `clang -dynamiclib` makes the framework binary from the same library. A second `type: .dynamic`
product was tried and abandoned: two library products in one package install the same dependency
objects, which fails an iOS archive with "Multiple commands produce .../ArgumentParser.o" (macOS
tolerated it). `BUILD_LIBRARY_FOR_DISTRIBUTION=YES` succeeds across the whole graph, C++ interop
included.

The dylib links `-all_load` and **without `-dead_strip`**: `NFKDynamicBackend` resolves providers
through `NSClassFromString`, so a class with no static reference is still reachable. It costs about
1.3 MB against a dead-stripped link, which is the right trade for a binary whose discovery mechanism
is by name.

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
Documentation informs and describes; it is not persuasive writing.

## InferKitMLX (companion package)

Everything here builds for iOS. `mlx-swift-examples` is gone: it was pulled in for one thing, the
bundled text-to-image model, and its pinned revision declared an iOS 16 floor while depending on
mlx-swift products that require 17 (upstream `main` has the same defect), which failed every iOS build
from deep inside the graph. `NFKMLXBackend` is now built from this package's own parts —
`NFKMLXSDTextEncoderNet`, `NFKMLXSDUNet`, `NFKDDIMScheduler`, `NFKMLXSDAutoencoder` — so the gating
(`condition: .when(platforms: [.macOS])` and the matching `#if os(macOS)`) is gone with it, along with
17 of the 18 transitive packages the companion used to resolve.

`InferKitMLX/` is a separate SwiftPM package (Apple Silicon, macOS 14 / iOS 17 — MLX's floor). It
depends on the core (`.package(path: "..")`) and `mlx-swift`, and on nothing else at runtime
(`swift-numerics` arrives through mlx-swift; the swift-docc-plugin chain is dev-only). It keeps MLX
out of the core so the core stays cross-platform and dependency-free.

Backends there adopt the same `NFKInferenceBackend` protocol from Swift:

- `NFKMLXBackend` (`@objc`) — a bundled Stable Diffusion release: `.stableDiffusion15`,
  `.stableDiffusion21Base`, or `.sdxlTurbo`. No image input runs text-to-image; a `CGImage` under
  `NFKInputImage` runs image-to-image (`NFKParameterStrength` controls source retention). It resolves
  the release's repository, downloads its files through the core's `NFKHFHub` (which caches at
  `<cache>/<repo>/<revision>/<path>`, reproducing the release's own tree), and builds
  `NFKMLXTextToImage` around them. SD 2.1 base is a **gated** repository: `NFKHFHub.accessToken` or
  `HF_TOKEN` supplies the credential. The backend loads a release at the precision it was published in
  (`precision`, an `@objc` `NFKMLXWeightPrecision`, default `.checkpoint`); the parity records were
  measured at `.float32`, which a half-precision release costs twice the memory to reach.
- `NFKMLXModuleBackend` — a bring-your-own MLX image model: supply a `@Sendable (MLXArray) -> MLXArray`
  forward closure; the backend handles the InferKit contract and the RGB `CGImage ↔ MLXArray` bridge.
- `NFKMLXMattingBackend` — a bring-your-own MLX matting model (keyer / background remover): plate under
  `NFKInputImage` + optional hint under `NFKInputMask` → `(plate, hint) -> [H,W,4]` closure → straight
  RGBA image. `NFKMattingConfiguration` adds the matte under `NFKOutputMask`, premultiply, color space,
  byte-level tiling for large plates, and `MTLTexture` output.
- `NFKMLXTensorBackend` — general named-tensor backend, `[String: MLXArray] -> [String: MLXArray]` over
  `NFKMLXTensorPort`s, for multi-input/multi-output image models.
- `NFKMLXDiffusionBackend` — a bring-your-own MLX diffusion model (iterative sampler, not a single
  forward). The consumer supplies `encode` (request + bridged image/mask → `NFKDiffusionContext`),
  `denoise` (latent + timestep + context + guidance → prediction), `decode` (latent → image tensor),
  and a scheduler; the backend owns the loop, per-step progress/cancellation, and the image bridge.
  No source latent runs text-to-image; a source latent runs image-to-image (`NFKParameterStrength`);
  a source latent + mask runs inpainting (kept region held to the source each step).
  `windowedContinuation(...)` produces output longer than the model's window, the `NFKMLXMusic3`
  mechanism lifted out for reuse: it tiles one axis into overlapping windows and keeps continuity
  Inside the sampler — each window's overlap is held to the previous window's finished latent,
  re-noised to the current step's level through `scheduler.addNoise` (the same primitive the inpaint
  path uses, so it is scheduler-agnostic rather than flow-specific), locked to it after the loop, and
  the windows stitched at the hop stride. It is a static helper over the shared-conditioning case
  (text-to-long-image, an extendable texture); a per-window `condition` encoder like the music path's
  keeps its own specialized loop. `windowStarts` pulls the final window back to end exactly at the
  total. Tested by single-window equivalence to the plain loop and that the overlap hold changes the
  result (`NFKMLXDiffusionWindowedTests`).
  `NFKDiffusionLatentPreview` gives the progress callback something to show. A full decode per
  step costs more than the sampling, so the preview is a 1×1 convolution over the channel axis —
  twelve weights and three biases for a four-channel latent — reported as the job's `partialResult`,
  the same mechanism a streaming text backend uses. `previewEverySteps` thins it. `.passthrough` suits
  a latent already in image space; `.stableDiffusion` and `.stableDiffusionXL` are the published
  latent-RGB factors, and what is measured about them is how closely they track a real decode: against
  the released SD 1.5 autoencoder, on a latent encoded from a real photograph, the shipped map
  reproduces the decode's structure at a mean-removed correlation of **0.93**.
  Raw cosine is the wrong measure here and the first version of that test used it — two nearly
  constant grey images score 0.98, so cosine cannot tell a preview showing the picture from a flat
  rectangle of the right brightness. It also fed the map a latent at the autoencoder's scale rather
  than the sampler's; the pipeline divides by `scaleFactor` before the autoencoder, so the two differ
  by 5.5× and the earlier numbers were artifacts of that. `fitted(latentChannels:decode:sample:)`
  derives a map from any decoder by least squares, which is the route for a model with no published
  factors — though a map fitted on noise scores 0.917 where the shipped coefficients score 0.931, so
  fit on latents encoded from real images. A preview is a progress indicator, so a mismatched map
  returns nil rather than failing the run it is only reporting on. `NFKDiffusionScheduler`
  is the sampler seam. `NFKDDIMScheduler` is at reference parity against diffusers' own
  `DDIMScheduler` (worst per-step latent cosine 0.9999999999999941, add-noise 0.9999999999999983,
  and the schedule it visits matches exactly). Reaching it corrected two things the sampler had wrong:
  it divided the training range evenly (999, 949, … 49) where the reference walks a fixed stride and
  adds **`steps_offset`** (951, 901, … 1), and its final step denoised against a signal ratio of 1
  where the released configurations set **`set_alpha_to_one: false`** and use the ratio at training
  step 0, which deliberately leaves a little noise. Both are now init parameters defaulting to the
  released Stable Diffusion values. The four `diffusion-*` reference stand-ins pass `setsAlphaToOne:
  true`, because an oracle that drives the loop to an exact target only lands on it when the last step
  denoises fully. `NFKDDIMScheduler` (epsilon/v/sample prediction types) and `NFKLCMScheduler`
  (few-step latent-consistency: consistency boundary `c_out·x₀ + c_skip·latent`, then fresh step-keyed
  deterministic noise from the same SplitMix64 stream) ship. Noise is a deterministic SplitMix64 +
  Box–Muller stream, so a run is repeatable without the MLX random state.
  Reference pipelines register by name via `NFKMLXReferenceModels`: `registerDiffusionUpscaler`
  (2× upscale), `registerDiffusionDepth` (Marigold-style depth), `registerDiffusionInpainter`,
  `registerControlNet` (`diffusion-controlnet`: a control map under the core key `NFKInputControl` →
  `context.conditioning["control"]`, the slot a real ControlNet `denoise` reads to inject residuals).
  To stay CI-runnable without real weights, their `denoise` is an oracle epsilon that drives the loop to
  a target derived from the input; a real integration swaps the oracle for a trained UNet forward.
  ControlNet/LCM need no full SD reimplementation: LCM is a scheduler swap, ControlNet is a `denoise`
  closure over `conditioning["control"]`; the UNet is brought by the consumer's `denoise` or a
  dynamically linked SD engine.
- `NFKMLXRealESRGAN` (`@objc`) — a real single-forward upscaler: the Real-ESRGAN generator (RRDBNet)
  in `MLXNN` (`Conv2d` + `leakyRelu`, norm-free), run through `NFKMLXModuleBackend` for ×4 upscaling.
  `+register` puts it in the registry under `real-esrgan-x4`, so ObjC/MetalForge builds it by name (and
  downloads weights via `NFKMLXHub`). The module structure and parameter names mirror the reference
  PyTorch `RRDBNet`, so `loadWeights(into:from:)` loads a safetensors checkpoint (`loadArrays` →
  `update(parameters:)`), transposing 4-D conv weights from PyTorch `[out,in,kH,kW]` to MLX
  `[out,kH,kW,in]`. A `.pth` release converts to safetensors first with
  `Tools/realesrgan-to-safetensors/convert.py`; the weight-load path is proven offline by a test that
  saves the net's params in PyTorch layout, reloads through the transpose, and confirms the forward
  matches. Adds the `MLXNN` product of `mlx-swift` to the target. Reference parity against the general
  ×4 release (cosine 0.9999947) and the anime release, a six-block generator where the general one has
  twenty-three (0.9999956).
- `NFKMLXDepthAnything` (`@objc`) — a real single-forward depth model: the Depth Anything V2 DINOv2 ViT
  encoder (`pretrained.*`) + DPT head (`depth_head.*`) in `MLXNN`, run through `NFKMLXModuleBackend`
  (image → grayscale depth under `NFKOutputImage`). `+register` under `depth-anything-v2-small`;
  `NFKMLXDepthConfiguration` holds the ViT-Small dims (Base/Large change them). The encoder runs a fixed
  518×518 (so `pos_embed` matches without interpolation) and the map resizes back. `loadWeights(into:from:remap:)`
  loads a **safetensors** checkpoint; the DPT key layout is intricate, so `Tools/depth-anything-to-safetensors/convert.py`
  is self-validating (matches every key against the module's expected layout, reports mismatches).
  Reference parity across all three released sizes: Small 0.99992 (encoder seam 0.9999924), Base
  0.99995, Large 0.99984, on the min-max-normalized 8-bit depth map. Two DPT-head fixes found while
  porting Depth Anything 3 raised these from ~0.998: the two `resize_layers` are `ConvTransposed2d`,
  whose PyTorch weight is `[C_in, C_out, kH, kW]` and needs the transposed-conv axis order `(1,2,3,0)` →
  `[C_out, kH, kW, C_in]`, not a regular convolution's `(0,2,3,1)` (both are square, so the wrong order
  loaded silently and scrambled the kernel, a small effect on the normalized map); and the
  FeatureFusionBlock upsamples bilinear with align_corners=true (`NFKMLXResample.resizeBilinearAlignCorners`),
  not nearest, the dominant fix. The oracle drives the authors' own `depth_anything_v2` package
  (`IK_DEPTH_VARIANT` picks the encoder config); it drove `transformers` until that package dropped the
  `depth_anything` model type, and the parity test kept passing throughout because it compares against a
  stored record, not a live oracle.
- `NFKMLXDepthAnything3` (`@objc`) — Depth Anything 3 monocular depth (DA3-SMALL). Reference parity
  against the authors' `depth_anything_3` package on the released weights: the four hooked backbone
  features and every DualDPT head stage ≥ 0.9999999999, the exp-depth map mean-removed 0.99999999992
  (max relative difference 5.5e-7). The backbone is a DINOv2 ViT variant rather than a reuse of the V2
  encoder: from block 4 it adds a 2-D rotary embedding, per-head query/key normalization, a learned
  camera token injected into the class-token slot, and alternating local/global attention whose
  cross-view "global" blocks collapse to query/key-normalized self-attention for a single image (their
  uniform rotary positions cancel in the score). Each hooked feature concatenates the preceding local
  block's output with the current global block's (`cat_token`), the final norm applied to the global
  half only, so the `DualDPT` head reads twice the embedding width (`dim_in` 768). Only the depth branch
  of the DualDPT is built; the ray branch, the camera decoder/encoder, and the aux heads are named as
  deliberately unimplemented (269 tensors loaded, 168 dropped, 0 unaccounted). The head adds a UV
  positional embedding and upsamples bilinear with align_corners, and the output convention is exp-depth
  (`exp(logits)`), not V2's relative disparity. `loadWeights` reads the released safetensors directly
  (`model.backbone.pretrained.*` + `model.head.*`); the two `resize_layers` take the transposed-conv
  axis order `(1,2,3,0)` (the same load-bearing detail as the V2 port). `depth()` applies the pipeline's
  ImageNet normalization. `+register` under `depth-anything-3-small`. The oracle is reproducible in-repo
  (`run_reference.py depth3`, the `da3` oracle env, `IK_REF_SRC` = the unpacked `depth_anything_3`
  wheel), recording the input, the four hooks, the four head stages, the fused map, the pre-exp logits,
  and the depth; the parity test compares every seam by cosine plus mean-removed correlation (raw cosine
  on the near-constant ~1.0 depth is misleading), and a coverage test asserts every released tensor is
  loaded or named as dropped. Weights: `depth-anything/DA3-SMALL` (Apache-2.0, ~80M). Base and Large are
  also at reference parity (`NFKMLXDepth3Configuration.base` / `.large`, `NFKMLXDepth3Variant`,
  registered as `depth-anything-3-base` / `-large`): Base is the same recipe at ViT-B (768 wide, 12
  heads, DPT features 128); Large is ViT-L (1024 wide, 24 blocks, 16 heads) hooked at blocks 11/15/19/23
  with the query/key norms starting at block 8 rather than 4, and DPT features 256. Base: hooks
  ≥ 0.99999999999436, depth mean-removed 0.99999999998975; Large: hooks ≥ 0.99999999999713, mean-removed
  0.99999999998452. `DA3-LARGE` is CC-by-NC.
- `NFKMLXU2Net` (`@objc`) — a real single-forward background remover: the U²-Net nested-U saliency
  network (Residual U-blocks) in `MLXNN`, run through `NFKMLXMattingBackend` (plate → straight
  foreground + saliency alpha, matte under `NFKOutputMask`). `+register` adds full `u2net` and light
  `u2netp`. Stage/side/`outconv` names match the reference; the RSU-internal convs are `enc`/`dec`
  arrays, and `Tools/u2net-to-safetensors/convert.py` renames `rebnconvN` → `enc`/`dec` so the file
  loads directly. Forward + matting round-trip tested under xcodebuild with the light config.
  Reference parity against U²-Net's own network on both releases (the full network 0.9992, `u2netp`
  0.9998); the light model is a separate class in the reference rather than a configuration of the full
  one.
- `NFKMLXNAFNet` (`@objc`) — a real single-forward restoration network (denoise / deblur): a U-shaped
  stack of NAFBlocks (SimpleGate + Simplified Channel Attention, channel LayerNorm = last-axis in NHWC,
  PixelShuffle up) in `MLXNN`, run through `NFKMLXModuleBackend` (image → restored image at input size,
  padding to `2^levels` and cropping back). `+register` under `nafnet`; `NFKMLXNAFNetConfiguration` sets
  width and block counts (default SIDD width 32). Block names (`conv1`…`conv5`, `norm1/2`, `beta`,
  `gamma`) match the reference; `Tools/nafnet-to-safetensors/convert.py` renames `middle_blks`/`ups.N.0`/
  `sca.1` so a real checkpoint loads directly. Reference parity against megvii-research's own NAFNet on
  the released SIDD width-32 denoiser (cosine 0.9999972, mean |difference| 0.00098 through the backend's
  8-bit bridge), every parameter covered on the first triage run. The weights are not Drive-only:
  `huggingface.co/nyanko7/nafnet-models` mirrors all five releases. The GoPro deblurrer is also at
  parity (0.9999973), a block-layout change rather than a width one: it puts twenty-eight of its blocks
  in the last encoder stage and one in the middle, where SIDD spreads `[2, 2, 4, 8]` with twelve, so a
  checkpoint fits only the geometry it was trained as. The REDS release is that same distribution at
  twice the width (0.9999971), which separates a wrong width from a wrong block layout. The two width-64
  releases are also at parity: `siddWidth64` (SIDD's distribution at width 64, 0.99999986) and
  `goProWidth64` (the REDS geometry trained on GoPro, 0.9999999966). The GoPro width-64 record must be
  made from the photographic plate: on the synthetic plate the reference itself diverges (output range
  −111 to 110), so a record made from it reproduces exactly while measuring nothing (the first run read
  0.527 against that record). `NFKMLXNAFNetVariant` (`.sidd`/`.goPro`/`.reds`/`.siddWidth64`/`.goProWidth64`)
  selects the geometry from Objective-C.
- `NFKMLXRIFE` (`@objc`) — real frame interpolation: the released HDv3 IFNet in `MLXNN`, run through
  `NFKMLXTensorBackend` (two frames under keys `frame0`/`frame1` → the middle frame under
  `NFKOutputImage`). Three identical IFBlocks (11 input channels, width 90) run coarse-to-fine at scales
  4/2/1. Each block's trunk is four groups of two convolutions, each group added back to its own input
  (not one residual over eight), and flow and mask leave through separate heads (`conv1`/`conv2`), each
  two transposed convolutions undoing `conv0`'s ×4 stride. The net applies every block twice per scale:
  once as given and once with the frames swapped, the mask negated, and the flow halves exchanged,
  averaging the two, because the network is trained symmetric in its inputs. The bilinear backward warp
  is `grid_sample(align_corners=True, padding_mode='border')` built from `take` gather (MLX has no
  grid_sample); the per-scale resampling is bilinear, as the reference interpolates. `+register` under
  `rife`. `remapReferenceKey` strips the training wrapper's `module.` prefix, maps `blockN` → `blocks.N`,
  and names the `Sequential` entries of `conv0`/`convblock0…3` (convolution, PReLU) and the heads
  (`up1`, `prelu`, `up2`); the checkpoint's `block_tea` teacher is ignored as an extra key. Reference
  parity against the released HDv3 `flownet.pkl` (interpolated cosine 0.9999999999993, mean |difference|
  2.7e-7). Warp, interpolate, remap, and round-trip tested. Weights: `huggingface.co/yow46228/RIFE`
  ships the checkpoint together with its own `IFNet_HDv3.py`, which pins the architecture.
- `NFKMLXRIFEv4` (`@objc`) — the third IFNet generation, a separate architecture from HDv3. Reference
  parity on the released `rife-flownet-4.13.2` weights (interpolated cosine 0.9999999999995, mean
  |difference| 2.3e-7). Four blocks instead of three (widths 192/128/96/64), a learned frame `encode`
  module (`Head`: three convolutions and a transposed one to eight feature channels) whose features are
  warped alongside the frames, `ResConv` trunk entries that scale their convolution by a learned
  per-channel `beta` before the residual add, an upsampling convolution emitting `4 × 6` channels that
  pixel-shuffles ×2, and a timestep channel that is what v4 adds: `interpolate(_:_:timestep:)` lands
  anywhere between the frames, not only the midpoint. Its convolutions activate with a parameter-free
  leaky ReLU where HDv3 used a PReLU; that difference surfaced as exactly eight uncovered parameters,
  the coverage guard naming a structural mistake rather than a number going quietly wrong. `+register`
  under `rife-v4`; pads to a multiple of 64 and runs scales `[8, 4, 2, 1]`. Oracle: the architecture
  vendored by ComfyUI-Frame-Interpolation (`rife_arch.py`, `arch_ver="4.17"`), whose own `IFNet.py`
  ships inside the model zip rather than in the repository; only one ComfyUI device helper needs
  stubbing.
- `NFKMLXRAFT` (`@objc`) — real optical flow: the RAFT pipeline in `MLXNN` (shared feature encoder,
  all-pairs correlation volume + pyramid + bilinear lookup via `take` gather, context encoder, an
  iterative ConvGRU update). Run through `NFKMLXTensorBackend` (two frames `frame0`/`frame1` → a packed
  flow map under `NFKOutputImage`; raw `[H,W,2]` flow via `NFKMLXRAFTNet.flow`, the eighth-resolution
  field via `flowLow`). `+register` under `raft`. Faithful to RAFT-large (feature 256, 4 levels, radius
  4), including the convex-mask upsampling: each output pixel is a combination of its coarse 3×3
  neighborhood weighted by the mask the last update predicts (scaled ×0.25 as the reference does), over
  a zero-padded unfold. The default iteration count is low (6). Normalization is per encoder, as the
  reference `norm_fn` is: `fnet` uses a parameter-free InstanceNorm (`affine: false`, so the checkpoint
  carries nothing for it) and `cnet` uses BatchNorm, with `makeNet` setting eval mode for the running
  statistics. `flow` takes images in `0...1` and rescales to the trained `-1...1`; the correlation
  neighborhood is emitted in the reference's plane order (outer index shifts x, inner shifts y), which
  the trained 1×1 `convc1` depends on. The correlation lookup samples like the reference's
  `grid_sample(padding_mode: "zeros")`: a corner outside the map contributes nothing, where an edge
  clamp costs real accuracy (the lookup radius is 4 and the coarsest pyramid level is a few cells wide,
  so most of that neighborhood is outside). Converter `Tools/raft-to-safetensors/convert.py` renames
  `update_block`/`downsample`/`flow_head`/`mask`. Reference parity against princeton-vl's own RAFT on
  raft-things (eighth-resolution flow cosine 0.9999999999989, full-resolution 0.9999999999998); both
  sides run the same iteration count. Flow + round-trip tested under xcodebuild.
  `NFKRAFTCorrelation` carries the package's first custom Metal kernel, written as a source string
  through `MLXFast.metalKernel` (no `.metal` file, nothing for a consumer's build to link, compiled and
  cached by MLX on first use). The elementwise path walks 81 planes per level doing four gathers each,
  so one lookup was over thirteen hundred dispatches, and the update runs it once per GRU iteration; the
  kernel does the same arithmetic with one thread per (pixel, plane). Measured at RAFT's own
  eighth-resolution geometry, 2753 ms against 3.98 ms at 60×80, about 730×. `gatherLookup` stays as the
  CPU-stream path, since a Metal kernel cannot dispatch there and the package lets a caller select the
  CPU; the kernel is held to it bit for bit, including the zero-padding at the edges where most of the
  neighborhood lies. A silent fallback to the gathers would fail nothing and make the model unusable, so
  `testTheDispatchChoosesTheFusedPathOnTheGPU` compares the dispatched result to the fused one exactly
  rather than timing anything.
- `NFKMLXSAM` (`@objc`) — real promptable segmentation (Segment Anything): a ViT image encoder, a prompt
  encoder (point → sparse tokens via a random-Fourier positional encoding), and a two-way-transformer
  mask decoder with a hypernetwork mask head, in `MLXNN`. Run through `NFKMLXMattingBackend` (plate +
  point under `NFKSAMPointKey` → mask alpha + matte). `+register` under `sam`. The ViT encoder uses real
  windowed attention (`windowSize`, `globalAttnIndexes`) with decomposed relative-position embeddings
  (`rel_pos_h`/`rel_pos_w`, added via `take` gather + batched matmul). `remapReferenceKey` maps the
  reference's nested MLP, positional neck/upscaling Sequentials, and `transformer` submodule; scope the
  `.mlp.lin` rule to the encoder, or it eats the decoder's. `NFKMLXSAMVariant`
  (`.compact`/`.vitB`/`.vitL`/`.vitH`) selects the geometry on both the local and the download
  factories; a released checkpoint fits only its own size. Reference parity against the official
  `segment-anything` predictor on ViT-B (encoder cosine 0.9999986, selected-mask cosine 0.99993, binary
  agreement 99.7%); the decisive fix was `skip_first_layer_pe` in the two-way transformer's first layer.
  ViT-L and ViT-H are also at parity (`NFKMLXSAMConfiguration.vitL`: 1024 wide, 24 blocks, 16 heads,
  global attention at 5/11/17/23; `.vitH`: 1280 wide, 32 blocks, global at 7/15/23/31): encoder
  0.99999799 / 0.99999669, binary mask agreement 1.0 on both. SAM 2 is `NFKMLXSAM2` below.
  Segment + round-trip tested under xcodebuild.
  `NFKMLXSAM2` ports SAM 2's Hiera image encoder, at reference parity against facebookresearch's own
  sources (finest FPN level 0.9999999999986, second 0.9999999999918, vision features 0.9999999999939),
  every parameter covered on the first triage run. Hiera is hierarchical where SAM's ViT is flat: four
  stages that halve the resolution and double the width, attention inside local windows except at
  designated global blocks, and stage transitions that max-pool the queries so one block changes both
  size and width (the shortcut takes the same projection and pooling so the residual still lines up). A
  transition block keeps the previous stage's window (the reference reads `window_spec[cur_stage - 1]`
  before advancing the stage), which is what makes block 10 use window 14 rather than 7; reversing that
  order makes the pooled windows the wrong size and the reassembly stops tiling. The position grid is
  resampled bicubically (`NFKMLXBicubic`, PyTorch's `a = -0.75` Keys kernel with half-pixel centers and
  border clamping) and a tiled window grid is added. The FPN neck projects each captured stage to 256
  and fuses top-down on the deeper levels only, dropping the coarsest (`scalp`).
  The prompt encoder and mask decoder are ported too, at reference parity (sparse prompt
  0.99999999999998, mask logits 0.9999999999950, object score matching to five figures), every
  parameter covered on the first triage. The decoder is SAM's two-way transformer (attention, block, and
  MLP shared with `NFKMLXSAM`) plus three SAM 2 additions: an object-score token leading the sequence
  with its own head, and high-resolution features from the FPN's two finer levels added during
  upscaling. Their `conv_s0`/`conv_s1` projections are the decoder's parameters but the reference
  applies them in its base model before calling it, so this port applies them internally and takes the
  levels as they come off the neck. Two traps: the query positional term is the original token embedding
  at every layer and again at the final attention (the running queries drift the masks without breaking
  anything visibly), and the reference shifts a click by half a pixel to the pixel's centre before
  normalizing.
  The video memory path is ported too, which completes the checkpoint: `NFKMLXSAM2MemoryEncoderNet`
  folds a frame's features and its predicted mask into a 64-channel memory, and
  `NFKMLXSAM2MemoryAttentionNet` conditions the next frame on that memory. Both are at reference parity
  (encoded memory cosine 0.9999999999999, attention output 0.9999999999997), every parameter covered on
  the first triage run. The encoder's mask downsampler is four stride-2 stages
  (1 → 4 → 16 → 64 → 256 channels), so it takes the mask at the full 1024 frame resolution, not at the
  decoder's low-resolution output; the tracker upsamples before encoding, and the total stride of 16
  lands it on the 64×64 feature grid. Its fuser blocks are ConvNeXt (7×7 depthwise, channel LayerNorm, a
  4× pointwise MLP, and a learned per-channel `gamma` scale). The attention applies axial rotary
  embeddings (`NFKMLXAxialRotary`): adjacent channel pairs are rotated, the first half of the pairs by
  an x-frequency and the second half by a y-frequency, matching the reference's `view_as_complex`
  layout. Its positional-encoding switches are asymmetric and each one matters: self-attention and
  cross-attention queries take no positional term, cross-attention keys do, and the input takes
  `0.1 × position` once. The trap is the reference's `batch_first=True`, which describes what its layers
  want: `MemoryAttention` takes its inputs sequence-first and transposes them itself, so handing it
  batch-first tensors makes the tokens the batch and every token attends only to itself (that scored
  0.86 and raised nothing). Checkpoint
  `dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt` (161 MB, 468 tensors, 39M
  parameters), of which 122 are the video path (`memory_attention`, `memory_encoder`); an image-only
  port needs the other ~250: `image_encoder` (154 trunk + 8 neck), `sam_prompt_encoder` (10), and
  `sam_mask_decoder` (~118). The `sam2` package cannot be installed here (it requires Python ≥ 3.10; this
  environment is 3.9), and the installed `transformers` is 4.33.3, which has no SAM 2, so the oracle
  vendors `backbones/{hieradet,image_encoder,utils}.py`, `position_encoding.py`, `sam2_utils.py`,
  `memory_attention.py`, `memory_encoder.py`, and `utils/misc.py`, all of which parse under 3.9, with
  `iopath` and `sam2.utils.misc` stubbed. The tiny config (embed 96, one head, stages `(1, 2, 7, 2)`,
  global attention at blocks 5/7/9, background window 7×7; neck `d_model` 256 over channels
  `[768, 384, 192, 96]`, top-down levels `[2, 3]`, nearest interpolation, `scalp` 1) loads strictly, and
  a 1024×1024 forward returns `vision_features [1, 256, 64, 64]` with FPN levels at 256/128/64.
  All four released Hiera sizes are at parity: small (`.small`: stages `[1, 2, 11, 2]`, windows
  `[8, 4, 14, 7]`, global attention at 7/10/13; level0 0.99999999999956, level1 0.99999999999791), large
  (level0 0.9999999999994, level1 0.9999999999986; 48 blocks against tiny's 12, weighted toward the
  third stage, a coarser window there, and global attention much later, its config overriding every axis
  the Hiera constructor defaults), and base_plus (level0 0.9999999999958, level1 0.9999999999895), which
  sets only the width and head count and takes the rest from the defaults, so it is the size that proves
  the defaults rather than the overrides. The oracle's `sam2` package is now in the manifest; it cannot
  be pip-installed here and had been vendored by hand, leaving that oracle unreproducible.
- `NFKMLXLaMa` (`@objc`) — a real single-forward inpainter: the LaMa FFC-ResNet generator in `MLXNN`
  (each Fast Fourier Convolution runs a spatial branch and an FFT spectral branch via `MLXFFT`
  `rfft2`/`irfft2`, orthogonally normalized), run through an `NFKMLXMattingBackend` (plate under
  `NFKInputImage`, mask under `NFKInputMask`). `+register` under `lama-inpaint`. The configuration
  defaults are big-lama's own `config.yaml`: 64 base channels, three downsampling stages, 18 residual
  blocks, ratio 0.75 through the trunk and at the last downsample only, sigmoid output.
  `remapReferenceKey` translates the checkpoint's flat `model.N` Sequential (the parameter-free entries
  — reflection pads, the tuple concatenation, the activations — still consume an index, so the
  upsampling triples start at 24), plus the spectral branch's `conv1.0`/`conv1.1` narrowing. The
  upsampling transposed convolutions carry `outputPadding` 1 (without it they land a pixel short of
  doubling, which an output resize can only paper over) and load with `transposed(1, 2, 3, 0)`, not the
  forward convolutions' axis order. Reference parity against advimman's own FFCResNetGenerator on the
  released big-lama (inpainted cosine 0.9999999999997, mean |difference| 5.9e-8). The convolutions
  reflection-pad, as the reference's `padding_mode='reflect'` does; edge padding instead scores 0.99857
  / mean 0.00503, so the approximation this model shipped with was a real defect, measured both ways.
- `NFKMLXSDUNet` / `NFKMLXSDAutoencoder` (`NFKMLXStableDiffusionModels.swift`) — the real
  `UNet2DConditionModel` and `AutoencoderKL` in `MLXNN`, in the diffusers layout. One implementation
  serves every latent-diffusion model here, sharing one structure and differing only in the scalars a
  configuration carries (channel widths, which levels attend, the cross-attention width,
  convolution-versus-linear transformer projections, a class embedding, how many transformer blocks an
  attention runs, whether a pooled embedding joins the timestep). Reference parity against diffusers on
  every released configuration: SD-1.5-inpainting UNet 0.9999999999993, Marigold UNet 0.99999999999,
  ×4-upscaler UNet 0.9999999999973, SD 2.1 UNet 0.9999999993, SDXL UNet 0.9999999999955, SD autoencoder
  0.99999999982 latent / 0.99999999917 decoded, upscaler autoencoder 0.99999999998 / 0.99999999992.
  Every one covered on the first triage run.
  SDXL adds two axes and no new structure: `transformerLayers` (`transformer_layers_per_block`,
  `[1, 2, 10]` — the coarsest level runs ten transformer blocks where every earlier release runs one)
  and `additionEmbedding` (`addition_embed_type: "text_time"` — six `time_ids`, the original size, the
  crop's top-left corner, and the target size, each embedded at 256 and run together into 1536, joined
  by the second tower's 1280-wide pooled embedding and projected to the timestep's width).
  Three details are load-bearing. `only_cross_attention` (the ×4 upscaler sets it on three levels)
  makes a block's first attention a cross-attention too, so its keys and values take the context's
  width; nothing in the checkpoint's key names says so, only the tensor shapes, and MLX adopts a
  checkpoint's shapes wholesale, so getting it wrong loads cleanly and fails later. The transformer's
  input normalization uses epsilon 1e-6 where every resnet uses 1e-5. The autoencoder's downsampling
  convolution pads asymmetrically (right and bottom only), where the UNet's pads evenly. The SD 1.5
  autoencoders also predate the diffusers attention rename, so `remapVAEKey` accepts
  `query`/`key`/`value`/`proj_attn` as well as `to_q`/`to_k`/`to_v`/`to_out.0`.
  Oracle: diffusers cannot be installed beside the other oracles here (it needs a newer transformers
  than the 4.33.3 the Whisper / CLIP / SegFormer records were measured against), so it lives in its own
  virtual environment and every `sd_*` mode of `run_reference.py` runs under that interpreter. The
  manifest's `oracle_environments` records where it is and how to rebuild it. Weights are the released
  diffusers layout, one checkpoint per network.
- `NFKMLXSDPipeline` — a UNet and an autoencoder together, plus the text conditioning the released
  checkpoints cross-attend to. The tower is not on this path: the caller brings the embedding
  (`loadTextContext(from:)` reads a `[tokens, dimensions]` tensor), which is what the image-conditioned
  models here do (a model that takes no prompt still expects the embedding of an empty one).
  `NFKMLXTextToImage` is the path that owns a tower, because a prompt is its input.
  `loadWeights(unetURL:vaeURL:)` reads the released layout; `loadWeights(from:)` reads a single file
  holding both under `unet.`/`vae.`, which is what `NFKMLXWeights.save` writes, so a fine-tuned
  pipeline reloads through one path.
- `NFKMLXStableDiffusionInpaint` (`@objc`) — Stable Diffusion inpainting on `NFKMLXDiffusionBackend`,
  over those real networks. `encode` VAE-encodes the plate and the masked plate and builds the
  nine-channel conditioning (noisy latent, then mask, then masked-image latent, in that order);
  `denoise` is the UNet (epsilon); `decode` is the VAE. The backend runs the DDIM loop and the per-step
  inpaint compositing. The masked-image latent is the plate with the hole blanked and then encoded, not
  the plate's latent with the hole blanked; the encoder is not local, so the two differ. `+register`
  under `sd-inpaint`. Pipeline and offline round-trip tested under xcodebuild.
- `NFKMLXTextToImage` (`@objc`) — Stable Diffusion **text-to-image**, on `NFKMLXDiffusionBackend`, and
  what `NFKMLXBackend` is built from. `encode` turns the prompt into a conditioning sequence, `denoise`
  is the UNet with classifier-free guidance, `decode` is the autoencoder; the backend owns the loop.
  Above a guidance of 1 the conditional and unconditional predictions run as one batch of two, which
  makes guidance one forward pass rather than two, ordered unconditional first as the reference reads it
  back. `NFKMLXSDTextToImageConfiguration` carries the three releases
  (`.stableDiffusion15`, `.stableDiffusion21` / `.stableDiffusion21V`, `.sdxlTurbo`), and
  `NFKMLXSDReleaseFiles(directoryURL:)` resolves a downloaded release's tree (`unet/`, `vae/`,
  `text_encoder/`, `tokenizer/`, and for SDXL `text_encoder_2/` and `tokenizer_2/`), accepting the
  `.fp16.safetensors` spelling a half-precision-only release uses.
  Reference parity end to end against diffusers' own pipelines, starting from the reference's own
  initial latent (matching a random source across two implementations proves nothing about either):
  SD 1.5 image cosine 0.999998948, SD 2.1 (v-prediction) 0.9999986, SDXL-Turbo 0.9999973, SDXL with
  guidance 0.9999990, SDXL with no negative prompt 0.9999988. The record also carries the per-step
  latents and the first guided prediction, so a whole-picture mismatch says which stage diverged.
  The sampler is DDIM in every case, including SDXL-Turbo, whose release names
  `EulerAncestralDiscreteScheduler`; both sides run DDIM at the release's own `timestep_spacing`, so
  the comparison measures this port rather than two different samplers. A caller wanting the released
  sampler exactly brings one through `NFKDiffusionScheduler`. Three details are load-bearing, and each
  one first showed up as a wrong picture. Stable Diffusion 2.x pads a prompt with `!` (id 0), not the
  end marker, and `special_tokens_map.json` overrides `tokenizer_config.json`, which is where that is
  written, so reading only the config pads with 75 end markers and the model reads a different sentence
  (that scored 0.825). A bfloat16 release turns a float32 module into a bfloat16 one, because MLX's
  `update(parameters:)` adopts a checkpoint's element type along with its values; the SD 2.1 text tower
  and autoencoder are published that way, and it cost three orders of magnitude (0.9999956 against
  0.9999999999841). `NFKMLXWeightPrecision` makes that a choice rather than an accident.
  `force_zeros_for_empty_prompt` acts on an absent negative prompt, not an empty one: the reference's
  condition is `negative_prompt is None`, and an empty string is a sentence the model is asked to encode.
- `NFKMLXSDTextEncoderNet` / `NFKMLXSDTextEncoder` — the CLIP text tower the releases cross-attend to,
  built from `NFKMLXCLIP`'s own blocks (one implementation, four configurations). `NFKSDTextOutput`
  spells the difference between the releases: SD 1.x and 2.x read the last hidden state after the final
  layer normalization (2.x drops the tower's 24th layer in its own configuration rather than skipping it
  here), while SDXL reads the penultimate one, before that normalization. Only SDXL's second tower
  carries a projection, and it is the pooled embedding SDXL's UNet conditions on; the pooled path runs
  the whole stack through the final normalization even when the sequence stops a layer short, and reads
  the position of the highest token id (the reference's rule, which the padding repeats, so the first
  occurrence is the one). The releases are a `transformers` CLIPTextModel, whose attention stores
  separate `q_proj`/`k_proj`/`v_proj` where the module keeps the reference's fused projection, so the
  remap concatenates three tensors into one, which a 1:1 key map cannot express, as SegFormer's `kv` is
  a two-into-one. The activation is `quick_gelu` for SD 1.x and the exact error-function GELU for the
  OpenCLIP towers, not the tanh approximation its neighbour `gelu_new` selects. Reference parity on
  every released tower: SD 1.5 0.9999999999986,
  SD 2.1 0.9999999999841, SDXL primary 0.9999999999986, SDXL secondary 0.9999999999257 (pooled
  0.9999999999820).
- `NFKMLXSDPromptTokenizer` — the release's `tokenizer/` directory driving the core's `NFKTokenizer`
  CLIP variant, plus the padding to the tower's context length. Token-for-token agreement with
  `transformers`' CLIPTokenizer over five prompts, including punctuation, an empty prompt, a multi-byte
  one, and the markers written out literally.
- `NFKMLXWhisper` / `NFKMLXWhisperBackend` (`@objc`) — real on-device speech-to-text: the Whisper
  encoder-decoder transformer in `MLXNN` (log-mel via MLXFFT `rfft` → audio encoder → greedy text
  decoder). Audio → text backend: reads `NFKInputAudio` (an `NFKAudioAsset` WAV via `NFKMLXWaveFile.read`,
  or NSData), returns `NFKOutputText`. `+register` under `whisper-tiny`. Module names follow OpenAI
  Whisper (`encoder`/`decoder`, `blocks.N`, `attn`/`cross_attn`, `mlp.0`/`mlp.2`), so a converted
  checkpoint loads with the conv transpose (`loadWeights` handles 4-D and 3-D Conv1d). `NFKTokenizer`
  (optional) detokenizes; else token ids. `Tools/whisper-to-safetensors/convert.py` targets the OpenAI
  `.pt`. Reference parity against openai-whisper itself (log-mel cosine 0.9999999999940,
  first-step decoder logit cosine 0.9999999997432, and an exact greedy token match).
  The decoder's suppression rules are the reference's, measured rather than approximated:
  `suppressTokens` masks the curated non-speech set at every step, and `suppressesBlankStart` masks a
  space and an immediate `<|endoftext|>` at the first sampled position only, which is `SuppressBlank`.
  `NFKMLXWhisperSuppression.nonSpeechTokens(using:)` computes that set from the model's own tokenizer
  by the reference's rule — a symbol contributes its first token when it encodes to exactly one token,
  a musical symbol contributes its first token however many it encodes to — because the ids differ
  between the English-only and the multilingual vocabulary. `backendWithWeightsURL:tokenizer:` wires it.
  The record carries both decodings: `output` under the plain special/timestamp mask, which isolates
  the network, and `ruled_tokens` under the reference's own policy, which the port reproduces exactly.
  They differ on the synthetic clip (five tokens against three), which is why the policy needed
  measuring rather than describing. Timestamped decoding is implemented and at an exact token
  match against the reference (`transcribeWithTimestamps`, and `emitsTimestamps` on the backend,
  which adds `NSArray<NFKAudioSegment *>` under `NFKOutputSegments`). It is a different decode rather
  than a different reading of one: the times only exist when `<|notimestamps|>` is left out of the
  prompt and the timestamp range stays unmasked, so the model is asked a different question and may
  answer it with different words — which is why it is off by default. `ApplyTimestampRules` then
  orders the result: a timestamp is followed by text and text by a timestamp, so they come in pairs;
  a timestamp never precedes an earlier one, and the `+ 1` in the reference's bound is what forbids an
  empty segment; the opening position must be a timestamp no later than `max_initial_timestamp`
  (one second, 50 ids); and where the timestamps together hold more probability than any single word,
  a timestamp is taken even though no single one leads. On the synthetic clip the reference and the
  port both emit `<|0.00|>` "Thank you." `<|3.00|>` — and the tone in that clip does stop at 3.0
  seconds, so the span is a real measurement rather than a shape check. `timestampBegin` is the id of
  `<|0.00|>`, one past `<|notimestamps|>`, so large-v3's extra language token shifts both together.
  The mel question an early note left open is settled by measurement, not assumption: `melFilters` is
  the Slaney scale with Slaney area normalization — librosa `mel(htk=False, norm='slaney')`, which is
  what OpenAI's precomputed filters hold — and the parity record scores the port's log-mel at
  0.999999999994 against the reference's own. HF-format checkpoints load too: `loadWeights`
  detects the transformers naming (`model.encoder.layers.N.self_attn.q_proj`) and remaps it onto the
  OpenAI layout, asserted by renaming a real checkpoint into HF form and getting the identical
  transcription. `small`, `medium`, and `large-v3` are ported too, each at an exact token match
  against the reference (mel cosine 0.99999999999). Every size shares one encoder-decoder structure and
  differs in a width, a head count, and a depth — except large-v3, which also produces 128 mel bands
  instead of 80 and carries one more language token, shifting `<|transcribe|>` and `<|notimestamps|>`
  up by one. Both come from the model's own tokenizer rather than the smallest size's constants; the
  parity record carries the prompt the reference used, so a shifted id surfaces as a prompt mismatch
  rather than a mysterious token difference. `base`, `large` (large-v1 and large-v2 share one
  geometry: 1280 wide, 20 heads, 32 layers, 80 mels, vocabulary 51865), and `largeV3Turbo` (large-v3's
  128-mel encoder over a four-layer decoder) complete the released sizes, each at an exact greedy token
  match; `NFKMLXWhisperVariant` carries all of them. The `.en` releases share the multilingual geometry
  and differ only in their tokenizer files.
- `NFKMLXParakeet` / `NFKMLXParakeetNet` / `NFKMLXParakeetBackend` (`@objc`) — **Parakeet-TDT 0.6B v2**
  (NVIDIA NeMo, CC-by-4.0), a second on-device speech recognizer beside Whisper and the fast one: a
  **FastConformer** encoder — the NeMo mel front end (128 mels, 25 ms / 10 ms, a 512-point transform,
  `log(x + 2^-24)`, **per-feature normalization** by each band's unbiased standard deviation over time
  plus 1e-5), a **depthwise-striding 8× subsampler** (a full 3×3 stride-2 conv, then two depthwise
  3×3 stride-2 + pointwise 1×1 pairs over the `(time, mel)` plane, flattening `(channel, mel)` into a
  4096→1024 linear), and 24 relative-position (Transformer-XL) conformer layers (half-weighted
  macaron feed-forwards, rel-pos attention with per-layer `pos_bias_u`/`pos_bias_v` and the appendix-B
  shift, a GLU → depthwise-9 → BatchNorm → Swish convolution module, **no biases** on any projection or
  convolution) — and a **token-and-duration transducer**: a two-layer LSTM prediction network (640) over a
  blank-as-pad embedding, a joint (`enc` 1024→640 + `pred` 640→640, ReLU, a linear to 1024 tokens + blank
  + 5 duration classes), and greedy TDT decoding — at each encoder frame the joint scores the next token
  And how many frames to skip (`[0, 1, 2, 3, 4]`), a non-blank advances the LSTM state, and a zero
  duration keeps decoding the same frame (at most 10 symbols). `NFKMLXParakeetBackend` reads
  `NFKInputAudio` (resampled to 16 kHz) → the transcript under `NFKOutputText` plus one
  `NFKAudioSegment` per token under `NFKOutputSegments` (an encoder frame is 80 ms, so every token
  carries its onset). The release is an unpacked `.nemo` tar (`model_weights.ckpt` through the native
  torch reader; the SentencePiece **BPE** `*_tokenizer.vocab` piece table — recognition only decodes,
  so the pieces alone reproduce SentencePiece's `decode`: concatenate, `▁` → space, drop the leading
  space); `@objc backendWithDirectoryURL:error:`.
  Reference parity on the released weights, on the real validation clip, seam by seam against
  NeMo's own EncDecRNNTBPEModel (`run_reference.py parakeet --checkpoint <the .nemo>`,
  `IK_PARITY_PARAKEET` + `IK_VAL_PARAKEET`, the new `nemo` oracle env): features 0.9999999996, the
  subsampler 0.99999999989, the first layer 0.99999999975, the encoder 0.99999999999, the joint
  0.99999999999997, and the greedy TDT decode reproducing the reference's 19 tokens and their frame
  timestamps exactly — transcription "The quick brown fox jumps over the lazy dog." — through the
  public backend too (`testParakeetBackendTranscribesTheValidationClip`).
  Three facts are load-bearing, all found by the parity run. NeMo's valid frame count is
  `floor((samples + n_fft − n_fft) / hop)` — without the `+ 1` the transform's own frame count
  carries — so the transform's last frame lies past the valid length: it is zeroed (`pad_value`),
  excluded from the normalization statistics, and masked through the encoder; cropping to the valid
  length is exactly that through the stride-2 subsampler (with it, features went from 0.9975 to
  0.9999999996 and every downstream seam became exact). The blank / start state of the prediction net
  feeds the LSTM a zero vector in place of an embedding (`blank_as_pad`) and the LSTM still runs — its
  output and state are what the joint and the next step read; returning raw zeros scored the frame-0
  joint at −0.52 and dropped the first word. And NeMo's `joint()` log-softmaxes on the CPU when
  `log_softmax` is null — a constant shift over the whole 1030-vector that leaves both argmaxes alone,
  so the port keeps raw logits and the seam is compared in log-softmax space. The STFT pads with
  **zeros** (`pad_mode="constant"`), where the older MarbleNet VAD front end here reflects — measured
  both ways, reflect scores 0.974. The LSTM loads through the shared PyTorch→MLX fold (`weight_ih_l<n>`
  / `hh` → `Wx` / `Wh`, biases summed) under `dec_rnn.lstm.<n>`; the `nn.Sequential` indices of the
  subsampler (ReLU at 1, 4, 7) and the joint (ReLU 0, Dropout 1, Linear 2) are kept with marker modules
  so every other key matches with no remap; the 4-D and 3-D convolutions transpose to channels-last.
  Weights: `nvidia/parakeet-tdt-0.6b-v2` (2.47 GB `.nemo`; v3 and the CTC release are the same encoder).
- `NFKMLXChatterbox` / `NFKMLXChatterboxTTS` (`@objc` factory) — **Chatterbox** (Resemble AI, MIT), zero-shot
  VOICE-CLONING text-to-speech, the third TTS beside FastSpeech2 and Kokoro and the first that takes a
  reference voice. Five networks, ported stage by stage with a parity record gating each
  (`run_reference.py chatterbox_voice` / `chatterbox_t3` / `chatterbox_s3gen`, the `chatterbox` oracle env,
  `IK_VAL_CHATTERBOX` = the release directory), all on the released weights and the validation clip:
  - **VoiceEncoder** (`NFKMLXChatterboxVoiceEncoderNet`, `ve.safetensors`): a 40-band unscaled power mel
    (no log; librosa Slaney to 8 kHz, every STFT frame kept) over the librosa-trimmed 16 kHz prompt
    (20 dB below the loudest 2048-sample frame, ported exactly), cut into 160-frame partials at step 77
    (`round((16000 / 1.3) / 160)`), a 3-layer LSTM → linear → ReLU → L2 per partial, mean → L2. Trim
    exact, mel 0.99999999999984, partials 0.999999999999, embedding 0.9999999999995.
  - **S3 speech tokenizer** (`NFKMLXS3TokenizerNet`, the `tokenizer.` subtree of `s3gen.safetensors`):
    Whisper's 128-band log-mel exactly (`NFKMLXMel.logMel` is reused), two stride-2 GELU convolutions
    (100 Hz → 25 Hz), six FSMN-attention blocks at 1280 over 20 heads (a depthwise 31-tap memory over
    the values added to the attention output; rotate-half rotary at 64, base 10000; the reference's
    `headDim^-0.25` on q and k is one `1/√headDim`), and an 8-channel base-3 FSQ (`round(tanh · 0.999) + 1`
    read as a base-3 number, 6561 codes). Mel 0.99999999998, states 0.9999999999, codes **87/87 exact** on
    both the six-second crop and the whole prompt.
  - **T3** (`NFKMLXT3Net`, `t3_cfg.safetensors`): the shipped dense decoder `NFKMLXLanguageNet` as a
    Llama 520M (30 layers, 16 heads × 64, no q/k norm, theta 500000) under **llama3 rope scaling**, which
    `NFKMLXRoPEScaling` now implements (factor 8, low/high frequency factors 1/4 over the 8192 window; at
    parity with transformers' `ROPE_INIT_FUNCTIONS` on two configurations, worst relative difference
    < 1e-5, added to the rope record). Conditioning is a 34-token prefix: `spkr_enc(speaker)`, a
    **Perceiver** (32 learned queries cross-attend to the prompt codes' `speech_emb + speech_pos_emb`, then
    self-attend once; 4 heads, one shared LayerNorm) and `emotion_adv_fc(exaggeration)`; then the text
    with learned positions and the start-of-speech token. Three reference quirks are reproduced, not
    fixed: `inference` feeds the start-of-speech embedding twice at speech position 0; the CFG
    unconditional row zeroes the text embedding but keeps the text positions; and generated code `i`
    takes speech position `i + 1`. The text tokenizer (`NFKMLXChatterboxTextTokenizer`) reads
    `tokenizer.json` directly — a plain character BPE with a `Whitespace` pre-tokenizer and literal added
    tokens, spaces replaced by `[SPACE]` first, plus `punc_norm` — token-exact against `tokenizers`.
    Sampling is the reference's chain (CFG 0.5 → repetition penalty 1.2 over every generated id → temperature
    0.8 → min-p 0.05 → top-p; `NFKMLXT3Sampler.processed`), pinned against transformers' own processors on
    the first step to 2e-5; temperature 0 is this port's greedy addition. Teacher-forced over the
    reference's own sampled sequence: cond 0.9999999999997, embeds 0.9999999999997, logits
    0.99999999999943 / 0.9999999999992 (both CFG rows), **argmax 79/79**. HF `hidden_states[-1]` Is the
    post-norm state (verified empirically on transformers 5.2, since T3 reads it as the head input).
  - **S3Gen token-to-mel** (`NFKMLXS3GenNet.flow`): the **CAMPPlus x-vector** over a ported Kaldi fbank
    (25/10 ms, 512-point, DC removed per frame, pre-emphasis 0.97 with replicate padding, the POVEY window
    `hann^0.85`, HTK mel 20 Hz–Nyquist, `log(max(x, ε))`, then the utterance mean removed) — an FCM 2-D
    head over the (frequency, time) plane whose stride hits frequency only, three CAM-dense TDNN blocks
    (12/24/16 layers, growth 32, a sigmoid gate from the utterance mean plus a 100-frame segment mean),
    statistics pooling (unbiased std), a 192-wide affine-free BatchNorm embedding; fbank 0.99999999995,
    x-vector 0.999999999999. The 24 kHz prompt mel (Matcha's: 1920/480, reflect pad 720, `sqrt(power +
    1e-9)`, log-clamp 1e-5) 0.9999999998. The **UpsampleConformerEncoder** (espnet `rel_pos` positions
    `T-1 … -(T-1)` with `√d` input scale, a 3-frame pre-lookahead convolution, six pre-norm rel-pos layers
    at epsilon 1e-12 with the appendix-B shift, nearest ×2 + left-padded 5-tap conv, four more layers, a
    final norm) 0.9999999999997; `mu` 0.9999999999998. The **CausalConditionalCFM**: ten Euler steps on
    `1 − cos(t·π/2)`, the estimator run as a batch of two (the unconditional row with zero mu, speaker, and
    prompt mel), `(1 + 0.7)·cond − 0.7·uncond`; the **ConditionalDecoder** (input `[x, mu, speaker, cond]` =
    320 channels; one down stage, twelve mid, one up, each a causal resnet block — left-padded 3-tap convs,
    LayerNorm over channels, Mish, the timestep MLP added between — plus four diffusers
    `BasicTransformerBlock`s with plain LayerNorms and a GELU FF; the released single level makes both
    resamples a causal 3-tap conv). First-step velocity 0.9999999999997, the solved mel 0.9999999999999.
  - **HiFT vocoder** (`NFKHiFTGeneratorNet`, `mel2wav.`): ConvRNNF0Predictor (five ELU convs, `|linear|`)
    0.99999999999997; the NSF harmonic source (nine harmonics, `2π·(cumsum(f0·k/sr) mod 1)`, voiced above
    **10 Hz**, `tanh(linear)`) **cosine 1.0** with the random phases and noise zeroed (the oracle zeroes
    them; the consumer path draws them); the generator — three transposed-conv upsamples (×8, ×5, ×3), the
    source's own 16-point STFT injected at each scale through strided convolutions, Snake residual blocks, a
    One-sample reflection pad before the last scale, the one bare `leaky_relu` at 0.01 before `conv_post`
    (the HiFi-GAN/Kokoro trap again), `exp` magnitude clipped at 100, `sin` phase, iSTFT at 16/4 through the
    shared `NFKKokoroSTFT` — waveform 0.99999999999, with the reference's 40 ms leading fade.
  End to end on the released weights: the validation clip as the voice, "The quick brown fox jumps
  over the lazy dog." → 3.38 s of 24 kHz speech that the package's own Parakeet (at parity) transcribes
  back exactly (`testChatterboxSynthesizesSpeechParakeetTranscribes`), and the release's built-in voice
  (`conds.pt`, read through the native torch reader: nested dicts flatten to `t3.` / `gen.` keys)
  speaks through the backend. `NFKMLXChatterboxTTS(directoryURL:)` loads all five; `conditionals(voice:
  sampleRate:)` is `prepare_conditionals` (resample to 24 kHz then to 16 kHz through
  `NFKMLXAudioRate.matched` — the reference's librosa/torchaudio resamplers are not bitwise, so the
  parity records take the recorded waveforms at each rate as inputs and the consumer path is a documented
  resampler approximation); `@objc chatterboxBackendWithDirectoryURL:voiceURL:error:` returns an
  `NFKMLXSpeechBackend` (24 kHz WAV under `NFKOutputAudio`; nil voice = `conds.pt`). **Two oracle
  traps**: `solve_euler` calls `estimator.forward` Directly, so a forward hook never fires (wrap the
  method); and the prompt mel's frame count can be odd (173) while the codes are trimmed to `mel // 2`
  (86), so the generated mel is `2n − 1` frames rather than `2n`. Weights: `ResembleAI/chatterbox`
  (`ve.safetensors` 7 MB, `t3_cfg.safetensors` 2.1 GB, `s3gen.safetensors` 1.1 GB, `tokenizer.json`,
  `conds.pt`). pyannote diarization stays blocked (gated `pyannote/segmentation-3.0`, no token here).
- `NFKMLXDemucs` / `NFKMLXDemucsBackend` (`@objc`) — real music stem separation: the time-domain Demucs
  U-Net in `MLXNN` (strided Conv1d + GLU encoder, transposed-conv decoder with skips, via `NFKMLXDemucsBackend`
  audio → four stems, each an `NFKAudioAsset` under its name "drums"/"bass"/"other"/"vocals"). `+register`
  under `demucs`. 1-D transposed conv is a `ConvTransposed2d` with a singleton width. Reference parity
  against the released Demucs v2 (per-stem-channel cosine 0.9999999995). `NFKMLXDemucsConfiguration`
  carries everything the two released families differ in, so one network serves both: the music model is
  stereo, six blocks deep, mixes decoder channels over a `context` of 3, runs a **bidirectional**
  bottleneck (`NFKDemucsBLSTM`: forward and reversed passes concatenated, then a linear projection), and
  resamples ×2 through `NFKDemucsFractionalResample` (the polyphase `julius.resample_frac`); the speech
  denoiser is mono, five deep, context 1, causal, and resamples ×4 through the half-sample-shift
  `NFKDemucsResample`. `validLength` follows the reference exactly (ceiling division, plus `context - 1`
  per encoder stage), skips are added with `center_trim`, and `centersOutput` selects the music model's
  centered result trim over the denoiser's head crop. `NFKMLXDemucs.loadWeights` is the shared reference
  loader for both. Parity, round-trip, and per-stem stereo WAV tested.
  Demucs v4 (htdemucs) is `NFKMLXHTDemucs`, a separate architecture — `NFKMLXDemucsNet` is the v2
  time-domain U-Net and no v4 checkpoint fits it.
- `NFKMLXHTDemucs` / `NFKMLXHTDemucsBackend` (`@objc`) — Demucs v4 (Hybrid Transformer Demucs), at
  reference parity on the released `htdemucs` checkpoint (separated stems cosine
  0.9999999999996, mean |difference| 7.7e-8), every parameter covered on the first triage run and
  every stage seam exact on the first numeric run. Two branches run **in parallel**: a spectrogram
  branch over a complex-as-channels STFT (`nFFT` 4096, hop 1024, `torch.stft(normalized:)`, so the
  real and imaginary parts of each audio channel are two feature channels) and a waveform branch over
  the samples. Each is a four-stage U-Net of `HEncLayer`/`HDecLayer` — a strided convolution, a
  dilated `DConv` residual branch (compress ×4, GroupNorm, GELU, expand, GLU, a learned `LayerScale`),
  and a gated rewrite. They never merge by injection: every `tencoder` here is non-empty, so the
  only path between the branches is the **cross-transformer** at the bottleneck — five layers per
  branch alternating self-attention and cross-attention, pre-norm with two `LayerScale` factors and a
  `MyGroupNorm` over the whole sequence, reached through 1×1 channel samplers that widen 384 to 512
  and narrow back. The reconstructions are added. Adds the `MLXFast` product for
  `scaledDotProductAttention`: the bottleneck runs thousands of tokens, where an explicit score matrix
  would be hundreds of megabytes.
  Three details are load-bearing. The spectrogram branch's tokens are **frame-major**
  (`b c fr t -> b (t fr) c`) while the channel sampler flattens the same grid **frequency-major** —
  one grid, two orders. The two positional encodings follow different conventions: the 2-D grid
  alternates sine and cosine with the width in the low channels and the height in the high, and the
  1-D sequence puts all cosines first. `ScaledEmbedding` stores its weight divided by 10 and
  multiplies it back in the forward, so the frequency embedding's effective factor is 2.0, not 0.2.
  `NFKHTDemucsSpectrum` is the transform pair; framing and overlap-add run over Swift buffers because
  MLX has no scatter-add, and a round-trip test asserts the inverse. `separate` pads a short clip to
  the release's 7.8-second training segment and trims back, as the reference does; the parity record
  runs with that off, which is a padding policy rather than a shape. Oracle: `demucs` 4.0.1 is
  installed and `demucs.states.load_model` builds the network from the checkpoint directly, so
  nothing is vendored — but demucs predates torch 2.6, so `torch.load` must be patched to
  `weights_only=False` first. Checkpoint
  `dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th` (81 MB, 533 tensors, 42M
  parameters).
  The other two releases are at parity too. `htdemucs_6s` (`.htdemucs6s`, `NFKMLXHTDemucsVariant.sixStem`,
  registered as `htdemucs-6s`) predicts six stems, guitar and piano after the four
  (`NFKMLXHTDemucsConfiguration.stemNames`), and sets **`bottom_channels` to 0**: it carries no
  `channel_upsampler`/`downsampler` pair at all and runs the cross-transformer at the deepest encoder's
  own width (384, `transformerWidth`) with a 1536-wide feed-forward — so the samplers are optional
  modules, built only when the width is set, and a strict load of the 6s file is what pins that.
  `htdemucs_ft` is a **bag**: four checkpoints of the base geometry, each fine-tuned for one stem,
  combined by per-source weights as the reference's `BagOfModels` does (`NFKMLXHTDemucsBag`, one-hot
  weights so each stem comes from its own model; `backendWithFineTunedWeightsURLs:error:`;
  `run_reference.py htdemucs_bag` drives the same four files). Separated stems 0.99999999999949 (6s)
  and 0.99999999999958 (ft).
- `NFKMLXPhonemizer` (protocol) + two paths for the TTS text→phoneme front-end. `NFKMLXEspeakPhonemizer`
  (macOS only) shells out to a **system-installed** espeak-ng — InferKit does not bundle it (GPLv3);
  `Tools/espeak/install.sh` installs it and the phonemizer uses it only when present (`isInstalled`).
  `NFKMLXNeuralG2P` is the in-toolkit path: a compact encoder-decoder transformer (reusing `NFKWhisperBlock`)
  mapping graphemes → phonemes, no external dependency, permissively licensed. Both conform to
  `NFKMLXPhonemizer`; the neural model has a `loadWeights` + round-trip test (grapheme/phoneme vocabs are
  load-time artifacts). These are the front-end for a full TTS chain (phonemes → acoustic → vocoder).
- `NFKMLXTTS` + `NFKMLXAcousticNet` + `NFKMLXHiFiGAN` — the complete text-to-speech voice.
  `NFKMLXAcousticNet` (FastSpeech2-style: phoneme embedding → transformer encoder → duration predictor
  → length regulator (gather-expand by rounded durations) → decoder → mel projection, reusing
  `NFKWhisperBlock`). `NFKMLXHiFiGANNet` is the vocoder (mel → waveform: `conv_pre` → transposed-conv
  upsampling via `NFKDemucsConvT1d` + multi-receptive-field dilated `NFKHiFiResBlock`s → `conv_post`/tanh).
  `NFKMLXTTS` chains a `NFKMLXPhonemizer` + acoustic + vocoder and exposes `makeSpeechBackend()`
  (text → WAV) via `NFKMLXSpeechBackend`. Acoustic/vocoder load safetensors separately; each has a
  round-trip test, and the full text→audio chain is tested end to end.
  The vocoder runs real released weights at reference parity: jik876's UNIVERSAL_V1 generator
  (whose geometry is this port's default configuration), cosine 0.9999999999341 against the
  reference's own `models.py` on a deterministic mel — a vocoder is a pure function of its mel, so
  nothing about speech needs assuming. The release stores every convolution weight-normalized
  (`weight_g`/`weight_v`); `Tools/hifigan-to-safetensors/convert.py` fuses `g·v/‖v‖`, which is the
  reference's own `remove_weight_norm`. Reaching parity found a real defect: the reference's one bare
  `F.leaky_relu(x)` before `conv_post` runs at PyTorch's default slope 0.01 where every other
  activation is 0.1 — with 0.1 the released weights score 0.99954, measured both ways. The
  upsampling stages load through the Demucs ConvT treatment (`[in, out, k]`, name-gated).
  The trained acoustic model is ported too, and the voice is complete. `NFKMLXFastSpeech2Net`
  is the espnet FastSpeech2 conformer (through the transformers layout, whose implementation is the
  oracle): relative-position attention with the Transformer-XL shifting trick, macaron post-norm
  conformer layers (`normalize_before: false`), a GLU convolution module with BatchNorm, conv-FFN
  blocks, duration/pitch/energy variance adaptors (pitch and energy predicted per phoneme and
  embedded before the durations stretch to frames), the length regulator, and the residual postnet.
  Reference parity on the released LJSpeech weights on the first numeric run: encoder
  0.9999999999998, durations exact frame for frame, pitch/energy/mel all ≥ 0.9999999999. Module keys
  are the checkpoint's names.
  `NFKMLXVoice` chains it with the vocoder and the release's own 78-symbol ARPAbet vocabulary (the
  matching phoneme table), exposed through `makeSpeechBackend(phonemize:)`. The vocoder must be the
  Paired release (`espnet/fastspeech2_conformer_with_hifigan`, `vocoder.` prefix, weight norm
  already fused): espnet's acoustic model emits mels normalized by its training statistics, and the
  universal jik876 generator — same geometry, raw-log-mel convention — turns them into loud garbage.
  Measured, not assumed: the end-to-end test synthesizes "hello world" and has the package's own
  Whisper (real weights, at parity) transcribe it — with the universal vocoder Whisper hears
  "(indistinct)", with the paired one **" hello, world."** — which closes the loop TTS → audio → ASR
  entirely inside this package on released weights.
- `NFKMLXKokoro` / `NFKMLXKokoroNet` (`@objc`) — **Kokoro-82M**, a StyleTTS2 / iSTFTNet text-to-speech
  voice (hexgrad, Apache-2.0), a second TTS beside FastSpeech2 and the most popular on-device one. The
  pipeline: a **PL-BERT (Albert)** phoneme encoder (12 parameter-shared layers over a factorized 128-wide
  embedding), a `bert_encoder` projection, a **duration predictor** (a DurationEncoder of bidirectional
  LSTM + AdaLayerNorm blocks that re-concatenates the style vector each layer, then an LSTM and a
  duration head), the **alignment** by the rounded durations, an **F0/energy predictor** (a shared LSTM
  then AdaIN residual blocks with a depthwise-ConvTranspose upsample), a separate **TextEncoder** (CNN +
  LSTM), and an **iSTFTNet decoder** — AdaIN residual blocks over the alignment-expanded encoding, then a
  generator with a **harmonic sine source** (`SourceModuleHnNSF`), upsampling transposed convolutions,
  Snake-activated `AdaINResBlock1`s, a noise band from the source's STFT, and an **inverse-STFT** head.
  Reference parity on the released weights, seam by seam against the vendored `KModel` (`run_reference.py
  kokoro`, `IK_PARITY_KOKORO` + `IK_VAL_KOKORO`, the `llm` env): the PL-BERT 0.99999999999, the projection
  0.99999999999, the DurationEncoder 0.99999999999, durations 0.99999999999, F0/energy 0.99999999999, the
  TextEncoder 0.99999999999, the asr 0.99999999999, and the decoder's encode and full decode stack exact
  (0.9999999999995 / 0.9999999999991 — every AdaIN residual block, the F0/N stride-2 convs, and the
  depthwise-ConvTranspose upsample). The vocoder is float-precision-limited, not a modeling gap: the
  reference multiplies the accumulated sine phase by the upsample scale (≈18000 radians — the NSF sine
  oscillates at F0 over the whole clip) before `sin`, and a float32 argument of that size holds ~two
  fractional digits, so a torch/MLX rounding difference bounds the sine source at 0.99999 and the waveform
  cosine near 0.996 — the same class as the Music3 e2e, so the deterministic seams are the ground and the
  audio is compared loosely. The full consumer path (`loadVocab` + `loadVoice` + `synthesize(phonemes:voice:)`,
  per-Unicode-scalar phoneme mapping) reproduces the reference waveform at 0.997.
  Three facts are load-bearing, all found by the parity run. The released `.pth` stores each top-level
  module's state_dict under a `module.` DataParallel prefix (`bert.module.embeddings…`) which KModel strips
  at load and the loader strips too. The sine source's voiced threshold is **10, not 0** — SourceModuleHnNSF
  passes `voiced_threshod=10` to SineGen. And the generator's one bare `leaky_relu` before `conv_post` runs
  at the default slope 0.01 where every other activation is 0.1 (the same HiFi-GAN trap). All the
  bidirectional LSTMs load through the shared PyTorch→MLX fold (`weight_ih_l0`/`hh` → `Wx`/`Wh`, biases
  summed, forward/reverse), the weight-norm convs fuse `g·v/‖v‖`, and the `AdaINResBlock1` Snake `alpha`
  ParameterLists stack into one parameter. The misaki phonemizer is not required (it pulls spaCy, which
  will not build here): the oracle vendors just `KModel` + `istftnet` + `modules` (torch/transformers/scipy,
  no spaCy), and the backend takes a phoneme string under `NFKInputPrompt` directly — a caller brings the
  grapheme→phoneme front end (`NFKMLXNeuralG2P` or espeak). The `@objc` `NFKMLXKokoro.backend(directoryURL:voiceName:)`
  returns an `NFKMLXSpeechBackend` (24 kHz WAV under `NFKOutputAudio`); a voicepack `.pt` is a bare tensor
  the native reader will not interpret, so it converts to a single-`voice` safetensors offline (the
  `Tools/kokoro-voice-to-safetensors` treatment), which `loadVoice` reads. The weights are
  `hexgrad/Kokoro-82M` (327 MB `kokoro-v1_0.pth` + `config.json` + `voices/*.pt`).
- `NFKMLXVideoBackend` / `NFKMLXVideoFile` — the first backend that produces video, and the AVFoundation
  decode/encode layer under it (the video counterpart of `NFKMLXWaveFile`). `NFKModalityVideo` and the
  `NFKInputVideo` / `NFKOutputVideo` keys were in the core's vocabulary with nothing emitting a clip.
  The backend reads an `NFKVideoAsset`, hands every frame to a `([MLXArray]) -> [MLXArray]` transform
  as `[H, W, 3]` in `0...1`, encodes what comes back, and returns a new `NFKVideoAsset`. The transform
  takes the whole sequence, not one frame, because that is what the models need: frame interpolation
  reads pairs and returns more frames than it took, and BasicVSR propagates state backward and forward
  through time, so upscaling a clip is not upscaling its frames independently. A per-frame model simply
  maps. `frameRateMultiplier` / `outputFramesPerSecond` carry the rate change a frame count change
  implies — a doubled clip written at the source rate is slow motion, not smoother footage, so the
  duration is what stays fixed. `NFKMLXRIFE.clipBackend` (`n` → `2n - 1` frames at twice the rate) and
  `NFKMLXVideoSR.clipBackend` (×4, `upscaleSequence`) are the shipped users. AVFoundation's synchronous
  property accessors are deprecated, so the reads go through a semaphore-blocked `loadTracks` /
  `load(.nominalFrameRate)`; the contract is synchronous and the caller is already off the render
  thread. H.264 needs even dimensions and one frame size per clip, and both are rejected explicitly
  rather than cropped or scaled where the change would be invisible in the result.
  Trained models are carried through the whole path in tests, not only the shape: a clip is built
  by translating a real photograph (a synthetic gradient measures nothing — the models are trained on
  photographs), and the assertions are about the result. RIFE's synthesized frame must correlate
  better with the true midpoint than with either neighbour (0.9981 against 0.9145 / 0.9104), which is
  what separates interpolation from copying a frame; BasicVSR's output must still be the source frame
  enlarged (0.9891). Both comparisons resize to a common size first — correlating a ×4 output against
  its small source compares the output's first rows and measures nothing, which read as a model
  failure at 0.805 until the comparison was fixed.
- `NFKMLXSpeechBackend` (`@objc`) — a bring-your-own MLX text-to-speech backend: supply a
  `@Sendable (String) -> MLXArray` closure returning a mono waveform in `-1...1`; the backend reads the
  prompt (`NFKInputPrompt` or `NFKInputMessages`), writes a 16-bit PCM WAV via `NFKMLXWaveFile`
  (Foundation-only, unit-tested), and returns an `NFKAudioAsset` under `NFKOutputAudio`.
  `NFKMLXReferenceModels.registerToneSpeech` is the shipped reference (`tone-speech`), so ObjC builds
  the text→audio path by name. This is the first backend for the audio modality.
- `NFKMLXMarigold` / `NFKMLXSDUpscaler` (`@objc`) — two image-conditioned latent-diffusion models on
  `NFKMLXDiffusionBackend`, over the same real networks. Marigold (`marigold-depth`, image → depth) is
  Stable Diffusion 2 geometry and denoises a depth latent concatenated with the image latent; the ×4
  upscaler (`sd-x4-upscaler`, image → ×4 image) denoises a high-resolution latent conditioned on the
  low-resolution image itself, with a **noise level** joining the timestep through a class embedding
  (`noiseLevel`, the release's own default 20). The upscaler's autoencoder is one level shallower than
  the others, which is where its ×4 comes from — narrowing a test configuration must keep the level
  count or the model silently becomes a ×2. Output size and round-trip tested.
- `NFKMLXStyleTransfer` (`@objc`) — a real single-forward stylizer: Johnson et al.'s `TransformerNet`
  (three downsampling convs → five residual blocks → two nearest-upsample convs → output conv, each
  instance-normalized) in `MLXNN`, run through `NFKMLXModuleBackend` (image → stylized image at input
  size). `+register` under `fast-style-transfer`; the style is baked into the weights (one checkpoint =
  one style). Reference parity against pytorch/examples (cosine 0.9999926). Reaching it required real
  reflection padding (`NFKMLXResample.reflectPadded`, a mirror gather — MLX pads with a constant or the
  edge value only): with edge padding the mean pixel error was 0.049, and only a quarter of that sat at
  the border, because the instance norms are global and carry a border approximation into every pixel.
  Names match the reference, so `Tools/style-transfer-to-safetensors/convert.py` only drops the
  deprecated InstanceNorm running-stats keys.
- `NFKMLXCLIP` (`@objc`) — real image+text embeddings (CLIP ViT-B/32): a ViT image tower (`visual.*`)
  and a causal text transformer (`token_embedding`/`transformer`/`ln_final`/`text_projection`), both
  L2-normalized into a shared space. Attention keeps the reference fused `in_proj_weight`/`out_proj`.
  Reference parity on both towers against transformers' `CLIPModel` (image cosine 0.9999965,
  **text cosine 0.9999999999988** — the text record carries the reference's own token ids, since the
  port embeds ids rather than text).
  `NFKMLXCLIPBackend` reads `NFKInputImage` → embedding under the new core key `NFKOutputEmbedding`; a
  text prompt encodes when a tokenizer is supplied (byte-level-BPE vocab is a load-time artifact — a
  caller can pass token ids through `encodeText`). `+register` under `clip-vit-b-32`, and the other
  released towers under `clip-vit-b-16` / `-l-14` / `-l-14-336` (`NFKMLXCLIPVariant`: `.vitB16`,
  `.vitL14` — vision 1024 wide, 24 blocks, 16 heads, embedding 768, text 768 / 12 / 12 — and
  `.vitL14At336`); B/16 and L/14 load their released checkpoints strictly and return unit embeddings,
  and being the B/32 blocks at another geometry they carry the numeric parity above.
  `Tools/clip-to-safetensors/convert.py` targets the OpenAI JIT/state-dict (names match). Forward,
  round-trip, and unit-length embedding tested.
- `NFKMLXSigLIP2` (`@objc`) — real image+text embeddings (SigLIP 2, base-patch16-224), the CLIP upgrade
  and the vision tower a VLM reads. The vision and text towers are the same transformer the SmolVLM
  SigLIP encoder uses (`NFKSigLIPLayer`/`NFKSigLIPAttention`/`NFKSigLIPMLP`/`NFKSigLIPEncoder` are reused
  directly); SigLIP 2 adds an **attention-pooling head** over the vision patches (`NFKSigLIP2ProbeAttention`:
  a learned probe token cross-attends over the patch features through `nn.MultiheadAttention`'s fused
  `in_proj_weight`, then a residual LayerNorm and MLP), a **text tower** over a 256k multilingual
  vocabulary (last-token pooled through a `head` projection), and learned `logit_scale`/`logit_bias` for
  the sigmoid similarity. The image embedding is the pooling head's output; the text embedding is the
  last token's; both are L2-normalized and `logit = scale·(text·image) + bias`. The vision embeddings
  read the position table row-major (SigLIP 2 does not use SmolVLM's fractional position buckets, which
  is why `NFKSigLIP2VisionEmbeddings` is separate from the SmolVLM one). `NFKMLXSigLIP2Backend` reads
  `NFKInputImage` → the image embedding under `NFKOutputEmbedding`; `imageEmbedding`/`textEmbedding` are
  the object accessors. `+register` under `siglip2-base-patch16-224`. The MAP head's `attention` is a
  real submodule, not a dotted parameter key — MLX splits parameter keys on `.`, so `@ParameterInfo(key:
  "attention.in_proj_weight")` would not nest into an `attention` child and the head weights load as
  random (measured: image cosine collapses to ~0.5 while the text tower stays exact, the seam that
  localized it). The release is already a PyTorch-layout safetensors the loader reads directly (it
  transposes the 4-D patch conv and maps the `vision_model.`/`text_model.` prefixes; Linear/embedding
  weights and the fused attention projection are 2-D and pass through). Reference parity against
  transformers on the released weights (`run_reference.py siglip2`, llm oracle env, transformers ≥ 4.51):
  image embedding cosine 0.999999999999, every text embedding 0.999999999999, and the sigmoid logits to
  1e-5. The architecture is SigLIP v1 (`model_type` "siglip"); the "2" is the training. Most of the
  1.5 GB checkpoint is the 256k text embedding table. Converter `Tools/siglip2-to-safetensors` is a
  passthrough normalizer. Every SigLIP 2 release is a preset (`NFKMLXSigLIP2Family` × patch × image
  size through `towers(_:patchSize:imageSize:)`: base at patch 16 / 32, large, so400m at patch 14 / 16,
  giant-opt, each at its released resolutions; the giant-opt text tower is 1152 wide and projects to
  1536, which is what `projectionSize` carries), with `NFKMLXSigLIP2Variant` (15 cases) selecting one
  from Objective-C and `register()` naming each under its release name. The fourteen beyond the
  measured base-224 are held to the module by shape against their released headers (408 / 792 / 888 /
  1096 tensors per family, 0 missing, 0 mismatched, 0 unaccounted).
- `NFKMLXIPAdapterImageProjection` / `NFKMLXIPAdapterAttention` — IP-Adapter, lightweight image
  conditioning for a diffusion model (steer a Stable Diffusion generation with a reference image, not
  only text). Two pieces: the image projection maps a CLIP image embedding to a short sequence of
  image-text tokens (`image_embeds` Linear → reshape → LayerNorm), and the decoupled cross-attention
  adds a second, image-conditioned attention beside the text cross-attention — `text_attn + scale·ip_attn`,
  sharing the query, through its own `to_k_ip` / `to_v_ip` projections (the reference stores those under a
  `processor.` prefix, stripped on load). The projection and the extra key/value weights are the only
  trained parameters; the base UNet is frozen, so an adapter is a small file over a shipped SD model.
  Reference parity against diffusers (`run_reference.py ip_adapter`, `ltx` env): the ImageProjection
  against `diffusers.models.embeddings.ImageProjection`, and the decoupled attention against
  `IPAdapterAttnProcessor2_0` — both cosine 0.9999999999999997. The adapter is wired into the shipped
  `NFKMLXSDUNet`: an optional `NFKSDImageConditioning` (tokens + scale, nil by default so the base UNet is
  byte-identical) threads down → mid → up → transformer block → `attn2` Only, and `NFKSDAttention` gains
  optional `to_k_ip`/`to_v_ip` attached through `update(modules:)` (assigning a `@ModuleInfo` optional
  after init does not register it). `NFKMLXIPAdapter.load(from:into:)` reads the released
  `ip-adapter_sd15.safetensors`; the adapter's sorted indices map onto the cross-attentions in
  down → up → mid order, because `UNet2DConditionModel` registers `mid_block` last. Validated in the real
  SD 1.5 UNet with the real adapter against diffusers (`run_reference.py ip_adapter_unet`,
  `IK_PARITY_IP_ADAPTER_UNET`): cosine 0.99999999999989, with the base UNet and SD 1.5 text-to-image still
  at parity. `NFKMLXTextToImage.imageAdapterBackend(configuration:directoryURL:adapterURL:scale:)` is the
  consumer path (a precomputed CLIP-ViT-H image embedding under `NFKMLXInputImageEmbedding`; under CFG the
  unconditional row takes zero image tokens, the reference's `negative_image_embeds`).
- `NFKMLXTAESD` (`@objc`) — TAESD (Tiny AutoEncoder for Stable Diffusion), the fast preview decoder a
  latent-diffusion pipeline uses: a small distilled autoencoder mapping an image to a four-channel latent
  and back (8× down/up). The encoder and decoder are flat `nn.Sequential` stacks of 3×3 convolutions and
  residual blocks; modeling them as `[Module]` arrays makes the numeric Sequential keys (`0.weight`,
  `1.conv.0.weight`, …) match with no remap — the case the MLX-runtime note calls out as the legitimate
  use of numeric keys. `NFKTAESDBlock` is three convs (ReLU between) added back to the input then a fusing
  ReLU (skip identity, all 64→64); the downsample convs are stride-2 bias-free; the decoder clamps its
  input `tanh(x/3)·3`, upsamples nearest ×2, and ends in a `conv→3`. Parameter-free ops (ReLU, the clamp,
  the upsample) are empty marker `Module`s occupying their Sequential index; the net's forward dispatches
  by type. `NFKMLXTAESDBackend` reads `NFKInputImage` → the reconstruction under `NFKOutputImage`;
  `encode`/`decode` are the object accessors (the preview-decode use). `+register` under `taesd`. The
  release is two `.pth` files (encoder + decoder, GitHub, ~5 MB each); `Tools/taesd-to-safetensors`
  combines them into `encoder.*`/`decoder.*` keys and the loader transposes the 4-D convs. Reference
  parity against madebyollin's own `taesd.py` on the first numeric run: latent cosine 0.9999999999996,
  decode cosine 0.9999999999998, mean |difference| 1.9e-7.
- `NFKMLXLTXVideoVAE` (`@objc`) — the LTX-Video VAE (`AutoencoderKLLTXVideo`), the toolkit's first piece of
  video generation (the DiT and a T5 text encoder are the remaining stages) and its first 3D model. A
  **causal 3D autoencoder**: a video compresses to a spatiotemporal latent and back. The encoder is causal
  in time — each temporal convolution left-pads by repeating the first frame `kernel-1` times, so a frame
  depends only on the past (`NFKLTXCausalConv3d`, a wrapper over MLX's `Conv3d` keeping the reference's
  `.conv` key); the decoder is non-causal (symmetric pad). Downsampling is a stride-2 causal conv;
  upsampling is a conv that widens the channels ×8 then a 3D pixel shuffle (`NFKLTXUpsampler`:
  interleave the extra channels into the frame/height/width axes, drop the first frame). Resnet blocks
  normalize with a parameter-free channel RMS norm (so norm1/norm2/norm_out carry no weights; only the
  shortcut's `norm3` LayerNorm does). Encoder: **patchify** (fold each `patchT×patch×patch` block into the
  channel axis, the reference's `channel, temporal, width, height` order) → conv_in → down blocks → mid →
  RMSNorm+SiLU → conv_out (latent+1 channels; the extra channel is the posterior's shared log-variance,
  dropped for the deterministic latent). Decoder mirrors it and unpatchifies. The whole thing works in
  NDHWC (`[B, T, H, W, C]`) where the reference is NCTHW, so every patchify/pixel-shuffle reshape is the
  reference's permute re-derived for channels-last; the 5-D Conv3d weights transpose `[out,in,kT,kH,kW]` →
  `[out,kT,kH,kW,in]` at load, and the causal-conv `.conv` naming makes the keys match with no remap.
  `NFKMLXLTXVideoVAE.encode`/`decode` (Swift, over `MLXArray`) and `vae(configuration:weightsURL:)` are the
  surface; the released diffusers safetensors loads directly (converter `Tools/ltx-vae-to-safetensors` is a
  passthrough). Reference parity against diffusers' `AutoencoderKLLTXVideo` on the first numeric run
  (`run_reference.py ltx_vae`, the `ltx` oracle env — diffusers ≥ 0.32, its own venv): every encoder seam
  exact (conv_in / first down block / mid ≥ 0.99999999999), latent cosine 0.99999999999, decode cosine
  0.99999999996. A weight-free test also asserts the encoder's temporal causality (two clips sharing their
  first frames but not the last produce the same first latent frame).
- `NFKMLXLTXTransformer` (`@objc`) — the LTX-Video DiT (`LTXVideoTransformer3DModel`), the denoising
  transformer of the video-generation pipeline (the stage after the VAE). A 2B sequence transformer over
  the VAE's flattened latent tokens: `proj_in` (128→2048), 28 `NFKLTXBlock`s, `norm_out`+adaLN, `proj_out`
  (2048→128). Each block is adaptive-layer-norm self-attention with 3D rotary + cross-attention to the
  text embedding + a gelu-approximate feed-forward, the six modulation parameters coming from the
  timestep through the block's own `scale_shift_table` (PixArt-α style). Attention (`NFKLTXAttention`)
  applies an across-heads RMS norm to the query and key (over the full 2048 width, before the head split),
  the 3D rotary to both (self-attention only), then SDPA; cross-attention reads the projected text and does
  not rotate. The **3D rotary** (`NFKLTXRotary`) is computed over the (frame, height, width) latent grid: a
  per-axis log-spaced frequency ramp times the scaled coordinate, cos/sin repeat-interleaved by 2, the
  leading `dim % 6` channels left unrotated. Timestep conditioning is `AdaLayerNormSingle` (a 256-wide
  sinusoidal embedding through an MLP, then SiLU + linear to `6·inner`); the text is a PixArt caption
  projection (4096→2048). The feed-forward's `net` is a `[Module]` array (`net.0.proj`, an activation
  marker, `net.2`) so the diffusers Sequential keys match. The module keys mirror the reference exactly
  (all 715 tensors), so the sharded release loads through `NFKMLXReleaseWeights.arrays` with a pass-through
  remap and no transpose (every weight ≤ 2-D). For parity the text embedding is supplied directly (the
  caption projection is inside the DiT), so the transformer is validated in isolation like the SD UNet — no
  T5 needed. Reference parity against diffusers on the first numeric run (`run_reference.py
  ltx_transformer`, the `ltx` oracle env, recorded random latent/text/timestep): every seam exact (rope
  cos/sin ≥ 0.9999999, proj_in and first block ≥ 0.99999999999) and the full 28-layer velocity cosine
  0.99999999999. The 2B weights are ~7.7 GB sharded.
- `NFKMLXT5Encoder` (`@objc`) — the T5 v1.1 text encoder (`T5EncoderModel`), the text conditioning for the
  LTX pipeline and a reusable building block (the same family conditions Wan / PixArt / SD3 / Flux). A
  stack of pre-normalized blocks with two T5-isms: the attention is unscaled and adds a bucketed
  **relative-position bias** (`NFKT5Attention.computeBias`, the Mesh-TensorFlow bidirectional bucketing,
  computed once from block 0's table and shared across all layers, passed as the SDPA additive mask), and
  the norm is **T5LayerNorm** — an RMS norm with a weight and no mean subtraction. The feed-forward is
  gated (`wo(gelu(wi_0(x)) · wi_1(x))`, tanh-approx GELU). Module keys mirror the reference
  (`shared`, `encoder.block.N.layer.0.SelfAttention.{q,k,v,o}`, `.layer.0.layer_norm`,
  `.layer.1.DenseReluDense.{wi_0,wi_1,wo}`, `.layer.1.layer_norm`, `encoder.final_layer_norm`), so the
  sharded release loads through `NFKMLXReleaseWeights.arrays` with no remap and no transpose (all 2-D).
  `NFKMLXT5Configuration.xxl` is T5-XXL (d_model 4096, 24 layers, 64 heads, d_ff 10240). Reference
  parity against transformers on the first numeric run (`run_reference.py ltx_t5`, the `ltx` oracle env):
  embedding seam exact, first block 0.9999999999996, full text embedding cosine 0.99999999998. ~19 GB fp32
  sharded — the memory crux of the LTX pipeline, which stages the encoders sequentially.
- `NFKMLXFlowMatchScheduler` — the rectified-flow sampler (`FlowMatchEulerDiscreteScheduler`), the sampler
  LTX / Flux / SD3 / Wan / Z-Image use, a value type with no parameters. The schedule is a sigma ramp from
  1 to 0 with **dynamic resolution-dependent shifting** (a per-sequence-length `mu = base_shift + slope·
  (seq − base_seq)` warps the ramp: `σ ← exp(mu)/(exp(mu) + 1/σ − 1)`) and a **terminal stretch** so the
  last non-zero sigma lands on `shiftTerminal` (0.1). A step is one Euler update `x + (σ_next − σ)·v`.
  **Verified against diffusers** (schedule exact: sigmas to 1e-4, terminal sigma 0.1) under `swift test`
  (pure Float math, no MLX eval).
- `NFKMLXLTXPipeline` — the LTX-Video text-to-video pipeline glue, chaining the three parity-verified
  stages: T5 encode → the DiT denoised over the flow schedule with classifier-free guidance → the VAE
  decode. With the DiT's patch size of 1 the latent packing to/from the token sequence is a single reshape
  of the VAE's NDHWC latent. The stages are held together for a run but a caller manages residency (the
  19 GB T5, the 7.7 GB DiT, and the VAE do not all fit resident on 32 GB, so they load and free in turn,
  the Music 3 pattern). Validated by a weight-free glue test on matching tiny configurations (the packing,
  the guided loop, unpacking, and decode produce a correct-shaped clip) plus the four stages' own parity —
  a sampled clip cannot be compared bitwise, as with Music 3. The VAE + DiT + T5 + flow are the complete
  LTX text-to-video path.
- `NFKMLXZImageTransformerNet` — the Z-Image S3-DiT (`ZImageTransformer2DModel`, Alibaba Tongyi), the
  denoising transformer of a 6B text-to-image model and the third DiT family beside the SD UNet and LTX.
  **Single-stream**: the image latent tokens and the caption tokens are concatenated and every layer's
  self-attention runs over the join, rather than a separate cross-attention branch. Three stages: a
  `noise_refiner` (2 modulated blocks over the image tokens alone), a `context_refiner` (2 UN-modulated
  blocks over the caption tokens alone), then 30 unified `layers` over the concatenation. The block is a
  Sandwich: `attention_norm1` before and `attention_norm2` after the attention, the same for the FFN,
  with a 4-chunk adaptive modulation (scale/gate for each of attention and FFN, gates `tanh`'d, scales
  `1 +`). SwiGLU FFN (hidden `dim/3·8` = 10240), per-head RMS q/k norm, and a **3-axis complex rotary**
  (`view_as_complex`, axes `[32,48,48]` summing to headDim 128, θ 256) over the (frame, height, width)
  grid. The caption is a Qwen3-4B embedding (`cap_feat_dim` 2560) projected in through
  `RMSNorm → Linear`; base text-to-image feeds only the latent + caption (the SigLIP visual-semantic
  tokens are the edit variant's input). Sequence lengths pad to a multiple of 32 with learned
  `x_pad_token`/`cap_pad_token` at (0,0,0) positions. Module keys mirror the reference exactly (73
  at the tiny config, `all_x_embedder.2-1`/`all_final_layer.2-1` ModuleDict keys included), so a release
  loads with no remap and no transpose (every weight ≤ 2-D). For parity the caption features are
  supplied directly (no Qwen3), so the DiT is validated in isolation, as the LTX DiT is. Reference
  parity against diffusers on the first numeric run (`run_reference.py z_image`, tiny random config,
  the `ltx` oracle env — diffusers 0.36 carries ZImageTransformer2DModel): the t_embedder seam exact and
  the full velocity cosine **0.9999999999999653**, with the pad-token path exercised (sequence lengths
  not a multiple of 32).
  The Flux VAE Z-Image encodes into is the shared `NFKMLXSDAutoencoder`: its `vae/config.json` is a
  diffusers `AutoencoderKL` (16 latent channels, `[128,256,512,512]`, `mid_block_add_attention`) that
  differs from Stable Diffusion's only in dropping the quant convolutions (`use_quant_conv: false`,
  `use_post_quant_conv: false`) and in a centering `shift_factor` 0.1159 / `scaling_factor` 0.3611. So
  `NFKMLXSDVAEConfiguration` gained `useQuantConv` (the two 1×1 convs are now optional `Conv2d?` — when
  absent the encoder's `conv_out` is the moments directly and decode reads the latent directly) and
  `shiftFactor`, and `.flux` is the preset. Reference parity against diffusers' AutoencoderKL at a
  tiny `use_quant_conv=False` config (`run_reference.py flux_vae`, `ltx` env): encoded-mean cosine
  0.9999998591898961, decode cosine 0.9999999948059418.
  `NFKMLXZImagePipeline` chains it end to end (S3-DiT denoised over the flow schedule with
  classifier-free guidance → Flux VAE decode). The caller supplies the caption embedding (the Qwen3-4B
  hidden states), as the SD pipeline takes a text context — Z-Image's text step is the shipped Qwen3
  decoder (already at reference parity), read for its penultimate hidden state (`hidden_states[-2]`, which
  is `NFKMLXLanguageNet.layerStates(tokens)[count − 2]` — the existing per-layer seam) after the Qwen chat
  template, run and freed separately. So no new port is needed for the text step; Qwen3 + the seam cover
  it. `NFKMLXFlowMatchConfiguration.zImage`
  is its schedule (a smaller resolution shift, no terminal stretch, `sigma_min` 0); the DiT timestep is
  `1 − σ` and the flow velocity is negated before the Euler step, both the reference's conventions.
  Validated by a weight-free tiny-config glue test (the guided loop, the timestep/latent conventions,
  the centered-latent decode) plus the DiT/VAE/flow parities — a sampled image is not bitwise-comparable,
  as with LTX and Music 3. The Qwen3-4B DiT + Flux VAE + flow are the complete Z-Image text-to-image path.
  `generate(image:strength:…)` is the image-to-image (edit) path — the Flux VAE encodes the source, the
  latent is noised to `strength` through the scheduler's flow `addNoise`, and the denoise runs from the
  matching step, diffusers' own `pipeline_z_image_img2img`. The paper's SigLIP-conditioned editing is a
  separate original-repo model not present in the diffusers Z-Image (its transformer takes only the image
  latent and the caption; neither diffusers pipeline references SigLIP), so there is no reference to port
  it against — the img2img path here is diffusers' actual edit capability.
- `NFKMLXSANATransformerNet` — the SANA linear-attention DiT (`SanaTransformer2DModel`, NVIDIA), the
  fourth DiT family. Two things set it apart: the self-attention is **linear** — ReLU feature maps,
  `O = ((V·1̂) @ ReLU(K)) @ ReLU(Q)` normalized by the ones row, O(N) rather than O(N²), which is what
  lets it run at high resolution — and the feed-forward is a **GLUMBConv** (an inverted-bottleneck gated
  Depthwise convolution over the 2-D token grid: `conv_inverted` 1×1 to `2·hidden`, SiLU, `conv_depth`
  3×3 depthwise, gate-split `x · silu(gate)`, `conv_point` 1×1, no bias), not a pointwise MLP. Each
  block also cross-attends to the Gemma text embedding through ordinary softmax attention (heads 20,
  head_dim 112). Conditioning is PixArt-α `AdaLayerNormSingle`: one shared timestep embedding plus a
  per-block learned `scale_shift_table` [6, inner], expanded to the six modulation parameters; the output
  norm reads the pre-linear embedded timestep against a top-level `scale_shift_table` [2, inner]. Patch
  embed is a stride-`patch` `Conv2d` (no positional embedding). The caption enters through
  `PixArtAlphaTextProjection` (Linear → GELU-tanh → Linear) and an RMS `caption_norm`. Module keys mirror
  the reference (54 at tiny); the convolution weights load transposed to MLX's NHWC. Reference parity
  against diffusers on the first numeric run (`run_reference.py sana`, tiny random config, `ltx` env):
  embedded-timestep seam exact, full velocity cosine **0.9999999999999959**.
  `NFKMLXDCAutoencoderNet` is SANA's Deep-Compression Autoencoder (`AutoencoderDC`), which compresses
  an image 32× spatially (against a Stable Diffusion VAE's 8×) — that is what keeps SANA's latent-token
  count low enough to run at high resolution. It is deterministic (`encode` returns one latent, not a
  Gaussian), built from two block families: `ResBlock`s at the shallow stages and `EfficientViTBlock`s
  (a multiscale ReLU linear attention — grouped multiscale-kernel q/k/v projections, then the same
  `O = ((V·1̂) @ ReLU(K)) @ ReLU(Q)` linear attention the DiT uses — plus a GLUMBConv with an RMS norm and
  a residual) at the deep stages. Down/up sampling is pixel-unshuffle / pixel-shuffle with a
  channel-averaging (`DCDownBlock`) or channel-repeating (`DCUpBlock`) shortcut; the released SANA uses a
  stride-2 Conv downsample and a nearest-interpolate upsample. The `pixelUnshuffle` / `NFKMLXPixelShuffle`
  helpers are shared with Real-ESRGAN / BiSeNet. The reference's `<blocks>.<i>.<j>` `nn.Sequential`
  indices map onto the module's `<i>.block.<j>` (a stage is an `NFKDCStage` holding a `[Module]`); the
  convolution weights load transposed to NHWC. Reference parity against diffusers' AutoencoderDC on
  the first numeric run (`run_reference.py dc_ae`, tiny random config, `ltx` env): latent cosine
  0.9999999999999344, decode cosine 0.9999999999999848. Also validated on the actual released SANA
  weights end to end (`NFKMLXDCAutoencoderNet.loadWeights` reads the diffusers checkpoint, remapping the
  `<blocks>.<i>.<j>` Sequential indices to `<i>.block.<j>` and transposing the convs): on the real 1.2 GB
  `Sana_600M` VAE over a 256×256 image at the full 32× compression, latent cosine 0.9999999999918, decode
  cosine 0.9999999998 (`run_reference.py dc_ae_real`, `IK_VAL_DCAE`) — the released config, the real
  checkpoint, and the loader confirmed, not just the tiny-config architecture.
  `NFKMLXSANAPipeline` chains it end to end (linear-attention DiT denoised over the flow schedule with
  classifier-free guidance → DC-AE decode). The caller supplies the caption embedding (the Gemma text
  encoder's last hidden state), as the SD pipeline takes a text context. The sampler is SANA's released
  `DPMSolverMultistepScheduler` (`NFKMLXDPMSolverScheduler`, at reference parity — see the scheduler
  entry below), not a stand-in. Validated by a weight-free tiny-config glue test plus the DiT/VAE
  parities. SANA's text encoder is `NFKMLXGemma2Net` (Gemma-2, ported here — see the Gemma-2 entry):
  the caller runs it for the caption features. The SANA text-to-image path is complete.
- `NFKMLXGemma2Net` — the Gemma-2 text decoder (`Gemma2Model`), SANA's text encoder (its DiT
  cross-attends to Gemma-2's last hidden state). Gemma 2 is a distinct architecture from the Gemma 3 /
  Gemma 4 text models here: it keeps the `(1 + w)` RMS normalization and the sandwich block (a norm
  before and after each of attention and the feed-forward), but it has no query/key norm, it soft-CAPS
  the attention logits (`tanh(logit/cap)·cap`, cap 50), it uses a single rotary base (10000) with
  sliding-window attention on the even layers, and it scales the query by `query_pre_attn_scalar^-0.5`.
  The attention is computed explicitly (matmul + soft-cap + softmax, not the fused SDPA) because of the
  soft-cap. Module keys are the checkpoint's (`embed_tokens`, `layers.N.self_attn.{q,k,v,o}_proj`,
  `layers.N.mlp.{gate,up,down}_proj`, the four sandwich norms, `norm`), no transpose. Reference parity
  against transformers' Gemma2Model at a tiny configuration with a small sliding window (so the
  alternating sliding/full layers differ): last hidden cosine 0.9999999999998679 on the first numeric
  run (`run_reference.py gemma2`, the `llm` oracle env). The released sizes are presets
  (`.gemma2_9B`: 3584 / 42 layers / 16 heads / 8 kv / head 256 / 14336, `query_pre_attn_scalar` 256;
  `.gemma2_27B`: 4608 / 46 / 32 / 16 / 128 / 36864, scalar 144), each held to its released headers by
  shape (464 / 508 tensors, 0 missing, 0 mismatched, 0 unaccounted).
- `NFKMLXWanTransformerNet` — the Wan text-to-video DiT (`WanTransformer3DModel`, Alibaba Wan), the fifth
  DiT family. A 3-D sequence transformer over a `Conv3d`-patchified video latent (patch `(1,2,2)`), with
  the same 3-axis interleaved rotary as Z-Image (`NFKZImageRope` reused, θ 10000, axes `t = headDim −
  2·h`, `h = w = 2·(headDim/6)`) over the (frame, height, width) grid. Each block runs self-attention
  (with rotary), cross-attention to the text embedding, and a gelu-approximate feed-forward
  (`ffn.net.0.proj`/`net.2`), under PixArt-α adaptive norms — a shared `time_proj` [6·inner] plus a
  per-block `scale_shift_table` [1,6,inner]; the cross-attention norm (`norm2`) is an affine LayerNorm
  applied UN-modulated, where `norm1`/`norm3` are non-affine and modulated. The q/k norm is
  `rms_norm_across_heads` — an RMS norm over the whole inner width before the head split, where
  Z-Image norms per head. The condition embedder is `time_embedder` (TimestepEmbedding) → `time_proj`,
  plus `text_embedder` (`PixArtAlphaTextProjection`). This is the text-to-video path (no
  image-conditioning branch, no `added_kv`). Module keys mirror the reference (69 at tiny); the 5-D
  `Conv3d` weight loads transposed to NDHWC. Reference parity against diffusers on the first numeric
  run (`run_reference.py wan`, tiny random config, `ltx` env): full velocity cosine
  **0.9999999999999767**.
  `NFKMLXWanVideoVAENet` is the Wan 3D causal VAE (`AutoencoderKLWan`, the Wan 2.2 residual path),
  the last stage of the Wan pipeline and the hardest port of this batch. It compresses a video 4× in
  time and 16× in space. Unlike the LTX VAE (a clean full-clip causal forward), it runs a stateful
  streaming loop: the encoder consumes frames in chunks (1, then 4 at a time) and the decoder emits one
  latent frame at a time, threading a per-convolution feature cache (`feat_cache`) that supplies each
  causal convolution's temporal context across chunk boundaries — the temporal up/downsampling happens
  Only on that cache path, so a one-shot forward would skip it. `NFKWanCache` holds the per-conv slots
  (persisting across chunks, the index reset per chunk) with the reference's `Rep` / `frames` / `empty`
  states; `wanCausal` runs the caching dance (borrow the previous chunk's last frame when a chunk has
  fewer than two). `NFKWanCausalConv3d` holds its `weight`/`bias` directly (the reference is an
  `nn.Conv3d` subclass) via the functional `conv3d`, zero-padding the time axis on the left only. The
  residual down/up blocks carry the cacheless `AvgDown3D` / `DupUp3D` reshape shortcuts; the temporal
  resample `time_conv` doubles the frame count by splitting its doubled channel and interleaving it as a
  new frame sub-axis. Reference parity against diffusers' AutoencoderKLWan at a tiny residual config
  on a 5-frame clip (`run_reference.py wan_vae`, `ltx` env): encoder moments cosine 0.9999999999999997,
  decode cosine 0.999999999999892. The one load-bearing trap was the patchify channel order: the
  reference packs the `patch²` spatial block into the channel as `(C, pw, ph)`, not `(C, ph, pw)` — the
  swapped order left the encoder at 0.998 and the decode at 0.26 (the per-stage seams were exact through
  the last up-block, which pinned the fault to the final unpatchify and, symmetrically, the conv_in
  patchify). The RMS `gamma` loads flattened from its `[C,1,1,1]` layout; the 5-D/4-D conv weights
  transpose to NDHWC/NHWC. The non-residual Wan 2.1 path is implemented too (`isResidual` config,
  `.wan21` / `.tiny21`): Wan 2.1 replaces the residual down/up blocks (AvgDown3D / DupUp3D shortcuts) with
  a flat down-block list and a halving upsampler (`NFKWanUpBlock`, whose `WanResample` defaults
  `upsample_out_dim` to `dim/2`, so an inner decoder stage's input is halved), drops the patchify, and
  uses 16 latent channels. The encoder/decoder branch on `isResidual` and hold the blocks as `[Module]`
  with a type-dispatched forward. Reference parity against diffusers' AutoencoderKLWan at a tiny
  non-residual config (`run_reference.py wan_vae_21`): latent cosine 0.9999999999999978, decode
  0.9999999999999261, with the 2.2 residual path still at parity.
  `NFKMLXWanPipeline` chains it end to end (DiT denoised over the flow schedule with classifier-free
  guidance → the 3D VAE decode, over the `[C,F,H,W]`↔`[1,F,H,W,C]` bridge and the release's per-channel
  latent mean/std). The caller supplies the umT5 text embedding (a T5-family encoder). The sampler is
  Wan's released `UniPCMultistepScheduler` (`NFKMLXUniPCScheduler`, at reference parity — see the
  scheduler entry below), not a stand-in. Validated by a weight-free tiny-config glue test plus the
  DiT/VAE parities. Wan's text encoder is umT5, now verified: `NFKMLXT5Encoder` gained a
  `perLayerBias` configuration (`.umt5XXL` / `.tinyUMT5`) — umT5 (`UMT5EncoderModel`) differs from plain
  T5 only in giving every layer its own relative-position bias (plain T5 shares block 0's across the
  stack); everything else (T5LayerNorm, the gated FFN, the unscaled attention) is the same code.
  Reference parity against transformers' UMT5EncoderModel at a tiny configuration (`run_reference.py
  umt5`, the `llm` env): text embedding cosine 0.9999999999999984, with the plain-T5 shared-bias path
  still at parity. The Wan text-to-video path is complete.
- `NFKMLXDPMSolverScheduler` / `NFKMLXUniPCScheduler` — the released multistep samplers SANA and Wan use,
  ported in their flow-prediction configurations. Both are value types with no weights, so both are
  verified exactly against diffusers with no downloads (`run_reference.py dpm_solver` / `unipc`, `ltx`
  env): a fixed velocity sequence is run through the reference and this port, and the whole sample
  trajectory is compared step by step (worst |difference| 1.7e-6 and 1.4e-6), with the sigma schedule
  exact. **`NFKMLXDPMSolverScheduler`** is DPM-Solver++ (`algorithm_type: dpmsolver++`, `solver_order: 2`,
  `solver_type: midpoint`, `final_sigmas_type: zero`): each step converts the flow velocity to a data
  prediction `x0` and takes a first- or second-order multistep update, the coefficients computed as
  `Float` scalars (so the `log 0` at the terminal zero sigma resolves to a clean `x0`) and applied to the
  `MLXArray`. **`NFKMLXUniPCScheduler`** is UniPC (`solver_order: 2`, `solver_type: bh2`, `predict_x0`):
  a predictor-corrector — from step 1 on it corrects the previous sample before predicting the next — with
  the order-2 corrector's 2×2 `B(h)` linear system solved in closed form. Both take their flow sigma ramp
  from the release's `flow_shift` (SANA 3.0, Wan 5.0) and truncate the flow timesteps to integers as
  diffusers does. The `NFKMLXFlowMatchScheduler.sana`/`.wan` presets remain for a caller who wants the
  plain rectified-flow sampler, but the pipelines now run the released multistep samplers.
- `NFKMLXSD3TransformerNet` — the Stable Diffusion 3 MMDiT (`SD3Transformer2DModel`, Stability AI), the
  sixth DiT family and the flagship of the SD3.x line. Dual-STREAM: the image latent tokens and the text
  tokens each carry their own query/key/value projections, feed-forward, and adaptive-norm modulation
  (`norm1`/`norm1_context`, `ff`/`ff_context`), while attention runs jointly over the concatenation
  (`JointAttnProcessor2_0`: the image adds `add_*_proj` text keys and values, concatenates `[image,
  text]` on the sequence, and splits back). This is a different design from Z-Image's single-stream,
  which shares weights per layer. The last block runs `context_pre_only` — the text stream contributes
  keys and values but drops its own output and feed-forward, since nothing downstream reads the text.
  Conditioning is `time_text_embed` (a sinusoidal timestep MLP plus the pooled-text projection, summed);
  the text sequence enters through `context_embedder` (4096 → inner). The patch embed is a stride-2
  convolution plus a sincos positional table precomputed at `pos_embed_max_size²` and CENTER-CROPPED to
  the latent grid (the table is a persistent buffer, loaded from the checkpoint and cropped). The final
  `AdaLayerNormContinuous` + `proj_out` unpatchify via `nhwpqc->nchpwq`. SD3.5 adds two things over
  SD3.0: RMS query/key normalization (`qk_norm`), and — on SD3.5-**medium** (MMDiT-X), not the large —
  Dual attention (`SD35AdaLayerNormZeroX` giving nine modulation chunks, a second image-only `attn2`
  gated in beside the joint one). Reference parity against diffusers at a tiny random configuration
  that exercises the dual attention, the RMS q/k norm, the `context_pre_only` last block, and the
  cropped positional table (patch seam 0.9999999999999942, velocity cosine 0.9999999999999865). Two
  facts were load-bearing: `SD35AdaLayerNormZeroX` appends `(shift_msa2, scale_msa2, gate_msa2)` after
  the six (indices 6/7/8, not 8/6/7 — a moderate error localized to the dual layer); and the timestep /
  pooled-text embedders name their linears `linear_1` / `linear_2` (real submodules), not a
  `nn.Sequential`. Module keys mirror the release exactly, so `loadWeights` transposes only the 4-D
  patch-embed convolution to NHWC. `configuration(fromHuggingFace:)` reads `transformer/config.json`;
  presets `.sd3Medium` (2B, 24 layers, no qk-norm), `.sd35Medium` (2.5B, 24 layers, RMS qk-norm, dual
  attention 0…12, `pos_embed_max_size` 384), `.sd35Large` (8B, 38 layers × 38 heads, RMS qk-norm, no
  dual attention). Held to the released headers by shape: SD3.5-large 1227 tensors, SD3.5-medium 909
  (the dual-attention path), each 0 missing, 0 mismatched, 0 unaccounted. Oracle `run_sd3`, `IK_PARITY_SD3`.
- `NFKMLXFluxTransformerNet` — the FLUX.1 transformer (`FluxTransformer2DModel`, Black Forest Labs), the
  seventh DiT family. Two block kinds. The double-stream blocks (`transformer_blocks`) are MMDiT
  joint-attention blocks like SD3's, but concatenate `[text, image]` (text first, the opposite of SD3)
  and carry RMS q/k norm and an axial rotary. The single-stream blocks (`single_transformer_blocks`)
  concatenate the two streams and run a parallel attention-and-MLP over the join under one adaptive-norm
  gate (`AdaLayerNormZeroSingle`, three chunks): `proj_out([attention ‖ act(proj_mlp)])`, gated into the
  residual, then split back. Position is a 3-axis rotary (`FluxPosEmbed`, `get_1d_rotary_pos_embed` with
  `repeat_interleave_real`, adjacent-pair rotation) over the token ids — text ids all zero, image ids
  the `(0, row, col)` grid — whose axes must sum to the head dimension. Conditioning is
  `time_text_embed`: the timestep, an optional guidance embedding (the guidance-distilled `[dev]`
  carries one, `[schnell]` does not), and the pooled CLIP-L projection, summed; the timestep and
  guidance are scaled by 1000 inside the forward. FLUX operates on a packed latent (`x_embedder`:
  64 → inner), and `proj_out` returns the packed velocity (the pipeline unpacks). FLUX conditions on the
  T5-XXL sequence and the CLIP-L pooled embedding (no CLIP-G, no CLIP sequence). Reference parity
  against diffusers at a tiny random configuration covering both block kinds, the guidance embedding,
  the axial rotary, and the `[text, image]` order (velocity cosine 0.9999999999998679, first Swift run).
  Reuses the SD3 shared helpers (`NFKSD3AdaLinear`, `NFKSD3FeedForward`, `NFKSD3MLP`,
  `sd3AffineFreeLayerNorm`, `sd3TimestepEmbedding`). Every weight is at most 2-D, so `loadWeights` needs
  no transpose. `configuration(fromHuggingFace:)`; presets `.dev` / `.schnell` (both 19 double + 38
  single blocks, 24 heads × 128). Held to the released headers by shape: FLUX.1 [schnell] 1156
  tensors, FLUX.1 [dev] 1160 (with the guidance embedder), each 0 missing / mismatched / unaccounted.
  Oracle `run_flux`, `IK_PARITY_FLUX`.
- `NFKMLXSD3Pipeline` / `NFKMLXFluxPipeline` — the SD3 and FLUX text-to-image pipeline glue, chaining
  the transformer (denoised over the rectified-flow schedule) and the autoencoder. The caller supplies
  the joint text embedding and the pooled projection (SD3: T5 + the two CLIP sequences on the sequence
  axis, the two CLIP pooled on the channel axis; FLUX: T5 sequence + CLIP-L pooled), as the SD pipeline
  takes a text context. The autoencoder is the shared `NFKMLXSDAutoencoder`: SD3's VAE keeps the
  quant convolutions (`scaling` 1.5305, `shift` 0.0609), FLUX's drops them (`scaling` 0.3611, `shift`
  0.1159, the Z-Image `.flux` VAE). SD3 runs classifier-free guidance; FLUX takes the guidance embedding
  (`[dev]`) or none (`[schnell]`) with no CFG. FLUX packs the latent — each 2×2 spatial block folds
  into the channel axis (`pack` / `unpack`, round-tripped in a test), and `imageIds(height:width:)` is
  the `(0, row, col)` grid. Both step `sample + (σ_next − σ)·velocity` with no negation (Z-Image negated
  its convention). `NFKMLXFlowMatchScheduler` gained `.sd3` (static shift 3.0), `.flux` (dynamic shift),
  and `.fluxSchnell` (static shift 1.0). Validated by weight-free glue tests on matching tiny
  configurations; a sampled image is not bitwise-comparable, as with the other DiT pipelines.
- `NFKMLXSD3ControlNetNet` / `NFKMLXSD3ControlNetPipeline` — the Stable Diffusion 3 ControlNet
  (`SD3ControlNetModel`, Stability AI / InstantX), a partial copy of the MMDiT that steers a generation
  with a spatial control image (Canny, depth, pose, blur, tile). It runs the first N joint blocks over
  the noisy latent plus a control latent and emits one zero-initialized residual per block; the base
  `NFKMLXSD3TransformerNet` adds them into its own non-`context_pre_only` blocks, strided over the
  residual list by the reference's `interval_control` (`blockControlnetHiddenStates`, threaded back into
  the base forward, nil for plain text-to-image so the base is byte-identical). The control latent is
  VAE-encoded and patch-embedded through a zero-initialized `pos_embed_input` that carries no positional
  table, then added to the noisy latent before the blocks. Two released shapes, both ported and
  configurable: the InstantX SD3-medium / SD3.5 ControlNets carry a `context_embedder` and reuse the
  full dual-stream `NFKSD3JointBlock` (all `context_pre_only=false`); Stability's official SD3.5-large 8B
  ControlNets (Blur, Canny, Depth) drop the position embedding and the context embedder and run
  single-stream `NFKSD3SingleBlock`s (`AdaLayerNormZero`, no qk-norm, no text stream) over image tokens
  the base transformer's `pos_embed` supplies (`usePosEmbed` / `useContextEmbedder`; `extraConditioningChannels`
  widens the control patch embed). Reference parity against diffusers at tiny random configurations
  (`run_reference.py sd3_controlnet` / `sd3_controlnet_single`, `IK_PARITY_SD3_CONTROLNET` /
  `_SINGLE`, the `ltx` oracle env): the dual-stream per-block residuals 0.9999999999999958 /
  0.9999999999999937 and the base transformer with them injected (a four-block base, two residuals, so
  the striding is exercised) 0.9999999999999859; the single-stream residuals 0.9999999999999984 /
  0.9999999999999982. `configuration(fromHuggingFace:)` reads `config.json` (a `joint_attention_dim`
  present selects the dual-stream shape). The pipeline runs the ControlNet once per CFG branch (its
  residuals depend on the text stream) and offsets each base pass by its own residuals. Only the 4-D
  patch-embed convolutions transpose to NHWC.
- `NFKMLXFluxControlNetNet` / `NFKMLXFluxControlNetPipeline` — the FLUX.1 ControlNet
  (`FluxControlNetModel`, Black Forest Labs / InstantX / Shakker-Labs), a partial copy of the FLUX
  transformer. It runs a few double-stream and single-stream blocks over the packed noisy latent plus a
  packed control latent (added through the zero-initialized `controlnet_x_embedder`) and emits a
  zero-initialized residual per block — a `controlnet_blocks` list for the double blocks and a
  `controlnet_single_blocks` list for the single. The base `NFKMLXFluxTransformerNet` adds each into its
  own two block stacks (`controlnetBlockSamples` / `controlnetSingleBlockSamples`), strided by the
  reference's **`ceil` interval** (distinct from SD3's non-ceil rule), nil by default so the base is
  byte-identical. The **union** variant (`numMode`) prepends a learned control-type embedding to the
  text sequence and one txt-id row, so one ControlNet serves several control types. The
  `input_hint_block` shape (`NFKFluxControlNetHintEmbedding`, the reference `ControlNetConditioningEmbedding`:
  a `conv_in`, three stride-2 downsampling stages over a `(16, 16, 16, 16)` channel pyramid, a zero-init
  `conv_out`, SiLU between every convolution) is built: instead of a packed VAE control latent it takes a
  Full-resolution control image `[B, C, H·8, W·8]`, downsamples it 8× to the packed grid, and flattens to
  the control tokens; `conditioningEmbeddingChannels` selects it, and the pipeline feeds the raw NCHW
  image rather than a VAE latent when it is present. Reference parity against diffusers end to end
  (`run_reference.py flux_controlnet` / `flux_controlnet_hint`, `IK_PARITY_FLUX_CONTROLNET` / `_HINT`,
  the `ltx` oracle env): the double-block residuals 0.9999999999999967, the single-block residuals
  0.9999999999999962 / 0.9999999999999942, and the base transformer with both injected (a three-block
  base, two residuals of each kind) 0.9999999999999251; the hint variant's residuals ≥ 0.9999999999999913
  and injected velocity 0.9999999999999771. FLUX has no CFG (the guidance embedding), so the ControlNet
  runs once per step. Only the `input_hint_block`'s 4-D convolutions transpose to NHWC.
- `NFKMLXRVM` (`@objc`) — real video matting (Robust Video Matting): the reference `MattingNetwork` —
  a torchvision **MobileNetV3-Large** encoder (inverted residuals with squeeze-and-excitation,
  hardswish, **BatchNorm epsilon 1e-3**, the last stage dilated), the reference LR-ASPP, and a
  recurrent decoder whose ConvGRU runs on half of each stage's channels (one fused gate
  convolution emits reset then update), threading four hidden states across frames. Run through
  `NFKMLXMattingBackend` (single frame) or `NFKMLXRVMNet.forward` (video, state threaded; a
  `downsampleRatio` below one runs the network on a reduced frame and lifts the result through the
  deep guided filter refiner — the reference's high-resolution recipe). The foreground head
  predicts a residual added to the source, and the alpha is a clamp, not a sigmoid. `+register` under
  `robust-video-matting`. `remapReferenceKey` maps the positional `backbone.features.N.block.M`
  Sequentials (what position `M` holds depends on each block's expand/SE shape) and every other
  positional Sequential — module keys are semantic because MLX's `update(parameters:)` parses a
  numeric key as an array index (see the MLX-runtime gotchas). Reference parity against PeterL1n's
  own MattingNetwork on the released `rvm_mobilenetv3` (alpha cosine 0.9999999999999786, foreground
  0.9999999999996709, guided-filter pass at ratio 0.5: 0.9999999999998509); the parity plate must
  produce a **non-degenerate alpha** — the network returns an all-zero matte on a synthetic ellipse,
  and a cosine of zero vectors measures nothing (`run_reference.py --image`). A photograph of a real
  subject is what satisfies that; the shipped plate is an animal rather than a person, and its
  reference alpha reaches full opacity over about 8% of the frame, so the recorded numbers are a real
  measurement. An earlier note said the plate must contain a person, which is stricter than the
  requirement. Forward, recurrent-state carry,
  guided-filter shape, remap, and round-trip tested. The fine-tune question is answered by
  measurement: `testAFineTuneMovesTheSqueezeExciteAndHardswishBlocks` trains the tiny configuration —
  which carries every block form — and asserts the loss falls and the squeeze-excitation parameters
  move, because a decreasing loss alone could ride on the decoder while the backbone stays frozen.
  The ResNet-50 release is at parity too (`.resNet50`, `NFKMLXRVMVariant`, registered as
  `robust-video-matting-resnet50`): the shared `NFKMLXResNetBackbone` with its last stage dilated,
  tapped after the stem's ReLU, stage 1, and stage 2 (`taps(_:)`), LR-ASPP 2048 → 256, decoder
  `[128, 64, 32, 16]`, and no ImageNet normalization on this encoder — the reference applies it to
  MobileNetV3 alone. `encode(_:)` dispatches on the backbone; the loader remaps `backbone.` through the
  ResNet's own `remapReferenceKey` and drops `num_batches_tracked`. Alpha 0.99999999995280,
  foreground 0.99999999999967, guided-filter refine at 0.5: 0.99999999997671.
- `NFKMLXRoPEScaling` — the rotary frequency scaling a release declares (`rope_scaling`), shared by the
  dense decoder and DeepSeek. At reference parity against `transformers`' own `ROPE_INIT_FUNCTIONS`
  for `linear` and `yarn` across five configurations (worst relative frequency difference < 1e-5),
  driven by `run_reference.py rope_scaling` — which needs no weights, since the scaling is a function
  of the rotary geometry and the config alone. A kind this does not implement (`dynamic`, `llama3`,
  `longrope`) is refused rather than approximated: all three appear in released configs, all compute
  different frequencies, and loading one under the wrong rotary runs and is wrong.
  YaRN's blend runs the opposite way to the intuitive guess, and the first draft here had it
  backwards: the fast channels are left unscaled and the slow ones are interpolated. A fast channel
  completes many turns inside the trained window, so it encodes local offset and a longer sequence does
  not change its meaning; a slow channel does not complete a turn even at the trained length, so past
  that length it reaches angles the model never saw. The parity record caught the error — in the prose
  and the assertions, not in the arithmetic, because the formula was ported rather than reasoned out.
  The attention factor is `0.1·ln(factor) + 1` unless the config states one, and it multiplies the
  queries and the keys alike, so a score carries its square.
- `NFKMLXLanguage` / `NFKMLXLanguageBackend` — on-device **text generation** through MLX, which the
  package had no path for: the core runs a Core ML language model and the Foundation Models companion
  wraps Apple's, and nothing here ran a Qwen or Llama. `NFKMLXLanguageNet` is the modern dense decoder
  — grouped-query attention with rotary embeddings, a SwiGLU feed-forward, RMS normalization
  throughout — which is what Qwen3 and Llama both are; they differ in a **configuration**, not in
  structure (`normalizesQueryAndKey` is Qwen3's per-head query/key norm, `attentionBias` is Qwen2's,
  `tiesWordEmbeddings` is the smaller sizes'). Module keys are the released checkpoint's names
  (`model.layers.N.self_attn.q_proj`), so a release loads with no remapping at all, and every weight is
  at most 2-D so none of the convolution transposes apply. `NFKMLXKeyValueCache` is what makes a token
  cost one step's work instead of the whole sequence's; `testACachedStepMatchesRecomputingThePrefix`
  is the assertion the generation path rests on. It holds its rows in a buffer that grows in blocks
  with a cursor at each end, rather than concatenating — concatenating copies the whole cache on every
  token, which turns decoding back into quadratic work in the one place that exists to avoid it.
  A `window` bounds it (`NFKMLXGenerationOptions.contextWindow`): the oldest positions are dropped,
  so memory stops growing with the conversation. The trim goes to `window - 1` Before the append, which
  is what lets a single-token step read exactly `window` positions and need no sliding mask at all.
  `offset` stays the absolute position count — a rotary angle depends on where a token is in the
  sequence, not where it sits in the buffer — while `maskCacheLength` is what a multi-token pass
  actually sees, and the mask is built against that: a mask sized to the offset would be wider than the
  keys it is applied to. The bound is off by default, because for a model whose attention is not
  natively windowed it is an approximation rather than a configuration — exact while the conversation
  fits inside the window, and dropping the beginning past that.
  The cache also quantizes (`NFKMLXGenerationOptions.cacheQuantization`, `NFKMLXKeyValueCache.Quantization`):
  keys and values are stored affine-packed (`quantized`/`dequantized`) beside per-group scales in the
  same block-growing buffers, and a step dequantizes the retained span for attention. It changes
  storage, not positions, so the offset/window/mask accounting is untouched — the float path is byte
  for byte the same, gated behind `guard let quantization`. It is lossy, so off by default; measured,
  8-bit decoding tracks the full-precision logits while shrinking the resident cache, which is what
  lets a long conversation reach further before the cache is the ceiling. `groupSize` must divide the
  head dimension. Measured on the released Qwen3-0.6B against the reference record
  (`testTheCacheQuantizationBitWidthsAgainstTheQwen3Record`, a token-by-token decode through the packed
  cache so every key and value is read back packed): 8-bit/64 last-logit cosine 0.99956, same argmax,
  the greedy continuation 16/16 with the reference — and **4-bit per-token collapses**: 0.577 at group
  64, 0.928 at group 32, a different argmax, 0/16 of the continuation. The axis is the cause, not the
  bit width (`testTheKeyValueQuantizationAxisDiagnostic`, 4-bit relative reconstruction error over 28
  layers and 132 positions): the keys lose 0.133 per token at g64 against 0.045 per channel (grouped
  along the sequence, the KIVI axis; worst layer 0.155 → 0.043), while the values barely care (0.101 →
  0.088). Qwen3's keys carry per-channel outliers that a per-token group has to span. So the cache
  now stores keys per channel by default (`Quantization.keyAxis`, `.sequence`; `.headDimension` is
  the old per-position layout; ObjC `cacheQuantizationPerChannelKeys`, default true): each channel's
  keys over `groupSize` consecutive positions share one scale, packed as `[B, H, D, groups · words]`
  in a buffer that grows along the group axis, with the positions of the unfinished group held in a
  full-precision residual `[B, H, r, D]` — so a prompt shorter than a group costs no precision at all.
  A window drops whole groups by moving the group cursor and a partial group by a `skip` count on the
  first retained one; a rollback returns to the residual first, then whole groups, and a group cut
  part way is dequantized back into the residual (lossy by construction, held to a cosine in the tests).
  Export/restore carry `key_groups` / `key_group_scales` / `key_group_biases` / `key_residual` /
  `key_skip`, and the prompt cache's metadata gains a `:sequence` suffix, so an older per-position
  file still loads as what it was. Values stay per position. Measured end to end on the 132-token
  prompt (`testThePerChannelKeyCacheOnQwen3`, one prefill then a 24-token greedy continuation
  against the float cache): 8-bit per-channel keys reproduce the float cache — last-logit cosine
  0.99997, continuation 24/24 — where 8-bit per-position keys read 0.994 and 1/24; 4-bit per-channel
  reads 0.994 at group 64 and 0.997 at group 32 against 0.68 / 0.97 per position (continuations 9/24
  and 1/24 — a greedy continuation compounds a near-tie flip, so the cosine is the stable reading).
  So 8-bit is now near-lossless and 4-bit per-channel is usable where 4-bit per-position was not.
  MLX packs groups of 32, 64, or 128 only — a group of 16 aborts the process at the first append, and
  the truncated xcodebuild run still printed "0 failures" (read the exit code). **Prefill chunks** (`prefillChunkSize`): a long prompt runs through
  the cache in slices, so the attention peak is bounded by the chunk rather than the prompt. It is
  Exact — each chunk attends through the cache to exactly the keys a single pass would — pinned by
  `testChunkedPrefillMatchesASinglePass`. The backend applies a chat template on request
  (`chatTemplate: .chatML`): an instruct release is trained on `<|im_start|>role … <|im_end|>` turns,
  and the old message-flattening prompted it outside that format; `.chatML` renders the turns with the
  release's own special tokens (which the byte-level tokenizer resolves), off by default because a base
  model wants the plain text. **A fit-before-load predicate** (`NFKMLXReleaseWeights.verifyFits`)
  refuses a release whose weights exceed the memory budget before materializing any, so a load that
  would kill the process becomes an error naming the shortfall; it is wired into the dense loader,
  where weights are ~all the file (Gemma and the hybrid load a subtree, so the file size over-counts,
  and they are left to a model-aware check).
  `NFKMLXWeights.apply` gained an opt-in `verifyShapes` (default off): a parameter supplied at a
  shape the module does not expect is normally adopted wholesale by `update(parameters:)` — the
  checkpoint loads clean and computes wrong numbers. Checking shapes turns that into a load-time error,
  but only where every built shape already equals the checkpoint's, which is the dense decoder alone.
  Shape adoption is load-bearing in several builders — Conv-TasNet's placeholder `.base` widths and
  Gemma E4B's feed-forward-doubling heuristic both load right only because adoption reshapes the module
  to the checkpoint — so a global shape check false-positives (both were caught turning it on), and it
  stays scoped to the Qwen3 dense loader where config.json widths are exact. Per-layer weight streaming
  (making a 27B that does not fit run, rather than fail cleanly) needs lazy module weights and is a
  larger separate change; the fit predicate delivers the clean-error half. All of these reach Objective-C: an ObjC consumer builds the LLM through the `@objc` `NFKMLXLanguage.backendWithDirectoryURL:error:` (reads config.json + tokenizer + shards) and sets every generation option per request through the `NFKMLXGenerationParameterKey` string constants (`contextWindow`, `cacheQuantizationBits`/`GroupSize`, `prefillChunkSize`, `chatTemplate`), the same mechanism the core `NFKParameter*` keys use — parity with the Swift `NFKMLXGenerationOptions` struct. The struct-taking factories stay Swift-only because `NFKMLXLanguageConfiguration` is Swift-only (the directory factory reads config.json instead), and the cache class stays Swift-only because it takes `MLXArray`; the config knob is what bridges, per the package's expose-what-bridges rule.
  Gemma 4, the hybrid, and DeepSeek do not use this cache: they take no cache argument at all and
  run prefill-only, so the window reaches the dense decoder alone. Sampling is greedy at temperature 0, otherwise
  temperature with optional nucleus (`topP`) and a seed for repeatability. The backend reads
  `NFKInputPrompt` / `NFKInputMessages` → `NFKOutputText` and honors the core's temperature, top-p,
  max-tokens, and seed parameters. Reference parity against transformers' own `Qwen3ForCausalLM` on
  the released Qwen3-0.6B: prefill logit cosine 0.9999999999943 with the same argmax at every
  position, and greedy generation reproducing the reference's continuation **token for token** —
  which is what proves the cache and the rotary offsets, since a single forward pass does not exercise
  them. A tied release still ships `lm_head.weight`; in Qwen3-0.6B it is **byte-identical** to
  `model.embed_tokens.weight` (verified, not assumed), so the loader drops the duplicate.
  Qwen3.5 and 3.6 are not this architecture — they are `Qwen3_5ForConditionalGeneration`,
  multimodal, interleaving `linear_attention` layers with full attention every fourth layer and gating
  the attention output. `configuration(fromHuggingFace:)` rejects them, and any mixture-of-experts
  config, rather than loading their weights into a dense stack and producing fluent nonsense. The
  oracle needs transformers >= 4.51, which is newer than the vision oracles run under, so it has its
  own interpreter recorded in the manifest's `oracle_environments`.
  1.7B and 4B are at parity too (logit cosine 0.9999999999975 and 0.999999999987, each reproducing
  the reference's greedy continuation token for token), which is what shows the family scales by
  configuration. 14B and 32B are presets (`.qwen3_14B`: 5120 / 40 layers / 40 heads / 8 kv /
  head 128 / 17408, untied; `.qwen3_32B`: 5120 / 64 / 64 / 8 / 128 / 25600, untied), each held to its
  released headers by shape (443 / 707 tensors, 0 missing / mismatched / unaccounted), as are
  Qwen3-Embedding-4B and -8B (398 each). Both are **sharded**: every release above 0.6B splits its weights across files with a
  `model.safetensors.index.json` naming which shard holds each tensor, so a loader reading only
  `model.safetensors` covers the smallest model and nothing else. 4B is the largest size this machine
  holds at float32 — about 16 GB of weights on each side, measured, with the oracle and the test run as
  separate processes.
  Prompt cache, speculative decoding, mixture of experts, constrained decoding (all 0.3.0,
  each measured). `NFKMLXKeyValueCache.rollback(by:)` moves the end cursors back and copies nothing;
  it returns false where a window has dropped what it would reach. `NFKMLXPromptCache` keeps the
  cache and its token ids between generations, `align(to:)` rolls back to the shared prefix (capped
  one short of the prompt so a token runs and produces logits), and `save(to:)`/`load(from:)` persist
  it — float or packed rows alike. `reusesPromptCache` makes the backend keep one; the backend
  serializes generation through a lock because two runs through one cache interleave their rows.
  **Speculative decoding** (`generate(prompt:options:draft:promptCache:report:onToken:)`,
  `backend(directoryURL:draftDirectoryURL:)`, ObjC `backendWithDirectoryURL:draftDirectoryURL:error:`,
  request key `draftTokens`) verifies `[next] + proposals` in one cached pass, keeps the leading
  agreeing run, rolls both caches back by the rejected count — through the prompt cache when one is
  present, or the KV cache is rolled back twice — and is greedy-exact by construction; above
  temperature 0 it is the standard rejection scheme. Measured on Qwen3-1.7B←0.6B at float32:
  token-identical, 73.5% acceptance, 1.01× wall clock — a 28-layer step here is launch-bound, so
  the draft costs nearly a target step. The bandwidth-bound case is measured too and does not pay
  (`testSpeculativeDecodingPaysOnABandwidthBoundTarget`, Qwen3-4B bf16 ← 0.6B bf16, warmed up, best of
  two): plain 26.3 tok/s, speculative 14.9 tok/s, **0.57×**, acceptance 0.435. At bf16 the two runs
  can part at a near-tie — at token 39 the target's own top two were the two divergent tokens, margin
  0.125 — because the batched verification pass and the single-token pass round differently; greedy
  exactness holds at float32 and up to that rounding at bf16, which the test asserts rather than
  assumes (an earlier 1.44× reading was a warm-up artifact: the plain run went first and paid the
  kernel compilation).
  **The routed feed-forward** (`NFKLMMixtureFeedForward`: a router `gate`, experts stacked as one
  `[E, out, in]` tensor per projection in `NFKLMSwitchLinear`, dispatched through `gatherMM` /
  `gatherQuantizedMM`; `NFKLMQuantizedSwitchLinear` conforms to `Quantized` so `save` records it and
  `matchStructure` rebuilds it) reads `qwen3_moe` (`num_experts`, `moe_intermediate_size`,
  `norm_topk_prob`; dense-interleaved layers refused) and `mixtral` (`num_local_experts`,
  `intermediate_size`, always renormalized; a sliding window refused). Softmax over all experts then
  renormalizing the selected is Mixtral's softmax over the selected, so one implementation serves
  both. `moduleKey(forRelease:)` maps `block_sparse_moe.experts.N.w1/w3/w2` onto
  `mlp.experts.N.gate_proj/up_proj/down_proj`, and `stackingExperts` stacks the per-expert tensors in
  index order. Reference parity against transformers' own Qwen3MoeForCausalLM and
  MixtralForCausalLM at tiny random configurations (`run_reference.py qwen3_moe` / `mixtral`,
  `IK_PARITY_QWEN3_MOE_TINY` / `IK_PARITY_MIXTRAL_TINY`): every hidden state exact layer by layer,
  logit cosine 0.99999999999999 / 0.9999999999999903, on the first numeric run. The released
  Qwen3-30B-A3B is accounted for by shape: `Tools/validation-assets/shapes.py` reads every shard's
  safetensors header by HTTP range request (config + 18,867 shapes, no weights), and
  `testEveryParameterMatchesTheReleasedQwen3MoeCheckpoint` consumes all 18,867 with 0 missing, 0
  mismatched, 0 unaccounted. The released sizes need quantized experts to fit 32 GB.
  Qwen2-MoE is read too (`qwen2_moe`: Qwen1.5-MoE-A2.7B, Qwen2-57B-A14B): the same routed
  feed-forward plus a shared expert every token runs — a dense SwiGLU of
  `shared_expert_intermediate_size` gated by `sigmoid(shared_expert_gate(x))`, summed with the routed
  output (`NFKLMMixtureFeedForward.sharedExpert` / `sharedExpertGate`, present only when the width is
  set so the other families' strict loads stay strict). Its releases leave `norm_topk_prob` False
  (the default the reader applies for this type) and carry query/key/value biases spelled `qkv_bias`,
  absent from the released config because true is its default — the reader now defaults the bias to
  true for `qwen2` and `qwen2_moe`, which also means a dense Qwen2 release loads where the old
  `attention_bias ?? false` default had refused its bias tensors. Reference parity against
  transformers' own Qwen2MoeForCausalLM at a tiny configuration (`run_reference.py qwen2_moe`,
  `IK_PARITY_QWEN2_MOE_TINY`): every hidden state exact, logit cosine 0.9999999999999813, first
  numeric run.
  gpt-oss is read too (`gpt_oss`: gpt-oss-20b / 120b, Apache-2.0), the fourth expert family and
  the one that needed new mechanisms rather than a configuration. Four differences from the other
  mixtures, each a flag on the shared dense decoder so the other families are byte-identical:
  alternating sliding-window and full attention (`slidingWindows`, per layer from `layer_types`;
  a sliding layer keeps every key in the cache and masks the ones further back than its window
  through a banded additive mask built from absolute positions, so a single-token step against a
  cache longer than the window is bounded too, and the cache accounting is unchanged); a learned
  attention sink per head (`sinks`, one extra softmax logit that drains mass and contributes no
  value — the fused kernel has no slot for it, so `explicitAttention` writes the softmax out, spreading
  the kv heads to the query heads as `repeat_kv` does; the same explicit path serves the sliding
  layers); biases on every attention projection and the output projection (`outputProjectionBias`)
  and **on the router** (`routerBias`) — its softmax over the selected top-k logits is the
  renormalized form the module already computes; and fused, interleaved, clamped experts
  (`NFKLMFusedSwitchGLU`, `clampedSwiGLU`): one `gate_up_proj` whose even columns gate and odd
  columns lift, read back with a stride-2 slice, biases on both projections (gathered per chosen
  expert), the gate clamped above at 7 and the up clamped to ±7, `(up + 1) · gate · sigmoid(1.702 ·
  gate)`. The release stores `gate_up_proj` as `[E, hidden, 2·width]` (`x @ W`), which
  `releaseWeights` transposes to the switch linear's `[E, out, in]`, keeping the interleave so a saved
  checkpoint round-trips through the same loader; `router.` maps to the module's `gate.`. Its YaRN
  leaves the correction band fractional (`truncate: false`), now a field of `NFKMLXRoPEScaling`
  (`truncatesCorrectionRange`, default true). Reference parity against transformers' own
  GptOssForCausalLM at a tiny configuration (`run_reference.py gpt_oss`, `IK_PARITY_GPT_OSS_TINY`,
  eager attention forced since the fused kernels take no sink, a window of 4 over 8 tokens so the
  sliding layers see less than the full ones): every hidden state exact, logit cosine
  0.9999999999999721, first numeric run.
  The released experts are MXFP4 and stay packed. `*_blocks` (`uint8 [E, out, in/32, 16]`) viewed
  as little-endian `uint32` Are MLX's `mxfp4` words in its own element order — measured two ways:
  `testMXFP4PackingIsTheOpenComputeLayout` hand-decodes MLX's packing (element i in bits 4·(i mod 8)
  of word i/8, the sixteen e2m1 values, an e8m0 scale byte per 32 biased by 127) and matches
  `dequantized` exactly, and `run_reference.py gpt_oss_quant` range-fetches the first 64 rows of the
  released layer-0 `gate_up_proj` and decodes them through transformers' own
  `convert_moe_packed_tensors`, which MLX's decode of the same bytes matches at **worst |difference|
  0.0**. So `releaseWeights` maps `_blocks` → `.weight` (viewed) and `_scales` → `.scales`, and
  `installPackedExperts` swaps each fused projection for an `NFKLMQuantizedSwitchLinear(packed:…, mode:
  .mxfp4)` (that class now carries a stored `mode` and a prepacked init) before the strict apply, so
  the packed arrays land on matching structure instead of being adopted into a float layer; the
  mxfp4 `gatherQuantizedMM` runs them as they are. A release whose `quantization_config.quant_method`
  is `mxfp4` loads at `.checkpoint` precision (bf16 attention, embeddings, and head; packed experts),
  which is also what makes `verifyFits` count the bytes that will be resident — 13.8 GB, where the
  float32 doubling would have refused it. The checkpoint contract records the mode
  (`inferkit.quantization` = `bits:groupSize[:mode]`; a module mixing affine layers with MXFP4 experts
  records the affine geometry, and `matchStructure` rebuilds only affine structure — the packed
  experts are recognized by their `uint8` scales on load, an affine save's float scales being the
  tell). `testAnMXFP4ExpertModuleRoundTripsThroughTheCheckpoint` saves and reloads a packed module
  to identical logits. The tokenizer is o200k_harmony, shipped as `tokenizer.json` alone: the core
  `NFKByteLevelBPETokenizer` gained the `o200k` pre-tokenization (words split by their case pattern —
  lower-led or one-capital-led, each optionally led by one non-letter and followed by a
  case-insensitive contraction; digits in runs of at most three; a punctuation run absorbing trailing
  newlines or slashes), `releaseTokenizer(inDirectory:)` picks it by the `\p{Lu}\p{Lt}` classes in the
  release's `Split` regex and extracts the vocabulary and merges when no `vocab.json` exists
  (`byteLevelFiles(fromTokenizerJSON:)`, the extraction ModernBERT's loader now shares). Token-exact
  against the `tokenizers` library over seven strings (`testTheReleaseTokenizerAgreesWithTokenizers`),
  the harmony markers included; eos is `<|return|>`. The released 20B is held to the module by shape
  from its own local shard headers (`testEveryParameterMatchesTheReleasedGPTOSSCheckpoint`, the
  fused projections against their `_blocks`/`_scales` geometry: 459 released tensors consumed, 0
  missing, 0 mismatched, 0 unaccounted) and generates through the ordinary backend
  (`testGPTOSSGeneratesOnTheReleasedWeights`, `IK_VAL_GPT_OSS`): "The capital of France is" → " Paris."
  in 4.6 s for 12 tokens, the 20B resident at 13.8 GB with its experts packed. A config key
  registered by hand while `fetch.py` is running is lost: it loads `~/.inferkit-validation.json` at
  start and rewrites it at the end, so register keys before or after a fetch, never during. Not ported: the harmony
  chat template's tool-calling structure (a raw prompt or a caller-rendered template works), and
  gpt-oss-120b, the same architecture at 65 GB.
  **Constrained decoding** (`NFKMLXConstrainedDecoding.swift`): `NFKMLXVocabulary` holds every id's
  bytes, read through the core's new `NFKTokenizer.bytesForTokenId:`; `NFKMLXByteConstraint<State>`
  walks a grammar byte by byte and caches the admissible mask per state (the uncached cost is the
  vocabulary times a few bytes; a run revisits a handful of states); `NFKMLXJSONConstraint` is JSON
  syntax with `root` (`.container`/`.object`/`.array`/`.any`), `NFKMLXChoiceConstraint` a fixed set.
  `NFKMLXJSONSchemaConstraint` (`NFKMLXJSONSchemaConstraint.swift`, 0.4.0) is the schema grammar:
  `NFKMLXJSONSchema` compiles a JSON Schema dictionary into nodes (`type` as a name or a list,
  `properties`/`required`/`additionalProperties`, `items`/`minItems`/`maxItems`, `enum`/`const` as
  byte-matched compact serializations, `anyOf`/`oneOf`, `$ref` into `$defs`/`definitions` with
  recursion, an empty schema or `true` as `any`), and the constraint walks it byte by byte over the same
  engine. The state is a set of deterministic machines, so an `anyOf` forks one machine per alternative
  the byte can open and the survivors rejoin as the bytes decide; an `any` value pushes a frame that
  delegates to the free JSON grammar. Keys are matched as raw bytes against the unwritten properties
  (a 64-bit seen mask, so keys come in any order and a duplicate is refused), an object closes only
  once every `required` key is written, a comma is refused once every key is written and unlisted ones
  are forbidden, and a key that outgrows every property becomes an unlisted one where
  `additionalProperties` allows it. `integer` refuses the dot and the exponent. Keywords that only
  narrow content (`pattern`, `format`, `minimum`, `minLength`) are ignored; ones that change what is
  admissible (`allOf`, `not`, `if`, `patternProperties`) are refused at compile time, as is a
  `required` name not under `properties`. Wired through the core's own `NFKParameterJSONSchema`
  (the key the remote backends read, so a structured-output request is engine-agnostic): the backend
  compiles it per request, throws on a schema it cannot enforce rather than running unconstrained, and
  hands the parsed document back under `NFKOutputStructured` beside the text whenever JSON was asked for
  (schema or `outputFormat`) — never guessed from JSON-looking text. Measured live on Qwen3-0.6B
  (`testASchemaConstrainedRequestOnQwen3ConformsAndReturnsStructuredOutput`): a `{city, country,
  population: integer, landlocked?}` schema comes back with exactly those keys and types. The free
  grammar's byte helpers (`advanceNumber`, `isTerminal`, `word`, `isWhitespace`, …) are module-internal
  statics so both grammars share one spelling of JSON's lexical rules. Request keys:
  `outputFormat` (`"json"`/`"json-object"`/`"json-array"`), `choices`, and the core `NFKParameterJSONSchema`. Two traps, both measured
  on Qwen3-0.6B: JSON admits unbounded whitespace, and with its preamble forbidden the greedy
  model emitted 96 tokens of blank lines — `maximumWhitespaceRun` (8 bytes) caps the detour; and
  a thinking model wants its `<think>` block, which the grammar forbids, and the leftover mass
  gave `{}` — the prompt closes the block (`<think>\n\n</think>\n\n` after the assistant marker),
  as the release's own no-think template does. **Two latent defects fixed on the way:** the release
  path passed no special tokens to the tokenizer (they live in `tokenizer_config.json`'s
  `added_tokens_decoder`, not `vocab.json`), so a ChatML marker encoded as plain text — now
  `specialTokens(inDirectory:)` supplies them and the `eos_token`; and no end-of-sequence stop was
  ever set, so generation ran to `maxTokens` — the release's eos is now the default stop when a
  request names none (a behavior change, recorded in the changelog). `NFKMLXLanguageBackend` is now
  `@objc(NFKMLXLanguageBackend)` with `hasDraftModel`, `promptCacheLength`, `resetPromptCache`.
  `Tools/reference-parity/run_reference.py` must stay parseable by Python 3.9: the LLM oracle
  environment is 3.9, and a backslash inside an f-string expression (legal from 3.12, written for
  the music oracle) had made every mode there unrunnable — found the first time the qwen3_moe mode
  ran, fixed by hoisting the literal.
- `NFKMLXTextEmbedder` / `NFKMLXQwen3Embedding` / `NFKMLXTextEmbeddingBackend` — on-device **text
  embeddings**, the capability the package lacked: it embedded images (CLIP) with no path for text, so
  no semantic search, retrieval, clustering, or reranking over a consumer's corpus. A text embedder is
  the decoder-only model with its output projection removed — the post-final-norm hidden states pooled
  to one vector and L2-normalized — which is exactly the seam `NFKMLXLanguageNet.hiddenStates(fromEmbeddings:)`
  already exposes, so nothing about the transformer is re-implemented. Qwen3-Embedding-0.6B is the
  Qwen3-0.6B dense decoder this package already runs (`configuration(fromHuggingFace:)` reads its
  `Qwen3ForCausalLM`/`qwen3` config unchanged; geometry equals `.qwen3_0_6B` but for `vocab_size` 151669
  and the tie), pooled at the **last token** over an appended `<|endoftext|>` (id 151643, not the chat
  `eos_token` 151645) and L2-normalized. `NFKMLXTextEmbedderConfiguration` carries the pooling
  (`.lastToken`/`.mean`), the appended token, normalization, and the Matryoshka `dimensions` a leading
  slice is a usable embedding at. Reference parity against the model card's own transformers recipe
  (`AutoModel` last hidden state, last-token pool, `F.normalize`) on the released 0.6B weights: query
  embedding cosine 0.99999999999, document 0.99999999999, retrieval score 0.76456 reproduced to 1e-6 end
  to end (`run_reference.py qwen3_embedding`, the `llm` oracle interpreter). A separate
  tokenizer-agreement test reproduces the reference's ids from the shared text — the **`qwen2`
  pre-tokenization** is what makes them right, the same trap the music tokenizer hit.
  Two release facts are load-bearing. The tokenizer appends `<|endoftext|>` and its hidden state is
  what the pooling reads, so the append is the model's geometry rather than the tokenizer's; and the
  released checkpoint is the **base model** (`AutoModel`/`Qwen3Model`), so its keys carry no `model.`
  prefix and no `lm_head` — the decoder keeps the causal-LM layout, so `NFKMLXQwen3Embedding.loadWeights`
  prepends `model.` and drops the absent projection (the shared `NFKMLXLanguage.loadedRelease` loader
  expects the prefix and would reject it, which is exactly what the first parity run reported). ObjC
  reaches it through `backendWithDirectoryURL:error:` and `backendWithDirectoryURL:outputDimensions:error:`
  (0 = full width); the `*Configuration` structs stay Swift-only, per the parity rule, and a Swift
  caller with token ids reads them through `embedding(forTokens:)`. `NFKMLXLanguage.releaseTokenizer(inDirectory:)`
  was extracted from `loadedRelease` so the embedder and the tokenizer-agreement test build the release
  tokenizer without loading the 1.2 GB of weights.
- `NFKMLXEmbeddingGemma` / `NFKMLXGemma3EncoderNet` / `NFKMLXGemmaTokenizer` — a second text embedder
  over a second architecture: EmbeddingGemma-300M, a bidirectional encoder where Qwen3-Embedding is a
  causal decoder. The backbone is the Gemma 3 text model (`gemma3_text`, `use_bidirectional_attention`),
  Not the causal Gemma 4 (`gemma4_text`) `NFKMLXGemmaLanguage` implements, so it is its own
  implementation `NFKMLXGemma3EncoderNet`: `(1 + w)` RMS normalization (Gemma 3; Gemma 4 uses `x · w`, the
  difference that first broke a Gemma port here), the sandwich norm (a norm before and after each of
  attention and the feed-forward), dual RoPE (local base 10000 for the sliding layers, global 1000000 for
  the full ones, the full head turned entirely — Gemma 3 carries no partial factor), per-head QK-norm
  before the rotary, a GeGLU `gelu_pytorch_tanh` feed-forward, the query scaled by `queryPreAttnScalar ^
  -0.5`, an embedding scaled by `√hidden`, no value norm, no per-layer embeddings, no softcapping. Every
  layer is bidirectional; a sliding layer sees a symmetric window (512), a full layer everything, and for
  an input shorter than the window (the common case) the mask is inert and every layer is full attention.
  The sentence-transformers head is mean pooling over every token → Dense 768→3072 → Dense 3072→768
  (both no-bias, Identity activation) → L2, with Matryoshka truncation before the final normalize.
  The Dense projections live in `2_Dense/`/`3_Dense/` subdirectories keyed `linear.weight`, which
  `NFKMLXReleaseWeights.files` does not read, so the loader takes them separately; the backbone's keys are
  the base-model checkpoint's (no `model.` prefix). Reference parity on the first numeric run against
  the sentence-transformers pipeline over transformers' own `Gemma3TextModel` (`run_reference.py
  embeddinggemma`, the `llm` oracle): every one of the 24 layers exact by the per-layer isolation harness,
  query and document embedding cosine 0.99999999999, retrieval score 0.60923 to 1e-7.
  The tokenizer is BPE, not unigram, and that was measured rather than assumed. Gemma's
  `tokenizer.model` scores are merge ranks (score ≈ −(id − constant)), so `NFKUnigramTokenizer`'s unigram
  Viterbi picks the wrong pieces (`ta`+`sk` over `task`) — caught by a tokenizer-agreement test before it
  reached anything else. `NFKMLXGemmaTokenizer` reads Gemma's byte-fallback BPE `tokenizer.json` Directly
  (no offline conversion): a metaspace normalizer (space → `▁`), the whole normalized string as one
  pre-token (the space split is a no-op after normalization), each character or its UTF-8 byte-fallback
  `<0xHH>` pieces, then the greedy merge-by-rank loop. It is neither the byte-level BPE the GPT-2/Qwen
  path uses nor unigram, so it is its own reader; token-for-token agreement with the reference is tested.
  The gated `google/embeddinggemma-300m` is mirrored ungated at `unsloth/embeddinggemma-300m` (the
  project uses mirrors for gated repos, as SD 2.1 does). ObjC reaches it through
  `backendWithDirectoryURL:error:` / `backendWithDirectoryURL:outputDimensions:error:` and the
  `query:`/`document:` prompt helpers; `NFKMLXTextEmbeddingBackend` now serves both embedders through the
  internal `NFKTextEmbedding` protocol and a tokenization closure (so a family with its own pooling,
  projection, and tokenizer plugs into one backend). The `*Configuration` structs stay Swift-only.
- `NFKMLXModernBERTReranker` / `NFKMLXModernBertRerankerNet` — a **cross-encoder reranker**, the third
  piece of the retrieval story after the two embedders. An embedder scores a query and a document
  independently and compares the vectors; a cross-encoder reads the pair together —
  `[CLS] query [SEP] document [SEP]` — through one bidirectional pass and predicts a single relevance
  logit, which is more accurate and is what reorders an embedder's shortlist. It is the released
  `gte-reranker-modernbert-base` (`ModernBertForSequenceClassification`). Because it takes a query and a
  List of documents rather than one input, it is a scoring object, not an `NFKInferenceBackend`:
  `scores(query:documents:)` / `rankedIndices(query:documents:)` (ObjC `scoresForQuery:documents:` /
  `rankedIndicesForQuery:documents:` / `scoreForQuery:document:`), built by `rerankerWithDirectoryURL:error:`.
  ModernBERT is a modernized BERT encoder: **RoPE** (a global base 160000 every third layer — `i %
  globalAttentionEvery == 0` — and a local base 10000 with a 128-token bidirectional sliding window
  elsewhere), a **GeGLU** feed-forward (`Wi` → input,gate; `gelu(input) * gate`; exact erf GELU, not
  tanh), LayerNorm throughout with no biases (`norm_bias` false), bias-free attention and MLP, no
  absolute position embeddings, and layer 0's `attn_norm` is the identity (the embeddings are
  pre-normed, so the checkpoint carries no weight for it and the module's is nil). The reranker head is
  mean pooling over the pair → a prediction head (dense + gelu + LayerNorm) → a single-logit classifier
  (with bias, though `classifier_bias` reads false — the checkpoint has `classifier.bias`, so trust the
  checkpoint). Module keys are the checkpoint's under `model.`/`head`/`classifier`, so nothing is
  remapped. Reference parity against transformers' own `ModernBertForSequenceClassification`
  (`run_reference.py modernbert_reranker`, the `llm` oracle): every one of the 22 layers exact by the
  per-layer isolation harness, and both the relevant and irrelevant scores to within 5e-3, plus the
  reranking order. Two things were measured, not assumed. The parity pair is deliberately long (95
  tokens) so it exceeds the 64-either-side local window and actually exercises the sliding layers — a
  short pair would leave a wrong window inert and pass silently. And ModernBERT's `output_hidden_states`
  does not apply the final norm to its last entry (the final norm goes only into `last_hidden_state`),
  unlike the Llama/Gemma convention, so `layerStates` returns the raw last layer output and the score test
  covers the final norm — the isolation reported a lone divergence at the last state until this was
  matched, while the score already agreed to 2e-6. The tokenizer is GPT-2-family byte-level BPE (not
  Gemma's char-BPE), which the core `NFKByteLevelBPETokenizer` reads; the release ships only
  `tokenizer.json`, so `byteLevelTokenizer(inDirectory:)` extracts its vocabulary and merges into the
  `vocab.json`/`merges.txt` the core reader takes (a temp directory), then wraps the pair in
  `[CLS]`/`[SEP]`; token-for-token agreement with the reference is tested.
- `NFKMLXSmolVLM` / `NFKMLXSigLIPNet` / `NFKMLXSmolVLMConnector` / `NFKMLXSmolVLMNet` — the package's
  **first vision-language model**, SmolVLM2-500M: an image and a question in, an answer out, which is
  the dominant 2026 on-device use (captioning, VQA, doc/screen understanding). Three parts: a **SigLIP
  vision encoder** (`NFKMLXSigLIPNet`: patch-embedding convolution + learned position embedding + 12
  bidirectional layers + post-norm; separate q/k/v/out projections with bias, gelu-tanh, LayerNorm eps
  1e-6), a **pixel-shuffle connector** (`NFKMLXSmolVLMConnector`: the Idefics3 pixel shuffle folds a
  4×4 patch neighborhood into one token at 16× the channels — 1024 patches per tile → 64 — then one
  bias-free `proj` to the decoder width), and a **Llama decoder** the dense `NFKMLXLanguageNet` already
  runs (`text_config` model_type llama, hidden 960, 32 layers, loaded from the checkpoint's
  `model.text_model.` subtree remapped onto `model.`). The **fusion** embeds the text, then splices the
  flattened projected vision tokens into the decoder's input embeddings at the `image_token_id` (49190)
  positions with a `where` over a gathered feature index (no per-row scatter), and the causal decoder
  reads the whole fused sequence. Reference parity against transformers' own
  `SmolVLMForConditionalGeneration` (`run_reference.py smolvlm`, the `llm` oracle — needs Pillow,
  torchvision, num2words), staged by the isolation harness: SigLIP embeddings 0.9999999999, layer 0
  0.9999999999, the full encoder 0.99999999997, the connector 0.9999999999, and the fused decoder
  logits predicting the reference's token at every one of the 1140 positions (last-position cosine
  0.9999999999) with the greedy continuation **token for token**. **Two bugs the isolation harness
  located, neither guessable:** SigLIP's position ids are not the row-major `0 … 1023` — the reference
  buckets each patch's fractional coordinate against `1/side … (side-1)/side` with a `1 - 1e-6` factor,
  so a full 32-patch row maps to `[0, 0, 1, …, 30]` (`positionIds` reproduces it, held as a Swift array
  so the stored constant does not enter `parameters()` — a stored `MLXArray` would, and the loader would
  then report it as an uncovered weight); and SmolVLM's `lm_head` is not tied to the embedding
  (byte-diff 0.85 despite the tied geometry), so the decoder loads its own `lm_head.weight` from the top
  level (`tiesWordEmbeddings: false`) rather than reusing the embedding — loading it tied scored logit
  cosine 0.83 and drifted after the sixth continuation token. **The consumer path:**
  `NFKMLXSmolVLMImageProcessor` tiles a `CGImage` the way SmolVLM does (longest edge scaled to 2048,
  split into `⌈h/512⌉ × ⌈w/512⌉` 512×512 sub-tiles plus a global 512×512 thumbnail appended last, each
  normalized to `-1 … 1`), and `NFKMLXSmolVLM.prompt(rows:cols:question:)` builds the expanded
  `User:` + per-tile `<fake_token_around_image><row_r_col_c>` + 64 `<image>` + `<global-img>` tile +
  question + `<end_of_utterance>\nAssistant:` string, which the byte-level BPE tokenizer (every added
  token registered as a special so the string segments on them) turns into ids **token-exactly** against
  the processor. The resize is CoreGraphics, not the reference's PIL LANCZOS, so a consumer caption is
  not token-identical to the reference — a documented approximation; the network is at parity on the
  reference's own pixel values, and end to end on the real Apollo-astronaut validation photo the model
  answers "a man is standing in front of a backdrop that resembles the moon. He is dressed in a white
  …". ObjC reaches it through `smolVLMWithDirectoryURL:error:` and `answerForImage:question:`.
- `NFKMLXQwen3VLVisionNet` / `NFKMLXQwen3VL` — the vision tower of a **second VLM**, Qwen3-VL-2B, and a
  second vision architecture beside SmolVLM's SigLIP. Qwen3-VL's encoder is a **2D-rotary ViT** whose
  patches are laid out in **2×2 merge blocks** (not row-major): a patch embedding (the reference's
  full-kernel `Conv3d` written as one `Linear` over the flattened `3·2·16·16` patch), a bilinearly
  interpolated position embedding (the learned 48×48 grid resampled to the image's grid, then
  reordered to merge-block order to line up with the patches), 24 blocks with 2D rotary and gelu-tanh
  MLP, a **merger** that folds each 2×2 block to `out_hidden` (2048), and a three-layer **deepstack**
  (feature maps from vision layers 5/11/17, each through its own merger with a post-shuffle norm). The
  2D rope pairs channel `i` with `i+32` over a `[row·freqs, col·freqs]` table; the merger groups four
  Consecutive patches, which is why the merge-block patch order is load-bearing. Reference parity on
  the first numeric run against transformers' own Qwen3-VL vision model (`run_reference.py qwen3vl`,
  the `llm` oracle — needs Pillow/torchvision): patch embedding 0.9999999999, position embedding
  0.9999999999, merged output 0.9999999997, and every one of the three deepstack features
  0.9999999999. The decoder is the Qwen3 dense stack `NFKMLXLanguageNet` already runs, loaded from the
  checkpoint's `model.language_model.` subtree. Qwen3-VL is much more than "reuses the Qwen3 decoder":
  the decoder adds interleaved M-RoPE (3D positions, `mrope_section [24,20,20]`) and deepstack injection
  at its first three layers, and `get_rope_index` computes the 3D positions from the token layout.
  That decoder integration is shipped, in the shared `NFKMLXLanguageNet` (the dense Qwen3, the
  embedders, SmolVLM, the music AR stage, and the Z-Image text step all reuse it, kept byte-identical
  through nil defaults): an opt-in M-RoPE (precomputed interleaved 3-D cos/sin in `NFKLMAttention`,
  rotate-half; the 64 frequency pairs interleave the T/H/W axes by `c % 3` for `c < 60`, then T, which is
  exactly `mrope_section [24, 20, 20]`) and deepstack injection (adding `features[i]` at the image-token
  positions after layers 0/1/2), both carried by an `NFKLMMultimodal` struct through
  `hiddenStates(fromEmbeddings:multimodal:)`. `NFKMLXQwen3VL` gained `decoder(directoryURL:)` (the
  `model.language_model.` subtree; Qwen3-1.7B geometry at `ropeTheta` 5e6), `ropePositionIds`
  (`get_rope_index`, single image), `mropeCosSin`, and `logits(...)`. Reference parity on the released
  4.25 GB Qwen3-VL-2B (`testQwen3VLDecoderMatchesTheReferenceOnReleasedWeights`, the recorded parity
  vision features fed in): logit cosine 0.99999999998, argmax 80/80, the first continuation token
  matching. The shared-decoder change was re-verified against the dense Qwen3 (0.99999999999), SmolVLM
  (argmax 1140/1140), and Qwen3-Embedding (0.99999999999) records. `NFKMLXQwen3VLImageProcessor` is the
  `smart_resize` + patchify input adapter (CoreGraphics resize, the documented approximation).
  The larger sizes are read from their own releases: `NFKMLXQwen3VLVisionConfiguration.configuration(fromHuggingFace:)`
  (model_type `qwen3_vl` or `qwen3_vl_moe`; the position grid's side is the square root of
  `num_position_embeddings`), `visionNet(directoryURL:)` (sharded through `NFKMLXReleaseWeights.arrays`),
  `decoderConfiguration(directoryURL:)` (the `text_config` through `NFKMLXLanguage.configuration(fromJSON:)`,
  tied unless the release ships a head), and `decoder(directoryURL:)`, which splits the 30B-A3B's fused
  `gate_up_proj [E, hidden, 2·inter]` into the module's `gate_proj` / `up_proj [E, inter, hidden]` and
  transposes `down_proj`. The 4B keeps the 2B's 24-block tower; the 8B, 32B, and 30B-A3B run the
  27-block one. Held to their released headers by shape, tower and decoder together: 713 / 750 /
  1058 / 930 tensors, 0 missing, 0 mismatched, 0 unaccounted.
- `NFKMLXHybridLanguage` — the hybrid decoder Qwen3.5, Qwen3.6, and **Qwen3.8** are built from
  (`Qwen3_5ForConditionalGeneration`), at reference parity on the released Qwen3.5-4B (logit
  cosine 0.9999999999962, every one of the 33 hidden states exact layer by layer). 4B is the smallest
  release of the family and the only one that fits here; Qwen3.8-27B is the same architecture at
  ~54 GB, so it is covered structurally (851 parameters, 0 mismatched) and its numerics rest on the
  4B measurement rather than on a run of its own.
  The per-layer isolation harness found three defects a shape check could not:
  the family normalizes with `x · (1 + w)` where the dense Qwen3 stack and Gemma 4 both scale by
  the weight directly — two conventions from the same vendor, indistinguishable by shape, different in
  every number (the gated norm inside the recurrence is the plain kind even here). The full-attention
  projection interleaves queries and gate per head: it is viewed as `[.., heads, 2·headDim]` and
  split on the last axis, so taking two contiguous halves of the flat width takes the wrong channels
  entirely. And the gate is applied as a plain `sigmoid`, despite the config field being named
  `output_gate_type: swish` — the implementation is what the weights were trained against. The
  recurrence itself also needed the query scaled by `1/√headDim` and the decay applied before reading
  the state rather than after.
  Three quarters of its layers replace attention with a **gated delta-rule recurrence** — a fixed-size
  state instead of a growing key-value cache, so cost is linear in sequence length — and every fourth
  layer is full attention whose output is gated. The shapes decode the design: `q_proj` is
  `[12288, 5120]` where 24 heads × 256 would be 6144, because the query projection also emits the
  output gate, which is applied as a plain sigmoid; `in_proj_qkv` is 10240 = two key streams of 16×128 plus a value stream of
  48×128; `A_log` and `dt_bias` are `[48]`, one decay and one step per value head; and
  `partial_rotary_factor` 0.25 turns only 64 of each head's 256 channels. Beside the 4B measurement,
  `NFKMLXHybridLanguageTests` checks every one of the 27B decoder's 851 parameters against that
  checkpoint's own safetensors headers, name by name and shape by shape — read with HTTP range
  requests, about a megabyte instead of 54 GB — with zero missing and zero mismatched, which is what
  carries the 4B result across to the size that cannot be run. The converse is asserted too, so the parts deliberately absent are named rather
  than overlooked: 333 tensors are the vision tower and 15 the multi-token-prediction head, and
  851 + 333 + 15 is the checkpoint's full 1199. A small configuration also runs end to end, and the
  recurrence is checked to be causal (appending tokens cannot change an earlier token's output).
  The one layout difference is the depthwise convolution: PyTorch stores `[channels, 1, kernel]` and
  MLX `[channels, kernel, 1]`, which the structural test compares as a loader would.
- `NFKMLXDeepSeek` — the DeepSeek V4 decoder (`DeepseekV4ForCausalLM`): Multi-head Latent Attention
  over a mixture of experts, a third architecture family beside the dense stack and the hybrid.
  Its arithmetic is measured — at a tiny all-sliding configuration against transformers' own
  plain-PyTorch implementation, which shipped after this port was written and is the third-party
  oracle DeepSeek's GPU-only inference code could not be (`run_reference.py deepseek_v4`,
  `IK_PARITY_DEEPSEEK_TINY`): every layer's hidden state ≥ 0.9999995 and the logits 0.9999999999,
  with the oracle saving its weights in the release naming so the module loads them strictly. With
  every layer sliding and the sequence shorter than the window, the reference degenerates to exactly
  the dense-with-sink path this port computes, so the measurement covers the MLA projections, the
  per-head query norm, the trailing interleaved rope, the sink softmax, the output de-rotation, the
  grouped output projection, both routers, the clamped SwiGLU experts, and the hyper-connections. The
  compressor and indexer stay outside it (no compressed window closes at that length). The released
  weights still cannot run here — the measurement is of the implementation, not the checkpoint.
  The measurement immediately found a wrong class the structural check could not: the head's
  collapse (`NFKDeepSeekHyperHead`) predicts only read gates, so `hc_head_fn` is `[copies, copies ×
  hidden]` — and the module had built the full block connection there, `[(2 + copies) × copies, …]`.
  The structural check compares declared shapes against the release and the declaration was right, so
  it passed while the module was wrong; MLX's `update(parameters:)` then adopts a checkpoint's shapes
  wholesale, so a real load would have crashed in the forward, not at load.
  The hybrid's checkpoint is bf16, so a float module's shapes match it exactly; this one is
  **quantized**: attention is fp8 with 128×128 block scales and a routed expert is 4-bit packed two
  to an int8 byte, so `w1` is stored `[2048, 2048]` where the float weight is `[2048, 4096]`. The
  structural check therefore derives what each float parameter looks like stored, and because that
  derivation is an assumption, `testTheQuantizedLayoutIsWhatTheReleaseUses` asserts it against the
  observed dtypes and shapes instead of trusting it.
  Implemented: low-rank queries (`wq_a` → norm → `wq_b`), one shared latent key-value per position
  (`wkv` → norm) which is what keeps the cache small, a grouped low-rank output (`wo_a` applied per
  group of heads, then `wo_b`), the learned per-head `attn_sink`, rotary on the head's trailing
  channels, and the mixture of experts — square-root-softplus scoring, a bias that steers selection
  without entering the weights, renormalize-then-scale by `routed_scaling_factor`, the clamped SwiGLU,
  and one shared expert every token passes through. The first `num_hash_layers` route by a
  `tid2eid` table indexed by token ID rather than by the hidden state, which changes which parameters
  those layers carry, so the boundary is asserted.
  The compressor and the sparse indexer are implemented. `NFKDeepSeekCompressor` pools
  `compressRatio` consecutive positions into one: each contributes a value (`wkv`) and a score
  (`wgate`), the scores softmax across the window so its positions compete, and `ape` is a learned
  per-slot bias so a position's weight depends on where it sits as well as on what it holds. At ratio
  4 the projections are twice as wide and a second, overlapping window is pooled alongside — shifted
  back by one window so a boundary is covered from both sides, with the first window's absent
  predecessor filled with zero values and −inf scores. `NFKDeepSeekIndexer` runs its own compressor,
  projects the layer's low-rank query into an index space, scores every compressed position, combines
  the heads by a learned weight, and keeps the best `index_topk` — masking any window not yet complete
  at the querying position, which the reference marks −1. The reference's Hadamard rotation is
  deliberately omitted: it is applied to both the query and the compressed keys before fp4
  quantization, and being orthogonal and shared it cancels in the dot product — it spreads information
  for quantization rather than changing the score, so omitting it and the quantization together gives
  the unquantized ranking the reference approximates. Prefill only; the incremental decode path keeps
  rolling state buffers a single forward pass never enters.
  Hyper-Connections are implemented, and finding them corrected a real mistake here. The `hc_*`
  parameters are not a hash-clustering head, which is what this file previously called them: the
  residual stream is `hc_mult` (4) parallel copies of the hidden state, so every block works on
  `[batch, length, 4, hidden]`. A block predicts its mixing weights per position from the copies
  themselves — `hc_*_fn` projects the flattened, RMS-normalized copies into a read weight per copy, a
  write weight per copy, and a copy-to-copy matrix that is softmaxed and then **Sinkhorn**-normalized
  toward doubly stochastic so the copies do not collapse into one another. The copies are reduced to
  one stream before attention and before the feed-forward, and expanded back after each; `hc_head_*`
  collapses them at the top.
  The structural check missed it entirely. It compared every parameter the
  port declares against the release and reported zero problems, because it only asked "does what I
  declare exist" and never "does everything in the release exist here". A whole mechanism sat in the
  checkpoint with no counterpart in the code. `testEveryReleasedTensorIsDeclaredOrNamed` now asserts
  the converse — 34223 declared, 38094 named as deliberately unimplemented, **0 unaccounted**, which is
  the checkpoint's full 72317. The Qwen3.8 and Gemma 4 checks always had that assertion; this one did
  not, which is exactly where the gap opened.
  The quantization the release is stored in is decoded, and this part is measured.
  `NFKMLXDeepSeekQuantization` dequantizes both formats — fp8 `e4m3` with 128×128 block scales for
  attention and the shared experts, and `e2m1` 4-bit packed two to a byte with 32-value block scales
  along the last axis for a routed expert — and `NFKMLXDeepSeek.dequantized(_:shapes:)` turns a shard's
  arrays into the float parameters a module holds. The scales are `e8m0`, an exponent with no sign and
  no mantissa, so a scale is exactly a power of two. Both decodes are **exact** against a reference
  built from real checkpoint bytes fetched by HTTP range request (`run_reference.py deepseek_quant`):
  torch 2.13 is the first here with `float8_e4m3fn` and `float8_e8m0fnu` on the CPU, and since it has
  no CPU kernel for `float4_e2m1fn_x2` at all, the 4-bit side decodes through `ml_dtypes`, the same
  format from a different vendor. So the checkpoint's storage is measured even though its arithmetic
  cannot be.
  Two details are load-bearing and only one of them is measurable. The 4-bit blocks run along the
  last axis, which the checkpoint's own values corroborate: the reference quantizer clamps a block to
  ±6 and rounds its scale to the power of two that puts the block's largest magnitude in `(3, 6]`, so
  under the right grouping every block lands in that range and under a wrong one about 1% do not.
  The nibble order is not measurable — a byte's pair decodes to the same two values either way and
  both stay inside one block, so no statistic separates them — so it follows the format's own
  convention (low nibble first) and a test pins it against a hand-encoded byte, the same treatment the
  rotary convention gets.
  Still not implemented, and named by `testEveryReleasedTensorIsDeclaredOrNamed` rather than merely
  absent: the multi-token-prediction and DSpark speculative-decoding stack, 4705 tensors. It serves
  speculative decoding, which needs a generation loop this port does not have, and no oracle here can
  run it, so it would be unmeasurable code serving an absent path. That test now accounts for the
  release exactly — 34223 declared, 33389 block scales each decoding a declared weight, 4705 named as
  unimplemented, **0 unaccounted**, which is the checkpoint's full 72317. Counting a scale as
  "unimplemented" was the weaker claim it used to make; it is now accounted for by the weight it
  decodes.
  The arithmetic measurement above post-dates the source audit below. When this port was written,
  DeepSeek's own `inference/model.py` was the only reference and it imports `sparse_attn` and the
  fp8/fp4 kernels from a GPU-only `tilelang` module — stubbing those would have made the oracle this
  port's own code, which proves nothing. The audit was therefore source-driven, applying the error
  classes the isolation harness exposed in the Qwen and Gemma decoders; the transformers oracle later
  confirmed its three findings numerically. Three were found and fixed by reading `model.py`:
  the rotary pairs adjacent channels (`view_as_complex`), where the other decoders here rotate
  halves — indistinguishable by shape, different in every value, so `NFKDeepSeekRotary` writes the
  convention out rather than selecting it with a flag; the attention output is DE-rotated on the
  way out, because the values share their latent with the keys, so the rotation has to be undone with
  the conjugate; and the learned per-head `attn_sink` was declared but never used — it is an extra
  logit that drains probability mass without contributing a value, which the fused attention call has
  nowhere to put, so the softmax is written out. Its normalization is the plain kind, which the port
  already had. Each is pinned by a test, since no measurement can catch them here. Verification enumerates the architecture
  **analytically** — 43 layers of 257 experts cannot be instantiated at float precision — and compares
  3975 parameters across the five layers whose headers were captured, with zero mismatches.
  DeepSeek V4 Pro (0813) gets the same treatment, and the enumeration generalized with no code
  change: 61 layers, hidden 7168, 384 experts, 128 heads, `q_lora_rank` 1536, `o_groups` 16 — 3540
  parameters compared across three captured layers with zero mismatches, and its 149782-tensor index
  accounted for exactly (71983 declared, 70790 block scales each decoding a declared weight, 7009
  MTP/DSpark, 0 unaccounted). Pro adds a YaRN `rope_scaling`, which carries no parameters and so is
  invisible to a structural check — the reason a run of Pro without it would be silently wrong rather
  than a load failure. It is implemented now, through the shared `NFKMLXRoPEScaling`, and the config
  parser reads it. Sources:
  DeepSeek ships `inference/model.py` in the release, which is what this was written from.
- `NFKMLXGemma3` / `NFKMLXGemma3Net` / `NFKMLXGemma3Model` / `NFKMLXGemma3Backend` — the Gemma 3 line,
  end to end (`gemma3_text` for the 270M and 1B, the multimodal `gemma3` for the 4B and up), at
  reference parity on the released weights against transformers' own Gemma 3 for every size and
  every stage, on the first numeric run. The decoder is the Gemma 3 block the EmbeddingGemma encoder
  already ran, now shared (`NFKGemma3Block`/`NFKGemma3Attention`/`NFKGemma3Norm` in
  `NFKMLXGemma3.swift`; the encoder is the same blocks under a bidirectional mask): `(1 + w)` RMS norms,
  the sandwich block, five sliding-window layers to one full layer (`layer_types`, or derived from
  `sliding_window_pattern`), the local rotary base 10000 on the sliding layers and the global 1e6 on the
  full ones, the 4B's `rope_scaling {linear, factor 8}` applied to the full layers only (`MLXFast.RoPE`
  with `scale: 1/factor`), per-head QK norm before the rotary, `query_pre_attn_scalar^-0.5`, GeGLU
  `gelu_tanh`, a tied head, both soft-caps supported (none released; an attention soft-cap runs the
  softmax explicitly). Generation runs through a hybrid cache (`NFKMLXGemma3Cache`: an unbounded
  `NFKMLXKeyValueCache` for the full layers and one bounded to the window for the sliding ones, the
  reference's hybrid cache), with the masks built per kind from absolute positions
  (`NFKMLXGemma3Masks`: a full layer causal, a sliding layer causal and `q − k < window` against the
  `min(offset, window − 1)` retained positions; a single cached step needs no mask). Measured: 270M
  logit cosine 0.99999999999368 and 1B 0.99999999999821 (every hidden state exact, argmax 6/6, the
  greedy continuation through the cache 12/12 token for token with the reference's own cached
  decode); a tiny record (`run_reference.py gemma3_tiny`, 4-position window over sliding/sliding/full,
  scaling 8, soft-caps 50/30) pins the cached step-by-step decode against a teacher-forced pass (worst
  step 1 − cosine 6.7e-8) and the continuation 6/6. **The multimodal 4B**: `NFKMLXGemma3VisionNet` is
  SigLIP so400m at 896×896 (27 layers, 1152 wide, patch 14 → 4096 patches) built from the shared
  SigLIP encoder with the SigLIP-2 row-major position embedding; `NFKMLXGemma3MultimodalProjector`
  average-pools the 64×64 patch grid in 4×4 cells to 256 soft tokens (a reshape-mean, row-major), a
  Gemma `(1 + w)` norm, then `x · W` with `mm_input_projection_weight` `[1152, 2560]` (a matmul, not a
  Linear). The prompt is the release's Jinja `chat_template.jinja` through `NFKMLXChatTemplateRenderer`
  (token-exact against `apply_chat_template`), the image spelled `<start_of_image>` in the last user
  turn and expanded in the text before tokenizing to `\n\n<start_of_image>` + 256 ×
  `<image_soft_token>` + `<end_of_image>\n\n` — the processor's own order, and load-bearing: `user\n` +
  `\n\n` tokenizes as one id (109), so expanding after tokenizing reads a different sentence; the whole
  processor prompt is reproduced id for id. An image's soft tokens attend to each other **bidirectionally**
  (the reference's `token_type_ids` blockwise rule: full layers `causal OR same-block`, sliding layers
  `window AND (causal OR same-block)`; `blockIds(for:)` numbers each soft-token run), the decode steps
  are plain causal as the reference's are. 4B numbers: text-only logit cosine 0.99999999999735 with every one of the 35 hidden states at
  1.0000000000 and the cached continuation 12/12; `gemma3_vision_real` patch embeddings
  0.99999999999953, layer 0 0.99999999999904, tower output 0.99999999875, the 256 projected soft tokens
  0.99999999971; `gemma3_conditional_real` the processor's 276-id prompt reproduced exactly, every
  layer ≥ 0.99999995, **argmax 275/275** over the fused sequence, last-16 logits 0.99999999565, and the
  greedy continuation token for token; on the real validation photograph through the CoreGraphics
  processor the 4B answers "The main subject of the image is a **puppy**." `NFKMLXGemma3Backend`
  reads `NFKInputPrompt` / `NFKInputMessages` (+ `NFKInputImage` on the 4B: `CGImage`, `CVPixelBuffer`,
  or texture through the image bridge, resized by CoreGraphics — the documented approximation of PIL
  bilinear), honors temperature / top-p / max-tokens / seed, streams each token through a submitted
  job's `partialResult` and cancels between tokens; `NFKMLXGemmaLanguage.backend(directoryURL:)` routes
  a Gemma 3 config here. ObjC: `[NFKMLXGemma3 backendWithDirectoryURL:error:]`,
  `gemma3WithDirectoryURL:error:` + `answerForImage:question:error:` / `answerForQuestion:error:`.
  Four facts were load-bearing. (1) The released 4B is in the transformers **4.x key layout**
  (`language_model.model.*`, `vision_tower.vision_model.*`, `multi_modal_projector.*`, no `model.`
  prefix) while a 5.x-written one nests them under `model.`; `decoderName(of:)` / `visionName(of:)` /
  `projectorName(of:)` accept both and drop any `lm_head.weight`. (2) transformers 5 **flattened
  `SiglipVisionModel`** (no `vision_model` child), so the oracle loads the tower's state dict without
  that prefix. (3) The processor's `apply_chat_template(tokenize=False)` text already spells `<bos>`, so
  the oracle tokenizes it with `add_special_tokens=False` — the reference's own double-BOS quirk is not
  reproduced. (4) EmbeddingGemma's bidirectional window was wrong and unmeasured: the release states
  `sliding_window: 512` and the reference turns it into the exclusive bound `sliding_window // 2 + 1 = 257`
  on `|q − k|`; the encoder had used 512, invisible on the ~20-token parity query. Now
  `NFKMLXGemma3EncoderConfiguration.geometry` applies the rule, and `run_reference.py
  gemma3_bidirectional_tiny` (span 6 → bound 4 over 12 tokens) measures it: last hidden 0.99999999999967.
  `NFKMLXGemmaTokenizer` now matches `added_tokens` as literals before the merge (a `<…>` scan, HF's
  own rule), so a rendered template or a 256-soft-token run encodes to ids; `decode(_:skipSpecial:)`
  drops the markers. Oracles: `gemma3` (a release directory; the chat ids and six tokenizer probes
  ride along), `gemma3_tiny`, `gemma3_bidirectional_tiny`, `gemma3_vision_real` (the tower + projector
  loaded selectively, the plate through the release's own image processor), `gemma3_conditional_real`
  (the full model; the record keeps the argmax at every position and the logits of the last 16 —
  the whole `[276, 262208]` matrix is 290 MB), all under the gemma interpreter (which gained Pillow +
  torchvision). Weights: `unsloth/gemma-3-{270m,1b,4b}-it`, ungated mirrors of the gated `google/`
  releases (536 MB / 2 GB / 8.6 GB bf16; the 4B is ~17 GB at float32, which fits beside nothing else).
  Not ported: pan-and-scan (off in every release). The 12B / 27B are the same architecture at 24 / 54 GB
  bf16 and are held to their released headers by shape through the 4B's configuration reader —
  decoder, vision tower, and projector together, 1065 / 1247 tensors, 0 missing, 0 mismatched,
  0 unaccounted. Gemma 3n is `gemma3n`, a separate family, and is refused here — `NFKMLXGemma3n` runs it.
- `NFKMLXGemma3n` / `NFKMLXGemma3nNet` / `NFKMLXGemma3nAudioNet` / `NFKMLXGemma3nVisionNet` — Gemma 3n,
  tri-modal and end to end, at reference parity on the released E2B weights for every stage, each on
  its first numeric run. A distinct architecture from Gemma 3 and Gemma 4, sharing the family name and
  almost nothing else; four mechanisms none of the others carry, and each changes the forward pass.
  - **AltUp** (Alternating Updates): the residual stream is `altup_num_inputs` (4) parallel copies, so a
    layer works on `[copies, batch, length, hidden]`. A learned per-token map `predict`s every copy from
    the active one before the block runs and `correct`s them from its output after. The prediction
    coefficients are reshaped `[n, n]` and transposed before the multiply; the correction coefficients
    take a `+ 1` so an untrained map is the identity; and the router's input scale is the hidden size's
    Reciprocal, not its inverse square root. Closest relative here is DeepSeek's hyper-connections.
  - **LAuReL** (Learned Augmented Residual Layer): a rank-64 detour beside the attention residual,
    normalized and added, with the sum divided by `sqrt(2)`.
  - **Per-layer embeddings**: a second 262144-row embedding gives every layer its own 256-wide slice,
    gated into that layer's output and added to the inactive copies only. The per-layer vocabulary is
    Smaller than the token vocabulary (262144 against 262400) — the ids past it are the vision and audio
    tokens, which carry no per-layer embedding and read row zero.
  - **Activation sparsity**: the first ten layers zero everything in the feed-forward's gate below
    `mean + Phi⁻¹(0.95)·sd` of that token's own gate, the deviation being the population one. There is no
    `erfinv` in Foundation, so `NFKGemma3nStatistics.standardNormalQuantile` is Acklam's approximation
    refined once by Halley's method against `erfc`.
  Two smaller differences are equally load-bearing: attention runs at scale 1.0, not
  `1/sqrt(headDim)` — the query normalization stands in for it — and the values carry their own
  normalization with no weight (`with_scale: false`), so the checkpoint holds no tensor for it. The norm
  is `x_norm · w`, the plain scale, which is Gemma 4's convention rather than Gemma 3's `(1 + w)`.
  **Key-value sharing**: the last `num_kv_shared_layers` (10 of E2B's 30) compute no keys or values at
  all and reuse the last non-shared layer OF THEIR own kind — a sliding layer reuses a sliding
  layer's, a full layer a full layer's — so the module declares no `k_proj`/`v_proj`/`k_norm`/`v_norm`
  for them and the released checkpoint carries none. Because a donor hands its full-length keys to the
  layers below it, `NFKMLXGemma3nCache` keeps every layer's keys whole and enforces the window by the
  Mask; a donor trimmed to its own window would hand a shorter history than the reference does.
  **Measured**: tiny (every mechanism, both attention kinds, a shared tail of each kind) every layer
  1.0000000000, logit cosine 0.9999999999999682, cached greedy continuation 6/6 with worst step
  1 − cosine 5.0e-14; released E2B every one of the 30 hidden states 1.0000000000, logit cosine
  0.9999999999943566, argmax 6/6, cached continuation 11/11 token for token.
  **The audio encoder** (`NFKMLXGemma3nAudioNet`) is a Universal Speech Model Conformer, a different
  network from the Gemma 4 Conformer rather than a configuration of it: two strided 2-D convolutions
  under a **cumulative group normalization** (a frame is normalized by every frame up to and including
  itself, and the variance takes each frame's deviations from THAT frame's cumulative mean before
  summing over time — not the running variance it resembles), then 12 blocks of feed-forward /
  chunked attention / causal depthwise convolution / feed-forward. The time axis is padded on the right
  only (`kernel − 1`, JAX's reverse-causal), the frequency axis by one each side. The attention is
  chunked with a Transformer-XL relative-position shift, its queries scaled by `1/sqrt(headDim)` divided
  by `softplus(0)` times a learned per-dimension softplus, and the activation clamp
  (`gradient_clipping`) runs at inference, six times a block. Two bugs, both found by seam
  isolation rather than guessed at: the block mask's bounds are relative to the query's row, not to
  its position in the context (`k >= q` and `k <= q + past + future`) — the two coincide when the right
  context is zero, which the release sets, so only the tiny configuration exposed it; and the validity
  mask is never skipped, because a block's context is zero-padded at both ends and the reference marks
  those frames invalid by padding the mask itself with false. Running with no mask at all left the first
  and last blocks attending to zeros (0.9987) while every weight was right. **Measured**: released E2B
  front end 0.9999999999999402, first block 0.9999999999998025, encoded 0.9999999999999257; tiny (which
  is where a non-zero right context is exercised) 0.9999999999999707.
  **The vision tower** (`NFKMLXGemma3nVisionNet`) is **MobileNetV5-300M**, a convolutional encoder rather
  than the SigLIP transformer every other vision model here carries, and it reaches the release through
  `timm` rather than transformers. Four stages of edge residuals, universal inverted residuals, and
  multi-query attention over the feature map (many query heads sharing one key and one value head,
  with no positional embedding of any kind) feed a fusion adapter that joins the last two stages.
  Five facts are load-bearing: padding is TensorFlow's `SAME`, asymmetric at stride two (3×3 pads
  (0, 1), 5×5 pads (1, 2) — symmetric padding gives the same output size and a shifted picture);
  there is **no BatchNorm anywhere**, every normalization being an RMS norm over the channel axis
  with a weight and no running statistics (the checkpoint's `bn` names are legacy); the activation is
  the **tanh-approximate** GELU; the inverted residual's first depthwise convolution runs before the
  expansion and carries no activation, with the stride on the second where the block downsamples; and
  the fusion concatenates the coarse stage after the fine one, nearest-upsampled, an order the
  `[3840, 1920, 1, 1]` weight it feeds cannot reveal. Measured on the released weights, first numeric
  run: stem 0.9999999999996717, stages 0.9999999999955526 / 0.9999999999982407 / 0.9999999999571081 /
  0.999999999990077, fused grid 0.9999999999981896.
  **The fusion** splices in two stages, unlike the other multimodal models here: a placeholder id is
  first embedded hard through a small per-modality table indexed by an offset into the token vocabulary,
  and the tower's soft tokens then overwrite those positions. Doing only the hard pass gives a model
  that runs and ignores the picture. The vision grid is scaled by `sqrt(2048)` before its embedder reads
  it. A clip shorter than 188 soft tokens is padded with the audio modality's last id.
  Measured end to end on the released E2B: the prompt this port builds reproduces the release
  processor's 272 ids exactly, argmax **272/272** over the fused sequence, last-16 logit cosine
  0.9999999999785188, and the cached greedy continuation reproduces the reference's caption token for
  token. The prompt is built as text and tokenized once, which is load-bearing and is the same trap
  Gemma 3 hit here: the `\n` closing `user` and the `\n\n` opening the image run merge into one id
  (109), so a prompt assembled from separately encoded pieces reads a different sentence to the model.
  The audio front end (`NFKMLXGemma3nAudioFeatures`) pairs HTK's mel scale with no area
  normalization (`norm=None`, where every other filterbank in this package is Slaney-normalized —
  adding it shifts each band's log by a constant and scored −0.149), cuts frames one sample longer than
  the window so HTK's pre-emphasis has a predecessor for every sample it keeps, and runs the transform
  at twice the window's length (`fft_overdrive`). Measured 0.999999999999857.
  `NFKMLXGemma3nBackend` reads `NFKInputPrompt` / `NFKInputMessages` (+ `NFKInputImage` and
  `NFKInputAudio`), honors temperature / top-p / max-tokens / seed, and streams through a submitted
  job's `partialResult`. ObjC: `[NFKMLXGemma3n backendWithDirectoryURL:error:]`,
  `gemma3nWithDirectoryURL:error:` + `answerForImage:question:error:` / `answerForQuestion:error:`.
  The image processor resizes to 768×768 and scales to `0...1` and **nothing more** — the release's
  preprocessor file states an `image_mean` and an `image_std` and then sets `do_normalize` false, so a
  `-1...1` frame is a plausible-looking mistake. Weights: `unsloth/gemma-3n-E2B-it` (10 GB, an ungated
  mirror of the gated `google/`). E4B is measured too, at the precision it ships in: 35 layers,
  fifteen of them sharing keys and values, read by the same configuration reader, and 16 GB of bf16
  that doubles past this machine at float32, so both sides run bf16 (`IK_GEMMA_DTYPE=bfloat16`,
  `.checkpoint` here) — logit cosine 0.99989 with the argmax matching at 5 of 6 positions, the one flip
  at the prompt's flattest position where the reference's own margin is half a logit. Whether that is
  rounding or a defect was measured, not argued: the E2B, exact at float32, recorded the same way at
  bf16 reads the same 0.99989 (`IK_PARITY_GEMMA3N_E2B_BF16`), so that is the floor the E4B is held to
  (`testGemma3nE4BMatchesTheReferenceLogits`: cosine above 0.999, at most one flip). Its decoder is
  also held to the released headers by shape: 806 tensors consumed, 870 named as dropped (the towers
  and the k/v projections the release still ships for its sharing layers), 0 unaccounted. Not ported:
  the MatFormer nesting that slices E2B out of E4B, which is a checkpoint operation rather than a
  forward pass.
- `NFKMLXGemmaLanguage` — the Gemma 4 text decoder (`gemma4_text`), a fourth architecture family, at
  reference parity against transformers' own implementation on the released E2B weights (logit
  cosine 0.9999999999994, and every one of the 36 hidden states exact layer by layer). Measuring it
  required installing Python 3.12 beside the system 3.9, because Gemma 4 is in no transformers that
  runs on 3.9; `oracle_environments` records that interpreter.
  This is the model that proved a structural check is not a numeric one. Its 600 parameters matched
  the release by name and shape while the forward scored **0.0044**, and four corrections read from the
  reference moved it only to 0.48 — two of them making it worse. What resolved it was the **per-layer
  isolation harness** (`testGemma4LayerByLayerAgainstTheReference`): the oracle records the state
  entering the stack and the state each layer produced, so the first divergence is located rather than
  guessed. It put the fault at layer 0's `input_layernorm` while that layer's input was exact, and a
  sub-step probe inside layer 0 narrowed it further. Three real defects came out of it:
  Gemma 4 normalizes with a plain scale, `x · w` — the `x · (1 + w)` convention is **Gemma 3's**,
  and assuming the family inherited it is what broke the port. The feed-forward uses Gemma's own
  activation (`gelu_pytorch_tanh`), not the SwiGLU's silu, which surfaced only once attention was
  exact. And the full-attention layers use the `proportional` rotary: frequencies computed over the
  Whole head width with the first `partial_rotary_factor` of the pairs real and the rest **zeroed**,
  which is not the same as rotating a contiguous leading slice — with rotate-half a pair is
  `(i, i + width/2)`, so a quarter-turned 512-wide head turns channels 0…63 and 256…319. That one left
  every sliding layer exact and every full layer subtly wrong, which is exactly what the harness showed
  at layer 4.
  Its distinguishing feature is **per-layer input embeddings**: beside the ordinary token embedding, a
  second much wider one (`embed_tokens_per_layer`, 262144 × 8960 = 35 layers × 256) gives every layer
  its own slice, gated into the residual after the feed-forward. Normalization is Gemma's `x · (1 + w)`
  rather than `x · w`, so a Gemma checkpoint in a plain RMSNorm produces near-zero activations and
  looks like a broken model instead of a convention mismatch. Logits are soft-capped through `tanh`.
  Two things the config does not say, both read from the checkpoint. A full-attention layer runs
  `global_head_dim` 512 where a sliding one runs `head_dim` 256, so seven of E2B's 35 layers have
  doubled attention widths — using `head_dim` throughout mismatched 42 tensors. And the layers that
  share keys and values run a doubled feed-forward: E2B's first fifteen are 6144 wide and its last
  twenty — exactly `num_kv_shared_layers` — are 12288, which no config field states. Both were found by
  the structural check rather than by reading, which is the argument for running it before trusting a
  port.
  Scope: the text decoder. The release is tri-modal, carrying a vision tower (659 tensors) and an audio
  Conformer (752); a test names them so they are known rather than overlooked. The configuration guard
  accepts `gemma4_text` and `gemma4` and rejects everything else: `gemma4_unified_text` (the 12B) is a
  Different decoder architecture (`Gemma4Unified*` classes), so it is refused rather than loaded into
  this stack and made to produce fluent nonsense. The 26B-A4B mixture (`enable_moe_block`) is
  implemented — see the mixture entry below; the dense sizes carry the expert fields nulled and set
  the flag false, so the flag distinguishes them.
  The Gemma 4 mixture of experts (`NFKGemmaRouter` / `NFKGemmaExperts`, the 26B-A4B family) runs a
  routed branch beside every layer's dense feed-forward and sums the two — not a dense-FFN swap. The
  block computes `post_ff_norm(post_ff_norm_1(mlp) + post_ff_norm_2(experts(pre_ff_norm_2(residual))))`,
  where the router reads the pre-feed-forward `residual` (not the normed copy) and the experts read a
  separately-normed copy of it. The router is a scale-free RMS norm, a learned per-channel `scale`
  times `hidden^-0.5`, a projection to the experts, a softmax, the top-k, a renormalization, and a
  learned `per_expert_scale` on the kept weights; the experts are a fused `gate_up_proj` `[E, 2·inter,
  hidden]` and a `down_proj` `[E, hidden, inter]` dispatched through `gatherMM`, and they apply the
  routing weights themselves (the reference's index-add), so the block sums rather than re-weights.
  Reference parity against transformers' own `Gemma4ForCausalLM` at a tiny two-layer configuration
  (`run_reference.py gemma4_moe`, `IK_PARITY_GEMMA4_MOE`, the gemma oracle): every hidden state exact
  layer by layer, logit cosine 0.9999999999996, on the first numeric run after the geometry was
  corrected. **Two geometry facts were load-bearing, both invisible on the E-series:** the per-layer
  input embedding is a fixed 262144 rows (`vocab_size_per_layer_input`), not the token vocabulary — the
  two coincide on the E-series because its token vocabulary is 262144, so the net had been reading the
  token vocabulary and it only mattered at a tiny test; and a full-attention layer runs a fixed 512-wide
  head (its own `head_dim`) where the sliding layers run theirs, so setting the global head width to the
  sliding one crashed the full layer's projection reshape. The mixture is read from `config.json`
  through `NFKMLXGemmaLanguage.configuration(fromHuggingFace:)` (the same entry the dense sizes use),
  which turns on the routed branch from `enable_moe_block`. The Gemma 4 decoders run through
  `NFKMLXGemmaBackend` (`NFKMLXGemmaLanguage.backend(directoryURL:)` / `@objc
  gemmaBackendWithDirectoryURL:error:`), which reads a release directory and dispatches on its config's
  model type; internally it builds through `configuration(fromHuggingFace:)` plus `makeNet` /
  `loadWeights` (the path the parity tests use). See the `NFKMLXGemmaBackend` entry below.
- `NFKMLXGemma4UnifiedNet` (`NFKMLXGemma4Unified.swift`) — the **12B `gemma4_unified_text` decoder**, a
  Different architecture from the E-series: no per-layer input embeddings and no mixture, only the
  sandwich block with a per-layer scalar. Its attention is the same one the E-series runs — learned
  query/key norms, a scale-free value norm, attention at scale 1, per-layer head widths (a full layer
  runs 512), and the proportional rotary on the full layers — so `NFKGemmaAttention` and
  `NFKGemmaFeedForward` are reused directly and only the block and model are new (this is why the port
  matched on the first numeric run rather than after a hunt). `NFKMLXGemmaLanguage.unifiedConfiguration(fromHuggingFace:)`
  reads a `gemma4_unified_text` config (rejecting the E-series and everything else); `makeUnifiedNet` /
  `loadUnifiedWeights`. Embeddings scale by `√hidden`, logits are tied, no softcap. Reference parity
  against transformers' own `Gemma4UnifiedForCausalLM` at a tiny sliding/sliding/full configuration
  (`run_reference.py gemma4_unified`, `IK_PARITY_GEMMA4_UNIFIED`, the gemma oracle): every layer exact
  (cosine 1.0000000000), logit cosine 0.9999999999995, on the first run.
- `NFKMLXGemma4VisionNet` (`NFKMLXGemma4Vision.swift`) — the Gemma 4 vision encoder, the image
  tower of the tri-modal release. It embeds flattened patches through one linear projection, adds a
  learned 2-D position embedding (an x-table and a y-table summed per patch, padding zeroed), and runs
  the sandwich block bidirectionally. The attention keeps the learned query/key norms, the scale-free
  value norm, and scale 1, and applies a **2-D rope** beside the learned position embedding — the head
  is split per spatial axis (`headDim/2` channels each) and each half is rotated (rotate-half) by its
  coordinate at rope base **100**. The rope was missing at first and the tiny random-weight test could
  not see it — at `head_dim` 8 with small positions it moved the tiny encoder by 2e-8 (0.9999999850,
  which read as ordinary imprecision), but on the released weights (`head_dim` 64, 16 layers) it
  compounded to a cosine of 0.77; the real-weight test is what caught it. The projections are
  `Gemma4ClippableLinear` (`NFKGemmaClippableLinear`) — a bias-free linear under a `.linear` key with
  optional input/output clamps. The release trains them with finite clamp bounds (`use_clipped_linears`,
  a quantization-aware-training artifact — ±12, ±2.4, …), which are load-bearing at inference; the tiny
  oracle used `use_clipped_linears=False`, so this too only surfaced on real weights. Both fixes are
  Measured load-bearing on the released weights (`testGemma4VisionRopeAndClampsAreLoadBearingOnTheReleasedWeights`):
  running the real encoder with an identity rope (cos 1, sin 0) drops it to 0.768, and with no clamp
  bounds to 0.837, against the 0.9999999999898 the pair reaches — each isolated by leaving the other in
  place. `makeVisionNet` /
  config `NFKMLXGemma4VisionConfiguration` (`ropeTheta`, `useClippedLinears`). Reference parity at a
  tiny configuration (`run_reference.py gemma4_vision`, `IK_PARITY_GEMMA4_VISION`, encoder 0.9999999999)
  And on the released E2B weights (`gemma4_vision_real`, `IK_PARITY_GEMMA4_VISION_REAL`: encoder
  0.9999999999898, pooled 0.9999999999975, projected 0.9999999999948 — the vision tower, pooler, and
  embedder loaded selectively from the tri-modal checkpoint). The pooler is implemented too
  (`softTokens(_:positionIds:)`): a position-based average pool that folds the patches falling into each
  `k × k` grid cell (with `k` read from the input patch count over the output token count) and scales by
  `√hidden`, producing the soft tokens a language model reads. The optional standardization is
  implemented too: `standardize` creates `std_bias`/`std_scale` (`@ParameterInfo` optionals, absent by
  default) and applies `(pooled − std_bias) · std_scale` after the pooler, as `Gemma4VisionModel` does. No
  released Gemma 4 enables it (E2B and E4B both `standardize: false`), so the `.tiny` configuration and
  the `gemma4_vision` oracle set it on to exercise the path (pooled cosine 0.99999999999999). The image
  processor is `NFKMLXGemma4ImageProcessor` below.
- `NFKMLXGemma4AudioNet` (`NFKMLXGemma4Audio.swift`) — the Gemma 4 audio Conformer, the most complex
  tower. A **2-D convolutional subsampler** (two stride-2 3×3 convolutions, a channel LayerNorm with no
  bias, a ReLU, then a linear projection of the flattened frequency-and-channel features — the frequency
  dim is tied to `subsampling_conv_channels[0]`) feeds **Conformer layers**: a macaron feed-forward, a
  **blocked relative-position attention**, a **light depthwise convolution**, a second macaron
  feed-forward, and sandwich norms, then an output projection. Built in two isolated phases, each at
  parity against transformers' own `Gemma4AudioModel` (`run_reference.py gemma4_audio`,
  `IK_PARITY_GEMMA4_AUDIO`): the subsampler (`makeAudioSubSample`, cosine 0.9999999999999845) and the
  Conformer fed the reference's post-subsample hidden, position encoding, and mask (`conformer(...)`,
  cosine 0.9999999999999974, the blocked-attention seam alone 0.9999999999999688).
  **The blocked attention is a Transformer-XL relative-position attention:** queries group into
  non-overlapping blocks of `chunk_size`, each block attends over a `contextSize = chunk + (left-1) +
  right` window extracted by a padded gather (MLX has no `unfold`), the content score adds a
  relative-position score built from `relative_k_proj` and reshaped through the appendix-B `relativeShift`
  (pad, reshape, drop, reshape), the logits are `tanh`-softcapped, the mask fills the invalid positions,
  and the queries carry a per-dimension **softplus** scale beside `q_scale`/`k_scale`. The light conv is
  a GLU, a **causal** depthwise convolution (left-padded so a frame sees no future), and a pointwise
  projection. The convolution kernels load transposed from PyTorch's `[out,in,kH,kW]` / `[ch,1,k]` to
  MLX's layouts. **The parity trap was in the oracle, not the port:** the shared `_randomized` helper
  perturbs the clippable-linear clamp buffers (`input_min`/`max`), which the released model ships at
  ±inf (identity) — leaving them randomized clamped the reference's activations aggressively and the
  first feed-forward scored 0.35; resetting them to ±inf in the oracle (the port models no clamp)
  restored parity. The full tower runs end to end (`callAsFunction(_ features:)`): it subsamples,
  builds its own sliding-window blocked mask (`blockedMask`), and runs the Conformer — reference parity
  against `Gemma4AudioModel` from the mel features (cosine 0.999999999999996). The mask construction was
  the one off-by-one: the window function admits `dist ∈ [0, leftWindow)` (strict), so a valid past
  distance is `< maxPast`, not `≤ maxPast` — the inclusive form scored 0.9987. Reference parity on the
  released E2B weights too (`gemma4_audio_real`, `IK_PARITY_GEMMA4_AUDIO_REAL`: encoded
  0.9999999999995, projected 0.9999999999994 — the audio tower and its embedder loaded from the tri-modal
  checkpoint, with the finite `use_clipped_linears` clamps that the tiny oracle left at ±inf). The audio
  configuration gained `useClippedLinears`, threaded to every projection, the same as the vision tower.
- `NFKMLXGemmaBackend` (`NFKMLXGemmaBackend.swift`) — the **text-generation backend** for the Gemma 4
  decoders, so a consumer runs them through the InferKit contract. `NFKMLXGemmaLanguage.backend(directoryURL:)`
  / `@objc gemmaBackendWithDirectoryURL:error:` reads a release's `config.json` and dispatches on its
  model type: the E-series and the 26B-A4B mixture through `configuration(fromHuggingFace:)` + `makeNet`,
  the 12B through `unifiedConfiguration` + `makeUnifiedNet`. **Gemma runs prefill-only** — the decoders
  carry no key-value cache — so generation re-runs the growing sequence each step (quadratic, fine for
  the short outputs an on-device assistant produces). The tokenizer is `NFKMLXGemmaTokenizer` (the core
  reader cannot produce Gemma's byte-fallback BPE), which gained `decode` and `id(forToken:)`; a raw
  prompt encodes after `<bos>`, and a message list builds Gemma's `<start_of_turn>role\n…<end_of_turn>`
  turns from the special-token IDS rather than encoding the markers as text. Generation is greedy at
  temperature 0, else temperature-sampled, stopping on `<eos>`/`<end_of_turn>`. Measured end to end
  on the released E2B (`testGemmaBackendGeneratesText`): "The capital of France is" → " Paris." The net
  itself is already at parity layer by layer, so the backend test exercises the tokenizer and the
  generation loop it adds. The networks and tokenizer cross the async job boundary through an
  `@unchecked Sendable` holder, as the core language backend does.
- `NFKMLXGemma4ImageProcessor` (`NFKMLXGemma4ImageProcessor.swift`) — the vision **image processor**:
  a `CGImage` to the flattened patches and `(x, y)` positions the tower reads. It resizes preserving
  aspect ratio to fit a patch budget (both sides a multiple of `poolingKernelSize · patchSize`),
  rescales to `0 … 1`, splits into patches flattened `(row, column, channel)`, and pads to the budget
  with `(-1, -1)` positions. The resize is CoreGraphics, not the reference's torchvision bicubic, so the
  patch pixels are a documented approximation (as SmolVLM's are); the resized dimensions, the patch
  layout, and the position ids are the reference's exactly, checked by `testGemma4ImageProcessorLayout`.
- `NFKMLXGemma4AudioFeatureExtractor` (`NFKMLXGemma4AudioFeatureExtractor.swift`) — the audio **mel front
  end**: raw 16 kHz audio to the log-mel features `[frames, 128]` the subsampler reads. Semicausal
  framing (prepend `frameLength / 2` zeros, unfold `frameLength + 1` and drop the last sample), a
  periodic Hann window, the magnitude (not power) of a 512-point real FFT, a 128-band HTK triangular mel
  filterbank (`log10`-mel, `norm=None`, distinct from the Slaney bank the Whisper front end uses), and
  `log(mel + 1e-3)`. Reference parity against transformers' own `Gemma4AudioFeatureExtractor`
  (`run_reference.py gemma4_mel`, cosine 0.9999999999928).
- `NFKMLXGemma4MultimodalEmbedder` / `NFKMLXGemma4Fusion` (`NFKMLXGemma4Fusion.swift`) — the **multimodal
  fusion**. The embedder projects a tower's soft tokens into the decoder's space (a scale-free RMS norm,
  then a bias-free linear to the text hidden size) at reference parity against
  `Gemma4MultimodalEmbedder` (`run_reference.py gemma4_embedder`, cosine 0.9999999999999927); `fuse`
  replaces the text embeddings at the placeholder positions with the projected soft tokens in order, the
  same `where`-over-a-gathered-index splice the SmolVLM fusion uses. The whole chain end to end
  (image/audio to an answer) additionally needs the released TRI-MODAL weights — the towers, the
  embedders, and the E-series decoder together — which the text-only E2B release does not carry, so the
  components are each verified at parity rather than the full run.
- `NFKMLXGemma4ConditionalGeneration` (`NFKMLXGemma4Fusion.swift`) — the full tri-modal chain wired end
  to end: an image and/or a waveform and a placeholder-carrying token sequence in, a generated
  continuation out. It runs the image through the processor and the vision tower to soft tokens (and the
  waveform through the mel front end and the audio tower), projects each through its embedder, splices
  them at the placeholder positions with `fusedEmbeddings`, and runs the E-series decoder prefill-only
  over the fused embeddings. The decoder gained an `embed` / `logits(fromEmbeddings:tokens:)` seam so the
  main stream is supplied pre-spliced while the per-layer input identity still reads the
  placeholder-padded token ids (the reference's split — the context projection reads the spliced
  embeddings, the identity reads the ids). A placeholder token embeds as the pad token and the soft
  token then replaces it. Reference parity on the released E2B weights, end to end
  (`testGemma4ConditionalGenerationOnTheReleasedWeights`, `run_reference.py gemma4_conditional_real`,
  `IK_PARITY_GEMMA4_CONDITIONAL_REAL`): an image over a 6×6 patch grid fills four image placeholder
  tokens, and the fused sequence's logits match transformers' own `Gemma4ForConditionalGeneration` at
  logit cosine 0.999999999959, argmax 8/8 (every position predicts the same token). The E2B release
  Is the full tri-modal `Gemma4ForConditionalGeneration` — its 10 GB checkpoint carries the vision
  tower, the audio Conformer, both embedders, and the decoder — so the numeric end-to-end run needed no
  separate weights; an earlier note wrongly called it text-only. The oracle and the Swift side each load
  the sub-towers selectively from that one checkpoint (`model.vision_tower.` / `model.embed_vision.` /
  `model.language_model.`), so neither has to hold the whole tri-modal graph at float32 at once.
- Nothing remains in the Gemma 4 family. The four architectures, both towers' full forwards (the
  optional vision standardization included), the input adapters, the fusion, and the
  conditional-generation chain are all at reference parity on the released weights; the text decoders
  generate through `NFKMLXGemmaBackend`.
  E4B is measured too, at the precision it ships in: its 16 GB of bf16 weights double past this
  machine's RAM at float32, so both sides run bf16 (`IK_GEMMA_DTYPE=bfloat16` for the oracle,
  `.checkpoint` here) — logit cosine 0.9998 with the same argmax at every position, and the strict
  load itself confirms the doubled feed-forward on its 18 kv-shared layers, since a wrong width fails
  loudly. The measurement surfaced a defect only bf16 could: the attention masks are built float32,
  and the fused attention refuses a mask that does not promote to a bf16 module's own type, so the
  mask now takes the queries' dtype — invisible at float32, which is why no float32 run ever raised
  it. The dense and hybrid decoders had the identical latent crash on any `.checkpoint` load and are
  fixed the same way, each pinned by a bf16-forward test.
- `NFKMLXQuantization` — runtime MLX quantization, the package's first path for running a model in
  MLX-quantized form (`NFKMLXDeepSeekQuantization` only decodes a stored format). `quantize(module:
  bits:groupSize:includeEmbeddings:)` packs `Linear` layers whose input width divides the group size
  into affine 4- or 8-bit `QuantizedLinear` (everything else computes as built; already-quantized
  layers are excluded, which matters because `QuantizedLinear` subclasses `Linear` and satisfies a
  type test silently). `includeEmbeddings` (default off) also packs `Embedding` layers into
  `QuantizedEmbedding`; it is off by default because a tied model reuses its input embedding as the
  logit head, so quantizing it quantizes the head too — a per-model cost. That cost is measured and
  small (`testTheTiedEmbeddingQuantizationCostAgainstTheRecord`, opt-in `IK_QWEN_EMB_PROBE=1`): on
  the tied Qwen3 0.6B and 1.7B, packing the embedding at the same width as the Linear layers moves the
  logit cosine by ~1e-5 at 8-bit and ~0.006 at 4-bit — the Linear bit width dominates, not the tied
  head (8-bit Linear scores 0.9962 / 0.9994, 4-bit only 0.9222 / 0.9488, so these small models want
  8-bit regardless). So `includeEmbeddings: true` is safe for a tied model quantized at 8-bit; the
  default stays off because the wrong-width case (4-bit) is where packing the head costs most, and a
  caller should choose it deliberately. The music LM opts in
  (`quantizeRelease` passes `includeEmbeddings: true`): that LM is untied, its `lm_head` is a separate
  packed `Linear`, and the input embedding is its largest tensor (200000×4096, 1.6 GiB) — quantizing
  it at 4-bit measured logits cosine 0.99933 against 0.99952 for the bf16 embedding and reclaims
  1.10 GiB. The checkpoint contract closes the packed-uint32 hazard: `NFKMLXWeights.save` detects
  quantized leaves and records `inferkit.quantization` = "bits:groupSize" in the metadata;
  `loadCheckpoint` reads it back, and a loader calls `NFKMLXQuantization.matchStructure(of:on:)`
  Before applying, so the packed arrays land on matching structure — without that, a packed weight
  loaded into a plain `Linear` adopts the wrong shape and dtype with no error. The metadata records
  one bits/groupSize, not which layer kinds were packed, so `matchStructure` reads whether the
  embedding was quantized from the checkpoint itself — a packed embedding weight is `uint32` where an
  unquantized one is a float — which keeps a file saved before embeddings were quantizable loadable. A
  quantized checkpoint loads at its stored dtypes whatever precision the caller requests (the packing
  is uint32 regardless, and the scales keep the precision the quantization was computed at). Wired
  through the language-model loaders (single-file releases route through `loadCheckpoint`; a
  quantized module saves as one file, so quantized-sharded does not arise) and the music loaders;
  round-tripped exactly by `testAQuantizedCheckpointRoundTripsThroughTheLoaders`, embedding-packed
  case included.
- `NFKMLXReleaseWeights` — one reader for a downloaded release's weights, single-file or sharded
  (`model.safetensors.index.json`, each shard read once), with a remap closure whose nil skips a
  tensor. The dense, hybrid, and Gemma loaders all read through it; before it each had its own copy,
  and Gemma's copy had no sharded path — a capability gap consolidation removed as a side effect.
  The per-family differences stay in the loaders where they belong: the tied-`lm_head` drop, the
  hybrid's `model.language_model.` remap and depthwise-conv transpose, Gemma's tower skip.
- `NFKMLXGGUF` / `NFKMLXGGUFFormat` — the **native GGUF reader**, the sequel to the native PyTorch
  checkpoint reader, and the format most quantized language models are distributed in. Same contract:
  pure Foundation below the MLX materialization (parsing and dequantization run under `swift test`;
  `NFKMLXGGUFFormat` is the Foundation layer, `NFKMLXGGUF` the `@objc` MLX face), and a type it does not
  implement is refused per-tensor, not per-file — an unknown GGML type leaves the tensor listed (so a
  consumer sees the whole model) and only reading it throws. The container is a header of typed key/value
  metadata (`readValue` covers the 13 GGUF value types, including nested arrays) and a tensor table
  (name, dims, GGML type, offset), then the tensor data aligned to `general.alignment` (default 32).
  GGUF stores the fastest-varying dimension first, so a tensor's row-major (MLX) shape is the
  Reverse of its stored dims — a Linear weight stored `ne=[in, out]` is shape `[out, in]`. Dequantizers:
  `F32`, `F16`, `Q4_0`, `Q5_0`, `Q8_0`, `Q4_K`, `Q6_K` — the k-quants (`Q4_K`: a 256-value super-block of
  eight 32-value sub-blocks, a block `d`/`dmin` scaling per-sub-block 6-bit scales/mins unpacked from 12
  packed bytes, each value `d·sc·q − dmin·min`; `Q6_K`: sixteen 16-value sub-blocks, a 6-bit quant from a
  4-bit low and 2-bit high part centered at 32, scaled by `d` and a per-sub-block int8 scale) plus the
  legacy 32-value blocks. The scalar per-block port matches the vectorized `gguf` reference exactly
  (both compute the one canonical dequantization). **Bit-exact** against the `gguf` package on the
  released SmolLM2-135M Q4_K_M (`run_reference.py gguf`, the `llm` oracle + the `gguf` package): the
  first tensor of each of F32/Q8_0/Q5_0/Q4_K/Q6_K dequantizes to **worst |difference| 0.0** across 262144
  values. A real `Q4_K_M` file mixes Q5_0 (the bulk here), Q4_K, Q6_K, Q8_0, and F32, so supporting Q5_0
  is what makes the file readable rather than mostly-refused. A stored `MLXArray` constant is not held on
  the format struct; the dequant returns `[Float]`, materialized to an `MLXArray` only in the face.
  Wired into the language-model loader (`NFKMLXGGUFLanguage.swift`): a GGUF release generates text
  end to end. `NFKMLXLanguage.configuration(fromGGUF:)` maps the metadata (`<arch>.block_count`,
  `<arch>.embedding_length`, …) onto an `NFKMLXLanguageConfiguration`, reading two structural facts
  from the tensors because the metadata carries no flag for them — a model is tied when it ships no
  `output.weight`, and it normalizes queries and keys when it ships `blk.0.attn_q_norm.weight` (Qwen3
  does, Qwen2/Llama do not). `loadWeights(into:fromGGUF:)` remaps the llama.cpp names
  (`blk.N.attn_q.weight` → `model.layers.N.self_attn.q_proj.weight`, `token_embd`/`output_norm`/`output`)
  and `ggufTokenizer` rebuilds the embedded byte-level BPE (the already-encoded tokens and merges written
  to a temp `vocab.json`/`merges.txt` for the core reader, specials read from `token_type` 3/4). Only
  the dense `llama`/`qwen2`/`qwen3` families are accepted; another architecture throws. Factory
  `backend(ggufURL:)` / `@objc backendWithGGUFURL:error:`.
  **The Q/K permute is load-bearing:** llama.cpp permutes the query and
  key projections during conversion so its interleaved rotary reads adjacent channels, where this
  decoder rotates split halves — loading the raw weights runs mostly-right and subtly wrong (logit
  cosine 0.95, the model saying "Paris" but diverging after). `unpermuteRotary` undoes it per head
  (reshape `[heads, headDim/2, 2, …]`, swap the split-half axes, reshape back — transformers'
  `_reverse_permute_weights`), with the query using the head count and the key the KV-head count. With
  it, reference parity against transformers loading the same GGUF (`run_reference.py gguf_lm`,
  `IK_PARITY_GGUF_LM`, the llm oracle now carrying `accelerate`): logit cosine 0.9999999999971 with the
  same argmax at every prompt position, the first greedy token identical, and the rebuilt tokenizer
  reproducing the reference's ids. A full greedy continuation is not asserted exact — two dequantization
  implementations flip an occasional near-tie on quantized weights — so the check is teacher-forced
  agreement (8/10, near-ties excepted) with the logit cosine as the tight bound. Measured on the
  released SmolLM2-135M-Instruct Q4_K_M.
- `NFKMLXChatTemplateRenderer` (`NFKMLXChatTemplate.swift` / `…Engine.swift` / `…Expr.swift`) — a
  native Jinja renderer for the release's `chat_template`, so the language backend reproduces an
  instruct model's trained input instead of the hand-coded ChatML approximation. Rendering the template
  wrong silently changes the model's input, the same failure class as the `qwen2` pre-tokenization
  defect, so the faithful path is to render the release's own template. A compact interpreter for the
  subset chat templates use: text with `{{ }}` output and `{% %}` control (for / if / elif / else /
  set), the whitespace model transformers compiles a template with (`trim_blocks` + `lstrip_blocks` +
  the explicit `{%-`/`-%}` markers), and the expression language — attribute/index access, slicing
  (`messages[::-1]`), `namespace`, the `loop` variable, `is` tests, string methods, and the
  `tojson`/`trim` filters. Pure Foundation below no runtime at all (no MLX), so it and its tests run
  under `swift test`. Reference parity against transformers' own `apply_chat_template` over six
  cases (`Tools/reference-parity/generate_chat_templates.py`, config key `IK_CHAT_TEMPLATE_REF`): the
  Qwen3 template (namespaces, reversed slicing, `is` tests, the tool-call and tool-role branches),
  Llama-3 (`bos_token` + `| trim` precedence), and Gemma (`%`, `!=` on booleans, `set role`, the
  `raise_exception` guards), each rendered byte-for-byte. **Two whitespace traps, both measured:**
  `lstrip_blocks` strips a block tag's line indentation only when the tag begins a source line (a
  trailing whitespace run after content on the same line stays — the first draft stripped it
  unconditionally, which dropped a real space); and `| trim` binds tighter than `+`, so
  `a + b | trim + c` trims only `b`. Wired into the backend as `NFKMLXChatTemplate.jinja(template:
  bosToken:eosToken:)`; from ObjC a `chatTemplate` request parameter carrying Jinja delimiters
  (`{%`/`{{`) is rendered the same way (the associated-value enum case is Swift-only, per the parity
  rule, and the string parameter is the bridge). **One documented divergence:** `tojson` emits an
  object's keys sorted, where transformers emits insertion order — Foundation dictionaries do not
  preserve it, so a faithful whole-object serialization needs an ordered-map pipeline; it affects a
  tool schema's key order, not a plain or multi-turn chat. The `NFKJinja*` types (value, namespace,
  parser, evaluator) are internal.
- `NFKMLXTorchCheckpoint` / `NFKMLXTorchFormat` — the native PyTorch checkpoint reader: a consumer's
  raw `.pth`/`.pt`/`.ckpt`/`.th`/HF `.bin` loads with no Python toolchain. `NFKMLXWeights.loadCheckpoint`
  sniffs a file's leading bytes (never the extension — an HF torch `.bin` and a safetensors `.bin` are
  told apart by content), so every `weightsURL:` factory accepts a raw checkpoint wherever it accepts a
  converted safetensors, reported as `needsConvTranspose: true`. Three layers, all pure Foundation
  below the MLX materialization, so the parsing tests run under `swift test`:
  `NFKMLXZipArchive` (central-directory ZIP with zip64 and deflate; a stored entry's contents are a
  zero-copy slice of the memory-mapped file), `NFKMLXPickle` (a restricted pickle machine, protocols
  2–5: no global ever executes — `collections.OrderedDict` is the only one the machine itself
  interprets, and every other construction becomes an inert opaque node that flattening drops), and
  `NFKMLXTorchFormat` (both containers: the modern zip and the pre-1.6 five-pickle stream, whose
  storages arrive after the pickles and whose persistent tuples carry a trailing view entry).
  Training wrappers unwrap in the converters' own precedence (`state_dict`, `model_state_dict`,
  `params_ema`, `params`, `model`, `generator`, `state`) before the root is flattened — a Lightning
  checkpoint keeps optimizer tensors beside its state_dict, so root-first sweeps those in.
  Whisper's releases store their Linear weights as transposed fp16 views, found by the first real
  parity run after the plan assumed state dicts are contiguous: `bytes(for:)` gathers a strided
  tensor to row-major, held to torch's own materialization by comparing the raw `whisper_tiny.pt`
  against its converted safetensors tensor for tensor. The byte oracle throughout is the offline
  converters' own output (raw in `~/.inferkit-validation/raw/`, `IK_RAW_<KEY>` written by fetch.py).
  `NFKMLXTorchCheckpoint` is the public `@objc` face: inspect `tensorNames`/`infoForTensor:`, read a
  tensor's bytes, or convert on device with `writeSafetensorsToURL:` (a hand-rolled pure-Swift
  safetensors writer — no Metal needed — whose output carries no `inferkit.layout` metadata, which is
  the PyTorch-layout marker; float64 narrows to float32 as the converters do). Refused with errors
  naming the offline converter: TorchScript archives (CLIP), an opaque module tree (YOLO), `.nemo`
  big-endian saves, sparse/quantized storages. There are no deferred models — YOLO, VAD, and CLIP
  all load. Three walks/unwraps, all non-executing (no class constructed, no serialized `code/`
  interpreted): (1) a checkpoint that pickled a live `nn.Module` tree (YOLO's ultralytics
  DetectionModel) is walked through the standard `_parameters`/`_buffers`/`_modules` state — `walkModule`,
  reproducing `nn.Module.state_dict()` exactly (parameters + persistent buffers, recurse `_modules`,
  skip a None param / non-persistent buffer / plain-attribute tensor), matched against the real
  yolov8n's 498-key state dict; (2) a **TorchScript archive** (CLIP) is walked through its
  attribute-keyed scripted-module state — `walkScriptedModule`, where each object's `BUILD` state is
  a flat dict of `name → tensor | submodule | scalar` rather than the eager layout, and the leaf
  tensors are the same `_rebuild_tensor_v2` records. The earlier "TorchScript needs its `code/` IR"
  claim was wrong: the probe showed `data.pkl` carries every attribute name (`visual.conv1.weight`,
  `transformer.resblocks.0.attn.in_proj_weight`), matched against the real ViT-B/32's 302-key
  state dict; (3) a `.nemo` PAX/ustar tar is unwrapped to the checkpoint inside it (`readTar`). The
  scripted walk is scoped to archives carrying `constants.pkl` (the TorchScript marker), so the eager
  path is untouched; its int config attributes (`input_resolution`) are surfaced and ignored by the
  loaders' coverage the way `num_batches_tracked` is.
  Every converter's rename/transform is ported into its model's Swift loader, so all non-excluded
  models load a raw checkpoint end to end, each verified by an `NFKMLXTorchParityTests` equivalence
  test: the raw file and the converted file must land identical parameters through the model's own
  `loadWeights` (u2net's legacy `rebnconvN` index rename, colorizer's Sequential table + ConvT
  permute, hifigan's weight-norm fusion — held to 1e-6, the one tolerance, because two float32
  evaluations of `g·v/‖v‖` differ in the last ulp — nafnet/raft/rife's renames, lama's `generator.`
  and fastspeech2's `model.` strips, and pose, whose raw and converted files carry identical key
  names differing only in deconv axis order, which is why `Checkpoint.isNativeTorch` exists).
  Conv-TasNet and the denoiser needed no change — their shape-keyed 3-D branches already read the
  raw layout — and that is verified, not assumed. RAFT found the package's newest MLX hazard:
  its reference reuses each block's `norm3` inside `downsample`, the rename collides the two names
  deliberately, and duplicate keys crash `ModuleParameters.unflattened` with a stack overflow —
  dedupe through a dictionary first (see the gotchas below and `Docs/mlx-runtime-hazards.md`). The
  nafnet/rife/lama/modnet raw checkpoints are in the validation manifest, which grew two acquisition
  routes to serve them: `gdrive` (a Google Drive id, MODNet) and `extract` (a member path inside a
  zip at `url`, LaMa's Lightning `best.ckpt`); the rest download from `url` as before. Their
  equivalence tests read the `IK_RAW_*` keys `fetch.py` stamps and skip when absent, like every other
  parity test.
- `NFKMLXMusic3` — MiniMax Music 3, the text+lyrics→music model, ported stage by stage with a
  measured parity record gating each stage. The full model is a hybrid: a Qwen3-8B autoregressive
  stage over 8 RVQ codebooks (one semantic of 16384 living inside the LM's own vocabulary, seven
  acoustic of 1024 filled per frame by a 4-layer depth decoder — there is no MusicGen delay pattern;
  the depth decoder is what replaces it), a condition encoder blending the 8 per-codebook hidden
  states (synthesis conditions on the hidden states, the codes only close the AR feedback loop), a
  36-layer flow-matching DiT over 8-second latent windows, and a DAC-style Snake vocoder. The oracle
  is diffusers' own implementation (>= 0.40.0 ships `MiniMaxMusic3Vocoder`,
  `MiniMaxMusic3RVQDepthDecoder`, `MiniMaxMusic3ConditionEncoder`, the DiT, and the modular
  pipeline), under its own `musicvenv` interpreter (`oracle_environments.music`) — a third-party
  reference, where the community MLX ports of this model have none. The `music_ar` and
  `music_tokenizer` oracles import the pipeline's own helper functions (`_sample_top_k`,
  `_generate_depth_codes`, `_embed_audio_frame`, `_clean_caption`, `_normalize_lyrics`), so the
  arithmetic compared against is the reference's, not a copy. All five networks are at measured
  parity, the prompt contract is token-exact, and the chained pipeline generates audio end to end.
  - **Prompt contract** (`NFKMusic3Prompt`): the caption/lyrics cleaners, the special-token
    template, the release byte-level BPE through the core `NFKTokenizer` (slow-format vocab/merges
    from `qwen_7B/qwen3-8B-tokenizer-music/`, special tokens from `added_tokens.json`), and the
    CFG-row substitution (interior tokens → `<|audio_cfg|>` 151654) — an exact token match against
    the reference tokenizer over the shared `MUSIC_PROMPTS` cases (markdown, `<|tag value|>`
    rewrites, structure tags, multi-byte text, whitespace forms), both rows. Reaching it required
    the core tokenizer's `qwen2` pretokenization (see the Tokenizers section): under the GPT-2
    default the same prompt encodes to different, valid-looking ids, which no output would ever
    reveal. The cleaners are additionally pinned against the reference's own intermediate strings,
    so a cleaning bug reads as a string diff before it reads as a token mismatch.
  - **`NFKMLXMusicBackend`** (`NFKMLXMusic3.backend(directoryURL:)`, registry `minimax-music3`,
    in `registerAll`): `NFKInputPrompt` + `NFKInputLyrics` (a new core key) →
    stereo 44.1 kHz `NFKAudioAsset` under `NFKOutputAudio`; honors `NFKParameterDurationSeconds` /
    `Seed` / `Steps` / `GuidanceScale`. The stages load from the release directory per run and each
    is freed when its part is done (`clearCache` between): the bf16 LM (16 GiB) and the float32 DiT
    (9.7 GB) together exceed a 32 GB machine's working set and the pipeline is strictly sequential.
    There is no random-weights form — the factory takes the release directory, and `isReady`
    reports presence. Cancellation is honored between stages and per flow step; progress reports
    through the job.
  - End to end, measured on the real weights
    (`testTheMusicBackendGeneratesAClipEndToEnd`): a 2-second request produces 1.997 s of stereo
    audio at RMS 0.051 in 77 s wall clock, the whole 27 GB stack staged through. A sampled song
    cannot be compared to the reference bitwise (the random streams differ by construction) — the
    per-stage records are the numeric ground; the e2e asserts duration, rate, channels, and that
    the clip is signal rather than silence or clipping. `IK_MUSIC3_KEEP_CLIP` keeps the WAV for
    listening.
  - Quantized releases and residency
    (`NFKMLXMusic3.quantizeRelease(at:to:bits:transformerBits:groupSize:)`): writes a quantized
    copy of the release in the release's own layout, so `backend(directoryURL:)` takes it
    unchanged — 27 GB falls to **7.7 GiB**. The split default is measured, not assumed: at 4-bit
    the language model holds (first-step logits cosine **0.99933** with its input embedding packed
    too, **0.99952** with the embedding left bf16; prefill 0.9905 against the same full-precision
    parity records) while the DiT's velocity falls to **0.9775** — the flow field is the
    quantization-sensitive stage — so the DiT defaults to **8-bit**, where it measures **0.99990**
    (6-bit measures 0.99844, an order of magnitude worse for ~0.6 GB, so it is not the default;
    `testTheDiTQuantizationBitWidthSweep` is the record). The LM's `Linear` layers and its input
    embedding pack — the model is untied, so the embedding is separate from the packed `lm_head`, and
    at 1.6 GiB it is the stack's largest tensor (`includeEmbeddings: true`, reclaiming 1.10 GiB). The
    vocoder and condition encoder copy through unquantized; the LICENSE copies too — it travels with
    the weights. Whether the stages stay loaded between runs is decided from the weights
    (`keepsStagesResident`): stack bytes + a 4 GB reserve (activations + the CFG pair's KV cache)
    against the recommended working set — deliberately not live free memory, which a resident
    backend's own weights would count against and evict themselves. The quantized stack goes
    resident (measured: two consecutive 2-s generations at 34.5 s / 32.5 s with `resident true`);
    the full-precision stack stages per run, with each stage now scoped so the language
    model releases before the DiT loads (the original code's locals lived to function exit, so the
    claimed staging never actually happened — found while making residency real).
  - **Vocoder** (`NFKMusic3VocoderNet`, latents `[B, T, 128]` → stereo `[B, T·512, 2]`): waveform
    cosine 0.9999999999990, worst |difference| 8.3e-7, all 121 tensors accounted both directions,
    first numeric run. Stereo folds the latent's channel halves into the batch through one shared
    decoder, pinned weight-free by `testSwappingTheLatentHalvesSwapsTheStereoChannels`. The released
    file is already safetensors, so there is no offline converter: `loadVocoderWeights` fuses the
    weight-norm pairs itself (`g·v/‖v‖`, norm over every axis but the first, which covers the
    forward and transposed convolutions alike) and transposes layouts, all gated on
    `needsConvTranspose` so a fine-tuned save round-trips. The Snake α is stored `[1, C, 1]` for the
    reference's NCL and held `[1, 1, C]` for NLC; the loader transposes it under the same gate.
  - **RVQ depth decoder** (`NFKMusic3DepthDecoderNet`, 4 causal layers, a learned 16-position
    embedding rather than a rotary, 7 heads over 1024 codes each, an offset-packed
    `audio_embeddings` table of 7 × 1024): forward 0.999999999997, heads 0.999999999996, projection
    0.9999999999995, embedding 1.0 — the record covers all four parameter families because the
    pipeline reads them through different paths and a forward alone touches only the first. 47/47
    tensors accounted both directions; ships bf16, loads at float32 by default.
  - **Condition encoder** (`NFKMusic3ConditionEncoderNet`): 0.9999999999997. A learned softmax
    blend of the 8 per-codebook hidden states, a scalar gain, a 3-wide convolution, and PyTorch's
    exact nearest-neighbor resample onto the latent rate — `floor(i · frames/latents)` with the
    scale at Float precision; 13 frames land on `int(13 · 44100/24000 · 960/512)` = 44 latents.
  - **Flow-matching DiT** (`NFKMusic3DiTNet`, 36 layers, partial rotary over the leading 32 of each
    head's 64 channels, the trained Fourier timestep prepended as token 0 and stripped after the
    blocks, `ff_in` splitting into `value · silu(gate)`, input `[latent, zeros, condition]` on
    channels where the zeroed block is the reference's unfilled audio-prompt slot): velocities
    0.999999999994–0.999999999999 at three timesteps and the zero-condition unconditional branch.
    The release is float32 and sharded under the diffusers spelling, which
    `NFKMLXReleaseWeights.files` now also resolves (`diffusion_pytorch_model.safetensors[.index.json]`).
    Its 9.7 GB cannot sit in a structural test, so `testTheDiTReleaseIsAccountedBothDirections`
    enumerates: the tiny module's key template expanded to 36 layers equals the shard index's own
    key set exactly. `NFKMusic3FlowSchedule` matches diffusers' `FlowMatchEulerDiscreteScheduler`
    (`invert_sigmas`: σ = 1 − linspace(1, 1/N, N) with a terminal 1; the model consumes σ directly
    as its timestep; a step is `x + (σ_next − σ)·v`), measured against the scheduler configured from
    the release's own config. `NFKMusic3FlowMatcher` is the windowed loop — 200-frame windows at hop
    100, the overlap re-blended toward the previous window's carry at every Euler step
    (`(1 − (1 − 1e-6)σ)·noise + σ·previous`) and locked after it, crops 86/258 latents at the
    stitch — with the boundary lock pinned weight-free.
  - **Autoregressive stage** (`NFKMusic3AutoregressiveStage` over `NFKMLXLanguageNet`, whose
    embed / hiddenStates-from-embeddings / logits-from-hidden seams were opened for it): parity
    bf16 both sides, the Gemma E4B treatment, because the LM's geometry counts to 8,584,475,648
    parameters (measured by `NFKMLXModelSizing` before any load: 16.0 GiB at 16-bit, 32.0 GiB at
    float32, which does not fit this machine). Teacher-forced with the reference's own sampled
    codes so the comparison measures the networks rather than two random streams: prompt prefill
    0.99993, first-step logits 0.999995, guided band 0.99998 with 51/52 shared top-50 candidates
    and the same argmax, fused frame hiddens 0.99994. The CFG pair is a batch of 2 through one
    cache; a frame is one position (the 8 code embeddings sum, scaled by 8^-0.5); the depth
    interleave replaces any MusicGen delay pattern; the warm-up decode step past `<|audio_start|>`
    is not an emitted frame; and guidance is gated to the conditional branch's top-50 before
    sampling. `NFKMusic3Sampler` takes its top-k threshold by CPU sort — `MLX.top` is unsorted, and
    reading its last slot as the threshold silently turns sampling into argmax.
  Diffusers enforces the prompt and frame limits the community ports drop (a > 5000-token
  prompt raises; frames cap at 9000), and this port additionally enforces what neither does: prompt
  + frames must fit the LM's 10240-position budget (`NFKMusic3Contract.positionBudget`), rejected
  before any forward runs. The music LM's config is transformers-5.x-shaped, which
  `NFKMLXLanguage.configuration(fromHuggingFace:)` now reads: `layer_types` listing only
  `full_attention` is dense (only a mixed stack is rejected), and `rope_theta` nests under
  `rope_parameters`.
  The weights are not permissively licensed (MiniMax-Music3 Community License: UI attribution in
  commercial products, separate authorization above USD 20M revenue, safeguard obligations for
  hosted generation) — recorded in `Docs/companions.md`, the manifest's MUSIC3 entry, and the LICENSE
  fetched beside the weights.
- `NFKMLXRetinaFace` (`@objc`) — real face detection with five-point landmarks, and the detector the
  CodeFormer reference pipeline runs through facexlib. The released **mobile0.25** model: a MobileNetV1
  backbone at quarter width (a plain stem then depthwise-separable blocks), a three-level FPN fusing
  top-down with **nearest** resampling, three SSH context modules (a 3×3 branch beside 5×5 and 7×7
  receptive fields built from stacked 3×3s, concatenated then activated together), and per-level class
  / box / landmark heads over two anchors a cell. Run through `NFKMLXDetectionBackend`
  (`NFKInputImage` → `NSArray<NFKDetection *>` under `NFKOutputDetections`, boxes normalized 0…1,
  origin top-left) or `detector(weightsURL:)` for the landmarks, which is what alignment needs.
  `+register` under `retinaface-mobile025`. The input is BGR 0…255 minus `[104, 117, 123]`, because
  the reference reads its frames through OpenCV. Feeding RGB is a quiet defect, not a loud one:
  measured on the validation portrait, the face is still found and the confidence is unchanged to three
  decimals (0.9971 against 0.9975 — the wrong order scores marginally higher), while the box moves to
  IoU 0.962 and the landmarks shift by up to 5.7 px, which is enough to move the aligned crop and
  therefore the restoration. Nothing in the values reveals the order, so
  `testTheChannelOrderIsLoadBearing` pins it by measuring that displacement; asserting on confidence
  would have passed with the swap in place. The scale is detectable where the order is not, so
  `prepared` asserts its input is `0...1` rather than `0...255`. Neither is reachable from the public
  API — `faces(in:)`, `detector(weightsURL:)`, and `backend(...)` all take a `CGImage` and convert
  internally — so this is a maintainer hazard rather than a consumer one. Anchors are
  generated per level from `minSizes` / `steps` and decoded with the reference's variances
  (`0.1`/`0.2`): a centre is the anchor's centre plus a variance-scaled offset of the anchor's size,
  and a size is the anchor's size times the exponential of its offset. The checkpoint's ImageNet
  classifier (`body.fc`, `body.avg`) and every `num_batches_tracked` counter are dropped rather than
  loaded — they are not parameters of the detector. Reference parity against facexlib's own
  RetinaFace over the whole pre-suppression tensor (box cosine 0.9999999999980, class
  0.9999999999999969, landmark 0.9999999999980, and an exact anchor grid), and end to end through
  decoding and suppression against its `detect_faces` (same face count, **box IoU 1.0**, landmarks
  within a pixel). Weights: `github.com/xinntao/facexlib/releases` `detection_mobilenet0.25_Final.pth`,
  1.7 MB — negligible beside CodeFormer's own checkpoint, which is why it is the recommended detector.
- `NFKMLXCodeFormer` (`@objc`) — real face restoration: the reference CodeFormer (sczhou) in `MLXNN` —
  a VQGAN encoder and generator built as the reference's flat heterogeneous `blocks` list (residual
  blocks with a 1×1 skip projection where the width changes, single-head spatial attention at
  resolution 16, asymmetric-pad stride-2 downsamples, nearest ×2 upsamples; GroupNorm at epsilon
  1e-6), a codebook under `quantize.embedding`, and a Transformer code-predictor whose queries and
  keys carry the position embedding while the values do not (the reference's fused
  `in_proj_weight`/`out_proj` layout is kept). The quantized features **always** take the degraded
  latent's per-channel statistics (`adaptive_instance_normalization`): the reference's signature
  defaults `adain=False`, but its released `inference_codeformer.py` passes `True`, and matching the
  shipping behavior rather than the signature default is a ratified decision. The
  **controllable feature transformation** (`NFKCFFuseBlock`, one per connect resolution 32/64/128/256)
  modulates the generator with a learned scale and shift weighted by the fidelity `w` — 0 is full
  generative quality, 1 keeps the degraded input's detail. Run through `NFKMLXModuleBackend` (aligned
  face → restored face at the model resolution). `photoBackendWithFidelity:weightsURL:` takes a whole
  Photograph: it detects every face, aligns each to the reference's five-point 512 template, restores
  it, and composites the result back through the inverse transform with a feathered edge. Detection and
  alignment are `NFKMLXFaceAlignment`, built on **Vision** — no weights, no download, no third-party
  code, which is the rule the core applies to its own backends. It is not facexlib's RetinaFace, so a
  crop here is not byte-identical to the reference pipeline's and a restored photograph differs slightly
  from it; what the model does to a crop is unchanged, and that is what the parity record measures. The
  alignment is a **similarity** transform (uniform scale, rotation, translation, no shear), solved in
  closed form as one complex multiply over the centered point sets — a full affine would stretch the
  face onto the template exactly and hand the model a distorted subject. Vision reports each feature as
  a contour rather than a point, so an eye is its centroid, the nose is the lowest point of its contour,
  and the mouth corners are the outer lip's extremes in x. An image with no detectable face passes
  through unchanged. The detector is selectable and defaults to RetinaFace, the reference
  pipeline's own, so the crop is the crop facexlib produces; `photoBackendWithFidelity:weightsURL:
  detectorWeightsURL:` takes its 1.7 MB checkpoint. `NFKMLXVisionFaceDetector` is the alternative when
  a download-free path matters more than matching the reference. They disagree measurably — on the validation portrait, box IoU
  0.65 and a worst landmark disagreement of 15.7 px over a 960×1200 frame — so the choice changes the
  restoration, and `NFKMLXFaceAlignmentTests` records that number rather than describing it.
  A landmark assertion cannot validate the crop — the transform maps landmarks
  onto the template by construction, so it stays true however the drawing lands. Only the crop's
  Content can: `testTheAlignedCropContainsTheFace` detects a face inside the crop and checks it fills
  and centers it. That is what caught a real defect here — CoreGraphics orients an image for a y-up
  space, so drawing inside the flipped context produced a crop mirrored about the image's centre (the
  subject's chest instead of the face) while every number stayed in tolerance. Both the crop and the
  paste-back therefore carry a second, per-draw flip. `IK_VAL_FACE` is the portrait the detection tests
  read: a NASA Apollo XI photograph, a US government work and public domain, fetched by
  `Tools/validation-assets/fetch.py` as an `input` asset (no conversion step). `+register` under
  `codeformer`. `w` is the Objective-C knob (`+backendWithFidelity:weightsURL:error:` and its two
  download peers, clamped to 0…1) — the same role the variant enums play for the models that have
  them, since one backend restores at one fidelity. `remapReferenceKey` translates the fuse dictionary's resolution keys and the
  positional Sequentials (`scale.0`, `idx_pred_layer.0/1`) — the coders' `blocks.N` indices land on
  real arrays and pass through. Reference parity against CodeFormer's own architecture on the
  released `codeformer.pth` at the real inference settings (w 0.5, AdaIN on): code logits cosine
  0.9999999999985, **code agreement 1.0**, restored face 0.9999999999987, and the public backend path
  0.9999968 (8-bit CGImage quantization). Forward, fidelity effect, geometry, remap, and round-trip
  tested.
- `NFKMLXZeroDCE` (`@objc`) — a real single-forward low-light enhancer: the Zero-DCE DCE-Net (seven
  3×3 convs with U-style skip concatenations → 24 curve-parameter channels) in `MLXNN`, run through
  `NFKMLXModuleBackend` (dark image → brightened image). Enhancement applies `x = x + r·(x²−x)` eight
  times. `+register` under `zero-dce`. Names match the reference (`e_conv1`…`e_conv7`), so
  `Tools/zero-dce-to-safetensors` only extracts. Forward + round-trip tested.
- `NFKMLXMODNet` (`@objc`) — real trimap-free portrait matting: the reference three-branch MODNet
  (ZHKKKe) in `MLXNN` over one **MobileNetV2** encoder — a low-resolution branch (squeeze-excitation
  on the deepest feature, then two 5×5 stages) deciding what the subject is, a high-resolution branch
  recovering boundary detail, and a fusion branch producing the matte. Run through
  `NFKMLXMattingBackend` (portrait → straight foreground + alpha). `+register` under `modnet`;
  factory sets `train(false)`. Its distinctive layer is **`IBNorm`**: the first half of a layer's
  channels are batch-normalized and the rest instance-normalized **without affine terms**, then
  concatenated — so the checkpoint carries parameters for only half the width. The input normalizes
  to `-1...1` (the demo's `Normalize(0.5, 0.5)`), and the strides need sides that are multiples of 32,
  so `matte(_:)` resizes for the network and resizes the alpha back. `remapReferenceKey` strips the
  `module.` prefix and the per-branch prefixes, translates the backbone's `features.N` and each
  inverted residual's `conv.M` (whose slots shift when the expansion is absent — the
  expansion-1 block's depthwise pair sits at 0/1, not 3/4), and unwraps every `Conv2dIBNormRelu`'s
  `layers` Sequential. The checkpoint stores the backbone **twice**, once per branch holding a
  reference to it; the copies are identical, so the loader keeps the first. Reference parity
  against MODNet's own network on the released photographic checkpoint (alpha cosine
  0.9999999999994, mean |difference| 1.4e-8), every parameter covered on the first triage run.
  Weights: `python3 -m gdown 1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz` (26 MB; the HF ONNX exports remain
  unreadable by this loader).
- `NFKMLXYOLO` (`@objc`) — real object detection: the reference **YOLOv8** (ultralytics) in `MLXNN` —
  a CSPDarknet backbone of `Conv` (convolution + **BatchNorm epsilon 1e-3** + SiLU) and `C2f` stages
  ending in SPPF (three chained 5×5 stride-1 max pools through `NFKMLXResample.maxPooled`), a PAN-FPN
  neck fusing strides 8/16/32 both ways, and a decoupled head with **distribution-focal box
  regression**: each box side is a softmax over 16 bins whose expectation (the `dfl` convolution,
  fixed to 0…15, loaded from the checkpoint) is a distance from the cell's anchor point. v8 has no
  objectness — confidence is the best class probability. Box decode and greedy per-class NMS in
  Swift. The suppression thresholds are deliberately the reference's 0.25 / 0.7, not the 0.45 this
  module first shipped with — a ratified behavior change, so do not "restore" the older value.
  `NFKMLXYOLOBackend` reads `NFKInputImage` → `NSArray<NFKDetection *>` under `NFKOutputDetections`;
  boxes are normalized 0…1, origin top-left.
  The `+backendWith…labels:` factory attaches class names. `+register` under `yolo`.
  `remapReferenceKey` maps the reference's `model.N` module list onto named stages and the head
  branches' positional Sequential (`cv2.i.0/1/2` → `conv1`/`conv2`/`out`). Reference parity
  against ultralytics' own YOLOv8n on the released `yolov8n.pt` over the full pre-suppression tensor
  (box cosine 0.9999999999999638, class cosine 0.9999999999944721, same top class at the same
  anchor), and against ultralytics' `predict` end to end on a 16:9 frame through the public backend
  (9/9 detections, same classes, worst box IoU 0.9999984). A frame is fitted the reference's way:
  scaled by the smaller ratio, padded with gray 114 to a multiple of 32 (`auto` mode, so a wide frame
  runs at 640×384 rather than wasting a third of the input), and the decoded boxes have that padding
  and scale undone before they are normalized against the caller's own frame. Forward, decode, NMS,
  letterbox, remap, and round-trip tested. YOLOv8s is at parity too (box cosine
  0.99999999999997, class 0.9999999999997): it has the **same depth** as the nano model at twice the
  width, because the releases scale by two independent multiples — reading only one of them right
  still loads and is still wrong. **YOLOv8m** is the first size where both multiples change — wider
  stages and deeper C2f repeats `[2, 4, 4, 2]` — and it matches too (box 0.99999999999996, class
  0.999999999998). `NFKMLXYOLOVariant` (`.nano`/`.small`/`.medium`/`.large`/`.extraLarge`) selects the size. l and x
  are at parity too (box 0.99999999999993 / 0.99999999999994): both run the full depth multiple, so
  their C2f stages repeat `[3, 6, 6, 3]`, and x is wider again. The records must be made at
  `--size 640`, where the reference's letterboxing is an identity — generating one at another size
  produces a different anchor count and looks like a model failure.
- `NFKMLXRTDetr` (`@objc`) — real object detection, the license-clean (Apache-2.0) alternative to the
  AGPL YOLO: RT-DETR (`RTDetrForObjectDetection`, PekingU/lyuwenyu) in `MLXNN` — a **ResNet-D**
  backbone (deep 3-conv stem, avgpool-in-shortcut bottleneck), a **hybrid encoder** (an AIFI transformer
  on the deepest feature plus a CSP-RepVGG FPN/PAN), **query selection** over generated anchors, and a
  **deformable-attention decoder** with iterative box refinement. Run through `NFKMLXRTDetrBackend`
  (`NFKInputImage` → `NSArray<NFKDetection *>` under `NFKOutputDetections`, boxes normalized 0…1, origin
  top-left). `+register` under `rtdetr`. DETR-family, so there is no non-max suppression — the
  one-to-one training makes the queries distinct — and the image processor **squashes** to 640×640 (no
  aspect-preserving pad), so a normalized box maps to the original frame unchanged.
  The decoder's cross-attention is multi-scale deformable attention, not DCNv2 deformable
  Convolution: a `grid_sample` bilinear gather at learned offset locations (`loc·size − 0.5`,
  `align_corners=false`, zero padding), which MLX expresses with `takeAlong` the way RAFT/RIFE do their
  warps — so unlike BiRefNet (which needs a deformable *conv* MLX has no op for), RT-DETR is portable.
  Anchors are generated per forward (`anchor_image_size=None`) as `logit(centre/size)` with a validity
  mask; query selection takes the top `num_queries` by the best class score; the decoder runs
  `sigmoid(reference)`, a per-layer `query_pos_head` MLP position, deformable cross-attention over the
  flattened encoder tokens, and `sigmoid(corner + inverse_sigmoid(reference))` box refinement, with
  `class_embed`/`bbox_embed` **cloned per layer** (`with_box_refine`). BatchNorm runs in eval (the
  reference freezes the backbone BNs). Reference parity against transformers' own
  RTDetrForObjectDetection, seam by seam at a tiny config: backbone 0.99999999999998, the PAN encoder
  0.9999999999999988, query-selection scores (`enc_class`) 0.9999999999999925 and boxes (`enc_coord`)
  0.9999999999999999, and the deformable-attention decoder exact over the reference's
  selection (logits 0.9999999999999958, boxes 0.9999999999999966). Also at parity on the released
  `PekingU/rtdetr_r50vd` weights end to end (`run_reference.py rtdetr_real`, `IK_PARITY_RTDETR_REAL` +
  `IK_VAL_RTDETR`): logits cosine 0.9999999999887, boxes 0.9999999999642 over the reference selection —
  the r50vd ResNet-50-vd geometry, the actual checkpoint, and the loader (the **stage-1 stride-1
  shortcut** re-indexed to `.0.`, the `model.` prefix stripped, the tied top-level `class_embed`/
  `bbox_embed` and every `num_batches_tracked` dropped) exercised, which the tiny config does not cover.
  Two facts are load-bearing, both found by the parity run. The oracle's shared `_randomized`
  randomizes all floating state including the BatchNorm `running_var` buffer, which can go negative, and
  `rsqrt(var + eps)` is then NaN in both the reference and the port; `run_rtdetr` therefore randomizes
  only the trainable parameters, leaving the BN buffers physical (mean 0, var 1). And the end-to-end
  top-k selection is float-tie-sensitive: `torch.topk` and MLX's `argSort` break a sub-ulp score tie
  differently, so one or two of the selected queries can swap (end-to-end boxes 0.981 where the decoder
  over the reference selection is exact) — the same near-tie class as the GGUF/Whisper greedy flips, so
  the decoder parity is measured over the reference's own selection and the end-to-end boxes are asserted
  at a tolerance that reflects the tie rather than a modeling error. `NFKMLXRTDetrConfiguration`
  (`.tiny`/`.r50vd`/`.r18vd`/`.r34vd`/`.r101vd`) selects the geometry; the backbone/encoder/decoder use
  `[Module]` arrays so the module keys mirror the checkpoint's nested `nn.Sequential` layout, with only
  the shortcut re-index in the remap. All four released sizes are at parity (`NFKMLXRTDetrVariant`,
  registered as `rtdetr-r18vd` / `-r34vd` / `-r101vd` beside `rtdetr`): r18vd and r34vd run ResNet
  **basic** blocks (`NFKRTDetrBasicLayer`, two ConvNorms and the avgpool-in-shortcut at stride 2),
  narrower stage widths, and three or four decoder layers; r101vd is the bottleneck backbone at depths
  `[3, 4, 23, 3]` with a 384-wide encoder and 2048 FFN. Over the reference's selection: logits
  0.99999999999881 / 0.99999999999911 / 0.99999999997880, boxes 0.99999999999285 / 0.99999999999693 /
  0.99999999979751. RT-DETR-v2 (a v2 deformable-attention variant) is the remaining RT-DETR candidate.
- `NFKMLXRFDetr` (`@objc`) — real object detection under Apache-2.0 (RF-DETR base, Roboflow), a two-stage
  Group-DETR detector ported from transformers' `RfDetrForObjectDetection`, at reference parity on
  both a tiny random config and the released weights (`testRFDetrMatchesTheReference` /
  `…OnReleasedWeights`). A **windowed DINOv2** backbone (each block partitions the patch grid into
  `num_windows²` local windows with a replicated CLS; a global-attention block — the out-index layers
  2/5/8/11 — unpartitions to one sequence per image before attending and re-partitions after; selected
  stages are layernormed, the CLS dropped, unpartitioned, and reshaped to feature maps), a C2f /
  RepVGG scale projector (concat the stage maps → a C2FLayer → a channels-first LayerNorm, which in NHWC
  is a plain last-axis LayerNorm), **two-stage query selection** (an `enc_output` Linear + LayerNorm, the
  `enc_out_class`/`bbox` heads over every token, top-k by the class max), **mixed queries** (a learned
  `reference_point_embed` refined by the top-k coords in direct normalized box space — cxcy = Δxy·wh + xy,
  wh = exp(Δwh)·wh, not the sigmoid space RT-DETR uses — plus a learned `query_feat`), and an **LW-DETR
  deformable decoder** (self-attention with the query position added to q and k and the value without
  position; deformable cross-attention over one feature level; no iterative box refine — the reference
  points are constant and one box refinement runs at the end in the head). Group-DETR collapses to one
  group at inference.
  Two seam bugs the parity ladder caught. The global-attention block re-partitions using the
  Unpartitioned shape — the reference reassigns `hidden_states` before reading `.shape`, so it takes
  `[B/windows², windows²·seq, C]`, not the original windowed shape (a wrong shape crashes the reshape).
  And the released backbone interpolates its 518-trained pos_embed to 560 (37→40 patches) with `bicubic,
  align_corners=false, ANTIALIAS=true`, which is not a no-op when upsampling: antialias uses the PIL cubic
  coefficient **a = -0.5** (torch's non-antialias bicubic, and `NFKMLXBicubic`, use -0.75) with per-output
  weight normalization — `antialiasResampleMatrix` builds the `[out, in]` operator (measured to reproduce
  torch to 2e-6). The width/height-swapped window reshape, the direct box space, the channels-first
  LayerNorm eps (conv norms 1e-5, the projector norm 1e-6), and the DETR sinusoidal position embedding
  were all correct as first written.
  The released file loads directly on device. It is prefix-free with the original Roboflow naming
  (`backbone.0.encoder.encoder.*`, `transformer.*`, `refpoint_embed`, a fused `self_attn.in_proj_*`);
  `remapReferenceKey` converts it to the module names — the exact map derived by tensor-identity matching
  the raw checkpoint (487 keys) against the converted state_dict (499), since transformers exposes no
  mapping dict — and `loadWeights` splits the fused `in_proj` (packed `[q; k; v]`) into q/k/v_proj. The
  base geometry was read from the release: `num_labels` 91, `decoder_n_points` 2 (not 4), resolution
  560. `+register` under `rf-detr`; `detect()` applies the image processor's ImageNet normalization
  (mean/std, 560 resize; the PIL-bilinear resize is a documented approximation). The four later
  releases are at parity too (`laterRelease(resolution:decoderLayers:)`, `NFKMLXRFDetrVariant`,
  registered as `rf-detr-nano` / `-small` / `-medium` / `-large`): a patch-16 DINOv2 at the release's own
  resolution (384 / 512 / 576 / 704) with two windows a side and out-indices 3/6/9/12, over 2 / 3 / 4 / 4
  decoder layers. Over the reference's selection, logits 0.99999999998936 / 0.99999999997724 /
  0.99999999994142 / 0.99999999988494 and boxes ≥ 0.99999999221948. The tiny parity loads the
  oracle's converted weights and the released parity loads the raw file through `loadWeights`, so both the
  network and the on-device naming conversion are measured. Oracle: `run_reference.py rf_detr` /
  `rf_detr_real` under the `rfdetr` env (transformers 5.16.1).
- `NFKMLXSegFormer` (`@objc`) — real semantic segmentation: the SegFormer MiT transformer encoder
  (efficient self-attention with spatially reduced keys/values + Mix-FFN depthwise conv, so no
  positional embedding) and an all-MLP decode head in `MLXNN`, run through `NFKMLXModuleBackend`. The
  argmax label map is emitted as a grayscale image under `NFKOutputImage`; recover the class index as
  `round(gray·(classCount−1))`. `+register` under `segformer-b0`. Reference parity against
  transformers' own `nvidia/segformer-b0-finetuned-ade-512-512` (logit cosine 0.99999992, label
  agreement 99.99%). `remapReferenceKey` regroups the reference's flat
  `segformer.encoder.block.<stage>.<index>` and its separate `patch_embeddings.N`/`layer_norm.N` lists
  onto per-stage names, and concatenates the reference's separate `key`/`value` into this port's one
  fused `kv` — a two-into-one a 1:1 key map cannot express. Forward, label-map, and round-trip tested.
- `NFKMLXSwinIR` (`@objc`) — real transformer super-resolution: SwinIR (shallow-feature conv → residual
  Swin Transformer blocks → pixel-shuffle upsampler) in `MLXNN`, with real window attention — window
  partition/reverse, cyclic shift with the standard attention mask, and a relative-position bias table
  gathered by a precomputed index. Run through `NFKMLXModuleBackend`; the input side must be a multiple
  of the window size. `+register` under `swinir-x4`. Reference parity against JingyunLiang's own
  `network_swinir.py` on the released `001_classicalSR_DIV2K_s48w8_SwinIR-M_x4` (cosine 0.99986, mean
  pixel |difference| 0.0037). The input is **RGB-mean centered** — the reference subtracts
  `(0.4488, 0.4371, 0.4040)`, scales by `img_range`, and restores it at the end; leaving that out was
  the fifth missing input normalization in this sweep.
  Non-power-of-two scaling is implemented and at parity on the released x3 (cosine 0.99987, mean
  0.0036). The reference `Upsample` reaches a power-of-two scale with repeated ×2 pixel-shuffle stages
  and a scale of three with one ×3 stage, because a factor-3 shuffle is not a composition of factor-2
  ones — so an x3 checkpoint packs `9·C` channels into a single stage where x4 packs `4·C` into each of
  two, and fits only its own geometry. `NFKMLXSwinIRVariant` selects the release, and `NFKMLXSwinIR.makeNet` throws
  `unsupportedConfiguration` for a scale the reference builds no upsampler for, rather than silently
  truncating `log2`. x8 is the same network with a third ×2 stage (0.99990). The lightweight
  release is not the classical network at a smaller size: it reconstructs through the reference's
  `pixelshuffledirect` — one convolution to `3·scale²` channels and a single shuffle, with neither the
  convolution before the upsampler nor the one after it — so `convBeforeUpsample` and `convLast` are
  absent rather than unused, and its checkpoint carries a single `upsample.0` (0.99991, mean 0.00097). The shuffle itself is
  the shared `NFKMLXPixelShuffle`, which BiSeNet, RIFE, VideoSR, and SwinIR all use. Forward, window
  helpers, and round-trip tested.
  Every released SwinIR now loads (`NFKMLXSwinIRVariant`: `.classicalX2`, `.lightweightSRX3`,
  `.lightweightSRX4`, `.realWorldX4Medium`, `.realWorldX4Large` beside the four above), each at float
  parity ≥ 0.9999999999993 with a mean pixel difference under 4e-7. The two real-world GAN releases
  reconstruct through the reference's **`nearest+conv`** upsampler (`NFKMLXSwinIRUpsampler.nearestConv`:
  a 64-wide tail, two nearest-×2 + convolution + leaky 0.2 stages, `conv_hr`, `conv_last`) instead of the
  pixel shuffle, and the large one is 240 wide over nine six-block groups with the **`3conv` residual
  connection** (`NFKMLXSwinIRResidualConnection.threeConv`: a 3×3 → 1×1 → 3×3 squeeze at a quarter width
  with leaky 0.2 between, after every RSTB and after the body; the remap moves `conv_after_body.N` and
  `layers.K.conv.N` onto `conv_after_body_3conv` / `layers.K.conv3`). Reaching them found a real
  defect in the classical tail: `conv_before_upsample` activates with a leaky ReLU at 0.01, not the
  plain ReLU this port had shipped with — on the released classical ×4 through the backend's 8-bit
  bridge the mean pixel difference fell from 0.0037 to 0.00136 (×3 0.0036 → 0.00119, ×8 0.0035 →
  0.00095); the lightweight release has no such tail and is unchanged.
- `NFKMLXColorizer` (`@objc`) — real colorization (Zhang et al. ECCV-16): eight VGG-style conv blocks
  (BatchNorm block ends; blocks 5–6 dilation 2) over the L channel predict a distribution over 313
  quantized ab bins; the annealed mean is the checkpoint's own `model_out` 1×1 conv (renamed
  `out_ab`), so no separate cluster file. `NFKLabColor` implements sRGB ↔ CIELAB (D65) in MLX ops,
  tested against CIE reference values (white L*=100, mid-gray L*=53.39) plus a full-gamut round-trip.
  Predicted ab recombines with the original full-resolution L, preserving luminance exactly. The
  factory sets `train(false)` so BatchNorm uses the checkpoint's running statistics.
  `Tools/colorizer-to-safetensors/convert.py` performs the complete `nn.Sequential` rename and the
  ConvTranspose axis swap (`[in,out,kH,kW]` → `[out,in,kH,kW]`), so the release loads directly —
  no remap is left to Swift. `+register` under `colorizer-eccv16`. Reference parity against the
  released eccv16 (ab cosine 0.9999999998, colorized sRGB cosine 0.9999971). The L and ab resampling is
  bilinear, as the reference's `nn.Upsample` is: nearest neighbour scored the ab prediction at 0.96, so
  this was a real defect. `abPrediction` exposes the network's output before the lightness
  goes back, so a parity failure says network or Lab conversion.
  `NFKMLXSiggraphColorizer` is the second released colorizer — a separate network, not a
  configuration of this one, and at reference parity against richzhang's own `siggraph17.py`
  (ab cosine 0.9999999999996, colorized 0.9999999999). It is 16 blocks (`model1…model10` plus `model8up`/`model9up`/`model10up` and
  the `model{3,2,1}short{8,9,10}` U-Net shortcuts), a **four-channel** input (L, an ab hint, and a
  hint mask), a `model_out` regression head emitting ab directly, and a 529-class `model_class`
  auxiliary head that only supervises training — the loader drops it. Blocks five and six **dilate**
  rather than downsample, and the downsampling elsewhere is a stride-2 **subsample**, not pooling.
  `remapReferenceKey` counts convolution slots per block because the encoder blocks open with a
  convolution while the decoder blocks open with a ReLU, so the same slot number means different
  layers. `+register` under `colorizer-siggraph17`; weights at
  `colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth`, converted with
  `Tools/colorizer-to-safetensors --passthrough` (the eccv16 rename does not apply). With an empty
  hint it colorizes automatically; `predictAB(lightness:hint:mask:)` takes user strokes. Forward, Lab math, bin softmax, and round-trip tested.
- `NFKMLXPose` (`@objc`) — real top-down pose estimation (SimpleBaseline): `NFKMLXResNetBackbone` as
  ResNet-50 and a transposed-convolution head produce one heatmap per joint in `MLXNN`; the argmax of
  each heatmap is a joint location, refined a quarter cell toward its larger neighbor as the reference
  decode does. `NFKMLXPoseBackend` reads `NFKInputImage` → `NSArray<NFKKeypoint *>` (a new core value
  type) under the new core key `NFKOutputPose`; positions are normalized 0…1, origin top-left. The
  `+backendWith…jointNames:` factory attaches joint names. `+register` under `pose-simplebaseline`.
  A person crop is taller than it is wide, so the trained geometry is 256×192 (`inputHeight`/`inputWidth`)
  and the input takes ImageNet normalization. Factory sets `train(false)` for BatchNorm running stats;
  the converter swaps the deconv ConvT axes. Reference parity against microsoft's own
  SimpleBaseline (heatmap cosine 0.9999999999961, peak agreement 1.0), on the mmpose ResNet-50 COCO
  release — whose keys are the reference's under a `backbone.`/`head.` prefix, so a strict load of the
  reference doubles as proof the two architectures are one. `remapReferenceKey` maps that prefix and the
  head's positional `deconv_layers.{0,3,6}`/`{1,4,7}` Sequential.
- `NFKMLXDeepLab` (`@objc`) — real semantic segmentation (DeepLabV3): `NFKMLXResNetBackbone` with its
  last two stages dilated (so features reach the head at stride 8) and an Atrous Spatial Pyramid Pooling
  head (1×1 + three dilated 3×3 branches + global image pooling, fused, then a 3×3 convolution before
  the classifier) in `MLXNN`, run through `NFKMLXModuleBackend`. Emits a grayscale class-label map under
  `NFKOutputImage` (same convention as `NFKMLXSegFormer`); the logits upsample before the argmax, and
  the input takes ImageNet normalization. `+register` under `deeplabv3`; factory sets `train(false)`.
  Reference parity against torchvision (logit cosine 0.9999999999999, label agreement 1.0).
  `remapReferenceKey` maps the reference's positional `classifier.N` Sequential onto the module's names.
  Complements `NFKMLXSegFormer` (CNN vs transformer segmentation).
- `NFKMLXResNetBackbone` (`NFKMLXResNet.swift`) — the shared bottleneck residual backbone (ResNet-50 and
  up) in the reference layout, including the stride-to-dilation substitution DeepLab depends on
  (`replaceStrideWithDilation`; the reference gives a stage's first block the previous stage's dilation).
  `remapReferenceKey` names the projection shortcut the reference keeps in a `Sequential`
  (`downsample.0/1` → `downsample_conv`/`downsample_bn`). Pose's ResNet-50 reuses this.
  Its stem pools through `NFKMLXResample.maxPooled` (see the MLX-runtime gotchas below).
- `NFKMLXConvTasNet` (`@objc`) — real time-domain speech separation: a 1-D convolutional encoder, a
  masking temporal convolutional network (depthwise-separable dilated Conv1d blocks with global layer
  normalization `NFKTasNetGlobalNorm` and PReLU), and a shared transposed-conv decoder in `MLXNN`.
  `NFKMLXConvTasNetBackend` reads `NFKInputAudio` → one `NFKAudioAsset` per speaker ("speaker-1",
  "speaker-2", …). `+register` under `conv-tasnet`. Reference parity against `asteroid`'s own
  ConvTasNet on `JorisCos/ConvTasNet_Libri2Mix_sepclean_16k` (per-speaker cosine 0.9999999995).
  `remapReferenceKey` unwraps asteroid's `filterbank` (`_filters`, no bias) and its positional
  `shared_block` Sequential. Every PReLU carries one slope by default, which is the reference's own
  shape — asteroid builds them as `nn.PReLU()`, whose `num_parameters` defaults to 1, and all 49
  slope tensors in the released checkpoint are `[1]`. An earlier note called per-channel slopes a
  sweep item as though the release used them; it does not, and making them the default would diverge
  from it. `perChannelPReLU` offers them anyway, for a fine-tune that wants the capacity: a shared
  slope applied to every channel is the same function, so `loadWeights` **widens** a released `[1]`
  slope to `[C]` against the widths the module reports, the model computes exactly what it computed
  before, and training moves the slopes apart from there. Tested by saving a shared-slope model,
  loading it into a per-channel one, and asserting the separation is unchanged. Forward, separation,
  and round-trip tested.
- `NFKMLXDenoiser` (`@objc`) — real speech noise suppression (Défossez et al.): the same Demucs
  time-domain U-Net as `NFKMLXDemucs` configured with `stems == 1`, so it reuses `NFKMLXDemucsNet` and
  `NFKMLXDemucs.loadWeights` (DRY). `NFKMLXDenoiserBackend` reads `NFKInputAudio` → one cleaned
  `NFKAudioAsset` under `NFKOutputAudio`. `+register` under `denoiser`. Reference parity against
  facebookresearch/denoiser dns48 (cosine 0.99999999999992), which also guards the shared network
  against a change made for the music model breaking the speech one. Single-output and round-trip tested.
- `NFKMLXVAD` (`@objc`) — real voice activity detection (MarbleNet): a mel front end feeding a stack of
  QuartzNet-style blocks — runs of time-channel-separable convolutions with an optional projected
  residual — and a two-class per-frame head; consecutive above-threshold frames merge into spans.
  `NFKMLXVADBackend` reads `NFKInputAudio` → `NSArray<NFKAudioSegment *>` (a new core value type) under
  the new core key `NFKOutputSegments`. `+register` under `vad-marblenet`; factory sets `train(false)`.
  Reference parity against NeMo (cosine 0.99999999999983). The front end (`NFKVADFrontEnd`) is the
  reference preprocessor — preemphasis, a centered 512-point transform, power spectrum through the mel
  filterbank, natural log with a `2⁻²⁴` guard, frames padded to a multiple of two — and it loads its
  window and filterbank from the checkpoint, which carries both; the defaults reproduce them for a
  randomly initialized net. Held in a plain box, not on the `Module`, so those constants stay out of
  `parameters()`. `remapReferenceKey` maps NeMo's flat positional `mconv` list (five entries per
  separable convolution, four per plain one) and its `res.0` shortcut onto the module's names.
  A clip arriving at another sample rate is resampled to 16 kHz through `NFKMLXAudioRate.matched`
  (the parity-proven `julius.resample_frac` port, with the ratio reduced by its greatest common
  divisor first — 44100 → 16000 would otherwise build 16000 polyphase kernels instead of 160). Frame
  times are computed at the model's rate, which is the caller's own seconds because resampling
  preserves duration.
- `NFKMLXSileroVAD` (`@objc`) — real voice activity detection (Silero VAD v6, snakers4), a second VAD
  architecture beside MarbleNet and the first of the per-modality roadmap adds. A learned-STFT
  front end (`Conv1d(1→258, k256, s128)`, no bias) → magnitude of `real[:129]`/`imag[129:]` → four
  `Conv1d+ReLU` (129→128→64→64→128, convs 2/3 stride-2) → a one-layer `LSTM(128→128)` → ReLU →
  `Conv1d(128→1, k1)` → sigmoid, scoring one speech probability per 512-sample chunk (32 ms). It streams:
  each chunk carries the previous chunk's last 64 samples as look-back (context roll; the first chunk
  zeros) and the LSTM state threads across chunks. The whole clip runs as one pass with the chunks on
  the LSTM's sequence axis, which reproduces the reference's chunk-by-chunk stream exactly (zero-init
  state, sequential). `NFKMLXSileroVADBackend` reads `NFKInputAudio` → `NSArray<NFKAudioSegment *>` under
  `NFKOutputSegments`; `+register` under `silero-vad`. v6 differs from v5 in the STFT padding alone:
  v5 pads the 576-sample (64 context + 512 chunk) input symmetrically by 128 and drops transform frame 0;
  v6 pads the right by 64 → 640 → four frames directly, no drop (`NFKMLXHTDemucs.reflectPadded(left:right:)`
  gives the right-only reflect). The LSTM reuses the Demucs bottleneck idiom — MLXNN's `LSTM`
  (`Wx`/`Wh`/fused `bias`, gate order `i,f,g,o`), PyTorch's `bias_ih`+`bias_hh` folded — which is what
  de-risked the port. Reference parity against the released snakers4 JIT (`silero_vad` 6.2.1) on the
  first numeric run: per-chunk cosine 0.9999999999998, max |difference| 6.9e-7, threshold agreement
  32/32. `remapReferenceKey` maps `_model.stft.forward_basis_buffer`→stft, `_model.encoder.{0..3}.reparam_conv`
  →conv1..4, `_model.decoder.rnn.weight_ih/hh`→`Wx`/`Wh`, `_model.decoder.decoder.2`→final; the released
  `.jit` also carries an 8 kHz `_model_8k.*` branch this port drops (`"_model."` is not a prefix of
  `"_model_8k."`, char 7 being `_` not `.`, and the loader skips `_model_8k` explicitly). The converter
  `Tools/silero-vad-to-safetensors` reads the `.jit` with `torch.jit.load` (torch alone, no `silero-vad`
  package) and keeps the 16 kHz `_model.*` in PyTorch layout; the native `.pth`/JIT reader reads the raw
  `.jit` too. The parity oracle (`run_reference.py silero_vad`, llm env, needs `silero-vad`+`torchaudio`)
  streams the JIT chunk by chunk. Resampled to 16 kHz through `NFKMLXAudioRate.matched`.
- The speech-restoration family shares two front-end primitives (`NFKMLXAudioSTFT.swift`):
  `NFKMLXComplexSTFT` reproduces `torch.stft` / `torch.istft` (`center=true`, `pad_mode="reflect"`,
  `normalized=false`) and returns either magnitude+phase (`transform`/`inverse`) or real+imag
  (`transformComplex`/`inverseComplex`) over one window-squared overlap-add; the window is a value the
  caller supplies (sqrt-Hann or Hann). `NFKMLXERB` is the Glasberg-Moore ERB filterbank. A second shared
  file `NFKMLXRecurrent.swift` carries `NFKMLXGRUCell` / `NFKMLXBiGRU` / `NFKMLXRecurrentFold` — the
  PyTorch `nn.GRU` weight fold (bidirectional → forward/backward cells; `b = bias_ih + [bias_hh[:2H], 0]`,
  `bhn = bias_hh[2H:3H]`). The GRU fix lives here: MLX's GRU drops the n-gate hidden bias `b_hn` at
  step 0 where PyTorch keeps it (`n₁ = tanh(W_in x + b_in + r₁·b_hn)`), so the cell adds `bhn` at every
  step; found in GTCRN, it improved MP-SENet too.
- `NFKMLXMPSENet` / `NFKMLXMPSENetFactory` (`@objc(NFKMLXMPSENet_Factory)`) — MP-SENet
  (`yxlu-0102/MP-SENet`, MIT), the first speech-restoration port: a time-frequency transformer that
  denoises the compressed magnitude and phase in parallel. A DenseEncoder, four **TS-transformer** blocks
  (`norm1 → MHSA → norm2 → FFN → norm3`, the FFN a bidirectional GRU over the shared `NFKMLXBiGRU`), and
  parallel mask / phase decoders producing a complex ratio mask. Reference parity against the
  reference `MPNet` on the released `g_best_dns` (waveform cosine > 0.999, every seam exact). The
  released core is the TS-TRANSFORMER, not the conformer the repo's `conformer.py` describes (it is
  unused). The bug was the `batch_first` trap: `nn.MultiheadAttention` / `nn.GRU` default
  `batch_first=False`, so the block runs over axis 0 of `[B·F, T, C]` — one `transposed(1, 0, 2)` around
  the block body fixed a 0.358 parity to > 0.999. `loadWeights` uses `NFKMLXRecurrentFold.fold` + a
  Sequential-index remap. Config: fftSize 400, hop 100, denseChannel 64, 4 blocks, 4 heads, compress
  0.3. The oracle (`run_reference.py mpsenet`) runs from the cloned source on `/usr/bin/python3` (3.9)
  via `IK_MPSENET_SRC`.
- `NFKMLXGTCRN` / `NFKMLXGTCRNFactory` — GTCRN (`Xiaobin-Rong/gtcrn`, MIT), a **~48.2K-parameter**
  real-time speech enhancer, the second restoration port. An ERB band merge/split (bias-free `Linear`s
  over the shared `NFKMLXComplexSTFT`), an SFE unfold, a grouped-convolution encoder/decoder, a **dual-path
  grouped RNN** (DPGRNN, intra/inter over the shared `NFKMLXGRUCell`), and a complex ratio mask.
  Reference parity on the released `model_trained_on_dns3` (waveform cosine > 0.999, every seam —
  encoder / skips / DPGRNN / decoder — exact). **Five bugs, each caught by seam isolation:** `erb_fc` /
  `ierb_fc` are `bias=False`; the grouped-deconv weight transpose is group-aware
  (`[in, out/g, kH, kW]` → `[out, kH, kW, in/g]`); the shared GRU dropped `b_hn` at step 0 (fixed in
  `NFKMLXRecurrent`); the deconv front-pads time for both conv and deconv with `ConvTranspose` padding
  `(2·dilation, 1)` and no crop; and MLX's grouped `ConvTranspose` does not match PyTorch, so a grouped
  deconv runs each group as its own `groups=1` transpose. STFT is sqrt-Hann. The oracle
  (`run_reference.py gtcrn`) runs from the cloned source via `IK_GTCRN_SRC` (torch only, 3.9).
- `NFKMLXSGMSE` (`@objc`) / `NFKMLXNCSNppNet` — SGMSE+ (`sp-uhh/sgmse`, MIT), score-based generative
  speech **dereverberation** / enhancement, the third restoration port and the first generative one. A
  forward OUVE variance-exploding SDE walks a clean complex spectrogram toward the observation; inference
  runs the reverse SDE (a predictor-corrector sampler) from `x_T = y + noise` to `x_0`, scored by an
  NCSN++ network, then inverts the STFT. `NFKMLXNCSNppNet` is ported as the reference's flat `all_modules`
  list (a `[Module]` array, keys match with no remap) walked by an index counter that mirrors the
  reference forward, plus a separate `output_layer`. The one new op is `NFKSGMSEFIR.upfirdn2d` — a
  depthwise FIR resample (kernel `[1,3,3,1]`) the DAC/SNAC resamplers resemble. The complex spectrogram
  packs into four real channels (`[xt.re, xt.im, y.re, y.im]`); the sampler is the OUVE SDE
  (`NFKMLXOUVEScheduler`, a value type with the closed-form mean / std / diffusion) driven by a
  reverse-diffusion predictor plus an annealed-Langevin corrector; the front end is a sqrt-Hann or Hann
  STFT with the `|X|^a · factor` amplitude compression and the time axis padded to a multiple of 64.
  Inference reads the EMA weights: `Tools/sgmse-to-safetensors` lets `torch_ema` apply them
  (`model.eval()` → `ema.copy_to(dnn)`) then dumps `dnn.state_dict()`. At reference parity on the
  released EMA weights — net-seam cosine 1.000000000000 on both released backbone variants: the classic
  `ncsnpp` (attention + progressive input/output skip, `sp-uhh/speech-enhancement-sgmse`) and `ncsnpp_48k`
  (no attention, `progressive='none'`, plain Hann, and the output projection applied before the sigma
  division; the ReverbFX release). The port is config-driven for both (`progressiveOutputSkip` /
  `windowPower` / `attentionResolutions`), and the oracle (`run_reference.py sgmse`, source via
  `IK_SGMSE_SRC`) records the net geometry from the DNN's own attributes so the Swift parity test builds a
  matching config. `backbone='ncsnpp_v2'` (a different two-arg forward) is out of scope. Two traps:
  `torch >= 2.6` defaults `weights_only=True` and refuses the pickled data module, so the converter and
  oracle patch `torch.load`; and the net geometry must match the front-end freq bins or the flat
  `all_modules` walk desyncs (the config carries both). A sampled clip is not bitwise-comparable (random
  stream), so the deterministic net seam is the numeric ground and the e2e asserts signal.
- `NFKMLXStoRM` (`@objc`) / `NFKMLXStoRMNet` — StoRM (`sp-uhh/storm`, MIT), a few-STEP stochastic-
  regeneration follow-on on SGMSE+. A discriminative predictor produces an initial denoised estimate,
  then the score network regenerates from it: the reverse SDE is re-centered on the denoised estimate and
  the score conditions on `[noisy, denoised]`, so the diffusion repairs only residual artifacts in far
  fewer steps (default corrector `none`). Both networks are `NFKMLXNCSNppNet`, which was generalized for
  the two roles — `inputChannels` (2 for the predictor, 6 for the `condition='both'` score), `conditional`
  (the predictor runs `discriminative=True` → no Gaussian-Fourier time embedding, the Dense weights load
  but are not applied), and `scaleBySigma` (off for the predictor) — and SGMSE+ stayed at parity as the
  `inputChannels=4`/conditional/scaled case. The sampler (`NFKSGMSESampler`) gained an `observation` (the
  SDE center = `y_denoised`) separate from the `conditioning` channels the score net reads, and a
  `useCorrector` flag. Keys mirror the reference `StochasticRegenerationModel` (`denoiser_net.*` /
  `score_net.*`), so the converted EMA safetensors loads with no remap. At reference parity at a tiny
  random configuration (denoiser seam and score seam cosine 1.000000000000): the released combined
  checkpoints are GDrive-only, and the NCSN++ backbone is already at released-weight parity via SGMSE+, so
  the tiny-random oracle (`run_reference.py storm`, built from the backbone registry, saving both nets'
  weights into the record under `w::…`) validates the new two-net architecture exactly. The StoRM clone
  omits `upfirdn2d_native.py` and its op imports the fused CUDA extension, so the oracle injects an inline
  native `upfirdn2d` + a leaky-ReLU shim into `sys.modules`. Registered under `storm`.
- `NFKMLXMossFormer2SENet` / `NFKMLXMossFormer2Factory` (`@objc(NFKMLXMossFormer2_Factory)`) — MossFormer2
  SE 48K (modelscope/ClearerVoice-Studio, Apache-2.0), full-band speech enhancement, at reference
  parity on the released `last_best_checkpoint.pt`, measured on the M1 at float32: the Kaldi fbank+Δ
  **1.0000000**, the encoder and FLASH block 0 **0.99999994**, FLASH block last and the 961-bin mask
  **1.0000000**, and the enhanced waveform **0.9999998**. The shared MossFormer2 backbone is a mask-predicting
  net over a Kaldi-fbank front end: a `GroupNorm(1)` input norm, a `Conv1d` bottleneck, a scaled sinusoidal
  positional embedding, 24 `MossformerBlock_GFSMN` layers, and a gated output to a real 961-bin mask
  (final ReLU). Each block interleaves `FLASH_ShareA_FFConvM` (gated single-head attention: quadratic
  ReLU-squared local attention within 256-groups + a linear global path, `to_hidden`/`to_qk` as `FFConvM`
  norm→Linear→SiLU→depthwise-`ConvModule`, an `OffsetScale(heads=4)`, adjacent-pair rotary over the first
  32 of the 128 qk dims, gate `(att_u·v)·sigmoid(att_v·u)`) and a `Gated_FSMN_Block` (a `UniDeepFsmn`
  depthwise `Conv2d[39,1]` memory, the Chatterbox-S3 FSMN family). Norms are the `ScaleNorm`/`CLayerNorm`/
  `LayerNorm(1e-6)` zoo. The front end is `NFKMLXKaldiFbank` — `torchaudio.compliance.kaldi.fbank`
  reproduced (DC-removal, pre-emphasis 0.97, Povey/hamming, pow2-padded FFT, kaldi-mel, log) + `compute_deltas`
  ×2 → 180-dim, with `dither` forced to 0 (it is random noise; parity is impossible with it on). The
  mask multiplies a hamming/`center=false` STFT (phase kept) and inverts. Two facts are load-bearing, both
  found on the M1: **`num_spks=2`** (the wrapper builds the MaskNet with the default, so `conv1d_out`
  widens to `dModel·2` and the net returns speaker 0 — the port slices the first `dModel` channels, exact
  because the 1×1 gated output and decoder are per-position), and the FLASH linear-attention path divides
  by the original sequence length `n = x.shape[-2]`, not the padded length (only wrong when the frame
  count is not a multiple of the group size; it dragged FLASH block 0 to 0.9975, localized by the block
  seams). `remapReferenceKey` strips the `mossformer.` wrapper prefix, drops the pos-enc/rotary buffers,
  and translates the `FFConvM.mdl`/`ConvModule`/`Gated_FSMN_Block.conv1`/output-gate Sequential indices.
  The oracle is `run_reference.py mossformer2_se` (dither=0; records the 180-dim feature, STFT, encoder,
  first/last block, mask, waveform), run against the source files (`IK_MOSSFORMER2_SE_SRC`) on
  `~/.inferkit-validation/llmvenv` (needs `rotary_embedding_torch` + `torchinfo`); the parity test feeds
  the recorded feature to isolate the backbone from the fbank. The SR sibling is `NFKMLXMossFormer2SRNet`
  below. Registered under `mossformer2-se`.
- `NFKMLXDeepFilterNet` / `NFKMLXDeepFilterNetBackend` (`@objc(NFKMLXDeepFilterNet_Factory)`) —
  **DeepFilterNet3** (Rikorose/DeepFilterNet, dual **MIT/Apache-2.0**), a ~2.3M-parameter real-time
  48 kHz speech denoiser, the cheap counterpart to `NFKMLXDenoiser`. The `DfNet` is a clean torch
  `nn.Module`: an **encoder** (an ERB convolution pathway `erb_conv0..3` + a DF convolution pathway
  `df_conv0/1`, summed — `enc_concat=False` — into a `SqueezedGRU_S` embedding), an **ERB decoder** (a
  second `SqueezedGRU_S` and a U-net of depthwise-1x1 skips + (transposed) convolutions → a 32-band
  sigmoid ERB mask), and a **DF decoder** (a `SqueezedGRU_S` + a grouped-linear skip → the deep-filter
  coefficients `[B, 5, T, 96, 2]`). The enhanced spectrum is the ERB mask applied to the full spectrum,
  with the lowest 96 bins replaced by a 5-tap causal complex deep filter (`MF.DF`, a per-frame
  complex MAC over a `(dfOrder-1-lookahead, lookahead)`-padded window). The STFT / ERB / normalization
  DSP is Rust `libdf` in the reference, reproduced in `NFKMLXDeepFilterNetDSP` (MLX + Swift) and each
  step validated numerically against a libdf recording: **analysis** left-pads by `nFFT-hop`, windows
  with the **VORBIS window** `sin(π/2·sin²(π(n+0.5)/N))`, rffts, and scales **`1/N`**; the **ERB
  feature** is `10·log10(|spec|²·erb_fb + 1e-10)` then a per-band EMA mean-normalization `(x−s)/40`
  (α = 0.99, `s` init `linspace(-60,-90,32)` = `MEAN_NORM_INIT`); **unit_norm** on the lowest 96 bins is
  `x/√s`, `s` init `linspace(0.001,0.0001,96)` = `UNIT_NORM_INIT`; **synthesis** is `irfft(spec·N)` with
  window-squared overlap-add. Reference parity on the released DeepFilterNet3 weights, seam by seam
  and end to end against the `deepfilternet` pip package: every net seam exact (encoder `e0..e3` /
  `emb` / `c0`, the ERB mask `m`, the deep-filter coefficients, `spec_e` all cosine 1.0000000), the DSP
  features exact (spec / feat_erb / feat_spec 1.0000000), and the enhanced waveform end to end
  **0.9999999**. `+register` under `deepfilternet3`.
  Seven facts are load-bearing, most found by the seam ladder. The `Conv2dNormAct` layout is
  derived from the shapes, not a per-layer flag: `groups = gcd(in, out)`, a pointwise 1x1 follows
  only when `groups > 1` And the kernel is not 1x1, and a causal time pad precedes the conv only when the
  time kernel exceeds 1 — so `erb_conv0` (`gcd(1,64)=1`) is a plain conv while `conv3p` (`gcd(64,64)=64`,
  1x1) is a **depthwise 1x1**. `SqueezedGRU_S`'s `linear_out` carries a trailing **ReLU** (missing it
  dropped `emb` to 0.68), and its `df_gru` variant has `output_size=None` → no `linear_out` (an
  Identity) while running its grouped linears at **8** groups where `emb_gru` runs 16. The DF skip fed
  to the decoders is `c0` (the `df_conv0` output), not `c1`. `df_fc_a` and `pad_spec` exist in the
  checkpoint/module but the reference `forward` never applies them (loaded, unused — there is **no alpha
  blend**). The ERB decoder's `convt2`/`convt1` are grouped depthwise `ConvTranspose2d` (the shared
  GTCRN per-group workaround + the `[in,out/g,kH,kW]→[out,kH,kW,in/g]` transpose, with `output_padding`).
  **The one oracle trap:** `df_state.synthesis(as_complex(spec_e).numpy())` writes IN place through the
  view, corrupting a `spec_e` recorded afterward — snapshot it before synthesis. The oracle is
  `run_reference.py deepfilternet --checkpoint <DeepFilterNet3 dir>` (the `dfnvenv`: `deepfilternet` +
  `deepfilterlib` + torch, Python 3.9; `init_df` downloads the model, and `IK_DEEPFILTERNET_WEIGHTS_OUT`
  dumps the state dict as the weights). No offline converter: the pip package ships the weights and the
  DSP oracle.
- `NFKMLXVoiceRestore` / `NFKMLXBigVGAN` / `NFKMLXVoiceRestoreBackend` (`@objc(NFKMLXVoiceRestore_Factory)`)
  — **VoiceRestore** (skirdey/voicerestore, **MIT**), a ~301M-parameter flow-matching (CFM) universal
  speech restorer, text-free, that fixes noise, reverberation, clipping, and band-limiting together in
  one model. **E2-TTS-derived, not F5** (deps pin `x-transformers==1.34.0`,
  `gateloop-transformer==0.2.5`). The pipeline (24 kHz): degraded audio → a BigVGAN log-mel → a CFM ODE
  `dx/dt = v_θ(x_t, t | degraded_mel)` from noise to the restored mel → BigVGAN → waveform. At
  reference parity on the released weights, seam by seam and end to end.
  **The velocity net** `NFKMLXVoiceRestore` is the vendored E2-TTS transformer: the condition is
  **Additive and frame-aligned** (`x = proj_in(x_t) + cond_proj(degraded_mel)`, not F5's concat), the
  sequence carries 32 learned register tokens prepended (unpacked after the blocks) plus an absolute
  positional embedding on the mel frames, and each of the 20 blocks is a **residual
  `SimpleGateLoopLayer`** → adaptive-RMSNorm attention (adaLN-zero gated) → adaptive-RMSNorm GEGLU
  feed-forward (adaLN-zero gated), with U-net concat skips over the second half. At reference parity
  against the `x-transformers 1.34.0` / `gateloop 0.2.5` reference (`run_reference.py voicerestore`):
  every seam exact (`x_in` / each block's gateloop, attention, feed-forward / the final velocity all
  cosine 0.99999994–1.0). The **`SimpleGateLoopLayer`** (the one research-grade piece) is a data-dependent
  gated linear recurrence: an RMSNorm, a bias-free `Linear(dim, 3·dim)` → q/kv/a, a **sigmoid** forget
  gate, a per-channel first-order recurrence `h_t = a_t·h_{t-1} + kv_t`, and the output `q_t·h_t` — a
  sequential scan, which is exact in MLX (no associative scan needed). The **x-transformers `Attention`**
  carries `gate_value_heads` (a per-head sigmoid value gate) and `softclamp_logits` (`tanh(logit/50)·50`);
  **`AdaptiveRMSNorm`** is `F.normalize(x)·√dim·(1 + γ)` and **`AdaLNZero`** is a `-2`-biased **sigmoid**
  gate; rotary is adjacent-pair over `dim_head`.
  **The vocoder** `NFKMLXBigVGAN` is BigVGAN v2 (`nvidia/bigvgan_v2_24khz_100band_256x`, MIT), a
  HiFi-GAN-style generator with two BigVGAN additions: **SnakeBeta** periodic activations
  (`x + (1/(exp(β)+1e-9))·sin²(exp(α)·x)`) and an **anti-aliased `Activation1d`** — a fixed kaiser-sinc
  up/down FIR (cutoff 0.25, half-width 0.3, kernel 12) around each activation, which this port
  **recomputes** (the Bessel-I0 Kaiser window in Swift) rather than loading. All convolutions are
  weight-normed (`g·v/‖v‖`, the shared `NFKMLXMusic3.fusedWeightNorm`); the `ups` are ConvTranspose1d.
  At reference parity against the released generator end to end (mel → waveform cosine 0.9999997,
  `run_reference.py bigvgan`). The mel front end reproduces `meldataset.mel_spectrogram` (n_fft 1024,
  hop 256, win 1024, fmin 0, fmax 12000, Hann, center=False with a reflect pad of `(n_fft-hop)/2`,
  magnitude `sqrt(·+1e-9)`, `log(clamp(·, 1e-5))`, the shared Slaney filterbank). **The sampler**
  reproduces `VoiceRestore.sample` (`torchdiffeq` fixed-step **midpoint**, `times = linspace(0, 1, steps)`,
  default 32 steps, classifier-free guidance 0.5). End to end at reference parity with the sampler
  seeded from the reference's `y0` (`run_reference.py voicerestore_e2e`): the mel 1.0, the restored mel
  0.9999999, the restored waveform ~1.0. `+register` under `voicerestore` (a directory holding the
  transformer checkpoint + `bigvgan_generator.pt`).
  **The costliest debugging was an oracle bug, not a port bug:** the transformer was correct as first
  written; the seam hooks fired during both the conditioned velocity pass and the CFG null pass, and the
  null pass overwrote every recorded seam — so conditioned inputs were being compared against null-pass
  seams. Removing the hooks before the null pass turned every seam green at once. The oracle runs in a
  dedicated `vrvenv` (Python 3.9, torch 2.2.2, the pinned x-transformers / gateloop / torchdiffeq, plus
  jaxtyping). Weights: `jadechoghari/VoiceRestore/pytorch_model.bin` (the transformer, keyed
  `transformer.*` / `proj_in` / `cond_proj` / `to_pred`, loaded through the native torch reader; the
  `abs_pos_emb` is 2000 rows) and `nvidia/bigvgan_v2_24khz_100band_256x/bigvgan_generator.pt`. No offline
  converter.
- `NFKMLXResembleEnhance` / `NFKMLXResembleEnhanceBackend` (`@objc(NFKMLXResembleEnhance_Factory)`) —
  **Resemble Enhance** (resemble-ai, **MIT**), a five-network general speech restorer (noise +
  reverberation + clipping + band-limiting together), the eighth restoration-vein port and the largest of
  the family. All five networks plus the mel front end are at reference parity on the released
  enhancer_stage2 weights, seam by seam and end to end (`run_reference.py reenhance_*`, the
  `reenhancevenv` oracle env — Python 3.12; the torch source is the only oracle, no community MLX port
  existed): mel 0.9999998, IRMAE encode 0.9999985 / decode 1.0000002, CFM velocity 1.0000001 / sample
  1.0000001, UnivNet 0.9999365, denoiser 0.9999996, and the full `enhance()` waveform 0.9999971. The
  `enhance()` path (`resemble_enhance/enhancer/{enhancer,inference}.py`, defaults nfe 32 / lambd 0.5 /
  tau 0.5): peak-normalize → mel → the mix mel and the denoised mel blended by `lambd` → the LCFM stage
  samples a latent from the encoded-prior-plus-noise (`tau`) → decode to the vocoder input → the UnivNet
  vocoder.
  - **Mel front end** (`NFKMLXResembleMel`): `resemble_enhance.melspec.MelSpectrogram` — preemphasis
    0.97, a torchaudio magnitude mel (Slaney scale + Slaney normalization, reusing `NFKMLXMel.melFilters`,
    a zero-centered STFT `pad_mode="constant"`, n_fft 2048 / hop 420 / 128 mels), `amp_to_db`
    (`clamp(1e-4).log10()·20`), and a headroom normalization `(s + 80) / 95`. `to_mel` drops the last
    frame. A global scalar `Normalizer` (`(x − mean) / std`, `std = sqrt(var + 1e-9)`) loaded from the
    checkpoint follows it.
  - **IRMAE** (`NFKMLXResembleIRMAE`): an implicit-rank-minimizing autoencoder. The encoder (a 1024-wide
    conv, four dilated GroupNorm/GELU ResBlocks, four bias-free 1×1 rank-minimizing convs, a Tanh)
    compresses the 128-mel to a 64-channel latent; the decoder mirrors it to the 160-channel
    (`num_mels + vocoder_extra_dim` 32) vocoder input. The training-only `head` and `estimator` are not
    built. `GroupNorm(pytorchCompatible: true)`.
  - **CFM** (`NFKMLXResembleCFM`): the flow-matching stage. The velocity net is a **WaveNet**
    (`NFKMLXResembleWN`, a DiffWave-style stack of 30 gated dilated-conv layers, dilation cycle 5, an
    InstanceNorm on the local condition, a `SinusodialTimeEmbedding`), not a transformer. The sampler is
    the reference's **exponential-decay midpoint ODE**: `ts = h(linspace(0,1,n+1))` with
    `h(t) = (a^t − 1)/(a − 1)`, `a` solving `h(1/4) = 0.5` (Newton, matching scipy `fsolve`); nfe 32 →
    16 midpoint steps. Distinct from VoiceRestore's plain-linspace midpoint.
  - **UnivNet** (`NFKMLXResembleUnivNet`): a GAN vocoder over **location-variable convolutions**. A noise
    input through `conv_pre` (reflect-padded), four `LVCBlock`s (each: an upsampling transposed conv at
    stride 7/5/4/3 = 420 = hop, an anti-aliased-SnakeBeta AMP block reusing the BigVGAN kaiser-sinc FIR
    family, then four dilated conv stages whose kernels a `KernelPredictor` generates per cond segment and
    applies through a GAU gate), then `conv_post` (LeakyReLU → conv → Tanh). The LVC (dilation 1) is a
    per-segment im2col matmul (MLX has no unfold). The noise is non-deterministic; parity feeds a recorded
    `z`.
  - **Denoiser** (`NFKMLXResembleDenoiser`): the stage-1 STFT-mask model — a complex STFT (reusing
    `NFKMLXComplexSTFT`, n_fft 1680 / hop 420), a 2-D (frequency × time) UNet predicting a magnitude mask
    and a phase residual, and the inverse STFT.
  Three facts are load-bearing, all found by seam localization. The KernelPredictor's LeakyReLU is
  slope **0.2** (the LVCBlock overrides the KernelPredictor's own 0.1 default) — with 0.1 the vocoder
  scores ~0.84. The LVCBlock `convt_pre` is `Sequential(LeakyReLU, ConvTranspose)`, so the activation runs
  Before the transposed conv. And the oracle must load the released `hparams.yaml`, because the enhancer
  default `lcfm_z_scale` is 5 but the release is **6** — a difference cosine cannot see (it changes only
  the encoded-prior/noise blend magnitude) and that surfaces only in the end-to-end path. The DeepSpeed
  shard nests everything under `module`, which the native reader does not unwrap, so the loader strips
  that prefix; it uses old `weight_g`/`weight_v` weight-norm (`fusedWeightNorm` handles it). `+register`
  under `resemble-enhance`; `@objc backendWithDirectoryURL:error:` (the `enhancer_stage2` dir holding
  `ds/G/default/mp_rank_00_model_states.pt`). Weights: `ResembleAI/resemble-enhance` (MIT). No offline
  converter (the native torch reader loads the shard). The oracle imports the released `resemble_enhance`
  leaf modules directly to avoid deepspeed (which the top-level `enhancer.py` pulls in).
- `NFKMLXMetricGANPlus` / `NFKMLXMetricGANPlusBackend` (`@objc`) — **MetricGAN+**
  (`speechbrain/metricgan-plus-voicebank`, Apache-2.0), the smallest member of the restoration family
  and the roadmap's "plumbing smoke test" for it: a magnitude-mask enhancer whose generator is a
  two-layer bidirectional LSTM (257 → 200 per direction) over `log1p(|X|)` frames, then Linear 400→300,
  LeakyReLU(0.3), Linear 300→257, and a per-bin learnable sigmoid `1.2 · sigmoid(slope · x)`. The
  enhanced magnitude is `expm1(mask · features)` under the noisy phase, inverted, then peak-normalized
  (`x / (max|x| + 1e-14)`). At reference parity on the released weights on the first numeric run
  against speechbrain's own `SpectralMaskEnhancement` (`run_reference.py metricgan`, the `llm` env plus
  the `speechbrain` package, `IK_PARITY_METRICGAN` + `IK_VAL_METRICGAN`): features 1.0, mask 1.0000001,
  enhanced waveform 1.0000001 (float32 cosines). Two front-end facts are load-bearing. speechbrain's
  STFT pads with zeros (`pad_mode="constant"`), where `torch.stft` and the shared `NFKMLXComplexSTFT`
  reflect; the shared transform gained `zeroPadded` for it (Parakeet's front end had the same fact,
  measured there at 0.974 the other way). And speechbrain's `resynthesize` calls `istft` with
  `sig_length = the input length`, which keeps the last frame's tail past the symmetric center trim
  (48000 samples where the plain inverse returns 47872), so the shared inverse gained `torch.istft`'s
  `length`. The window is a 512-sample periodic Hamming at hop 256. The release is a plain state dict
  the native torch reader opens; the two LSTM layers fold through the shared PyTorch→MLX
  `Wx`/`Wh`/`bias` treatment under `blstm.N.forward` / `.reverse` (the gate order matches, so the
  matrices transfer as they are), and `Learnable_sigmoid.slope` loads by name. `+register` under
  `metricgan-plus`; `backendWithWeightsURL:` and the repo / async peers; the gallery example runs it.
- `NFKMLXCMGAN` / `NFKMLXCMGANNet` / `NFKMLXCMGANBackend` (`@objc`) — **CMGAN** (`ruizhecao96/CMGAN`,
  MIT), a 1.83M-parameter conformer-based metric GAN whose generator `TSCNet` denoises a
  power-compressed (`mag^0.3`) complex spectrogram: a dense encoder (a 1×1 convolution, a four-layer
  dilated dense net, a `(1,3)` stride-`(1,2)` convolution halving the frequency axis) over
  `[magnitude, real, imaginary]`, four **two-stage conformer blocks** (a lucidrains conformer over time
  with each frequency a sequence, then one over frequency with each frame a sequence, each residual:
  half-weighted macaron feed-forwards, attention with Shaw's relative position embedding — a learned
  `[1025, 16]` table indexed by the clamped query-key distance, dotted with the query — a GLU →
  depthwise-31 → BatchNorm → Swish convolution module, and a post-norm), then a magnitude-mask decoder
  (the dense net, the MP-SENet sub-pixel frequency upsample, a `(1,2)` convolution to one channel, a
  per-bin `prelu_out` initialized at -0.25) and a complex-residual decoder. The output is
  `mask · mag` under the noisy phase plus the residual, decompressed `^(1/0.3)`. The front end is
  `evaluation.enhance_one_track`: the clip scaled to unit RMS (`c`), padded to a multiple of the hop by
  repeating its first samples, a 400-point periodic-Hamming STFT at hop 100 (center, reflect), and the
  output divided by `c` and trimmed. Only the generator runs; the discriminator is a training device.
  The MP-SENet blocks are reused directly (`NFKMPSEDenseConv`, `NFKMPSESubpixelUp`) — MP-SENet
  descends from CMGAN — and the reference's `nn.Sequential`s are held as `[Module]` arrays so the
  numeric keys match with no remap; only the dense nets' flat `conv{i}` / `norm{i}` / `prelu{i}`
  attributes and the `TSCB_{i}` blocks map onto arrays. At reference parity on the released weights on
  the first numeric run against the repository's own `TSCNet` (`run_reference.py cmgan`, the `llm` env,
  `IK_CMGAN_SRC` = the cloned `src/`, `IK_PARITY_CMGAN` + `IK_VAL_CMGAN`, the repository's own noisy
  VCTK-DEMAND clip `p232_052`): compressed spectrum 0.99999994, encoder 1.0, TSCB 1–4 1.0 / 1.0 /
  0.9999998 / 1.0, mask 1.0, complex residual 1.0, final real / imaginary 0.99999994 / 1.0, enhanced
  waveform 0.99999994. The released `ckpt` is a plain state dict the native torch reader opens.
  `+register` under `cmgan`.
- `NFKMLXFRCRN` / `NFKMLXFRCRNNet` / `NFKMLXFRCRNBackend` (`@objc`) — **FRCRN SE 16K**
  (modelscope/ClearerVoice-Studio, `alibabasglab/FRCRN_SE_16K`, Apache-2.0), a frequency-recurrent
  complex CRN: two complex UNets (`unet`, then `unet2` reading the first's raw output) over a
  convolutional STFT (a 640-point square-root periodic Hann at hop 320, **no centering**, the
  reference's `ConvSTFT` kernel, whose pseudo-inverse is the windowed irfft), the mask
  `tanh(unet2) + tanh(unet1)` applied as a complex product. Each UNet is seven complex encoders
  (`(5,2)` kernels over `(frequency, time)` at stride `(2,1)` padding `(0,1)`, so every stage halves
  the 321 bins to one and adds a frame; the decoders' `(·,2)` transposed convolutions remove it) with
  a **frequency-recurrent FSMN** before each encoder but the first (`ComplexUniDeepFsmn_L1`: each
  Frame is a sequence over the frequency axis, a causal 20-tap depthwise memory added to a
  Linear→ReLU→Linear projection, the whole residual; the complex form pairs `re`/`im` sub-nets as
  `re(x_re) − im(x_im)`, `re(x_im) + im(x_re)`), a **complex squeeze-excite** after each (`SELayer`:
  real and imaginary parts pooled and gated separately, the two gates combined as a complex product,
  then applied part by part, an elementwise scale rather than a complex multiply), a two-layer FSMN over
  Time at the one-bin bottleneck, and the encoders' excited outputs concatenated on channels into the
  decoders. Complex BatchNorm is two BatchNorms (eval), LeakyReLU at 0.01. The released checkpoint
  stores every stage twice, as flat `encoder{i}` / `fsmn_enc{i}` / `se_layer_enc{i}` attributes and
  as the `ModuleList`s `encoders.{i}` …; the module is keyed by the lists and the loader drops the flat
  copies (the MODNet backbone trap again). Three tensors exist but never run (`fsmn_enc0`,
  `fsmn_dec6`, `se_layer_dec5`) and are declared so the strict load holds. The consumer path
  reproduces `decode_one_audio_frcrn_se_16k`'s zero padding (to a 1 s window, to window + 0.75 s
  stride, or past that by `t − ⌊(t − window)/stride⌋·stride` off the stride grid): the FSMN memories
  and the squeeze-excites' global pools read the padded clip, so the padding changes every output
  sample and is part of the model's input; the output is trimmed back to the input length. At
  reference parity on the released weights on the first numeric run against ClearerVoice's own
  `DCCRN` (`run_reference.py frcrn`, the `llm` env, `IK_FRCRN_SRC` = the curled `models/frcrn_se/`
  sources, `IK_PARITY_FRCRN` + `IK_VAL_FRCRN`, the CMGAN noisy clip padded to 58368): conv-STFT
  spectrum 0.9999999, encoder 0 1.0, its squeeze-excite 1.0, the bottleneck FSMN 1.0, decoder 0
  0.99999994, the first UNet 1.0, mask 1.0, masked spectrum 1.0, enhanced waveform 1.0. The shared
  `NFKMLXComplexSTFT` gained `centered: false` for it. The FSMN's `[C, 1, order, 1]` depthwise memory
  loads as a 1-D `[C, order, 1]` convolution; the transposed convolutions through `(1, 2, 3, 0)`.
  `+register` under `frcrn`; weights `alibabasglab/FRCRN_SE_16K/last_best_checkpoint.pt` (161 MB).
- `NFKMLXMossFormer2SRNet` / `NFKMLXMossFormer2SRGenerator` / `NFKMLXMossFormer2SRFactory`
  (`@objc(NFKMLXMossFormer2SR_Factory)`) — **MossFormer2 SR 48K** (modelscope/ClearerVoice-Studio,
  `alibabasglab/MossFormer2_SR_48K`, Apache-2.0), speech super-resolution (bandwidth extension), the
  SR sibling the SE entry promised. Three stages plus a DSP post-process: the HiFi-GAN log-mel
  (`meldataset.mel_spectrogram` at 48 kHz, 1024/256, 80 bands to 8 kHz — the shared
  `NFKMLXVoiceRestoreMel`, now parameterized), the **mel-to-mel MossFormer2 backbone** (the shipped
  `NFKMLXMossFormer2SENet` under `NFKMLXMossFormer2Configuration.superResolution`: 80 in, 80 out,
  `num_spks` 1 — the reference's block / FSMN / conv-module sources are byte-identical to the SE
  ones, so nothing in the backbone is new; its final ReLU stays, so the restored log-mel is clipped
  at zero), a **Snake HiFi-GAN generator** (`ResBlock1` with per-channel Snake activations in place
  of every leaky ReLU, a Snake before each of the four transposed-convolution upsamples `[8, 8, 2, 2]`
  = the 256 hop, `snake_post`, `conv_post`, `tanh`; the DAC Snake `NFKMusic3Snake` is reused), and
  **`bandwidth_sub`**, the decode path's scipy post-process ported in double precision
  (`NFKMossBandwidthSubstitution`): the input's effective bandwidth is the first bin where a 256-point
  Hann STFT's cumulative energy (zero boundary padding, `scipy.signal.stft` defaults) reaches 0.9996,
  the input is kept below it through a fourth-order Butterworth low-pass and the generator's output
  added above it through the matching high-pass (both `scipy.signal.butter` — prototype poles,
  pre-warp, bilinear at fs 2, `zpk2tf` — under `filtfilt`'s odd extension of 15, `lfilter_zi`
  initial state, forward and backward passes), and the result crossfades from the input over the
  first 100 ms. At reference parity on the released weights on the first numeric run against
  ClearerVoice's own `Mossformer` + `Generator` + `bandwidth_sub` (`run_reference.py mossformer2_sr`,
  the `llm` env plus `pydub` for the `meldataset` import, `IK_MOSSFORMER2_SR_SRC`,
  `IK_PARITY_MOSSFORMER2_SR` + `IK_VAL_MOSSFORMER2_SR`; the CMGAN clean 16 kHz clip resampled to
  48 kHz by torchaudio and recorded, so both sides read one waveform): mel 1.0, backbone 1.0,
  generator 0.9999999999987, the detected cutoff 6937.5 Hz exactly, the substitution 1.0, and the
  whole path 0.9999999999992. The backbone checkpoint is `{"mossformer": {"mossformer.…"}}`: the
  container name is not one the native reader unwraps, so it lands as a doubled prefix the loader
  strips before the SE remap; the generator checkpoint (`{"generator": …}`) is weight-normed
  (`fusedWeightNorm`) with the Snake `alpha` moving from `[1, C, 1]` to `[1, 1, C]` through the same
  3-D transpose the convolutions take. `+register` under `mossformer2-sr`; `@objc
  backendWithDirectoryURL:error:` (the directory holding `last_best_checkpoint_m.pt` and
  `last_best_checkpoint_g.pt`, 220 MB each). The reference's long-clip sliding window (past 20 s) is
  not reproduced; a clip runs whole.
- `NFKMLXNUWave2` / `NFKMLXNUWave2Net` / `NFKMLXNUWave2Backend` (`@objc`) — **NU-Wave 2** (maum-ai,
  BSD-3), diffusion bandwidth extension and the family's first generative up-sampler: a
  WaveGrad-style noise predictor whose 15 residual blocks are **short-time Fourier convolutions**
  (`FFC`): the 64 channels split into a local half (3-tap convolutions) and a global half whose
  `SpectralTransform` takes every channel's normalized STFT (1024/256, periodic Hann, center reflect,
  `normalized=true` on both transforms), interleaves the real and imaginary parts on the channel axis
  (`2c`, `2c + 1`), modulates them per bin by **BSFT** (the input's bandwidth as a one-hot over the
  513 bins through a shared 3-tap convolution to a per-bin scale and shift), ReLU, a bias-free 1×1
  convolution across those 64 channels, and the inverse STFT; the halves cross-connect, a gated
  activation gathers its gate and filter from both, and a 1×1 projection splits the residual (`/√2`)
  and the skip. The noise level is `(logsnr_max − logsnr) / 40` through a 50000-scaled sinusoidal
  embedding and two SiLU projections, added per block. The sampler is `denoise_ddim` over the
  released eight-value logSNR schedule (`[-2.6, -0.8, 2.0, 6.4, 9.8, 12.9, 14.4, 17.2]`, the last
  step landing on `logsnr_max` 20): from standard-normal noise, `x̂ = (y − σ_t ε) / α_t`,
  `y_s = α_s x̂ + σ_s ε` with `α² = sigmoid(logSNR)`, then a clamp to `1 − ε_fp16`. At reference
  parity on the official checkpoint on the first numeric run against the repository's own
  `Diffusion` (`run_reference.py nuwave2`, the `llm` env plus `omegaconf`, `IK_NUWAVE2_SRC`,
  `IK_PARITY_NUWAVE2` + `IK_VAL_NUWAVE2`), from the reference's own seeded start noise: the diffusion
  embedding 1.0, the first block's residual and skip 1.0, the step-0 noise prediction 1.0, every one
  of the eight DDIM steps 1.0, the clamped output 1.0. The conditioning follows `inference.py`: the
  clip peak-normalized, upsampled to 48 kHz (the reference's scipy `resample_poly`; the consumer path
  uses the shared `NFKMLXAudioRate.matched`, a documented approximation — the parity reads the
  recorded upsampled clip), trimmed to a multiple of the hop, and the band the first
  `int((rate / 2) / 24000 · 513)` bins (171 for a 16 kHz source, the reference's own float
  arithmetic). `NFKNUWaveSpectrum` is the batched normalized STFT pair over `[N, L]`: reflect padding
  and framing by gathers, and an overlap-add that reshapes each frame into `fftSize / hop` chunks
  and sums the shifted chunk sequences (MLX has no scatter-add). **Three oracle facts.** The official
  checkpoint is a Lightning file whose pickled callbacks need a `pytorch_lightning` stub to unpickle
  (a stub module whose `__getattr__` must still raise on dunders, or `inspect` inside `torch.load`
  breaks); the state dict sits under `model.model.` and the STFT window buffers are dropped; and the
  repository predates torch 2.x, handing `istft` the real `(…, 2)` view — the oracle patches
  `torch.istft` to take the complex view of the same numbers. `NFKParameterSeed` fixes the diffusion
  start; a step count other than eight walks the logSNR range evenly, as the reference does.
  `+register` under `nuwave2`; weights: the README's Google Drive checkpoint (20.9 MB, manifest route
  `gdrive`), which the native torch reader opens.
- `NFKMLXApollo` / `NFKMLXApolloNet` / `NFKMLXApolloBackend` (`@objc`) — **Apollo** (JusperLee,
  **CC-by-SA-4.0** code and weights, `JusperLee/Apollo/pytorch_model.bin`, 66 MB), music restoration
  of lossy-codec artifacts (MP3 at 24–128 kbps → lossless), the last of the audio fillers and the one
  music-leaning model. An 80-band split of a 20 ms STFT at 44.1 kHz (882/441, periodic Hann, center
  reflect, un-normalized): 79 bands of 5 bins and a 47-bin remainder over the 442 bins, each band's
  real and imaginary parts divided by the band's power (`sqrt(Σ|X|² + ε_fp32)`) and joined by the
  log power, an RMS norm and a 1×1 projection to 256 per band (`BN[i]`). Six **band-sequence layers**
  (`BSNet`): a Roformer across the 80 bands (every frame a sequence: an RMS-normed fused q/k/v 1×1
  projection whose 768 channels are head-major with q, k, v inside each of 8 heads, adjacent-pair rotary
  over a 100-position table, non-causal fused attention, a bias-free output projection with a residual,
  and a gated MLP — `silu` over the whole `8·dim` projection, then `silu(gate) · z` over its halves, so
  the gate is silu'd twice, reproduced as written), then an **ICB along time** (every band a sequence:
  three `ConvActNorm1d` blocks of a depthwise 7-tap convolution, an RMS norm, a 1×1 expansion ×4, SiLU,
  a 1×1 projection, residual). A head per band (an RMS norm, a 1×1 to `4·width`, a GLU to the band's
  real and imaginary bins, the real bins first) and the inverse STFT at the input's length. Every RMS
  norm is over the channels at eps 1e-5 (MLXNN's `RMSNorm`). The reference's `nn.Sequential` indices
  in `BN[i]` / `output[i]` map onto `norm` / `conv`; the rotary tables are recomputed (held as Swift
  arrays, off the parameters). At reference parity on the released weights on the first numeric
  run against the repository's own `Apollo` (`run_reference.py apollo`, the `llm` env, `IK_APOLLO_SRC`
  = the curled `look2hear` package, `IK_PARITY_APOLLO` + `IK_VAL_APOLLO`, channel 0 of the
  repository's own `asserts/input_wav.wav` for two seconds): band features 0.9999988, band 0's
  bottleneck 0.9999999, the first band-sequence layer 1.0, the last 1.0, band 0's head 1.0, and the
  restored waveform 0.9999973. The batched STFT pair (`NFKNUWaveSpectrum`) gained `normalized: false`
  and `torch.istft`'s `length` for it. The network runs each channel on its own; the backend runs the
  mono clip the WAV reader yields, at 44.1 kHz. The `inference.py` chunked overlap-add for long files
  is not reproduced; a clip runs whole. `+register` under `apollo`.
- `NFKMLXDAC` (`@objc`) — the Descript Audio Codec, the toolkit's first neural audio codec and the class a
  codec-token speech-LLM generates into. Three parts: a convolutional **encoder** (a wide first conv,
  then downsampling stages of three dilated residual units + Snake + a strided conv, doubling the width
  and halving the resolution), a **residual vector quantizer** (`NFKDACResidualVectorQuantize`: a stack
  of quantizers, each projecting the latent to the codebook width through `in_proj`, matching each frame
  to its nearest L2-normalized codebook entry, projecting the raw chosen entry back through `out_proj`,
  and coding the residual the previous ones left), and a **decoder** (the mirror, upsampling through
  transposed convs). The Snake activation, the dilated residual unit, and the decoder's upsample block
  are the shared Music 3 vocoder blocks (`NFKMusic3Snake`/`NFKMusic3ResidualUnit`/`NFKMusic3VocoderBlock`);
  the strided encoder and the RVQ are what the codec adds. `NFKMLXDAC.encode(_:)` returns the codebook
  tokens `[[Int]]` (codebook × frame) — the codec's product — and `decode(_:)` reconstructs; the
  `NFKMLXDACBackend` runs the round trip (audio → codes → audio under `NFKOutputAudio`). `+register`
  under `dac`. The released convolutions are weight-normalized, so the loader **fuses `g·v/‖v‖`** (reusing
  `NFKMLXMusic3.fusedWeightNorm`) and transposes; `remapReferenceKey` translates the reference's nested
  `nn.Sequential` names (`encoder.block.N.block.M.block.K`, `decoder.model.N.block.M`,
  `quantizer.quantizers.N`). The nearest-neighbor search compares normalized vectors (maximizing the dot
  product of unit vectors is minimizing Euclidean distance); the reconstruction uses the raw codebook
  entry, as the reference does. Reference parity against `descript-audio-codec` on the released 44.1
  kHz model (`run_reference.py dac`, llm oracle env, needs `descript-audio-codec`), on the first numeric
  run: codebook tokens matching exactly (783/783 over 9 codebooks × 87 frames) and the decoder
  reconstructing the reference's waveform from its codes at cosine 0.99999999999986. `NFKMLXDACConfiguration`
  carries the released `.dac44kHz`/`.dac24kHz`/`.dac16kHz` geometries (all four encoder rates, differing
  in the rates, codebook count, and sample rate); the native torch reader loads the released `.pth`
  directly (its `state_dict` is unwrapped, the weight norm fused). Converter `Tools/dac-to-safetensors`.
- `NFKMLXSNAC` (`@objc`) — SNAC, the toolkit's second neural audio codec and the first multi-SCALE one.
  Its codebooks code at different temporal rates: the residual is average-pooled before a coarse codebook
  quantizes it and repeat-interleaved back afterward, so codebook 0 emits one token per `vqStrides[0]`
  frames, codebook 1 per `vqStrides[1]`, and so on (`[4, 2, 1]` for the 24 kHz model). This is the 24 kHz
  Speech model — the codec the common speech codec-token LLMs use — with depthwise-separable convolutions,
  a decoder **noise block**, three codebooks, and no bottleneck attention (the release's
  `attn_window_size` is null). Structure like DAC with three SNAC-specific pieces: the depthwise blocks
  (`NFKSNACResidualUnit`'s dilated conv is grouped over every channel), the `NFKSNACNoiseBlock` (a learned
  per-position scale times a fresh Gaussian, added in the decoder), and the multi-scale RVQ
  (`NFKSNACResidualVectorQuantize`: `avgPool` before `in_proj`, `repeatInterleave` after `out_proj`). The
  Snake activation and the transposed-conv upsample are the shared Music 3 blocks. `NFKMLXSNAC.encode(_:)`
  returns the per-codebook token streams `[[Int]]` at their own rates (the coarser emit fewer),
  `decode(_:deterministic:)` reconstructs, and `NFKMLXSNACBackend` runs the round trip under
  `NFKOutputAudio`. `+register` under `snac`. The noise block is non-deterministic (`torch.randn`), so
  a decode is reproducible only with `deterministic: true` (which skips it); parity is measured that way,
  the noise's expected contribution being zero. The release weight-normalizes through torch's
  parametrization API (`parametrizations.weight.original0` = g, `original1` = v), where DAC used the older
  `weight_g`/`weight_v`; `NFKMLXSNAC.fusedWeightNorm` fuses that form. `remapReferenceKey` translates the
  nested `encoder.block.N.block.M.block.K` / `decoder.model.N.block.M` (0 snake, 1 transposed conv, 2 the
  noise block, 3..5 residual units) / `quantizer.quantizers.N` names. Reference parity against the
  `snac` package on the released 24 kHz model (`run_reference.py snac`, llm oracle env, needs `snac`), on
  the first numeric run: per-codebook tokens matching exactly (42/42 over the three multi-scale codebooks)
  and the decoder reconstructing at cosine 0.9999999999998. Native torch reader loads the release
  directly. Converter `Tools/snac-to-safetensors`.
  The two music models are at parity too (`.snac32kHz` / `.snac44kHz`, `NFKMLXSNACVariant.music32kHz`
  / `.music44kHz`, registered as `snac-32khz` / `snac-44khz`): encoder 64 wide at rates `[2, 3, 8, 8]`, a
  1536-wide decoder, four codebooks at strides `[8, 4, 2, 1]`, and — what the speech model omits — a
  windowed local attention at the bottleneck of both encoder and decoder (`NFKSNACLocalAttention`,
  `attentionWindow` 32: a LayerNorm, bias-free `to_qkv` / `to_out`, `headDim = min(64, dim)`, rotate-half
  rotary over the positions within the window, fused attention per window), which shifts every later
  `Sequential` slot by one in the remap and raises the padding multiple to `hop · lcm(stride₀, window)`.
  Codes 60/60 exact on both, reconstruction 0.99999999999978 / 0.99999999999984.
- `NFKMLXAudioTagger` (`@objc`) — real audio tagging (PANNs Cnn14): a log-mel spectrogram, normalized
  across its mel bands (`bn0`), feeds six VGG-style blocks (two 3×3 convolutions and an average pooling
  each), and the result pools over time — max plus mean — into an independent score per class; the top
  scores become tags. `NFKMLXAudioTaggerBackend` reads `NFKInputAudio` → `NSArray<NFKClassification *>`
  (a new core value type, most-confident first) under the new core key `NFKOutputClassifications`; the
  `+backendWith…labels:` factory attaches class names. `+register` under `audio-tagger-panns`.
  Reference parity against PANNs' own Cnn14 (mel cosine 0.99999999, embedding 0.99999994, tag
  0.99999988, same top class). The front end is 32 kHz / 1024-point / hop 320 over a 50 Hz–14 kHz
  filterbank, which the **checkpoint ships** (`logmel_extractor.melW`), so it loads rather than being
  recomputed — held in `NFKAudioTaggerFrontEnd`, off the `Module`, or it inflates `parameters()`.
  Its decibel scale is `10·log₁₀` floored at `1e-10`, not the natural log the other front ends use.
  The last block pools with a window of one, i.e. not at all; pooling it like the others cost
  embedding cosine 0.9942 and moved the top class. `remapReferenceKey` maps the reference's 1-based
  `conv_blockN` onto the module's array. A clip arriving at another sample rate is resampled to
  32 kHz through `NFKMLXAudioRate.matched`: the filterbank is built for one rate, so feeding another
  puts every frequency in the wrong mel bin — wrong tags, with nothing that looks like an error.
- `NFKMLXBiSeNet` (`@objc`) — real real-time semantic segmentation: the reference **BiSeNetV1**
  (CoinCheung) in `MLXNN` — a shallow Spatial Path (three strided convolutions and a 1×1 projection)
  preserving the detail a deep path discards, a Context Path over **ResNet-18** with Attention
  Refinement at strides 16 and 32 plus a globally pooled branch, and a Feature Fusion Module that adds
  a channel-gated copy of the concatenated result. Run through `NFKMLXModuleBackend`; emits a
  grayscale class-label map under `NFKOutputImage` (same convention as `NFKMLXSegFormer`/
  `NFKMLXDeepLab`). `+register` under `bisenet`; factory sets `train(false)`. The context path
  upsamples with **nearest** (`nn.Upsample(scale_factor: 2)` defaults to it) while the output head is
  bilinear ×8 — two different resamplings in one network. Inputs take ImageNet normalization, and the
  Context Path's stride-32-onto-16 addition only lines up when both sides are multiples of 32, so
  `segment` resizes for the network and resizes the **logits** (never the labels — interpolating class
  indices invents classes) back. The two auxiliary heads are built because the checkpoint carries
  them, though inference never reads them. `remapReferenceKey` names the ResNet block's positional
  projection shortcut (`downsample.0/1`). Reference parity against BiSeNetV1's own network on the
  released Cityscapes checkpoint (logit cosine 0.9999999999989, **label agreement 1.0**), every
  parameter covered on the first triage run. Weights: the `CoinCheung/BiSeNet` GitHub release
  (`model_final_v1_city_new.pth`) — they were never actually unavailable.
- `NFKMLXBiSeNetV2` (`@objc`) — the second BiSeNet, a separate architecture rather than a variant, at
  reference parity on the released Cityscapes checkpoint (logit cosine 0.9999999999992, **label
  agreement 1.0**), every parameter covered on the first triage run. It replaces V1's ResNet context
  path with a purpose-built pair: a wide shallow **detail** branch and a narrow deep **semantic**
  branch of Gather-and-Expansion layers (each a residual block at stride one; at stride two it
  downsamples through two depthwise stages and carries a depthwise-then-pointwise shortcut), joined by
  a **bilateral aggregation** layer where each branch gates the other at its own scale. `+register`
  under `bisenet-v2`; emits the same grayscale label map as the other segmenters, aligning sides to a
  multiple of 32 and resizing the logits back.
  Its released checkpoint predates the repository's current head: it emits `classes × upFactor²`
  channels and **pixel-shuffles** to full resolution, where master now emits `classes` and
  interpolates. Everything before the heads is unchanged, so the oracle substitutes the older head
  rather than pinning a historical commit — the substitution stays visible. A factor-eight shuffle is
  not three ×2 shuffles (the channel interleaving differs), so `NFKBiSeNetPixelShuffle` implements
  PyTorch's `c·r² + i·r + j` order directly and a test asserts it. The four `aux*` heads supervise
  training only and are neither built nor loaded — they hold the largest tensors in the file.
- `NFKMLXVideoSR` (`@objc`) — real video super-resolution: the complete **BasicVSR** (mmediting
  `BasicVSRNet`, ×4). **SPyNet** estimates flow between neighbors (coarse-to-fine pyramid, six
  five-conv modules; its ImageNet normalization tensors load from the checkpoint), and two
  propagation branches — backward and forward through time — warp their hidden features along that
  flow (`flowWarp`, a bilinear `grid_sample(align_corners=True)`: `zeros` padding for propagation,
  `border` inside SPyNet; the flow upsampling between pyramid levels is `align_corners=True` bilinear
  ×2, a separate grid from `resizeBilinear`). Fusion + two `PixelShufflePack` stages + `conv_hr`/
  `conv_last` reconstruct over a bilinear ×4 base. Bidirectionality means a frame draws on frames
  after it, so a clip goes through `NFKMLXVideoSRNet.upscaleSequence` whole; the backend upscales a
  single frame. `+register` under `video-super-resolution`. `remapReferenceKey` strips the
  checkpoint's `generator.` wrapper and maps the branches' positional `main.0`/`main.2` Sequential.
  Reference parity against mmediting's own BasicVSRNet on the released REDS4 checkpoint, over a
  three-frame translated clip (clip cosine 0.9999999999997946, mean |difference| 1.1e-7).
  Forward, bidirectional propagation, flow-warp identity/shift/padding-mode, remap, sequence, and
  round-trip tested.
- `NFKMLXTrainer` — the supervised and zero-reference training loop, for customizing a shipped model
  on a consumer's own data, in the app. Two entry points (`batch:loss:` with a target,
  `sample:loss:` without one) share a private loop: gradient clipping, per-step progress and early
  stop, and periodic checkpoints. The toolkit owns the loop, the caller owns the loss, mirroring
  `NFKMLXModuleBackend`. Which parameters train is set by **freezing** the rest before calling in —
  `valueAndGrad` differentiates only `trainableParameters()`, so a frozen backbone costs neither
  gradients nor optimizer state, which is what makes on-device fine-tuning viable at all. Adds the
  `MLXOptimizers` product.
  - **Output is input**: `NFKMLXWeights.save` writes a plain safetensors that the model's existing
    `backendWith…weightsURL:` factory loads, so a customized model needs no separate route. The file
    records its layout in metadata (`inferkit.layout`), and `NFKMLXWeights.loadCheckpoint` reports
    `needsConvTranspose` so a `loadWeights` **skips** its PyTorch transpose for a fine-tuned file.
    Skipping rather than inverting is what keeps the models whose transpose is not the common one
    correct (SAM's `up1`/`up2` use `transposed(1,2,3,0)`, Whisper handles 3-D Conv1d). Every model
    reads through `loadCheckpoint`, and every transpose in a loader is gated on the flag — including
    the branches keyed on a name rather than a rank (LaMa's `up.`, Demucs's and Conv-TasNet's 3-D
    transposed convolutions). An ungated loader double-transposes a fine-tuned file and loads silently
    wrong weights: `NFKMLXCheckpointRoundTripTests` saves and reloads through each model's own loader,
    and removing one gate makes it fail with a transposed shape rather than a bad number.
  - Writes are atomic (scratch file then replace), because a periodic checkpoint overwrites the only
    copy of a run's progress. A non-finite loss throws `NFKMLXError.trainingDiverged` **before** that
    step can reach a checkpoint, so divergence cannot replace good weights with ruined ones.
  - `clipGradientNorm` runs through `bounded(_:maxNorm:)`, which sanitizes before it clips. A
    gradient set can be entirely finite while the sum of its squares is not — 3e20 is an ordinary
    `Float` and its square is not — so a directly computed norm comes back infinite, `maxNorm/∞` is
    zero, and the whole update is scaled to nothing; the optimizer then builds its moments from zeros
    and later steps produce non-finite parameters while the loss stays finite throughout, so the
    divergence guard never fires. Non-finite entries are zeroed first, then the norm is taken relative
    to the largest magnitude present. Reported by RVC-MLX from a real run, pinned here by
    `testAGradientWhoseSquaresOverflowIsStillScaledToTheNorm`.
  - Optimizer state is **not** checkpointed: mlx-swift keeps `stateStorage` internal and `innerState()`
    unkeyed. `SGD` resumes exactly; `Adam` rebuilds its moment estimates.
- `NFKMLXTrainingData` / `NFKMLXBatchSampler` — the app-data side of training. `tensor` / `batch` /
  `matte` / `labels` convert a consumer's `CGImage`s into what the trainer takes, reusing
  `NFKMLXImageBridge` so a training batch and an inference input are built identically. `labels`
  inverts the label-map convention the segmentation backends emit (`index / (classCount − 1)`), so
  a mask painted in an app and a mask a model outputs are the same encoding. A mixed-size batch throws
  rather than resizing: crop versus scale changes what the model learns, so the choice stays with the
  caller. `NFKMLXBatchSampler` draws reshuffled passes from a seed (SplitMix64 Fisher-Yates, matching
  the schedulers' repeatable-randomness idiom) — cycling a handful of examples in a fixed order lets
  the optimizer chase the sequence rather than the data.
- `NFKMLXCLIPProbe` / `NFKMLXCLIPProbeBackend` — a consumer's own image classifier over a frozen CLIP
  embedding, and the cheapest useful customization in the package. Because both towers stay frozen,
  `NFKMLXCLIP.embeddings(for:using:)` computes each embedding **once** and `trainProbe` then runs over
  cached vectors: a step is one 512-wide matrix multiply rather than a transformer forward. A contrastive
  fine-tune of CLIP itself is not a device workload (it needs large batches for negatives); a probe is.
  The backend emits `NSArray<NFKClassification *>` under the core key `NFKOutputClassifications`,
  softmaxed and ranked. A probe is a **separate small model**, so what it saves is a companion file
  rather than modified CLIP weights.
- `NFKMLXWhisperTraining` — domain adaptation for speech (jargon, accents, recording conditions), and
  the recipe LoRA exists for. `NFKMLXWhisperObjective` is teacher forcing: the decoder sees the whole
  target sequence at once and each position is scored on the next token, so a step is one forward pass
  rather than one per token. `loss(logits:tokens:)` is separable from the forward for the oracle, as in
  the other two objectives. Only `decoder.blocks.*.{query,value}` are adapted — the reference LoRA
  choice, and the encoder's audio features transfer across domains. `NFKMLXWhisper.spectrogram` pads or
  trims to the 30-second window, the single biggest accuracy factor in reaching reference parity. **`NFKWhisperAttention.query`/`value`/`out` gained `@ModuleInfo`** so they
  can receive adapters; the wrapper keys equal the property names, so checkpoints are unchanged.
- `NFKMLXLoRA` / `NFKMLXLoRALinear` — low-rank adaptation, for the models with no small head to train
  (CLIP, Whisper: adapting them means reaching into the attention blocks, and doing that fully needs
  optimizer state proportional to the whole model). `NFKMLXLoRALinear` **subclasses `Linear`**, which is
  what `update(modules:)` requires — a replacement must be assignable to the `@ModuleInfo` ivar's type,
  the same idiom `QuantizedLinear` uses. `apply(to:rank:alpha:where:)` does the tree surgery via
  `leafModules().flattened()`, then freezes everything and reopens only `lora_a`/`lora_b`; it returns
  the count, so a predicate that matched nothing is visible rather than silent, and it skips
  already-adapted layers. `apply` and `merge` throw: MLX's non-throwing `update(modules:)` wraps a
  `try!`, so selecting a layer stored in a plain property aborts the process. Both call the throwing
  variant and report `NFKMLXError.loRANotApplicable` naming the `@ModuleInfo` requirement. The `B` factor starts at zero, so an adapted model produces exactly what it
  produced before training. `merge(into:)` folds each detour into its base weights (`Linear` computes
  `x·Wᵀ`, so the delta is `(A·B)ᵀ·scale`) and leaves plain layers: the saved file carries no adapter
  keys, so there is no adapter format and no second loading path.
  Neither `apply` nor `merge` will touch a quantized model. `QuantizedLinear` subclasses `Linear`,
  so it satisfies a type-based predicate silently while its `weight` holds packed integers — adapting
  one builds a detour around something that is not a weight. And merging a low-rank delta into a
  quantized base then requantizing rounds the delta away, so the training is discarded while the file
  loads without complaint. Merge at float precision, then quantize, in that order.
- `NFKMLXSegFormerTraining` — the head-only recipe. A consumer rarely wants ADE20K's 150 classes and
  usually wants their own few, which is a decode-head problem: `NFKMLXSegFormerTrainable.decodeHead`
  (the default) freezes the four encoder stages, so the run's memory falls to the head's share.
  `NFKMLXSegFormer.network(weightsURL:classCount:)` drops the checkpoint's `classifier.*` when the
  class count differs and loads the rest with `strict: false` — not optional, because MLX's
  `update(parameters:)` adopts a checkpoint's shapes wholesale rather than validating them, so keeping
  it would silently restore the old class set. `NFKMLXSegFormerObjective` upsamples the stage-1 logits
  to the label resolution before cross-entropy, as the reference does, rather than downsampling the
  labels. The ImageNet input normalization moved into `NFKMLXSegFormerNet.normalized`, shared by
  `segment` and the objective: a fine-tune that normalized differently would optimize for a
  distribution the model never sees at inference, which is the bug four models here have already
  shipped. Reference parity: `NFKMLXSegFormerObjective.loss(logits:labels:)` is separable from the
  forward pass so the oracle can score identical logits, and matches transformers'
  `SegformerForSemanticSegmentation` loss (`run_reference.py segformer_loss`,
  `IK_PARITY_SEGFORMER_LOSS`).
- `NFKMLXZeroDCETraining` — the first customization recipe, and the template for the rest.
  Zero-DCE is zero-reference: the reference trains it with no ground truth, so a consumer
  customizes it from their own dark photos with nothing to annotate, which is the only kind of data an
  end user has. `NFKMLXZeroDCELoss` ships the four losses (exposure, color constancy, illumination
  smoothness, spatial consistency); `NFKMLXZeroDCEObjective` weights them, and its `wellExposedLevel`
  is the consumer-facing knob (preferred brightness). `NFKMLXZeroDCE.network(weightsURL:)` builds the
  trainable net; `fineTune` runs it. Reference parity: all four losses agree with the reference to
  float precision (`run_reference.py zero_dce_losses`, `IK_PARITY_ZERO_DCE_LOSSES`). A wrong loss is
  invisible in a fine-tune's output, so the oracle is the only check that catches it — the first
  implementation, written from the paper rather than the code, was wrong in all four. Three reference
  expressions that read like slips are reproduced deliberately and marked where they occur.
- `NFKStableDiffusionProvider` (`@objc`) — the bridge that lets the core activate the bundled
  `NFKMLXBackend` (Stable Diffusion) without depending on InferKitMLX. It conforms to the core's
  `NFKDynamicBackendProvider` and is named exactly the default the core tries for its `stable-diffusion`
  capability, so linking InferKitMLX makes `NFKDynamicBackend.stableDiffusionBackend()` return a working
  SD backend, built lazily. It returns SD 1.5: the ungated release, so the capability activates with
  no credential. See "Dynamic backend discovery" below.
- `NFKMLXImageBridge` — shared `CGImage`/`MTLTexture` ↔ `MLXArray` conversion, preserving alpha, so a
  matting model returns a cutout rather than three channels. Its CoreGraphics/Metal byte halves are
  unit-tested.
- Direct `@objc` factories (the primary ObjC path for shipped real models, no registry): every real
  model class exposes `+backendWith[Variant:]weightsURL:error:` (local weights, nil → random weights,
  `isReady` true) and a download companion `+backendWith[Variant:]repo:weightsPath:revision:cacheDirectoryURL:error:`
  (downloads via `NFKHFHub` then builds; blocking — run off the render thread). Each blocking download
  factory has an async peer `+backendWith[Variant:]repo:weightsPath:revision:cacheDirectoryURL:completionHandler:`
  (the handler gets `(id<NFKInferenceBackend>, NSError *)`; it runs on the download's background queue over
  the core's async `NFKHFHub`), so the caller does not hand-thread the fetch. `NFKMLXHub` has the same
  `…completionHandler:` peer. Variant models take an
  `@objc` enum: `NFKMLXRealESRGANVariant` (x4/anime/x2), `NFKMLXDepthVariant` (small/base/large),
  `NFKMLXDepth3Variant` (small/base/large), `NFKMLXNAFNetVariant` (sidd/goPro/reds/siddWidth64/goProWidth64),
  `NFKMLXYOLOVariant` (nano/small/medium/large/extraLarge), `NFKMLXU2NetVariant` (full/light),
  `NFKMLXWhisperVariant`, `NFKMLXSAMVariant` (compact/vitB/vitL/vitH), `NFKMLXSwinIRVariant`,
  `NFKMLXRVMVariant` (mobileNetV3/resNet50), `NFKMLXRTDetrVariant` (r50vd/r18vd/r34vd/r101vd),
  `NFKMLXRFDetrVariant` (base/nano/small/medium/large), `NFKMLXCLIPVariant` (vitB32/vitB16/vitL14/vitL14At336),
  `NFKMLXSigLIP2Variant` (every release), `NFKMLXSNACVariant` (speech24kHz/music32kHz/music44kHz),
  `NFKMLXHTDemucsVariant` (fourStem/sixStem); single-config models omit it. Every size a family's
  authors released is a case on its enum — that was audited against the release lists in September
  2026 and closed (`NFKMLXReleasedSizesTests`), so a missing size is a defect, not a backlog item. Each `register()` delegates to the
  local factory (DRY), so registry/`registerAll()`/`NFKMLXHub` behavior is unchanged. The `*Configuration`
  structs stay Swift-only; the enum is the ObjC knob. The shared `NFKMLXDownload` helper wraps both the
  blocking (`weightsURL`) and async (`backend(…build:completionHandler:)`) download (used by `NFKMLXHub`
  and these factories). All
- `NFKMLXModelRegistry` (`@objc`) — lets an Objective-C consumer build and run an MLX model without
  writing Swift: a model author registers a factory by name from Swift (capturing the `MLXArray`
  forward), and ObjC calls `+backendNamed:weightsURL:error:` to get an `id<NFKInferenceBackend>`. The
  bring-your-own-closure backends (`NFKMLXModuleBackend`/`MattingBackend`/`TensorBackend`/`SpeechBackend`)
  stay registry/Swift-only (their init takes a Swift closure); the shipped real models use the direct
  factories above.
  `NFKMLXReferenceModels.registerGreenScreenKeyer` is the shipped reference; a learned keyer
  (CorridorKey's GreenFormer) registers the same way. `InferKitMLXObjCExamples` proves the ObjC path.
  `NFKMLXReferenceModels.registerAll` registers every shipped model at once — the real models
  (`real-esrgan-x4` + `-anime`, `depth-anything-v2-small`/`-base`/`-large`, `depth-anything-3-small`/`-base`/`-large`, `lama-inpaint`, `sd-inpaint`,
  `fast-style-transfer`, `clip-vit-b-32`/`-b-16`/`-l-14`/`-l-14-336`, `siglip2-base-patch16-224` and every other SigLIP 2 release under its own name, `taesd`, `robust-video-matting` + `-resnet50`, `codeformer`, `zero-dce`, `modnet`, `yolo`,
  `segformer-b0`, `swinir-x4`, `colorizer-eccv16`, `pose-simplebaseline`, `deeplabv3`, `conv-tasnet`, `denoiser`,
  `vad-marblenet`, `silero-vad`, `dac`, `snac`/`snac-32khz`/`snac-44khz`, `audio-tagger-panns`, `bisenet`, `video-super-resolution`, `htdemucs`/`htdemucs-6s`, `rtdetr`/`rtdetr-r18vd`/`-r34vd`/`-r101vd`, `rf-detr`/`rf-detr-nano`/`-small`/`-medium`/`-large`, `birefnet`, `mpsenet`, `gtcrn`, `sgmse`, `storm`, `mossformer2-se`, `deepfilternet3`, `voicerestore`, `resemble-enhance`, `metricgan-plus`, `cmgan`, `frcrn`, `mossformer2-sr`, `nuwave2`, `apollo`)
  and the reference stand-ins (`green-screen-keyer`, `tone-speech`, and the `diffusion-*` oracle
  pipelines, which are distinct from the real models of the same task). Depth `register` uses the
  `NFKMLXDepthConfiguration.small`/`.base`/`.large` presets; Real-ESRGAN `register` varies `blocks`
  (23 vs 6).
- `NFKMLXHub` (`@objc`) — combines the core's `NFKHFHub` download with the registry:
  `+backendNamed:repo:weightsPath:revision:cacheDirectoryURL:error:` downloads a model's weights from
  Hugging Face and builds the registered backend around them (fail-fast if the name is unregistered).
  `NFKHFHub` stays MLX-free in the core; this companion helper is where HF-download and MLX meet.
- `NFKMLXRandom` / `NFKMLXGPU` / `NFKMLXDevice` (`@objc`) — thin `NSObject` wrappers that expose MLX's
  global runtime knobs to Objective-C, which mlx-swift ships as a free `seed(_:)` function, a
  `GPU`/`Memory` enum, and a `Device` struct (none of which bridges to ObjC). `NFKMLXRandom.seed:` seeds
  the global RNG for reproducible weight init/sampling; `NFKMLXGPU` surfaces GPU memory management
  (`activeMemory`/`cacheMemory`/`peakMemory`/`cacheLimit`/`memoryLimit` getters,
  `setCacheLimit:`/`setMemoryLimit:`/`clearCache`/`resetPeakMemory`, all bytes). Use `MLX.Memory.*` (not
  the deprecated `MLX.GPU.*` memory members) to stay warning-free. It also reports what the machine
  has — `physicalMemory` (sysctl), `recommendedWorkingSetSize` (Metal's own budget, which is what a
  model should be sized against rather than the physical total), `reclaimableMemory` (the cache, which
  `clearCache` returns), `memoryPressure`, and `deviceArchitecture` — and
  `applyStandingLimits(cacheBytes:fractionOfRecommendedWorkingSet:)` sets a standing cache cap plus a
  soft memory limit derived from that budget. The standing cap is the version of `clearCache()` that
  does not have to be remembered at every model boundary.
  There is deliberately no `setWiredLimit:`. mlx-swift 0.31.6 admits a wired limit only through an
  async, scoped ticket — its synchronous `withWiredLimit` is deprecated and a documented no-op — so a
  persistent setter could only be a knob that silently did nothing. `NFKMLXGPU.withWiredLimit(_:_:)` is
  the scoped async Swift form, and stays Swift-only for the same reason the closure backends do.
  `NFKMLXDevice` selects the compute device — `currentType` and
  `performOnDeviceType:block:`, over the **scoped** `withDefaultDevice(_:_:)` rather than the deprecated
  global `setDefault(device:)`. The selection is task-local, so it does not cross a dispatch:
  measured, a block dispatched asynchronously inside the scope reports the global default, and so does a
  fresh `Thread`, while a synchronous call on the calling thread inherits it. That makes it the wrapper
  for `runInferenceForRequest:` and not for `submitInferenceJobForRequest:`, whose background queue takes
  the global device; a caller wanting a whole inference on the CPU runs the synchronous call inside the
  block on their own thread, which is where the contract puts a multi-second inference anyway. The `@objc`
  case names are given explicitly (`NFKMLXDeviceTypeCPU`/`…GPU`), since Swift would generate `…Cpu`/`…Gpu`.
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
- HF vs MLX: `NFKHFHub` is a download/cache layer, not a runtime. Every model here downloads through
  it, the bundled Stable Diffusion releases included (`NFKMLXBackend.cacheDirectoryURL` chooses where).
  A gated repository needs a credential: `NFKHFHub.accessToken` sends it as a bearer token and falls
  back to the `HF_TOKEN` environment variable, which is where the tooling around the hub keeps one.
- Weights are downloaded at runtime, not bundled at build time (size / redistribution-licensing /
  update reasons). `NFKHFHub.downloadRepo:` resolves `<endpoint>/<repo>/resolve/<revision>/<path>`,
  fetches with `NSURLSession`, and caches at `<cacheDirectoryURL>/<repo>/<revision>/<path>`. The download
  **blocks** (a semaphore wait), so the caller runs it off the main/render thread — or uses the async
  `downloadRepo:…completionHandler:` (Swift `try await`). `cacheDirectoryURL` is host-supplied
  (security-scoped for sandboxed apps); `+defaultCacheDirectoryURL` gives a ready `Application
  Support/InferKit/models` location, and the `NFKMLXDownload`/`NFKMLXHub` factories substitute it when a
  caller passes `nil`. The core hub stays strict (explicit `nil` cache → fails, asserted by a test).
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

Swift↔ObjC gotchas seen here: an `MLXArray` is not `Sendable`, so a backend hands its networks to the
generation task through an `@unchecked Sendable` holder rather than capturing them across an isolation
boundary; the `@unchecked Sendable` extensions on the ObjC value types carry `@retroactive`.

## InferKitFoundationModels (companion package)

`InferKitFoundationModels/` is a separate SwiftPM package (macOS 26 / iOS 26 — the Foundation
Models floor; the model itself needs Apple Intelligence enabled). It depends only on the core.

- `NFKFoundationModelsBackend` adopts `NFKInferenceBackend` and wraps `LanguageModelSession`:
  `NFKInputPrompt` or `NFKInputMessages` (a system message becomes the session instructions),
  `NFKParameterTemperature` / `NFKParameterMaxTokens` map to `GenerationOptions`, and
  `streamResponse` feeds the job's `partialResult`. `isReady` mirrors
  `SystemLanguageModel.default.availability`. Multi-turn history seeds a `Transcript`
  (system → `.instructions`, user → `.prompt`, assistant → `.response`) so the model replays real
  conversation; `Transcript.TextSegment` / `Transcript.Prompt` / `Transcript.Response` are the entry
  constructors.
- Tool calling: `NFKFoundationTool` (name, description, typed `NFKFoundationToolParameter`s, a handler)
  registers on `backend.tools`. `NFKToolAdapter` adapts each to Apple's `Tool` with a runtime schema
  (`DynamicGenerationSchema` per parameter → `GenerationSchema`); the model's arguments arrive as
  `GeneratedContent`, read out with `content.value(T.self, forProperty:)`. No compile-time `@Generable`
  type is needed. Tools combine with the seeded transcript via `LanguageModelSession(tools:transcript:)`.
- Structured output: `backend.responseSchema` (a `[NFKFoundationToolParameter]`) switches generation
  to `session.respond(to:schema:)`; the result carries the parsed dictionary under `NFKOutputStructured`
  (a core key) and JSON under `NFKOutputText`. `NFKSchema` builds the schema and reads `GeneratedContent`
  back for both tools and structured output.
- Generation tests skip (`XCTSkipUnless`) where the model is unavailable, so CI stays green.
- `NFKFoundationModelsProvider` (`@objc`) conforms to the core's `NFKDynamicBackendProvider` and is
  named the default the core tries for `NFKCapabilityTextGeneration`, so linking this package activates
  on-device LLM through `NFKDynamicBackend.backendForCapability:` with no registration (mirrors
  InferKitMLX's `NFKStableDiffusionProvider` / `NFKMLXWhisperProvider`).
- The reverse bridge (Apple's `LanguageModel` / `LanguageModelExecutor` provider protocols, WWDC26)
  needs the macOS 27 / iOS 27 SDK; it is documented in the package README, not built.
- Gotchas: SwiftPM tools 5.9 spells the platform `.macOS("26.0")` (`.v26` needs newer tools); the
  `NFKInferenceError` cases import into Swift as `.error_InferenceNotReady` style.

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

## Completing an InferKitMLX model to parity (the documentation checklist)

A model is not done when its parity test passes — it is done when every listing is updated too. A
partial update leaves the model missing from some indexes, which is the failure this checklist exists
to prevent. Update all of these, in the modality's existing section, mirroring the sibling rows:

- `CLAUDE.md` — a full per-model entry in the InferKitMLX model list, and the model's registered name
  in the `registerAll` prose list. Use the actual measured cosines, not the `> 0.999` test threshold.
- `AGENTS.md` — one document, two names: after editing `CLAUDE.md`, run `cp CLAUDE.md AGENTS.md`.
- `README.md` — the model's name in the modality bullet of the model list.
- `Docs/companions.md` — the full model gallery (README links here as "the full model gallery").
- `Docs/model-index.md` — the index row (entry class, network, configuration, registered name, the
  Swift + Objective-C copy-and-paste, base backend).
- `Docs/model-parity.md` — the parity row with the real recorded numbers (capture them by temporarily
  printing the measured cosines in the parity test, then revert the prints).
- `Docs/examples.md` — the modality's gallery snippet.
- `Docs/inference-guide.md` — the Roadmap, if the model was a roadmap item (mark it shipped).
- `InferKitMLX/Sources/InferKitMLX/InferKitMLX.docc/InferKitMLX.md` — the gallery Topics list.
- `InferKitMLX/Sources/InferKitMLX/InferKitMLX.docc/ModelGallery.md` — the gallery table row.
- `InferKitMLX/Sources/InferKitMLX/InferKitMLX.docc/ModelIndex.md` — the index table row and the
  per-section copy-and-paste code block.
- `InferKitMLX/Examples/MLXModelGalleryExamples.swift` — the live per-model gallery example (a build +
  representative forward). This is a compiled test, so run it (`InferKitMLXExamples` scheme).
- `Tools/validation-assets/manifest.json` — the checkpoint/record/config entry (and an
  `oracle_environments` note if the model needs a new interpreter or extra packages).
- `~/.inferkit-validation.json` — the model's `IK_VAL_*` / `IK_PARITY_*` keys, as absolute paths into
  the local validation store, so the full check exercises the model by default rather than only when
  those keys are set in the environment. The test class must read them through
  `NFKMLXValidationConfig.environment` (the process environment merged with that JSON), not
  `ProcessInfo.processInfo.environment` directly, or the JSON keys never reach it. This file is a
  machine's local config, not a tracked repository file, so its "update" is provisioning rather than a
  commit — but skipping it leaves the model's parity test silently skipped on a plain run (green with
  nothing behind it), which is exactly the gap a full models test exists to catch.

Not a listing, so not required per model: `InferKitMLX/ObjCExamples/MLXObjCExample.m` is a curated
illustrative set, not an exhaustive gallery.

The code/wiring that accompanies the docs (the model file, `NFKMLXReferenceModels.registerAll`
registration, the `run_reference.py` oracle mode, and the parity test) is covered by the shipped-model
pattern; a converter under `Tools/<model>-to-safetensors/` is optional because the native `.pth`/`.pt`
reader loads most released checkpoints directly.

### Entry house style (keep the whole consistent)

These files accreted across many sessions and drifted into several voices for the same thing. A new
entry matches its neighbors in the file it lands in, not the last entry a different session happened to
write. The "Documentation Style (enforced)" section above governs the prose (no em-dash dramatic asides,
no antithesis, no rule-of-three, one fact per sentence, present tense, American English); it applies to
these entries too, and where an older entry breaks it, the break is not a precedent to copy. The
per-file shape:

- CLAUDE.md / AGENTS.md model list — one bullet, `` `NFKMLXFoo`` `` (`` `(@objc)` `` only when the
  class is), then ` — `, then a lowercase noun phrase naming what it is and its reference
  (`the X (`ReferenceClass`, Vendor)`). State the architecture, the load-bearing facts, and the measured
  parity with the ACTUAL cosines. Say "at reference parity" in running prose — one casing, lowercase, no
  bold on the phrase itself; reserve bold for a specific load-bearing noun, not for emphasis. "at parity"
  is only for a short back-reference to a result already stated (a second size, another variant). Depth
  is proportional to the model's novelty, not to how recently it was added; a configuration-only variant
  is a sentence, not a section.
- **README.md** — the model's name in the modality bullet, nothing more.
- **Docs/companions.md** — a prose gallery bullet. Describe the model and its variants/presets and say
  "at reference parity against <reference>"; do NOT quote a tiny-config cosine here (those live in
  model-parity.md). A released-weight cosine may appear when it is the headline result, matching the
  neighbors that do.
- Docs/model-index.md and the DocC `ModelIndex.md` — one table row (entry class, network,
  configuration, registered name or "Swift API", the construction line, base backend). In the DocC file,
  the section also carries a copy-and-paste code block that is a per-FAMILY cheat-sheet — one
  construction line per model family, not a mirror of every table symbol — so every family in the table
  has at least one line there, and a code line never names a family the table omits. The drift to stop is
  a family present in one and absent from the other (a new family's table row with no cheat-sheet line,
  as SD3/FLUX/ControlNet were), not a per-symbol mismatch.
- Docs/model-parity.md and the DocC `ModelGallery.md` — the parity row (and gallery row) with the
  REAL recorded cosines at full precision, not rounded or a `> 0.999` threshold. Multiple seams are
  `seam A x; seam B y; final z`.
- **DocC `InferKitMLX.md`** — the symbol(s) in the modality's Topics list.
- **`MLXModelGalleryExamples.swift`** — a build-plus-forward example, for a REGISTERED `@objc` model.
  A generative pipeline (SD3/FLUX/Z-Image/…) is not registered and is absent here by design, so a new
  one that is also unregistered stays absent — consistently, not by omission.

When an edit touches a file, leave that file MORE consistent than you found it: if the neighbors already
share a shape, conform to it; if a genuinely better shape is warranted, do not introduce a third — raise
it so the whole file moves together rather than one more divergent entry accruing.

## Safeguards (Anti-Patterns)

Required without exception:

- **NEVER** run `git clone/mv/restore/rm/add/branch/commit/merge/rebase/reset/pull/push/fetch`
  without developer approval first.
- **NEVER** run `rm` on any path without developer approval first.
- **NEVER** erase or overwrite files for the task of unit testing — the changes being tested must be
  preserved.
- **NEVER** delete a file or folder until its associated task is completely finished.
