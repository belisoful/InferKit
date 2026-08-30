# InferKit Agent Guidelines

## Project Overview

**InferKit** is a small, cross-platform inference toolkit for Objective-C. It provides a swappable
backend protocol, request/result value types, an async job handle, the shipped backends (mock,
in-process Core ML, an on-device Core ML language-model runner, OpenAI-compatible chat and
transcription clients, a submit-poll-fetch base, and runtime discovery — plus the companion-package
MLX and Foundation Models backends), a texture-tensor conversion, tokenizers, and a Hugging Face
model-download layer. It has no FxPlug or
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

# The MLX companion (Apple Silicon, macOS 14 / iOS 17) is a separate package
cd InferKitMLX && swift build && swift test

# The Foundation Models companion (macOS 26 / iOS 26) is a separate package
cd InferKitFoundationModels && swift build && swift test

# Validate the CocoaPods spec (fast, no build)
pod lib lint InferKit.podspec --quick
```

### Full Check (required before commit)

Code is commit-ready only when every check below passes.

`.github/workflows/ci.yml` runs the hosted-runner subset on every push and pull request: the core's
build (zero warnings) + tests, the iOS and tvOS compile legs, the analyzer at a fresh derived-data
path, the podspec lint, and compile checks for both companions. The MLX test schemes stay a LOCAL
step: they evaluate real MLX arrays (Metal) and read multi-gigabyte checkpoints from
`~/.inferkit-validation`, which a runner does not have — there they would skip, proving nothing.

1. `swift build` + `swift test` on the host — **0 warnings**, all tests green.
2. `xcodebuild build` for a `generic/platform=iOS` destination (cross-platform compile), and for tvOS
   through the SDK: `-workspace InferKit.xcworkspace -scheme InferKit -sdk appletvos26.5 -arch arm64`.
   **A tvOS DESTINATION does not resolve here and that is not the same as tvOS being unbuildable.**
   Xcode derives destinations from installed platform *support*, which a machine without the tvOS
   platform lacks even when the tvOS SDK is present — and the SDK is what a compile needs. Naming the
   SDK builds the core for tvOS. This check recorded that leg as unverified for as long as it used the
   destination form. A bare package rejects `-sdk` (it demands `-destination`), so the invocation goes
   through the workspace.
2b. `xcodebuild analyze -scheme InferKit -derivedDataPath <FRESH DIR>` — **0 analyzer issues**. Use a
   fresh derived-data path: the analyzer is cached, and reusing one silently reports nothing.
3. `InferKitMLX/` `swift build` + `swift test` when a change touches the MLX companion. MLX cannot
   evaluate under `swift test`, so the real run goes through xcodebuild — and the package has **three**
   test targets, each with its own shared scheme in `.swiftpm/xcode/xcshareddata/xcschemes/`. Run all
   three (`-destination 'platform=macOS' -skipPackagePluginValidation` throughout):

   ```
   xcodebuild test -scheme InferKitMLXTests        …    # 563 — the model and API suite
   xcodebuild test -scheme InferKitMLXExamples     …    #  35 — the Swift documented snippets
   xcodebuild test -scheme InferKitMLXObjCExamples …    #  22 — the Objective-C ones
   ```

   **`-scheme InferKitMLX` is the LIBRARY scheme and runs only the first testable**, which is the
   collapse the core's workspace note above describes: it executed the `InferKitMLXTests` methods
   and silently ran neither examples target, so 57 tests went unclaimed while the command reported
   success. The examples targets are what keeps a documented snippet from rotting, which is exactly
   what a silent skip defeats. The per-target schemes exist to make that impossible; keep one testable
   in each. Only MLX is FORCED onto xcodebuild — `swift test` runs every test target a package
   declares, so the core (177) and `InferKitFoundationModels` (24) are covered by step 1 and step 4
   whatever Xcode does with their schemes.
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

## Project Structure

```
./
├── Sources/InferKit/
│   ├── include/InferKit/            # Public headers (the API surface)
│   │   ├── InferKit.h               # Umbrella (imports every public header)
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
├── Tools/reference-parity/          # Offline: runs a model's (or a training objective's) reference implementation, records input + result for numeric comparison
├── Package.swift                    # SwiftPM manifest
├── InferKit.podspec                 # CocoaPods source spec
├── AGENTS.md / CLAUDE.md
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

**No preset carries a default model name.** Model identifiers change faster than a release does, and a
stale default fails at the first call with a message about the model rather than about the default.
Each provider's `modelsURL` is where its list lives. Readiness is not the same question:
`NFKAnthropicBackend` reports not-ready without a model because the API requires one, while an
OpenAI-compatible backend is ready with an endpoint alone, which **llama.cpp depends on** — its server
answers for whatever model it has loaded.

**Deliberately absent.** Midjourney has no official public API (its API host does not resolve), so
shipping a preset would imply one exists. `opencode.ai` answers `Not Found` on its API path — it is a
coding agent that calls other providers rather than an inference service. Codex is OpenAI's coding
agent, not a separate endpoint; it is the `openai` preset.

`NFKRemoteProviderTests` stubs the transport to assert Anthropic's request shape without a network, and
carries one live test gated on `INFERKIT_LIVE_LOCAL_MODEL` that runs against a local server when one is
listening.

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

- `bpe-bytelevel` → `NFKByteLevelBPETokenizer`, the GPT-2 / Qwen scheme.
- `clip` → `NFKCLIPTokenizer`, which the CLIP image-text model and every Stable Diffusion text encoder
  take. It **subclasses** the byte-level tokenizer through four hooks — `normalizedText:`,
  `pretokenizationPattern`, `symbolsForWord:`, `finalizedText:` — because CLIP shares byte-level BPE
  and differs in exactly those places: text lowercases and its whitespace collapses; the pattern takes
  a run of letters, **one** digit, or a run of other non-space characters, with no leading space (so
  "2024" is four tokens); a word's last character carries `</w>`, which the vocabulary distinguishes;
  and decoding turns `</w>` back into a space.
- `unigram` / `sentencepiece` → `NFKUnigramTokenizer`; `wordpiece` → `NFKWordPieceTokenizer`.

`encode:` returns the ids for the text alone. A model input's start and end markers and its padding
are the model's geometry, not the tokenizer's, so they are added where the context length is known —
`NFKMLXSDPromptTokenizer` for the diffusion path.

## Dynamic backend discovery

`NFKDynamicBackend` (core, Foundation-only) activates an optional engine **only when its classes are
linked into the consumer's build**, with no build dependency on that engine. InferKit ships only
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

**What made it possible.** MLX needs `default.metallib`, which SwiftPM delivers as a
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

**The static xcframework carries its own shaders, and that is delivery only.** An xcframework is a
build-time container that never ships, and Xcode copies nothing out of a static one, so the consumer
still places the bundle in their own product — measured, not assumed: a `binaryTarget` consuming a
slice with the bundle inside builds and then throws `Failed to load the default metallib`, and copying
the bundle beside the binary makes the same probe pass. What co-location buys is that the shaders
cannot arrive separately from the library, at 1.7 MB compressed against the two forms an earlier layout
shipped in a second asset. SwiftPM accepts the extra file in a slice without complaint. The earlier
separation was also justified here by a claim that a `binaryTarget` zip may hold nothing beside the
xcframework; **that claim was never tested** — SwiftPM requires `https` for a `binaryTarget` URL, so it
cannot be checked against a local server — and the question is now moot, because the zip holds exactly
one `.xcframework` either way.

**Neither variant is smaller.** A consumer using one model links to 13.8 MB static against 14.2 MB
dynamic — dead-stripping buys about 3%, because `Cmlx.o` is one merged object and MLX's runtime is
densely interconnected. Of the ~18 MB a slice weighs, MLX's C++ is 72% of the code and its shaders are
3.6 MB; **InferKitMLX and the core together are about 320 KB**. So the choice is deployment mechanics:

- Static needs no embedding and no code signing, and vends both modules from one plain modulemap.
- Dynamic is one self-contained drop-in and is shared between several consumers, but **clang refuses a
  non-framework module inside a framework**, so its `InferKit` module needs a modulemap passed with
  `-fmodule-map-file`. An Objective-C consumer is better served by the static variant.

**Linking recipes and the release matrix live in `Docs/installation.md`**, one per consumer shape, each
verified by building and running it: Xcode static and dynamic, SwiftPM `binaryTarget` for both variants
(SwiftPM wires the static one's headers and modulemap automatically; the dynamic one still needs
`CoreHeaders` passed through `-fmodule-map-file`), plug-in bundles, and a consumer's own static library.
The artifacts are NOT committed — this repository is source-distributed, so a consumer resolving it
clones its history. Three compressed release assets, one per variant carrying every slice: core 0.7 MB,
static 28 MB, dynamic 19 MB. The core's `build.sh` cleans only its OWN artifact — both scripts share
`.xcframework-build/`, and an `rm -rf` of the whole directory once discarded a twenty-minute MLX build
beside a twenty-second core one.

`--variant static|dynamic|both`, `--slices macos,ios,iossim`, and `--no-swift-interfaces` trim the
output. `--slices` also trims the build: each slice compiles those 289 translation units again, so it
is the difference between about seven minutes and twenty. `--no-swift-interfaces` applies only to the
dynamic framework — the static library never carried interfaces, and is Objective-C-only by
construction. `verify-mlx.sh` reports what it skipped rather than failing when a variant or the
macOS slice is absent; an iOS binary cannot be executed on the host.

**Nothing in `Package.swift` changes for distribution.** One `xcodebuild archive` of the ordinary
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
- **Backward compatibility** — the toolkit is pre-1.0. Until the first tagged release, prefer the
  correct design over source compatibility. Once shipped, point releases stay backward compatible.
- Document the introducing version on new public methods and classes.
- **Backends are the extension seam.** A consumer brings a heavier runtime (MLX, a C/Rust engine) by
  adopting `NFKInferenceBackend`. The core ships only backends with zero third-party dependencies and
  no license entanglement (pure Apple frameworks).
- Inference by contract is synchronous and multi-second; a caller runs it off the main/render thread
  and prefers `submitInferenceJobForRequest:` (or the `NFKInferenceSubmit` wrapper) for progress and
  cancellation.

## Code Comments

The bar for a comment is high. Code should explain itself through clear naming and structure;
comments are reserved for what the code cannot express.

- Use HeaderDoc `/*! ... */` blocks for public types, methods, and properties. Put multi-sentence
  rationale in the method's `@discussion`.
- Write an inline `//` comment ONLY when it documents something non-obvious the code cannot state on
  its own: a subtle invariant (e.g. "must run on the lock"), an external constraint or platform quirk
  (e.g. an SPM header-search-path requirement), or a deliberate omission a maintainer might otherwise
  "fix".
- Do NOT narrate what the code obviously does, restate the method name, or leave historical /
  "FIX:" / "previously the code did X" justifications. The diff and commit message carry that.
- The same bar applies to test code: the test method name describes intent; add a comment only for a
  non-obvious setup or invariant.

## Documentation Style (enforced)

HeaderDoc blocks and comments are technical documentation written with direct technical statements.
Language and National Variety: English — American.
Qualities of the writing: clear, thorough, easy to comprehend, not verbose (brevity), timeless,
integrated, wholistic.
Tense: Present — tuned for ease of comprehension.

_Banned constructions_:
- **Antithesis / "not merely X — it Ys"**: no "does not just X, it Ys", "is not a Y, it's a Z",
  "rather than X, it Ys". State what it does, once.
- **Em-dash dramatic asides** used for emphasis or reveal ("— and that's the point"). Use a period
  or plain clause.
- **Editorializing / filler.**
- **Rule-of-three rhetorical lists** and build-up sentences. One fact per sentence.

Prefer subject–verb–object declaratives, and bullet lists of `condition → result` where appropriate.
Documentation informs and describes; it is not persuasive writing.

## InferKitMLX (companion package)

**Everything here builds for iOS.** `mlx-swift-examples` is gone: it was pulled in for one thing, the
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
  a source latent + mask runs inpainting (kept region held to the source each step). `NFKDiffusionScheduler`
  is the sampler seam. **`NFKDDIMScheduler` is at reference parity against diffusers' own
  `DDIMScheduler`** (worst per-step latent cosine 0.9999999999999941, add-noise 0.9999999999999983,
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
  **ControlNet/LCM need no full SD reimplementation**: LCM is a scheduler swap, ControlNet is a `denoise`
  closure over `conditioning["control"]`; the UNet is brought by the consumer's `denoise` or a
  dynamically linked SD engine.
- `NFKMLXRealESRGAN` (`@objc`) — a real single-forward model, not a stand-in: the Real-ESRGAN
  generator (RRDBNet) implemented in `MLXNN` (`Conv2d` + `leakyRelu`, norm-free), run through
  `NFKMLXModuleBackend` for ×4 upscaling. `+register` puts it in the registry under `real-esrgan-x4`,
  so ObjC/MetalForge builds it by name (and downloads weights via `NFKMLXHub`). The module structure
  and parameter names mirror the reference PyTorch `RRDBNet`, so `loadWeights(into:from:)` loads a
  **safetensors** checkpoint (`loadArrays` → `update(parameters:)`), transposing 4-D conv weights from
  PyTorch `[out,in,kH,kW]` to MLX `[out,kH,kW,in]`. A `.pth` release converts to safetensors first with
  `Tools/realesrgan-to-safetensors/convert.py`.
  The weight-load path is proven offline: a test saves the net's params in PyTorch layout, reloads
  through the transpose, and confirms the forward matches (no download). This added the `MLXNN`
  product of `mlx-swift` to the target. **Reference parity** on the general ×4 release (cosine
  0.9999947) and on the **anime** release, a six-block generator where the general one has
  twenty-three (0.9999956).
- `NFKMLXDepthAnything` (`@objc`) — a real single-forward depth model: the Depth Anything V2 DINOv2 ViT
  encoder (`pretrained.*`) + DPT head (`depth_head.*`) in `MLXNN`, run through `NFKMLXModuleBackend`
  (image → grayscale depth under `NFKOutputImage`). `+register` under `depth-anything-v2-small`;
  `NFKMLXDepthConfiguration` holds the ViT-Small dims (Base/Large change them). The encoder runs a fixed
  518×518 (so `pos_embed` matches without interpolation) and the map resizes back. `loadWeights(into:from:remap:)`
  loads a **safetensors** checkpoint; the DPT key layout is intricate, so `Tools/depth-anything-to-safetensors/convert.py`
  is self-validating (matches every key against the module's expected layout, reports mismatches).
  **Reference parity across all three released sizes** — Small 0.99817 (encoder seam 0.9999924),
  **Base 0.99807**, **Large 0.99846**. The oracle drives the authors' own `depth_anything_v2` package
  (the real directory on `sys.path`; `IK_DEPTH_VARIANT` picks the encoder config, and both depth modes
  take a `--checkpoint`). It drove `transformers` until that package stopped registering the
  `depth_anything` model type and every size began raising `KeyError` — the parity test kept passing
  because it compares against a stored record, not a live oracle. Rebuilding on the original
  repository reproduced the previous numbers to five decimals, which is what proves the replacement
  faithful rather than merely green.
- `NFKMLXU2Net` (`@objc`) — a real single-forward background remover: the U²-Net nested-U saliency
  network (Residual U-blocks) in `MLXNN`, run through `NFKMLXMattingBackend` (plate → straight
  foreground + saliency alpha, matte under `NFKOutputMask`). `+register` adds full `u2net` and light
  `u2netp`. Stage/side/`outconv` names match the reference; the RSU-internal convs are `enc`/`dec`
  arrays, and `Tools/u2net-to-safetensors/convert.py` renames `rebnconvN` → `enc`/`dec` so the file
  loads directly. Forward + matting round-trip tested under xcodebuild with the light config.
  **Reference parity** on both releases: the full network 0.9992 and `u2netp` 0.9998 — the light model
  is a separate class in the reference, not a configuration of the full one.
- `NFKMLXNAFNet` (`@objc`) — a real single-forward restoration network (denoise / deblur): a U-shaped
  stack of NAFBlocks (SimpleGate + Simplified Channel Attention, channel LayerNorm = last-axis in NHWC,
  PixelShuffle up) in `MLXNN`, run through `NFKMLXModuleBackend` (image → restored image at input size,
  padding to `2^levels` and cropping back). `+register` under `nafnet`; `NFKMLXNAFNetConfiguration` sets
  width and block counts (default SIDD width 32). Block names (`conv1`…`conv5`, `norm1/2`, `beta`,
  `gamma`) match the reference; `Tools/nafnet-to-safetensors/convert.py` renames `middle_blks`/`ups.N.0`/
  `sca.1` so a real checkpoint loads directly. **Reference parity** against megvii-research's own NAFNet
  on the released SIDD width-32 denoiser (cosine 0.9999972, mean |difference| 0.00098 through the
  backend's 8-bit bridge), every parameter covered on the first triage run. The weights are **not**
  Drive-only as previously recorded — `huggingface.co/nyanko7/nafnet-models` mirrors all five releases.
  The **GoPro deblurrer** is also at parity (0.9999973). It is not a width change: it puts twenty-eight
  of its blocks in the last encoder stage and one in the middle, where SIDD spreads `[2, 2, 4, 8]` with
  twelve — so a checkpoint only fits the geometry it was trained as. The **REDS** release is that
  same distribution at twice the width (0.9999971), which is what separates a wrong width from a wrong
  block layout. `NFKMLXNAFNetVariant` (`.sidd`/`.goPro`/`.reds`) selects the geometry from
  Objective-C.
- `NFKMLXRIFE` (`@objc`) — real frame interpolation: the released **HDv3** IFNet in `MLXNN`, run
  through `NFKMLXTensorBackend` (two frames under keys `frame0`/`frame1` → the middle frame under
  `NFKOutputImage`). Three identical IFBlocks (11 input channels, width 90) run coarse-to-fine at
  scales 4/2/1. Each block's trunk is **four groups of two convolutions, each group added back to its
  own input** — not one residual over eight — and flow and mask leave through **separate heads**
  (`conv1`/`conv2`), each two transposed convolutions undoing `conv0`'s ×4 stride. The net applies
  every block **twice per scale**: once as given and once with the frames swapped, the mask negated,
  and the flow halves exchanged, averaging the two, because the network is trained symmetric in its
  inputs. The bilinear backward warp is `grid_sample(align_corners=True, padding_mode='border')` built
  from `take` gather (MLX has no grid_sample); the per-scale resampling is bilinear, as the reference
  interpolates. `+register` under `rife`. `remapReferenceKey` strips the training wrapper's `module.`
  prefix, maps `blockN` → `blocks.N`, and names the `Sequential` entries of `conv0`/`convblock0…3`
  (convolution, PReLU) and the heads (`up1`, `prelu`, `up2`); the checkpoint's `block_tea` teacher has
  no counterpart and is ignored as an extra key. **Reference parity** against the released HDv3
  `flownet.pkl` (interpolated cosine 0.9999999999993, mean |difference| 2.7e-7). Warp, interpolate,
  remap, and round-trip tested. Weights: `huggingface.co/yow46228/RIFE` ships the checkpoint together
  with its own `IFNet_HDv3.py`, which is what pins the architecture.
- `NFKMLXRIFEv4` (`@objc`) — the third IFNet generation, a separate architecture rather than a
  variant of HDv3, at **reference parity** on the released `rife-flownet-4.13.2` weights (interpolated
  cosine 0.9999999999995, mean |difference| 2.3e-7). **Four** blocks instead of three (widths
  192/128/96/64), a learned frame `encode` module (`Head`: three convolutions and a transposed one to
  eight feature channels) whose features are **warped alongside the frames**, `ResConv` trunk entries
  that scale their convolution by a learned per-channel **`beta`** before the residual add, an
  upsampling convolution emitting `4 × 6` channels that **pixel-shuffles** ×2, and a **timestep**
  channel — which is what v4 adds: `interpolate(_:_:timestep:)` lands anywhere between the frames, not
  only the midpoint. Its convolutions activate with a parameter-free **leaky ReLU** where HDv3 used a
  PReLU; that difference surfaced as exactly eight uncovered parameters, which is the coverage guard
  naming a structural mistake rather than a number going quietly wrong. `+register` under `rife-v4`;
  pads to a multiple of 64 and runs scales `[8, 4, 2, 1]`. Oracle: the architecture vendored by
  ComfyUI-Frame-Interpolation (`rife_arch.py`, `arch_ver="4.17"`) — the version's own `IFNet.py` ships
  inside the model zip rather than in the repository, and only one ComfyUI device helper needs
  stubbing.
- `NFKMLXRAFT` (`@objc`) — real optical flow: the RAFT pipeline in `MLXNN` — shared feature encoder,
  all-pairs correlation volume + pyramid + bilinear lookup (via `take` gather), context encoder, and an
  iterative ConvGRU update. Run through `NFKMLXTensorBackend` (two frames `frame0`/`frame1` → a packed
  flow map under `NFKOutputImage`; raw `[H,W,2]` flow via `NFKMLXRAFTNet.flow`, the eighth-resolution
  field via `flowLow`). `+register` under `raft`.
  Faithful to RAFT-large (feature 256, 4 levels, radius 4), including the **convex-mask upsampling** —
  each output pixel is a combination of its coarse 3×3 neighborhood weighted by the mask the last update
  predicts (scaled ×0.25 as the reference does), over a **zero**-padded unfold. The default iteration
  count is low (6). Normalization is per encoder, as in
  the reference `norm_fn`: `fnet` uses a parameter-free InstanceNorm (`affine: false`, so the checkpoint
  carries nothing for it) and `cnet` uses BatchNorm, with `makeNet` setting eval mode for the running
  statistics. `flow` takes images in `0...1` and rescales to the trained `-1...1`; the correlation
  neighborhood is emitted in the reference's plane order (outer index shifts x, inner shifts y), which
  the trained 1×1 `convc1` depends on. The correlation lookup samples like the reference's
  `grid_sample(padding_mode: "zeros")`: a corner outside the map contributes **nothing**, where an
  edge clamp costs real accuracy (the lookup radius is 4 and the coarsest pyramid level is a few cells
  wide, so most of that neighborhood is outside). Converter `Tools/raft-to-safetensors/convert.py`
  renames `update_block`/`downsample`/`flow_head`/`mask`. **Reference parity** against princeton-vl's
  own RAFT on raft-things (eighth-resolution flow cosine 0.9999999999989, full-resolution 0.9999999999998);
  both sides run the same iteration count. Flow + round-trip tested under xcodebuild (the correlation
  lookup is many gather ops, so it is slow).
- `NFKMLXSAM` (`@objc`) — real promptable segmentation (Segment Anything): a ViT image encoder, a prompt
  encoder (point → sparse tokens via a random-Fourier positional encoding), and a two-way-transformer
  mask decoder with a hypernetwork mask head, in `MLXNN`. Run through `NFKMLXMattingBackend` (plate +
  point under `NFKSAMPointKey` → mask alpha + matte). `+register` under `sam`. The ViT encoder uses real
  windowed attention (`windowSize`, `globalAttnIndexes`) with decomposed relative-position embeddings
  (`rel_pos_h`/`rel_pos_w`, added via `take` gather + batched matmul). `remapReferenceKey` maps the
  reference's nested MLP, positional neck/upscaling Sequentials, and `transformer` submodule; scope the
  `.mlp.lin` rule to the encoder, or it eats the decoder's. `NFKMLXSAMVariant` (`.compact`/`.vitB`)
  selects the geometry on both the local and the download factories — the released `sam_vit_b`
  checkpoint fits only `.vitB`. **Reference parity** against the official `segment-anything` predictor
  (encoder cosine 0.9999986, selected-mask cosine 0.99993, binary agreement 99.7%); the decisive fix was
  `skip_first_layer_pe` in the two-way transformer's first layer. SAM 2 is `NFKMLXSAM2` below.
  Segment + round-trip tested under xcodebuild.
  **`NFKMLXSAM2` ports SAM 2's Hiera image encoder**, at **reference parity** against
  facebookresearch's own sources (finest FPN level 0.9999999999986, second 0.9999999999918, vision
  features 0.9999999999939), every parameter covered on the first triage run. Hiera is hierarchical
  where SAM's ViT is flat: four stages that halve the resolution and double the width, attention
  inside local windows except at designated global blocks, and stage transitions that **max-pool the
  queries** so one block changes both size and width — the shortcut takes the same projection and
  pooling so the residual still lines up. A **transition block keeps the previous stage's window**
  (the reference reads `window_spec[cur_stage - 1]` before advancing the stage), which is what makes
  block 10 use window 14 rather than 7; reversing that order makes the pooled windows the wrong size
  and the reassembly stops tiling. The position grid is resampled **bicubically**
  (`NFKMLXBicubic`, PyTorch's `a = -0.75` Keys kernel with half-pixel centers and border clamping)
  and a tiled window grid is added. The FPN neck projects each captured stage to 256 and fuses
  top-down on the deeper levels only, dropping the coarsest (`scalp`).
  **The prompt encoder and mask decoder are ported too**, at **reference parity** (sparse prompt
  0.99999999999998, mask logits 0.9999999999950, object score matching to five figures), every
  parameter covered on the first triage. The decoder is SAM's two-way transformer — attention,
  block, and MLP are shared with `NFKMLXSAM` — plus three SAM 2 additions: an **object-score token**
  leading the sequence with its own head, and **high-resolution features** from the FPN's two finer
  levels added during upscaling. Their `conv_s0`/`conv_s1` projections are the decoder's parameters
  but the reference applies them in its base model before calling it, so this port applies them
  internally and takes the levels as they come off the neck. Two traps: the query positional term is
  the **original** token embedding at every layer and again at the final attention (the running
  queries drift the masks without breaking anything visibly), and the reference shifts a click by half
  a pixel to the pixel's centre before normalizing.
  **The video memory path is ported too**, which completes the checkpoint: `NFKMLXSAM2MemoryEncoderNet`
  folds a frame's features and its predicted mask into a 64-channel memory, and
  `NFKMLXSAM2MemoryAttentionNet` conditions the next frame on that memory. Both are at **reference
  parity** (encoded memory cosine 0.9999999999999, attention output 0.9999999999997), every parameter
  covered on the first triage run. The encoder's mask downsampler is **four** stride-2 stages
  (1 → 4 → 16 → 64 → 256 channels), so it takes the mask at the **full 1024 frame resolution**, not at
  the decoder's low-resolution output — the tracker upsamples before encoding, and the total stride of
  16 is what lands it on the 64×64 feature grid. Its fuser blocks are ConvNeXt (7×7 depthwise, channel
  LayerNorm, a 4× pointwise MLP, and a learned per-channel `gamma` scale). The attention applies
  **axial rotary embeddings** (`NFKMLXAxialRotary`): adjacent channel pairs are rotated, the first
  half of the pairs by an x-frequency and the second half by a y-frequency, matching the reference's
  `view_as_complex` layout. Its positional-encoding switches are asymmetric and each one matters —
  self-attention and cross-attention **queries** take no positional term, cross-attention **keys** do,
  and the input takes `0.1 × position` once. The trap is the reference's `batch_first=True`, which
  describes what its layers want: `MemoryAttention` takes its inputs **sequence-first** and transposes
  them itself, so handing it batch-first tensors makes the tokens the batch and every token attends
  only to itself (that scored 0.86 and raised nothing). Checkpoint
  `dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt` (161 MB, 468 tensors, 39M
  parameters), of which 122 are the video path (`memory_attention`, `memory_encoder`); an
  image-only port needs the other ~250: `image_encoder` (154 trunk + 8 neck), `sam_prompt_encoder`
  (10), and `sam_mask_decoder` (~118). The `sam2` package **cannot be installed
  here** (it requires Python ≥ 3.10; this environment is 3.9), and the installed `transformers` is
  **4.33.3**, which has no SAM 2 — so the oracle vendors `backbones/{hieradet,image_encoder,utils}.py`,
  `position_encoding.py`, `sam2_utils.py`, `memory_attention.py`, `memory_encoder.py`, and
  `utils/misc.py`, all of which parse under 3.9, with `iopath` and `sam2.utils.misc` stubbed. Verified: the tiny config (embed 96, one head, stages
  `(1, 2, 7, 2)`, global attention at blocks 5/7/9, background window 7×7; neck `d_model` 256 over
  channels `[768, 384, 192, 96]`, top-down levels `[2, 3]`, nearest interpolation, `scalp` 1) loads
  **strictly**, and a 1024×1024 forward returns `vision_features [1, 256, 64, 64]` with FPN levels at
  256/128/64.
  **All three released Hiera sizes are at parity**: large (level0 0.9999999999994, level1
  0.9999999999986) — 48 blocks against tiny's 12, weighted toward the third stage, a coarser window
  there, and global attention much later, its config overriding every axis the Hiera constructor
  defaults — and base_plus (level0 0.9999999999958, level1 0.9999999999895), which sets only the
  width and head count and takes the rest from the defaults, so it is the size that proves the
  DEFAULTS rather than the overrides. The oracle's `sam2` package **is now in the manifest**: it
  cannot be pip-installed here and had been vendored by hand, leaving that oracle unreproducible.
- `NFKMLXLaMa` (`@objc`) — a real single-forward inpainter: the LaMa FFC-ResNet generator in `MLXNN`
  (each Fast Fourier Convolution runs a spatial branch and an FFT spectral branch via `MLXFFT`
  `rfft2`/`irfft2`, orthogonally normalized), run through an `NFKMLXMattingBackend` (plate under
  `NFKInputImage`, mask under `NFKInputMask`). `+register` under `lama-inpaint`. The configuration
  defaults are big-lama's own `config.yaml`: 64 base channels, three downsampling stages, **18**
  residual blocks, ratio 0.75 through the trunk and at the last downsample only, sigmoid output.
  `remapReferenceKey` translates the checkpoint's flat `model.N` Sequential — the parameter-free
  entries (reflection pads, the tuple concatenation, the activations) still consume an index, so the
  upsampling triples start at 24 — plus the spectral branch's `conv1.0`/`conv1.1` narrowing. The
  upsampling transposed convolutions carry `outputPadding` 1 (without it they land a pixel short of
  doubling, which an output resize can only paper over) and load with `transposed(1, 2, 3, 0)`, not
  the forward convolutions' axis order. **Reference parity** against advimman's own
  FFCResNetGenerator on the released big-lama (inpainted cosine 0.9999999999997, mean |difference|
  5.9e-8). The convolutions **reflection-pad**, as the reference's `padding_mode='reflect'` does:
  flipping them back to edge padding scores 0.99857 / mean 0.00503, so the approximation this model
  shipped with was a real defect, measured both ways rather than assumed.
- `NFKMLXSDUNet` / `NFKMLXSDAutoencoder` (`NFKMLXStableDiffusionModels.swift`) — the real
  `UNet2DConditionModel` and `AutoencoderKL` in `MLXNN`, in the diffusers layout. **One implementation
  serves every latent-diffusion model here**: they differ in scalars a configuration carries
  (channel widths, which levels attend, the cross-attention width, convolution-versus-linear
  transformer projections, a class embedding, how many transformer blocks an attention runs, whether a
  pooled embedding joins the timestep), not in structure. **Reference parity against diffusers
  on every released configuration**: SD-1.5-inpainting UNet 0.9999999999993, Marigold UNet
  0.99999999999, ×4-upscaler UNet 0.9999999999973, SD 2.1 UNet 0.9999999993, SDXL UNet
  0.9999999999955, SD autoencoder 0.99999999982 latent / 0.99999999917 decoded, upscaler autoencoder
  0.99999999998 / 0.99999999992. Every one covered on the first triage run.
  **SDXL adds two axes and no new structure**: `transformerLayers` (`transformer_layers_per_block`,
  `[1, 2, 10]` — the coarsest level runs ten transformer blocks where every earlier release runs one)
  and `additionEmbedding` (`addition_embed_type: "text_time"` — six `time_ids`, the original size, the
  crop's top-left corner, and the target size, each embedded at 256 and run together into 1536, joined
  by the second tower's 1280-wide pooled embedding and projected to the timestep's width).
  Three details are load-bearing. **`only_cross_attention`** (the ×4 upscaler sets it on three levels)
  makes a block's FIRST attention a cross-attention too, so its keys and values take the context's
  width — nothing in the checkpoint's key *names* says so, only the tensor shapes, and MLX adopts a
  checkpoint's shapes wholesale, so getting it wrong loads cleanly and fails later. The transformer's
  input normalization uses epsilon **1e-6** where every resnet uses 1e-5. The autoencoder's
  downsampling convolution pads **asymmetrically** (right and bottom only), where the UNet's pads
  evenly. The SD 1.5 autoencoders also predate the diffusers attention rename, so `remapVAEKey`
  accepts `query`/`key`/`value`/`proj_attn` as well as `to_q`/`to_k`/`to_v`/`to_out.0`.
  Oracle: diffusers cannot be installed beside the other oracles here (it needs a newer transformers
  than the 4.33.3 the Whisper / CLIP / SegFormer records were measured against), so it lives in its own
  virtual environment and every `sd_*` mode of `run_reference.py` runs under that interpreter. The
  manifest's `oracle_environments` records where it is and how to rebuild it. Weights are the released
  diffusers layout, one checkpoint per network.
- `NFKMLXSDPipeline` — a UNet and an autoencoder together, plus the text conditioning the released
  checkpoints cross-attend to. **The tower is not on this path**: the caller brings the embedding
  (`loadTextContext(from:)` reads a `[tokens, dimensions]` tensor), which is what the image-conditioned
  models here do — a model that takes no prompt still expects the embedding of an empty one.
  `NFKMLXTextToImage` is the path that owns a tower, because a prompt is its input.
  `loadWeights(unetURL:vaeURL:)` reads the released layout; `loadWeights(from:)` reads a single file
  holding both under `unet.`/`vae.`, which is what `NFKMLXWeights.save` writes, so a fine-tuned
  pipeline reloads through one path.
- `NFKMLXStableDiffusionInpaint` (`@objc`) — Stable Diffusion inpainting on `NFKMLXDiffusionBackend`,
  over those real networks. `encode` VAE-encodes the plate and the masked plate and builds the
  nine-channel conditioning (noisy latent, then mask, then masked-image latent — that order);
  `denoise` is the UNet (epsilon); `decode` is the VAE. The backend runs the DDIM loop and the
  per-step inpaint compositing. The masked-image latent is the *plate with the hole blanked, encoded*,
  not the plate's latent with the hole blanked — the encoder is not local, so the two differ.
  `+register` under `sd-inpaint`. Pipeline and offline round-trip tested under xcodebuild.
- `NFKMLXTextToImage` (`@objc`) — Stable Diffusion **text-to-image**, on `NFKMLXDiffusionBackend`, and
  what `NFKMLXBackend` is built from. `encode` turns the prompt into a conditioning sequence, `denoise`
  is the UNet with classifier-free guidance, `decode` is the autoencoder; the backend owns the loop.
  Above a guidance of 1 the conditional and unconditional predictions run as **one batch of two**,
  which makes guidance one forward pass rather than two, ordered unconditional first as the reference
  reads it back. `NFKMLXSDTextToImageConfiguration` carries the three releases
  (`.stableDiffusion15`, `.stableDiffusion21` / `.stableDiffusion21V`, `.sdxlTurbo`), and
  `NFKMLXSDReleaseFiles(directoryURL:)` resolves a downloaded release's tree (`unet/`, `vae/`,
  `text_encoder/`, `tokenizer/`, and for SDXL `text_encoder_2/` and `tokenizer_2/`), accepting the
  `.fp16.safetensors` spelling a half-precision-only release uses.
  **Reference parity end to end against diffusers' own pipelines**, starting from the reference's own
  initial latent (matching a random source across two implementations proves nothing about either):
  SD 1.5 image cosine 0.999998948, SD 2.1 (v-prediction) 0.9999986, SDXL-Turbo 0.9999973, SDXL with
  guidance 0.9999990, SDXL with no negative prompt 0.9999988. The record also carries the per-step
  latents and the first guided prediction, so a whole-picture mismatch says which stage diverged.
  **The sampler is DDIM in every case**, including SDXL-Turbo, whose release names
  `EulerAncestralDiscreteScheduler`; both sides run DDIM at the release's own `timestep_spacing`, so
  the comparison measures this port rather than two different samplers. A caller wanting the released
  sampler exactly brings one through `NFKDiffusionScheduler`.
  Three details are load-bearing, and each one first showed up as a wrong picture:
  **Stable Diffusion 2.x pads a prompt with `!` (id 0), not the end marker** — and
  `special_tokens_map.json` overrides `tokenizer_config.json`, which is where that is written, so
  reading only the config pads with 75 end markers and the model reads a different sentence
  (that scored 0.825). **A bfloat16 release turns a float32 module into a bfloat16 one**, because
  MLX's `update(parameters:)` adopts a checkpoint's element type along with its values; the SD 2.1
  text tower and autoencoder are published that way, and it cost three orders of magnitude
  (0.9999956 against 0.9999999999841). `NFKMLXWeightPrecision` makes that a choice rather than an
  accident. **`force_zeros_for_empty_prompt` acts on an ABSENT negative prompt, not an empty one** —
  the reference's condition is `negative_prompt is None`, and an empty string is a sentence the model
  is asked to encode.
- `NFKMLXSDTextEncoderNet` / `NFKMLXSDTextEncoder` — the CLIP text tower the releases cross-attend to,
  built from `NFKMLXCLIP`'s own blocks (one implementation, four configurations). `NFKSDTextOutput`
  spells the difference between the releases: SD 1.x and 2.x read the **last** hidden state after the
  final layer normalization (2.x drops the tower's 24th layer in its own configuration rather than
  skipping it here), while SDXL reads the **penultimate** one, before that normalization. Only SDXL's
  second tower carries a projection, and it is the pooled embedding SDXL's UNet conditions on — the
  pooled path runs the whole stack through the final normalization even when the sequence stops a
  layer short, and reads the position of the highest token id (the reference's rule, which the padding
  repeats, so the FIRST occurrence is the one). The releases are a `transformers` CLIPTextModel, whose
  attention stores separate `q_proj`/`k_proj`/`v_proj` where the module keeps the reference's fused
  projection, so the remap **concatenates three tensors into one** — a three-into-one a 1:1 key map
  cannot express, as SegFormer's `kv` is a two-into-one. The activation is `quick_gelu` for SD 1.x and
  the **exact error-function** GELU for the OpenCLIP towers, not the tanh approximation its neighbour
  `gelu_new` selects. **Reference parity on every released tower**: SD 1.5 0.9999999999986,
  SD 2.1 0.9999999999841, SDXL primary 0.9999999999986, SDXL secondary 0.9999999999257 (pooled
  0.9999999999820).
- `NFKMLXSDPromptTokenizer` — the release's `tokenizer/` directory driving the core's `NFKTokenizer`
  CLIP variant, plus the padding to the tower's context length. **Token-for-token agreement with
  `transformers`' CLIPTokenizer** over five prompts, including punctuation, an empty prompt, a
  multi-byte one, and the markers written out literally.
- `NFKMLXWhisper` / `NFKMLXWhisperBackend` (`@objc`) — real on-device speech-to-text: the Whisper
  encoder-decoder transformer in `MLXNN` (log-mel via MLXFFT `rfft` → audio encoder → greedy text
  decoder). Audio → text backend: reads `NFKInputAudio` (an `NFKAudioAsset` WAV via `NFKMLXWaveFile.read`,
  or NSData), returns `NFKOutputText`. `+register` under `whisper-tiny`. Module names follow OpenAI
  Whisper (`encoder`/`decoder`, `blocks.N`, `attn`/`cross_attn`, `mlp.0`/`mlp.2`), so a converted
  checkpoint loads with the conv transpose (`loadWeights` handles 4-D and 3-D Conv1d). `NFKTokenizer`
  (optional) detokenizes; else token ids. `Tools/whisper-to-safetensors/convert.py` targets the OpenAI
  `.pt`. **Reference parity** against openai-whisper itself (log-mel cosine 0.9999999999940,
  first-step decoder logit cosine 0.9999999997432, and an exact greedy token match).
  **The decoder's suppression rules are the reference's**, measured rather than approximated:
  `suppressTokens` masks the curated non-speech set at every step, and `suppressesBlankStart` masks a
  space and an immediate `<|endoftext|>` at the first sampled position only, which is `SuppressBlank`.
  `NFKMLXWhisperSuppression.nonSpeechTokens(using:)` computes that set from the model's own tokenizer
  by the reference's rule — a symbol contributes its first token when it encodes to exactly one token,
  a musical symbol contributes its first token however many it encodes to — because the ids differ
  between the English-only and the multilingual vocabulary. `backendWithWeightsURL:tokenizer:` wires it.
  The record carries BOTH decodings: `output` under the plain special/timestamp mask, which isolates
  the network, and `ruled_tokens` under the reference's own policy, which the port reproduces exactly.
  They differ on the synthetic clip (five tokens against three), which is why the policy needed
  measuring rather than describing. **Timestamped decoding is implemented and at an exact token
  match** against the reference (`transcribeWithTimestamps`, and `emitsTimestamps` on the backend,
  which adds `NSArray<NFKAudioSegment *>` under `NFKOutputSegments`). It is a different DECODE rather
  than a different reading of one: the times only exist when `<|notimestamps|>` is left OUT of the
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
  0.999999999994 against the reference's own. **HF-format checkpoints load too**: `loadWeights`
  detects the transformers naming (`model.encoder.layers.N.self_attn.q_proj`) and remaps it onto the
  OpenAI layout, asserted by renaming a real checkpoint into HF form and getting the identical
  transcription. **`small`, `medium`, and `large-v3` are ported too**, each at an exact token match
  against the reference (mel cosine 0.99999999999). Every size shares one encoder-decoder structure and
  differs in a width, a head count, and a depth — except **large-v3, which also produces 128 mel bands
  instead of 80 and carries one more language token**, shifting `<|transcribe|>` and `<|notimestamps|>`
  up by one. Both come from the model's own tokenizer rather than the smallest size's constants; the
  parity record carries the prompt the reference used, so a shifted id surfaces as a prompt mismatch
  rather than a mysterious token difference.
- `NFKMLXDemucs` / `NFKMLXDemucsBackend` (`@objc`) — real music stem separation: the time-domain Demucs
  U-Net in `MLXNN` (strided Conv1d + GLU encoder, transposed-conv decoder with skips, via `NFKMLXDemucsBackend`
  audio → four stems, each an `NFKAudioAsset` under its name "drums"/"bass"/"other"/"vocals"). `+register`
  under `demucs`. 1-D transposed conv is a `ConvTransposed2d` with a singleton width. **Reference parity**
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
  **Demucs v4 (htdemucs) is `NFKMLXHTDemucs`, a separate architecture** — `NFKMLXDemucsNet` is the v2
  time-domain U-Net and no v4 checkpoint fits it.
- `NFKMLXHTDemucs` / `NFKMLXHTDemucsBackend` (`@objc`) — Demucs v4 (Hybrid Transformer Demucs), at
  **reference parity** on the released `htdemucs` checkpoint (separated stems cosine
  0.9999999999996, mean |difference| 7.7e-8), every parameter covered on the first triage run and
  every stage seam exact on the first numeric run. Two branches run **in parallel**: a spectrogram
  branch over a complex-as-channels STFT (`nFFT` 4096, hop 1024, `torch.stft(normalized:)`, so the
  real and imaginary parts of each audio channel are two feature channels) and a waveform branch over
  the samples. Each is a four-stage U-Net of `HEncLayer`/`HDecLayer` — a strided convolution, a
  dilated `DConv` residual branch (compress ×4, GroupNorm, GELU, expand, GLU, a learned `LayerScale`),
  and a gated rewrite. They **never merge by injection**: every `tencoder` here is non-empty, so the
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
  **The vocoder runs REAL released weights at reference parity**: jik876's UNIVERSAL_V1 generator
  (whose geometry is this port's default configuration), cosine 0.9999999999341 against the
  reference's own `models.py` on a deterministic mel — a vocoder is a pure function of its mel, so
  nothing about speech needs assuming. The release stores every convolution weight-NORMALIZED
  (`weight_g`/`weight_v`); `Tools/hifigan-to-safetensors/convert.py` fuses `g·v/‖v‖`, which is the
  reference's own `remove_weight_norm`. Reaching parity found a real defect: the reference's ONE bare
  `F.leaky_relu(x)` before `conv_post` runs at PyTorch's default slope 0.01 where every other
  activation is 0.1 — with 0.1 the released weights score 0.99954, measured both ways. The
  upsampling stages load through the Demucs ConvT treatment (`[in, out, k]`, name-gated).
  **The trained acoustic model is ported too, and the voice is COMPLETE.** `NFKMLXFastSpeech2Net`
  is the espnet FastSpeech2 conformer (through the transformers layout, whose implementation is the
  oracle): relative-position attention with the Transformer-XL shifting trick, macaron POST-norm
  conformer layers (`normalize_before: false`), a GLU convolution module with BatchNorm, conv-FFN
  blocks, duration/pitch/energy variance adaptors (pitch and energy predicted per PHONEME and
  embedded before the durations stretch to frames), the length regulator, and the residual postnet.
  **Reference parity on the released LJSpeech weights ON THE FIRST NUMERIC RUN**: encoder
  0.9999999999998, durations exact frame for frame, pitch/energy/mel all ≥ 0.9999999999. Module keys
  are the checkpoint's names.
  `NFKMLXVoice` chains it with the vocoder and the release's own 78-symbol ARPAbet vocabulary (the
  matching phoneme table), exposed through `makeSpeechBackend(phonemize:)`. **The vocoder must be the
  PAIRED release** (`espnet/fastspeech2_conformer_with_hifigan`, `vocoder.` prefix, weight norm
  already fused): espnet's acoustic model emits mels NORMALIZED by its training statistics, and the
  universal jik876 generator — same geometry, raw-log-mel convention — turns them into loud garbage.
  Measured, not assumed: the end-to-end test synthesizes "hello world" and has the package's own
  Whisper (real weights, at parity) transcribe it — with the universal vocoder Whisper hears
  "(indistinct)", with the paired one **" hello, world."** — which closes the loop TTS → audio → ASR
  entirely inside this package on released weights.
- `NFKMLXVideoBackend` / `NFKMLXVideoFile` — the first backend that PRODUCES video, and the AVFoundation
  decode/encode layer under it (the video counterpart of `NFKMLXWaveFile`). `NFKModalityVideo` and the
  `NFKInputVideo` / `NFKOutputVideo` keys were in the core's vocabulary with nothing emitting a clip.
  The backend reads an `NFKVideoAsset`, hands every frame to a `([MLXArray]) -> [MLXArray]` transform
  as `[H, W, 3]` in `0...1`, encodes what comes back, and returns a new `NFKVideoAsset`. **The transform
  takes the whole sequence, not one frame**, because that is what the models need: frame interpolation
  reads pairs and returns MORE frames than it took, and BasicVSR propagates state backward and forward
  through time, so upscaling a clip is not upscaling its frames independently. A per-frame model simply
  maps. `frameRateMultiplier` / `outputFramesPerSecond` carry the rate change a frame count change
  implies — a doubled clip written at the source rate is slow motion, not smoother footage, so the
  duration is what stays fixed. `NFKMLXRIFE.clipBackend` (`n` → `2n - 1` frames at twice the rate) and
  `NFKMLXVideoSR.clipBackend` (×4, `upscaleSequence`) are the shipped users. AVFoundation's synchronous
  property accessors are deprecated, so the reads go through a semaphore-blocked `loadTracks` /
  `load(.nominalFrameRate)`; the contract is synchronous and the caller is already off the render
  thread. H.264 needs even dimensions and one frame size per clip, and both are rejected explicitly
  rather than cropped or scaled where the change would be invisible in the result.
  **Trained models are carried through the whole path in tests**, not only the shape: a clip is built
  by translating a real photograph (a synthetic gradient measures nothing — the models are trained on
  photographs), and the assertions are about the result. RIFE's synthesized frame must correlate
  better with the TRUE midpoint than with either neighbour (0.9981 against 0.9145 / 0.9104), which is
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
  one style). **Reference parity** against pytorch/examples (cosine 0.9999926). Reaching it required real
  reflection padding (`NFKMLXResample.reflectPadded`, a mirror gather — MLX pads with a constant or the
  edge value only): with edge padding the mean pixel error was 0.049, and only a quarter of that sat at
  the border, because the instance norms are global and carry a border approximation into every pixel.
  Names match the reference, so `Tools/style-transfer-to-safetensors/convert.py` only drops the
  deprecated InstanceNorm running-stats keys.
- `NFKMLXCLIP` (`@objc`) — real image+text embeddings (CLIP ViT-B/32): a ViT image tower (`visual.*`)
  and a causal text transformer (`token_embedding`/`transformer`/`ln_final`/`text_projection`), both
  L2-normalized into a shared space. Attention keeps the reference fused `in_proj_weight`/`out_proj`.
  **Reference parity** on both towers against transformers' `CLIPModel` (image cosine 0.9999965,
  **text cosine 0.9999999999988** — the text record carries the reference's own token ids, since the
  port embeds ids rather than text).
  `NFKMLXCLIPBackend` reads `NFKInputImage` → embedding under the new core key `NFKOutputEmbedding`; a
  text prompt encodes when a tokenizer is supplied (byte-level-BPE vocab is a load-time artifact — a
  caller can pass token ids through `encodeText`). `+register` under `clip-vit-b-32`.
  `Tools/clip-to-safetensors/convert.py` targets the OpenAI JIT/state-dict (names match). Forward,
  round-trip, and unit-length embedding tested.
- `NFKMLXRVM` (`@objc`) — real video matting (Robust Video Matting): the reference `MattingNetwork` —
  a torchvision **MobileNetV3-Large** encoder (inverted residuals with squeeze-and-excitation,
  hardswish, **BatchNorm epsilon 1e-3**, the last stage dilated), the reference LR-ASPP, and a
  **recurrent decoder whose ConvGRU runs on half of each stage's channels** (one fused gate
  convolution emits reset then update), threading four hidden states across frames. Run through
  `NFKMLXMattingBackend` (single frame) or `NFKMLXRVMNet.forward` (video, state threaded; a
  `downsampleRatio` below one runs the network on a reduced frame and lifts the result through the
  **deep guided filter refiner** — the reference's high-resolution recipe). The foreground head
  predicts a residual added to the source, and the alpha is a clamp, not a sigmoid. `+register` under
  `robust-video-matting`. `remapReferenceKey` maps the positional `backbone.features.N.block.M`
  Sequentials (what position `M` holds depends on each block's expand/SE shape) and every other
  positional Sequential — module keys are semantic because MLX's `update(parameters:)` parses a
  numeric key as an array index (see the MLX-runtime gotchas). **Reference parity** against PeterL1n's
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
  which carries every block form — and asserts the loss falls AND the squeeze-excitation parameters
  move, because a decreasing loss alone could ride on the decoder while the backbone stays frozen.
- `NFKMLXLanguage` / `NFKMLXLanguageBackend` — on-device **text generation** through MLX, which the
  package had no path for: the core runs a Core ML language model and the Foundation Models companion
  wraps Apple's, and nothing here ran a Qwen or Llama. `NFKMLXLanguageNet` is the modern dense decoder
  — grouped-query attention with rotary embeddings, a SwiGLU feed-forward, RMS normalization
  throughout — which is what Qwen3 and Llama both are; they differ in a **configuration**, not in
  structure (`normalizesQueryAndKey` is Qwen3's per-head query/key norm, `attentionBias` is Qwen2's,
  `tiesWordEmbeddings` is the smaller sizes'). **Module keys ARE the released checkpoint's names**
  (`model.layers.N.self_attn.q_proj`), so a release loads with no remapping at all, and every weight is
  at most 2-D so none of the convolution transposes apply. `NFKMLXKeyValueCache` is what makes a token
  cost one step's work instead of the whole sequence's; `testACachedStepMatchesRecomputingThePrefix`
  is the assertion the generation path rests on. Sampling is greedy at temperature 0, otherwise
  temperature with optional nucleus (`topP`) and a seed for repeatability. The backend reads
  `NFKInputPrompt` / `NFKInputMessages` → `NFKOutputText` and honors the core's temperature, top-p,
  max-tokens, and seed parameters. **Reference parity** against transformers' own `Qwen3ForCausalLM` on
  the released Qwen3-0.6B: prefill logit cosine 0.9999999999943 with the same argmax at every
  position, and greedy generation reproducing the reference's continuation **token for token** —
  which is what proves the cache and the rotary offsets, since a single forward pass does not exercise
  them. A tied release still ships `lm_head.weight`; in Qwen3-0.6B it is **byte-identical** to
  `model.embed_tokens.weight` (verified, not assumed), so the loader drops the duplicate.
  **Qwen3.5 and 3.6 are NOT this architecture** — they are `Qwen3_5ForConditionalGeneration`,
  multimodal, interleaving `linear_attention` layers with full attention every fourth layer and gating
  the attention output. `configuration(fromHuggingFace:)` rejects them, and any mixture-of-experts
  config, rather than loading their weights into a dense stack and producing fluent nonsense. The
  oracle needs transformers >= 4.51, which is newer than the vision oracles run under, so it has its
  own interpreter recorded in the manifest's `oracle_environments`.
  **1.7B and 4B are at parity too** (logit cosine 0.9999999999975 and 0.999999999987, each reproducing
  the reference's greedy continuation token for token), which is what shows the family scales by
  configuration. Both are **sharded**: every release above 0.6B splits its weights across files with a
  `model.safetensors.index.json` naming which shard holds each tensor, so a loader reading only
  `model.safetensors` covers the smallest model and nothing else. 4B is the largest size this machine
  holds at float32 — about 16 GB of weights on each side, measured, with the oracle and the test run as
  separate processes.
- `NFKMLXHybridLanguage` — the hybrid decoder Qwen3.5, Qwen3.6, and **Qwen3.8** are built from
  (`Qwen3_5ForConditionalGeneration`), at **reference parity** on the released Qwen3.5-4B (logit
  cosine 0.9999999999962, every one of the 33 hidden states exact layer by layer). 4B is the smallest
  release of the family and the only one that fits here; **Qwen3.8-27B is the same architecture at
  ~54 GB**, so it is covered structurally (851 parameters, 0 mismatched) and its numerics rest on the
  4B measurement rather than on a run of its own.
  The per-layer isolation harness found three defects a shape check could not:
  **the family normalizes with `x · (1 + w)`** where the DENSE Qwen3 stack and Gemma 4 both scale by
  the weight directly — two conventions from the same vendor, indistinguishable by shape, different in
  every number (the gated norm inside the recurrence is the plain kind even here). **The full-attention
  projection interleaves queries and gate PER HEAD**: it is viewed as `[.., heads, 2·headDim]` and
  split on the last axis, so taking two contiguous halves of the flat width takes the wrong channels
  entirely. And **the gate is applied as a plain `sigmoid`**, despite the config field being named
  `output_gate_type: swish` — the implementation is what the weights were trained against. The
  recurrence itself also needed the query scaled by `1/√headDim` and the decay applied BEFORE reading
  the state rather than after.
  Three quarters of its layers replace attention with a **gated delta-rule recurrence** — a fixed-size
  state instead of a growing key-value cache, so cost is linear in sequence length — and every fourth
  layer is full attention whose output is gated. The shapes decode the design: `q_proj` is
  `[12288, 5120]` where 24 heads × 256 would be 6144, because the query projection also emits the
  output gate, which is applied as a plain sigmoid; `in_proj_qkv` is 10240 = two key streams of 16×128 plus a value stream of
  48×128; `A_log` and `dt_bias` are `[48]`, one decay and one step per VALUE head; and
  `partial_rotary_factor` 0.25 turns only 64 of each head's 256 channels. Beside the 4B measurement,
  `NFKMLXHybridLanguageTests` checks **every one of the 27B decoder's 851 parameters** against that
  checkpoint's own safetensors headers, name by name and shape by shape — read with HTTP range
  requests, about a megabyte instead of 54 GB — with zero missing and zero mismatched, which is what
  carries the 4B result across to the size that cannot be run. The converse is asserted too, so the parts deliberately absent are named rather
  than overlooked: 333 tensors are the vision tower and 15 the multi-token-prediction head, and
  851 + 333 + 15 is the checkpoint's full 1199. A small configuration also runs end to end, and the
  recurrence is checked to be causal (appending tokens cannot change an earlier token's output).
  The one layout difference is the depthwise convolution: PyTorch stores `[channels, 1, kernel]` and
  MLX `[channels, kernel, 1]`, which the structural test compares as a loader would.
- `NFKMLXDeepSeek` — the DeepSeek V4 decoder (`DeepseekV4ForCausalLM`): **Multi-head Latent Attention
  over a mixture of experts**, a third architecture family beside the dense stack and the hybrid.
  **Its arithmetic IS measured** — at a tiny all-sliding configuration against transformers' own
  plain-PyTorch implementation, which shipped after this port was written and is the third-party
  oracle DeepSeek's GPU-only inference code could not be (`run_reference.py deepseek_v4`,
  `IK_PARITY_DEEPSEEK_TINY`): every layer's hidden state ≥ 0.9999995 and the logits 0.9999999999,
  with the oracle saving its weights in the RELEASE naming so the module loads them strictly. With
  every layer sliding and the sequence shorter than the window, the reference degenerates to exactly
  the dense-with-sink path this port computes, so the measurement covers the MLA projections, the
  per-head query norm, the trailing interleaved rope, the sink softmax, the output de-rotation, the
  grouped output projection, both routers, the clamped SwiGLU experts, and the hyper-connections. The
  compressor and indexer stay outside it (no compressed window closes at that length). The RELEASED
  weights still cannot run here — the measurement is of the implementation, not the checkpoint.
  **The measurement immediately found a wrong class the structural check could not**: the head's
  collapse (`NFKDeepSeekHyperHead`) predicts only read gates, so `hc_head_fn` is `[copies, copies ×
  hidden]` — and the module had built the FULL block connection there, `[(2 + copies) × copies, …]`.
  The structural check compares DECLARED shapes against the release and the declaration was right, so
  it passed while the module was wrong; MLX's `update(parameters:)` then adopts a checkpoint's shapes
  wholesale, so a real load would have crashed in the forward, not at load.
  The hybrid's checkpoint is bf16, so a float module's shapes match it exactly; this one is
  **quantized**: attention is fp8 with 128×128 block scales and a routed expert is **4-bit packed two
  to an int8 byte**, so `w1` is stored `[2048, 2048]` where the float weight is `[2048, 4096]`. The
  structural check therefore derives what each float parameter looks like stored, and because that
  derivation is an assumption, `testTheQuantizedLayoutIsWhatTheReleaseUses` asserts it against the
  observed dtypes and shapes instead of trusting it.
  Implemented: low-rank queries (`wq_a` → norm → `wq_b`), ONE shared latent key-value per position
  (`wkv` → norm) which is what keeps the cache small, a grouped low-rank output (`wo_a` applied per
  group of heads, then `wo_b`), the learned per-head `attn_sink`, rotary on the head's TRAILING
  channels, and the mixture of experts — square-root-softplus scoring, a bias that steers selection
  without entering the weights, renormalize-then-scale by `routed_scaling_factor`, the clamped SwiGLU,
  and one shared expert every token passes through. **The first `num_hash_layers` route by a
  `tid2eid` table indexed by TOKEN ID rather than by the hidden state**, which changes which parameters
  those layers carry, so the boundary is asserted.
  **The compressor and the sparse indexer are implemented.** `NFKDeepSeekCompressor` pools
  `compressRatio` consecutive positions into one: each contributes a value (`wkv`) and a score
  (`wgate`), the scores softmax ACROSS the window so its positions compete, and `ape` is a learned
  per-slot bias so a position's weight depends on where it sits as well as on what it holds. At ratio
  4 the projections are twice as wide and a second, OVERLAPPING window is pooled alongside — shifted
  back by one window so a boundary is covered from both sides, with the first window's absent
  predecessor filled with zero values and −inf scores. `NFKDeepSeekIndexer` runs its own compressor,
  projects the layer's low-rank query into an index space, scores every compressed position, combines
  the heads by a learned weight, and keeps the best `index_topk` — masking any window not yet complete
  at the querying position, which the reference marks −1. **The reference's Hadamard rotation is
  deliberately omitted**: it is applied to BOTH the query and the compressed keys before fp4
  quantization, and being orthogonal and shared it cancels in the dot product — it spreads information
  for quantization rather than changing the score, so omitting it and the quantization together gives
  the unquantized ranking the reference approximates. Prefill only; the incremental decode path keeps
  rolling state buffers a single forward pass never enters.
  **Hyper-Connections are implemented, and finding them corrected a real mistake here.** The `hc_*`
  parameters are not a hash-clustering head, which is what this file previously called them: the
  residual stream is `hc_mult` (4) PARALLEL COPIES of the hidden state, so every block works on
  `[batch, length, 4, hidden]`. A block predicts its mixing weights per position from the copies
  themselves — `hc_*_fn` projects the flattened, RMS-normalized copies into a read weight per copy, a
  write weight per copy, and a copy-to-copy matrix that is softmaxed and then **Sinkhorn**-normalized
  toward doubly stochastic so the copies do not collapse into one another. The copies are reduced to
  one stream before attention and before the feed-forward, and expanded back after each; `hc_head_*`
  collapses them at the top.
  **The structural check missed it entirely, and that is the lesson.** It compared every parameter the
  port declares against the release and reported zero problems, because it only asked "does what I
  declare exist" and never "does everything in the release exist here". A whole mechanism sat in the
  checkpoint with no counterpart in the code. `testEveryReleasedTensorIsDeclaredOrNamed` now asserts
  the converse — 34223 declared, 38094 named as deliberately unimplemented, **0 unaccounted**, which is
  the checkpoint's full 72317. The Qwen3.8 and Gemma 4 checks always had that assertion; this one did
  not, which is exactly where the gap opened.
  **The quantization the release is stored in is decoded, and this part IS measured.**
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
  Two details are load-bearing and only one of them is measurable. **The 4-bit blocks run along the
  last axis**, which the checkpoint's own values corroborate: the reference quantizer clamps a block to
  ±6 and rounds its scale to the power of two that puts the block's largest magnitude in `(3, 6]`, so
  under the right grouping EVERY block lands in that range and under a wrong one about 1% do not.
  **The nibble order is not measurable** — a byte's pair decodes to the same two values either way and
  both stay inside one block, so no statistic separates them — so it follows the format's own
  convention (low nibble first) and a test pins it against a hand-encoded byte, the same treatment the
  rotary convention gets.
  Still NOT implemented, and named by `testEveryReleasedTensorIsDeclaredOrNamed` rather than merely
  absent: the multi-token-prediction and DSpark speculative-decoding stack, 4705 tensors. It serves
  speculative decoding, which needs a generation loop this port does not have, and no oracle here can
  run it, so it would be unmeasurable code serving an absent path. That test now accounts for the
  release exactly — 34223 declared, 33389 block scales each decoding a declared weight, 4705 named as
  unimplemented, **0 unaccounted**, which is the checkpoint's full 72317. Counting a scale as
  "unimplemented" was the weaker claim it used to make; it is now accounted for by the weight it
  decodes.
  **The arithmetic measurement above post-dates the source audit below.** When this port was written,
  DeepSeek's own `inference/model.py` was the only reference and it imports `sparse_attn` and the
  fp8/fp4 kernels from a GPU-only `tilelang` module — stubbing those would have made the oracle this
  port's own code, which proves nothing. The audit was therefore source-driven, applying the error
  classes the isolation harness exposed in the Qwen and Gemma decoders; the transformers oracle later
  confirmed its three findings numerically. Three were found and fixed by reading `model.py`:
  **the rotary pairs ADJACENT channels** (`view_as_complex`), where the other decoders here rotate
  halves — indistinguishable by shape, different in every value, so `NFKDeepSeekRotary` writes the
  convention out rather than selecting it with a flag; **the attention output is DE-rotated** on the
  way out, because the values share their latent with the keys, so the rotation has to be undone with
  the conjugate; and **the learned per-head `attn_sink` was declared but never used** — it is an extra
  logit that drains probability mass without contributing a value, which the fused attention call has
  nowhere to put, so the softmax is written out. Its normalization is the PLAIN kind, which the port
  already had. Each is pinned by a test, since no measurement can catch them here. Verification enumerates the architecture
  **analytically** — 43 layers of 257 experts cannot be instantiated at float precision — and compares
  3975 parameters across the five layers whose headers were captured, with zero mismatches.
  **DeepSeek V4 Pro (0813) gets the same treatment**, and the enumeration generalized with no code
  change: 61 layers, hidden 7168, 384 experts, 128 heads, `q_lora_rank` 1536, `o_groups` 16 — 3540
  parameters compared across three captured layers with zero mismatches, and its 149782-tensor index
  accounted for exactly (71983 declared, 70790 block scales each decoding a declared weight, 7009
  MTP/DSpark, 0 unaccounted). Pro adds a YaRN `rope_scaling`, which carries no parameters and so is
  invisible to a structural check; the port does not implement it, and a run of Pro would need it. Sources:
  DeepSeek ships `inference/model.py` in the release, which is what this was written from.
- `NFKMLXGemmaLanguage` — the Gemma 4 text decoder (`gemma4_text`), a fourth architecture family, at
  **reference parity** against transformers' own implementation on the released E2B weights (logit
  cosine 0.9999999999994, and every one of the 36 hidden states exact layer by layer). Measuring it
  required installing Python 3.12 beside the system 3.9, because Gemma 4 is in no transformers that
  runs on 3.9; `oracle_environments` records that interpreter.
  **This is the model that proved a structural check is not a numeric one.** Its 600 parameters matched
  the release by name and shape while the forward scored **0.0044**, and four corrections read from the
  reference moved it only to 0.48 — two of them making it worse. What resolved it was the **per-layer
  isolation harness** (`testGemma4LayerByLayerAgainstTheReference`): the oracle records the state
  entering the stack and the state each layer produced, so the first divergence is located rather than
  guessed. It put the fault at layer 0's `input_layernorm` while that layer's input was exact, and a
  sub-step probe inside layer 0 narrowed it further. Three real defects came out of it:
  **Gemma 4 normalizes with a PLAIN scale, `x · w`** — the `x · (1 + w)` convention is **Gemma 3's**,
  and assuming the family inherited it is what broke the port. The feed-forward uses Gemma's own
  activation (`gelu_pytorch_tanh`), not the SwiGLU's silu, which surfaced only once attention was
  exact. And the full-attention layers use the `proportional` rotary: frequencies computed over the
  WHOLE head width with the first `partial_rotary_factor` of the pairs real and the rest **zeroed**,
  which is not the same as rotating a contiguous leading slice — with rotate-half a pair is
  `(i, i + width/2)`, so a quarter-turned 512-wide head turns channels 0…63 and 256…319. That one left
  every sliding layer exact and every full layer subtly wrong, which is exactly what the harness showed
  at layer 4.
  Its distinguishing feature is **per-layer input embeddings**: beside the ordinary token embedding, a
  second much wider one (`embed_tokens_per_layer`, 262144 × 8960 = 35 layers × 256) gives every layer
  its own slice, gated into the residual after the feed-forward. Normalization is Gemma's `x · (1 + w)`
  rather than `x · w`, so a Gemma checkpoint in a plain RMSNorm produces near-zero activations and
  looks like a broken model instead of a convention mismatch. Logits are soft-capped through `tanh`.
  **Two things the config does not say, both read from the checkpoint.** A FULL-attention layer runs
  `global_head_dim` 512 where a sliding one runs `head_dim` 256, so seven of E2B's 35 layers have
  doubled attention widths — using `head_dim` throughout mismatched 42 tensors. And the layers that
  share keys and values run a DOUBLED feed-forward: E2B's first fifteen are 6144 wide and its last
  twenty — exactly `num_kv_shared_layers` — are 12288, which no config field states. Both were found by
  the structural check rather than by reading, which is the argument for running it before trusting a
  port.
  Scope: the text decoder. The release is tri-modal, carrying a vision tower (659 tensors) and an audio
  Conformer (752); a test names them so they are known rather than overlooked. The configuration guard
  is EXACT (`gemma4_text`), not a prefix: `gemma4_unified_text` (the 12B) is a different architecture
  and the 26B-A4B is a 128-expert mixture, and both satisfy a `gemma4` prefix — the dense sizes also
  carry the expert fields NULLED, so the guard reads the number rather than testing for the key.
  **E4B is measured too**, at the precision it ships in: its 16 GB of bf16 weights double past this
  machine's RAM at float32, so both sides run bf16 (`IK_GEMMA_DTYPE=bfloat16` for the oracle,
  `.checkpoint` here) — logit cosine 0.9998 with the same argmax at every position, and the strict
  load itself confirms the doubled feed-forward on its 18 kv-shared layers, since a wrong width fails
  loudly. The measurement surfaced a defect only bf16 could: the attention masks are built float32,
  and the fused attention refuses a mask that does not promote to a bf16 module's own type, so the
  mask now takes the queries' dtype — invisible at float32, which is why no float32 run ever raised
  it. The dense and hybrid decoders had the identical latent crash on any `.checkpoint` load and are
  fixed the same way, each pinned by a bf16-forward test.
- `NFKMLXReleaseWeights` — one reader for a downloaded release's weights, single-file or sharded
  (`model.safetensors.index.json`, each shard read ONCE), with a remap closure whose nil skips a
  tensor. The dense, hybrid, and Gemma loaders all read through it; before it each had its own copy,
  and Gemma's copy had NO sharded path — a capability gap consolidation removed as a side effect.
  The per-family differences stay in the loaders where they belong: the tied-`lm_head` drop, the
  hybrid's `model.language_model.` remap and depthwise-conv transpose, Gemma's tower skip.
- `NFKMLXTorchCheckpoint` / `NFKMLXTorchFormat` — the native PyTorch checkpoint reader: a consumer's
  raw `.pth`/`.pt`/`.ckpt`/`.th`/HF `.bin` loads with NO Python toolchain. `NFKMLXWeights.loadCheckpoint`
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
  `params_ema`, `params`, `model`, `generator`, `state`) BEFORE the root is flattened — a Lightning
  checkpoint keeps optimizer tensors beside its state_dict, so root-first sweeps those in.
  **Whisper's releases store their Linear weights as transposed fp16 VIEWS**, found by the first real
  parity run after the plan assumed state dicts are contiguous: `bytes(for:)` gathers a strided
  tensor to row-major, held to torch's own materialization by comparing the raw `whisper_tiny.pt`
  against its converted safetensors tensor for tensor. The byte oracle throughout is the offline
  converters' own output (raw in `~/.inferkit-validation/raw/`, `IK_RAW_<KEY>` written by fetch.py).
  `NFKMLXTorchCheckpoint` is the public `@objc` face: inspect `tensorNames`/`infoForTensor:`, read a
  tensor's bytes, or convert on device with `writeSafetensorsToURL:` (a hand-rolled pure-Swift
  safetensors writer — no Metal needed — whose output carries no `inferkit.layout` metadata, which IS
  the PyTorch-layout marker; float64 narrows to float32 as the converters do). Refused with errors
  naming the offline converter: TorchScript archives (CLIP), an opaque module tree (YOLO), `.nemo`
  big-endian saves, sparse/quantized storages. **There are NO deferred models — YOLO, VAD, and CLIP
  all load.** Three walks/unwraps, all non-executing (no class constructed, no serialized `code/`
  interpreted): (1) a checkpoint that pickled a live `nn.Module` tree (YOLO's ultralytics
  DetectionModel) is walked through the standard `_parameters`/`_buffers`/`_modules` state — `walkModule`,
  reproducing `nn.Module.state_dict()` exactly (parameters + persistent buffers, recurse `_modules`,
  skip a None param / non-persistent buffer / plain-attribute tensor), matched against the real
  yolov8n's 498-key state dict; (2) a **TorchScript archive** (CLIP) is walked through its
  attribute-keyed scripted-module state — `walkScriptedModule`, where each object's `BUILD` state is
  a flat dict of `name → tensor | submodule | scalar` rather than the eager layout, and the leaf
  tensors are the same `_rebuild_tensor_v2` records. **The earlier "TorchScript needs its `code/` IR"
  claim was WRONG**: the probe showed `data.pkl` carries every attribute name (`visual.conv1.weight`,
  `transformer.resblocks.0.attn.in_proj_weight`), matched against the real ViT-B/32's 302-key
  state dict; (3) a `.nemo` PAX/ustar tar is unwrapped to the checkpoint inside it (`readTar`). The
  scripted walk is scoped to archives carrying `constants.pkl` (the TorchScript marker), so the eager
  path is untouched; its int config attributes (`input_resolution`) are surfaced and ignored by the
  loaders' coverage the way `num_batches_tracked` is.
  **Every converter's rename/transform is ported into its model's Swift loader**, so all non-excluded
  models load a raw checkpoint end to end, each verified by an `NFKMLXTorchParityTests` equivalence
  test: the raw file and the converted file must land IDENTICAL parameters through the model's own
  `loadWeights` (u2net's legacy `rebnconvN` index rename, colorizer's Sequential table + ConvT
  permute, hifigan's weight-norm fusion — held to 1e-6, the one tolerance, because two float32
  evaluations of `g·v/‖v‖` differ in the last ulp — nafnet/raft/rife's renames, lama's `generator.`
  and fastspeech2's `model.` strips, and pose, whose raw and converted files carry IDENTICAL key
  names differing only in deconv axis order, which is why `Checkpoint.isNativeTorch` exists).
  Conv-TasNet and the denoiser needed NO change — their shape-keyed 3-D branches already read the
  raw layout — and that is verified, not assumed. **RAFT found the package's newest MLX hazard**:
  its reference reuses each block's `norm3` inside `downsample`, the rename collides the two names
  deliberately, and duplicate keys crash `ModuleParameters.unflattened` with a stack overflow —
  dedupe through a dictionary first (see the gotchas below and `Docs/mlx-runtime-hazards.md`). The
  nafnet/rife/lama/modnet raw checkpoints are in the validation manifest, which grew two acquisition
  routes to serve them: `gdrive` (a Google Drive id, MODNet) and `extract` (a member path inside a
  zip at `url`, LaMa's Lightning `best.ckpt`); the rest download from `url` as before. Their
  equivalence tests read the `IK_RAW_*` keys `fetch.py` stamps and skip when absent, like every other
  parity test.
- `NFKMLXRetinaFace` (`@objc`) — real face detection with five-point landmarks, and the detector the
  CodeFormer reference pipeline runs through facexlib. The released **mobile0.25** model: a MobileNetV1
  backbone at quarter width (a plain stem then depthwise-separable blocks), a three-level FPN fusing
  top-down with **nearest** resampling, three SSH context modules (a 3×3 branch beside 5×5 and 7×7
  receptive fields built from stacked 3×3s, concatenated then activated together), and per-level class
  / box / landmark heads over **two anchors a cell**. Run through `NFKMLXDetectionBackend`
  (`NFKInputImage` → `NSArray<NFKDetection *>` under `NFKOutputDetections`, boxes normalized 0…1,
  origin top-left) or `detector(weightsURL:)` for the landmarks, which is what alignment needs.
  `+register` under `retinaface-mobile025`. The input is **BGR 0…255 minus `[104, 117, 123]`**, because
  the reference reads its frames through OpenCV. **Feeding RGB is a quiet defect, not a loud one**:
  measured on the validation portrait, the face is still found and the confidence is unchanged to three
  decimals (0.9971 against 0.9975 — the wrong order scores marginally HIGHER), while the box moves to
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
  loaded — they are not parameters of the detector. **Reference parity** against facexlib's own
  RetinaFace over the whole pre-suppression tensor (box cosine 0.9999999999980, class
  0.9999999999999969, landmark 0.9999999999980, and an exact anchor grid), and end to end through
  decoding and suppression against its `detect_faces` (same face count, **box IoU 1.0**, landmarks
  within a pixel). Weights: `github.com/xinntao/facexlib/releases` `detection_mobilenet0.25_Final.pth`,
  1.7 MB — negligible beside CodeFormer's own checkpoint, which is why it is the recommended detector.
- `NFKMLXCodeFormer` (`@objc`) — real face restoration: the reference CodeFormer (sczhou) in `MLXNN` —
  a VQGAN encoder and generator built as the reference's **flat heterogeneous `blocks` list** (residual
  blocks with a 1×1 skip projection where the width changes, single-head spatial attention at
  resolution 16, asymmetric-pad stride-2 downsamples, nearest ×2 upsamples; GroupNorm at epsilon
  1e-6), a codebook under `quantize.embedding`, and a Transformer code-predictor whose **queries and
  keys carry the position embedding while the values do not** (the reference's fused
  `in_proj_weight`/`out_proj` layout is kept). The quantized features **always** take the degraded
  latent's per-channel statistics (`adaptive_instance_normalization`): the reference's signature
  defaults `adain=False`, but its released `inference_codeformer.py` passes `True`, and matching the
  shipping behavior rather than the signature default is a ratified decision. The
  **controllable feature transformation** (`NFKCFFuseBlock`, one per connect resolution 32/64/128/256)
  modulates the generator with a learned scale and shift weighted by the fidelity `w` — 0 is full
  generative quality, 1 keeps the degraded input's detail. Run through `NFKMLXModuleBackend` (aligned
  face → restored face at the model resolution). **`photoBackendWithFidelity:weightsURL:` takes a whole
  PHOTOGRAPH**: it detects every face, aligns each to the reference's five-point 512 template, restores
  it, and composites the result back through the inverse transform with a feathered edge. Detection and
  alignment are `NFKMLXFaceAlignment`, built on **Vision** — no weights, no download, no third-party
  code, which is the rule the core applies to its own backends. It is NOT facexlib's RetinaFace, so a
  crop here is not byte-identical to the reference pipeline's and a restored photograph differs slightly
  from it; what the model does to a crop is unchanged, and that is what the parity record measures. The
  alignment is a **similarity** transform (uniform scale, rotation, translation, no shear), solved in
  closed form as one complex multiply over the centered point sets — a full affine would stretch the
  face onto the template exactly and hand the model a distorted subject. Vision reports each feature as
  a contour rather than a point, so an eye is its centroid, the nose is the lowest point of its contour,
  and the mouth corners are the outer lip's extremes in x. An image with no detectable face passes
  through unchanged. **The detector is selectable and defaults to RetinaFace**, the reference
  pipeline's own, so the crop is the crop facexlib produces; `photoBackendWithFidelity:weightsURL:
  detectorWeightsURL:` takes its 1.7 MB checkpoint. `NFKMLXVisionFaceDetector` is the alternative when
  a download-free path matters more than matching the reference. They disagree measurably — on the validation portrait, box IoU
  0.65 and a worst landmark disagreement of 15.7 px over a 960×1200 frame — so the choice changes the
  restoration, and `NFKMLXFaceAlignmentTests` records that number rather than describing it.
  **A landmark assertion cannot validate the crop** — the transform maps landmarks
  onto the template by construction, so it stays true however the drawing lands. Only the crop's
  CONTENT can: `testTheAlignedCropContainsTheFace` detects a face inside the crop and checks it fills
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
  real arrays and pass through. **Reference parity** against CodeFormer's own architecture on the
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
  inverted residual's `conv.M` (**whose slots shift when the expansion is absent** — the
  expansion-1 block's depthwise pair sits at 0/1, not 3/4), and unwraps every `Conv2dIBNormRelu`'s
  `layers` Sequential. The checkpoint stores the backbone **twice**, once per branch holding a
  reference to it; the copies are identical, so the loader keeps the first. **Reference parity**
  against MODNet's own network on the released photographic checkpoint (alpha cosine
  0.9999999999994, mean |difference| 1.4e-8), every parameter covered on the first triage run.
  Weights: `python3 -m gdown 1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz` (26 MB; the HF ONNX exports remain
  unreadable by this loader).
- `NFKMLXYOLO` (`@objc`) — real object detection: the reference **YOLOv8** (ultralytics) in `MLXNN` —
  a CSPDarknet backbone of `Conv` (convolution + **BatchNorm epsilon 1e-3** + SiLU) and `C2f` stages
  ending in SPPF (three chained 5×5 stride-1 max pools through `NFKMLXResample.maxPooled`), a PAN-FPN
  neck fusing strides 8/16/32 both ways, and a decoupled head with **distribution-focal box
  regression**: each box side is a softmax over 16 bins whose expectation (the `dfl` convolution,
  fixed to 0…15, loaded from the checkpoint) is a distance from the cell's anchor point. **v8 has no
  objectness** — confidence is the best class probability. Box decode and greedy per-class NMS in
  Swift. The suppression thresholds are **deliberately the reference's 0.25 / 0.7**, not the 0.45 this
  module first shipped with — a ratified behavior change, so do not "restore" the older value.
  `NFKMLXYOLOBackend` reads `NFKInputImage` → `NSArray<NFKDetection *>` under `NFKOutputDetections`;
  boxes are normalized 0…1, origin top-left.
  The `+backendWith…labels:` factory attaches class names. `+register` under `yolo`.
  `remapReferenceKey` maps the reference's `model.N` module list onto named stages and the head
  branches' positional Sequential (`cv2.i.0/1/2` → `conv1`/`conv2`/`out`). **Reference parity**
  against ultralytics' own YOLOv8n on the released `yolov8n.pt` over the full pre-suppression tensor
  (box cosine 0.9999999999999638, class cosine 0.9999999999944721, same top class at the same
  anchor), and against ultralytics' `predict` end to end on a 16:9 frame through the public backend
  (9/9 detections, same classes, worst box IoU 0.9999984). A frame is fitted the reference's way:
  scaled by the smaller ratio, padded with gray 114 to a multiple of 32 (`auto` mode, so a wide frame
  runs at 640×384 rather than wasting a third of the input), and the decoded boxes have that padding
  and scale undone before they are normalized against the caller's own frame. Forward, decode, NMS,
  letterbox, remap, and round-trip tested. **YOLOv8s is at parity too** (box cosine
  0.99999999999997, class 0.9999999999997): it has the **same depth** as the nano model at twice the
  width, because the releases scale by two independent multiples — reading only one of them right
  still loads and is still wrong. **YOLOv8m** is the first size where both multiples change — wider
  stages and deeper C2f repeats `[2, 4, 4, 2]` — and it matches too (box 0.99999999999996, class
  0.999999999998). `NFKMLXYOLOVariant` (`.nano`/`.small`/`.medium`/`.large`/`.extraLarge`) selects the size. **l and x
  are at parity too** (box 0.99999999999993 / 0.99999999999994): both run the full depth multiple, so
  their C2f stages repeat `[3, 6, 6, 3]`, and x is wider again. The records must be made at
  `--size 640`, where the reference's letterboxing is an identity — generating one at another size
  produces a different anchor count and looks like a model failure.
- `NFKMLXSegFormer` (`@objc`) — real semantic segmentation: the SegFormer MiT transformer encoder
  (efficient self-attention with spatially reduced keys/values + Mix-FFN depthwise conv, so no
  positional embedding) and an all-MLP decode head in `MLXNN`, run through `NFKMLXModuleBackend`. The
  argmax label map is emitted as a grayscale image under `NFKOutputImage`; recover the class index as
  `round(gray·(classCount−1))`. `+register` under `segformer-b0`. **Reference parity** against
  transformers' own `nvidia/segformer-b0-finetuned-ade-512-512` (logit cosine 0.99999992, label
  agreement 99.99%). `remapReferenceKey` regroups the reference's flat
  `segformer.encoder.block.<stage>.<index>` and its separate `patch_embeddings.N`/`layer_norm.N` lists
  onto per-stage names, and **concatenates the reference's separate `key`/`value` into this port's one
  fused `kv`** — a two-into-one a 1:1 key map cannot express. Forward, label-map, and round-trip tested.
- `NFKMLXSwinIR` (`@objc`) — real transformer super-resolution: SwinIR (shallow-feature conv → residual
  Swin Transformer blocks → pixel-shuffle upsampler) in `MLXNN`, with real window attention — window
  partition/reverse, cyclic shift with the standard attention mask, and a relative-position bias table
  gathered by a precomputed index. Run through `NFKMLXModuleBackend`; the input side must be a multiple
  of the window size. `+register` under `swinir-x4`. **Reference parity** against JingyunLiang's own
  `network_swinir.py` on the released `001_classicalSR_DIV2K_s48w8_SwinIR-M_x4` (cosine 0.99986, mean
  pixel |difference| 0.0037). The input is **RGB-mean centered** — the reference subtracts
  `(0.4488, 0.4371, 0.4040)`, scales by `img_range`, and restores it at the end; leaving that out was
  the fifth missing input normalization in this sweep.
  **Non-power-of-two scaling is implemented and at parity** on the released x3 (cosine 0.99987, mean
  0.0036). The reference `Upsample` reaches a power-of-two scale with repeated ×2 pixel-shuffle stages
  and a scale of three with ONE ×3 stage, because a factor-3 shuffle is not a composition of factor-2
  ones — so an x3 checkpoint packs `9·C` channels into a single stage where x4 packs `4·C` into each of
  two, and fits only its own geometry. `NFKMLXSwinIRVariant` selects the release, and `NFKMLXSwinIR.makeNet` throws
  `unsupportedConfiguration` for a scale the reference builds no upsampler for, rather than silently
  truncating `log2`. **x8 is the same network with a third ×2 stage** (0.99990). **The lightweight
  release is not the classical network at a smaller size**: it reconstructs through the reference's
  `pixelshuffledirect` — ONE convolution to `3·scale²` channels and a single shuffle, with neither the
  convolution before the upsampler nor the one after it — so `convBeforeUpsample` and `convLast` are
  absent rather than unused, and its checkpoint carries a single `upsample.0` (0.99991, mean 0.00097). The shuffle itself is
  the shared `NFKMLXPixelShuffle`, which BiSeNet, RIFE, VideoSR, and SwinIR all use. Forward, window
  helpers, and round-trip tested.
- `NFKMLXColorizer` (`@objc`) — real colorization (Zhang et al. ECCV-16): eight VGG-style conv blocks
  (BatchNorm block ends; blocks 5–6 dilation 2) over the L channel predict a distribution over 313
  quantized ab bins; the annealed mean is the checkpoint's own `model_out` 1×1 conv (renamed
  `out_ab`), so no separate cluster file. `NFKLabColor` implements sRGB ↔ CIELAB (D65) in MLX ops,
  tested against CIE reference values (white L*=100, mid-gray L*=53.39) plus a full-gamut round-trip.
  Predicted ab recombines with the original full-resolution L, preserving luminance exactly. The
  factory sets `train(false)` so BatchNorm uses the checkpoint's running statistics.
  `Tools/colorizer-to-safetensors/convert.py` performs the complete `nn.Sequential` rename and the
  ConvTranspose axis swap (`[in,out,kH,kW]` → `[out,in,kH,kW]`), so the release loads directly —
  no remap is left to Swift. `+register` under `colorizer-eccv16`. **Reference parity** against the
  released eccv16 (ab cosine 0.9999999998, colorized sRGB cosine 0.9999971). The L and ab resampling is
  bilinear, as the reference's `nn.Upsample` is: nearest neighbour scored the ab prediction at 0.96, so
  this was a real defect, not a nicety. `abPrediction` exposes the network's output before the lightness
  goes back, so a parity failure says network or Lab conversion.
  **`NFKMLXSiggraphColorizer` is the second released colorizer** — a separate network, not a
  configuration of this one, and at **reference parity** against richzhang's own `siggraph17.py`
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
  the converter swaps the deconv ConvT axes. **Reference parity** against microsoft's own
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
  **Reference parity** against torchvision (logit cosine 0.9999999999999, label agreement 1.0).
  `remapReferenceKey` maps the reference's positional `classifier.N` Sequential onto the module's names.
  Complements `NFKMLXSegFormer` (CNN vs transformer segmentation).
- `NFKMLXResNetBackbone` (`NFKMLXResNet.swift`) — the shared bottleneck residual backbone (ResNet-50 and
  up) in the reference layout, including the stride-to-dilation substitution DeepLab depends on
  (`replaceStrideWithDilation`; the reference gives a stage's FIRST block the previous stage's dilation).
  `remapReferenceKey` names the projection shortcut the reference keeps in a `Sequential`
  (`downsample.0/1` → `downsample_conv`/`downsample_bn`). Pose's ResNet-50 reuses this.
  Its stem pools through `NFKMLXResample.maxPooled` (see the MLX-runtime gotchas below).
- `NFKMLXConvTasNet` (`@objc`) — real time-domain speech separation: a 1-D convolutional encoder, a
  masking temporal convolutional network (depthwise-separable dilated Conv1d blocks with global layer
  normalization `NFKTasNetGlobalNorm` and PReLU), and a shared transposed-conv decoder in `MLXNN`.
  `NFKMLXConvTasNetBackend` reads `NFKInputAudio` → one `NFKAudioAsset` per speaker ("speaker-1",
  "speaker-2", …). `+register` under `conv-tasnet`. **Reference parity** against `asteroid`'s own
  ConvTasNet on `JorisCos/ConvTasNet_Libri2Mix_sepclean_16k` (per-speaker cosine 0.9999999995).
  `remapReferenceKey` unwraps asteroid's `filterbank` (`_filters`, no bias) and its positional
  `shared_block` Sequential. **Every PReLU carries one slope by default, which is the reference's own
  shape** — asteroid builds them as `nn.PReLU()`, whose `num_parameters` defaults to 1, and all 49
  slope tensors in the released checkpoint are `[1]`. An earlier note called per-channel slopes a
  sweep item as though the release used them; it does not, and making them the default would diverge
  from it. `perChannelPReLU` offers them anyway, for a fine-tune that wants the capacity: a shared
  slope applied to every channel is the same function, so `loadWeights` **widens** a released `[1]`
  slope to `[C]` against the widths the module reports, the model computes exactly what it computed
  before, and training moves the slopes apart from there. Tested by saving a shared-slope model,
  loading it into a per-channel one, and asserting the separation is unchanged. Forward, separation,
  and round-trip tested.
- `NFKMLXDenoiser` (`@objc`) — real speech noise suppression (Défossez et al.): the same Demucs
  time-domain U-Net as `NFKMLXDemucs` configured with `stems == 1`, so it **reuses `NFKMLXDemucsNet` and
  `NFKMLXDemucs.loadWeights`** (DRY). `NFKMLXDenoiserBackend` reads `NFKInputAudio` → one cleaned
  `NFKAudioAsset` under `NFKOutputAudio`. `+register` under `denoiser`. **Reference parity** against
  facebookresearch/denoiser dns48 (cosine 0.99999999999992), which also guards the shared network
  against a change made for the music model breaking the speech one. Single-output and round-trip tested.
- `NFKMLXVAD` (`@objc`) — real voice activity detection (MarbleNet): a mel front end feeding a stack of
  QuartzNet-style blocks — runs of time-channel-separable convolutions with an optional projected
  residual — and a two-class per-frame head; consecutive above-threshold frames merge into spans.
  `NFKMLXVADBackend` reads `NFKInputAudio` → `NSArray<NFKAudioSegment *>` (a new core value type) under
  the new core key `NFKOutputSegments`. `+register` under `vad-marblenet`; factory sets `train(false)`.
  **Reference parity** against NeMo (cosine 0.99999999999983). The front end (`NFKVADFrontEnd`) is the
  reference preprocessor — preemphasis, a centered 512-point transform, power spectrum through the mel
  filterbank, natural log with a `2⁻²⁴` guard, frames padded to a multiple of two — and it **loads its
  window and filterbank from the checkpoint**, which carries both; the defaults reproduce them for a
  randomly initialized net. Held in a plain box, not on the `Module`, so those constants stay out of
  `parameters()`. `remapReferenceKey` maps NeMo's flat positional `mconv` list (five entries per
  separable convolution, four per plain one) and its `res.0` shortcut onto the module's names.
  A clip arriving at another sample rate is **resampled to 16 kHz** through `NFKMLXAudioRate.matched`
  (the parity-proven `julius.resample_frac` port, with the ratio reduced by its greatest common
  divisor first — 44100 → 16000 would otherwise build 16000 polyphase kernels instead of 160). Frame
  times are computed at the model's rate, which is the caller's own seconds because resampling
  preserves duration.
- `NFKMLXAudioTagger` (`@objc`) — real audio tagging (PANNs Cnn14): a log-mel spectrogram, normalized
  across its mel bands (`bn0`), feeds six VGG-style blocks (two 3×3 convolutions and an average pooling
  each), and the result pools over time — max plus mean — into an independent score per class; the top
  scores become tags. `NFKMLXAudioTaggerBackend` reads `NFKInputAudio` → `NSArray<NFKClassification *>`
  (a new core value type, most-confident first) under the new core key `NFKOutputClassifications`; the
  `+backendWith…labels:` factory attaches class names. `+register` under `audio-tagger-panns`.
  **Reference parity** against PANNs' own Cnn14 (mel cosine 0.99999999, embedding 0.99999994, tag
  0.99999988, same top class). The front end is 32 kHz / 1024-point / hop 320 over a 50 Hz–14 kHz
  filterbank, which the **checkpoint ships** (`logmel_extractor.melW`), so it loads rather than being
  recomputed — held in `NFKAudioTaggerFrontEnd`, off the `Module`, or it inflates `parameters()`.
  Its decibel scale is `10·log₁₀` floored at `1e-10`, not the natural log the other front ends use.
  **The last block pools with a window of one**, i.e. not at all; pooling it like the others cost
  embedding cosine 0.9942 and moved the top class. `remapReferenceKey` maps the reference's 1-based
  `conv_blockN` onto the module's array. A clip arriving at another sample rate is **resampled to
  32 kHz** through `NFKMLXAudioRate.matched`: the filterbank is built for one rate, so feeding another
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
  projection shortcut (`downsample.0/1`). **Reference parity** against BiSeNetV1's own network on the
  released Cityscapes checkpoint (logit cosine 0.9999999999989, **label agreement 1.0**), every
  parameter covered on the first triage run. Weights: the `CoinCheung/BiSeNet` GitHub release
  (`model_final_v1_city_new.pth`) — they were never actually unavailable.
- `NFKMLXBiSeNetV2` (`@objc`) — the second BiSeNet, a separate architecture rather than a variant, at
  **reference parity** on the released Cityscapes checkpoint (logit cosine 0.9999999999992, **label
  agreement 1.0**), every parameter covered on the first triage run. It replaces V1's ResNet context
  path with a purpose-built pair: a wide shallow **detail** branch and a narrow deep **semantic**
  branch of Gather-and-Expansion layers (each a residual block at stride one; at stride two it
  downsamples through two depthwise stages and carries a depthwise-then-pointwise shortcut), joined by
  a **bilateral aggregation** layer where each branch gates the other at its own scale. `+register`
  under `bisenet-v2`; emits the same grayscale label map as the other segmenters, aligning sides to a
  multiple of 32 and resizing the logits back.
  Its released checkpoint **predates the repository's current head**: it emits `classes × upFactor²`
  channels and **pixel-shuffles** to full resolution, where master now emits `classes` and
  interpolates. Everything before the heads is unchanged, so the oracle substitutes the older head
  rather than pinning a historical commit — the substitution stays visible. A factor-eight shuffle is
  not three ×2 shuffles (the channel interleaving differs), so `NFKBiSeNetPixelShuffle` implements
  PyTorch's `c·r² + i·r + j` order directly and a test asserts it. The four `aux*` heads supervise
  training only and are neither built nor loaded — they hold the largest tensors in the file.
- `NFKMLXVideoSR` (`@objc`) — real video super-resolution: the complete **BasicVSR** (mmediting
  `BasicVSRNet`, ×4). **SPyNet** estimates flow between neighbors (coarse-to-fine pyramid, six
  five-conv modules; its ImageNet normalization tensors load from the checkpoint), and **two
  propagation branches — backward and forward through time** — warp their hidden features along that
  flow (`flowWarp`, a bilinear `grid_sample(align_corners=True)`: `zeros` padding for propagation,
  `border` inside SPyNet; the flow upsampling between pyramid levels is `align_corners=True` bilinear
  ×2, a separate grid from `resizeBilinear`). Fusion + two `PixelShufflePack` stages + `conv_hr`/
  `conv_last` reconstruct over a bilinear ×4 base. Bidirectionality means a frame draws on frames
  after it, so a clip goes through `NFKMLXVideoSRNet.upscaleSequence` whole; the backend upscales a
  single frame. `+register` under `video-super-resolution`. `remapReferenceKey` strips the
  checkpoint's `generator.` wrapper and maps the branches' positional `main.0`/`main.2` Sequential.
  **Reference parity** against mmediting's own BasicVSRNet on the released REDS4 checkpoint, over a
  three-frame genuinely translated clip (clip cosine 0.9999999999997946, mean |difference| 1.1e-7).
  Forward, bidirectional propagation, flow-warp identity/shift/padding-mode, remap, sequence, and
  round-trip tested.
- `NFKMLXTrainer` — the supervised and zero-reference training loop, for **customizing a shipped model
  on a consumer's own data, in the app**. Two entry points (`batch:loss:` with a target,
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
    correct (SAM's `up1`/`up2` use `transposed(1,2,3,0)`, Whisper handles 3-D Conv1d). **Every model
    reads through `loadCheckpoint`**, and every transpose in a loader is gated on the flag — including
    the branches keyed on a name rather than a rank (LaMa's `up.`, Demucs's and Conv-TasNet's 3-D
    transposed convolutions). An ungated loader double-transposes a fine-tuned file and loads silently
    wrong weights: `NFKMLXCheckpointRoundTripTests` saves and reloads through each model's own loader,
    and removing one gate makes it fail with a transposed shape rather than a bad number.
  - Writes are atomic (scratch file then replace), because a periodic checkpoint overwrites the only
    copy of a run's progress. A non-finite loss throws `NFKMLXError.trainingDiverged` **before** that
    step can reach a checkpoint, so divergence cannot replace good weights with ruined ones.
  - Optimizer state is **not** checkpointed: mlx-swift keeps `stateStorage` internal and `innerState()`
    unkeyed. `SGD` resumes exactly; `Adam` rebuilds its moment estimates.
- `NFKMLXTrainingData` / `NFKMLXBatchSampler` — the app-data side of training. `tensor` / `batch` /
  `matte` / `labels` convert a consumer's `CGImage`s into what the trainer takes, reusing
  `NFKMLXImageBridge` so a training batch and an inference input are built identically. `labels`
  **inverts the label-map convention the segmentation backends emit** (`index / (classCount − 1)`), so
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
  trims to the 30-second window; that is not a detail, it was the single biggest accuracy factor in
  reaching reference parity. **`NFKWhisperAttention.query`/`value`/`out` gained `@ModuleInfo`** so they
  can receive adapters; the wrapper keys equal the property names, so checkpoints are unchanged.
- `NFKMLXLoRA` / `NFKMLXLoRALinear` — low-rank adaptation, for the models with no small head to train
  (CLIP, Whisper: adapting them means reaching into the attention blocks, and doing that fully needs
  optimizer state proportional to the whole model). `NFKMLXLoRALinear` **subclasses `Linear`**, which is
  what `update(modules:)` requires — a replacement must be assignable to the `@ModuleInfo` ivar's type,
  the same idiom `QuantizedLinear` uses. `apply(to:rank:alpha:where:)` does the tree surgery via
  `leafModules().flattened()`, then freezes everything and reopens only `lora_a`/`lora_b`; it returns
  the count, so a predicate that matched nothing is visible rather than silent, and it skips
  already-adapted layers. **`apply` and `merge` throw**: MLX's non-throwing `update(modules:)` wraps a
  `try!`, so selecting a layer stored in a plain property aborts the process. Both call the throwing
  variant and report `NFKMLXError.loRANotApplicable` naming the `@ModuleInfo` requirement. The `B` factor starts at zero, so an adapted model produces exactly what it
  produced before training. `merge(into:)` folds each detour into its base weights (`Linear` computes
  `x·Wᵀ`, so the delta is `(A·B)ᵀ·scale`) and leaves plain layers: **the saved file carries no adapter
  keys**, so there is no adapter format and no second loading path.
- `NFKMLXSegFormerTraining` — the head-only recipe. A consumer rarely wants ADE20K's 150 classes and
  usually wants their own few, which is a decode-head problem: `NFKMLXSegFormerTrainable.decodeHead`
  (the default) freezes the four encoder stages, so the run's memory falls to the head's share.
  `NFKMLXSegFormer.network(weightsURL:classCount:)` **drops the checkpoint's `classifier.*` when the
  class count differs** and loads the rest with `strict: false` — not optional, because MLX's
  `update(parameters:)` adopts a checkpoint's shapes wholesale rather than validating them, so keeping
  it would silently restore the old class set. `NFKMLXSegFormerObjective` upsamples the stage-1 logits
  to the label resolution before cross-entropy, as the reference does, rather than downsampling the
  labels. The ImageNet input normalization moved into `NFKMLXSegFormerNet.normalized`, shared by
  `segment` and the objective: a fine-tune that normalized differently would optimize for a
  distribution the model never sees at inference, which is the bug four models here have already
  shipped. **Reference parity**: `NFKMLXSegFormerObjective.loss(logits:labels:)` is separable from the
  forward pass so the oracle can score identical logits, and matches transformers'
  `SegformerForSemanticSegmentation` loss (`run_reference.py segformer_loss`,
  `IK_PARITY_SEGFORMER_LOSS`).
- `NFKMLXZeroDCETraining` — the first customization recipe, and the template for the rest.
  Zero-DCE is **zero-reference**: the reference trains it with no ground truth, so a consumer
  customizes it from their own dark photos with nothing to annotate, which is the only kind of data an
  end user has. `NFKMLXZeroDCELoss` ships the four losses (exposure, color constancy, illumination
  smoothness, spatial consistency); `NFKMLXZeroDCEObjective` weights them, and its `wellExposedLevel`
  is the consumer-facing knob (preferred brightness). `NFKMLXZeroDCE.network(weightsURL:)` builds the
  trainable net; `fineTune` runs it. **Reference parity**: all four losses agree with the reference to
  float precision (`run_reference.py zero_dce_losses`, `IK_PARITY_ZERO_DCE_LOSSES`). A wrong loss is
  invisible in a fine-tune's output, so the oracle is the only check that catches it — the first
  implementation, written from the paper rather than the code, was wrong in all four. Three reference
  expressions that read like slips are reproduced deliberately and marked where they occur.
- `NFKStableDiffusionProvider` (`@objc`) — the bridge that lets the **core** activate the bundled
  `NFKMLXBackend` (Stable Diffusion) without depending on InferKitMLX. It conforms to the core's
  `NFKDynamicBackendProvider` and is named exactly the default the core tries for its `stable-diffusion`
  capability, so linking InferKitMLX makes `NFKDynamicBackend.stableDiffusionBackend()` return a working
  SD backend, built lazily. It returns **SD 1.5**: the ungated release, so the capability activates with
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
  `NFKMLXNAFNetVariant` (sidd/goPro/reds), `NFKMLXYOLOVariant` (nano/small/medium),
  `NFKMLXU2NetVariant` (full/light); single-config models omit it. Each `register()` delegates to the
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
  (`real-esrgan-x4` + `-anime`, `depth-anything-v2-small`/`-base`/`-large`, `lama-inpaint`, `sd-inpaint`,
  `fast-style-transfer`, `clip-vit-b-32`, `robust-video-matting`, `codeformer`, `zero-dce`, `modnet`, `yolo`,
  `segformer-b0`, `swinir-x4`, `colorizer-eccv16`, `pose-simplebaseline`, `deeplabv3`, `conv-tasnet`, `denoiser`,
  `vad-marblenet`, `audio-tagger-panns`, `bisenet`, `video-super-resolution`, `htdemucs`)
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
  the deprecated `MLX.GPU.*` memory members) to stay warning-free.
  `NFKMLXDevice` selects the compute device — `currentType` and
  `performOnDeviceType:block:`, over the **scoped** `withDefaultDevice(_:_:)` rather than the deprecated
  global `setDefault(device:)`. **The selection is task-local, so it does not cross a dispatch**:
  measured, a block dispatched asynchronously inside the scope reports the global default, and so does a
  fresh `Thread`, while a synchronous call on the calling thread inherits it. That makes it the wrapper
  for `runInferenceForRequest:` and NOT for `submitInferenceJobForRequest:`, whose background queue takes
  the global device; a caller wanting a whole inference on the CPU runs the synchronous call inside the
  block on their own thread, which is where the contract puts a multi-second inference anyway. The `@objc`
  case names are given explicitly (`NFKMLXDeviceTypeCPU`/`…GPU`), since Swift would generate `…Cpu`/`…Gpu`.
  **Principle**: expose Swift-only API to ObjC where opportunistic and reasonable — a global utility or a
  simple-typed method gets an `@objc` wrapper. What stays Swift-only is what *cannot* bridge: the
  bring-your-own-closure backends (their init takes an `MLXArray` closure), `MLXArray` itself, the schedulers,
  and the `*Configuration` structs (the `@objc` variant enums are the ObjC configuration knob by design).
- HF vs MLX: `NFKHFHub` is a download/cache layer, not a runtime. Every model here downloads through
  it, the bundled Stable Diffusion releases included (`NFKMLXBackend.cacheDirectoryURL` chooses where).
  A gated repository needs a credential: `NFKHFHub.accessToken` sends it as a bearer token and falls
  back to the `HF_TOKEN` environment variable, which is where the tooling around the hub keeps one.
- Weights are **downloaded at runtime, not bundled at build time** (size / redistribution-licensing /
  update reasons). `NFKHFHub.downloadRepo:` resolves `<endpoint>/<repo>/resolve/<revision>/<path>`,
  fetches with `NSURLSession`, and caches at `<cacheDirectoryURL>/<repo>/<revision>/<path>`. The download
  **blocks** (a semaphore wait), so the caller runs it off the main/render thread — or uses the async
  `downloadRepo:…completionHandler:` (Swift `try await`). `cacheDirectoryURL` is host-supplied
  (security-scoped for sandboxed apps); `+defaultCacheDirectoryURL` gives a ready `Application
  Support/InferKit/models` location, and the `NFKMLXDownload`/`NFKMLXHub` factories substitute it when a
  caller passes `nil`. The core hub stays strict (explicit `nil` cache → fails, asserted by a test).
- MLX runtime: MLX needs `default.metallib` **whether or not anything runs on the GPU** — the first
  stream request initializes the scheduler, which constructs the Metal device, so even
  `mlx_default_cpu_stream_new` throws without it (measured: a probe pinned to `Device(cpu, 0)` aborts
  with the library absent and runs convolutions and a full backend inference with it present). It is
  built and bundled by the **Xcode build system**
  (`mlx-swift_Cmlx.bundle/Contents/Resources/`) but a plain CLI `swift build`/`swift test` does not —
  so MLX array evaluation aborts under `swift test`. Run MLX-eval tests via
  `xcodebuild test -scheme InferKitMLX -destination 'platform=macOS' -skipPackagePluginValidation`
  (the `-skip…` flag gets past the unrelated `CudaBuild` plugin's validation). The matting round-trip
  tests auto-detect this: they skip when the test bundle is under `.build` and run under xcodebuild.
- **Never pass `padding:` to an mlx-swift pooling layer.** `Pool.callAsFunction` (mlx-swift 0.31.6)
  builds its pad widths as `[0, 0] + padding + [0, 0]`, two entries too many: a four-axis input gets the
  first four, so a 2-D pool pads width and channels instead of height and width. It raises nothing —
  the output shape is silently wrong, and the failure surfaces later as a channel mismatch reported
  against an innocent layer. Every `Pool` subclass (1-D/2-D/3-D, max and average) shares that
  initializer. Pool through `NFKMLXResample.maxPooled` / `.averagePooled`, which border the input
  explicitly and then window at padding zero. Pooling with no padding is unaffected.
- **A test process that loads many models back to back must clear MLX's cache between them.** The
  GPU cache survives from test to test, and the accumulation starves the largest float32 forward
  (Gemma E2B, ~20 GB) into a Metal command-buffer TIMEOUT — a process kill that truncates the run
  with "0 failures" reported. Measured: the same test passes in 25 s alone and dies mid-suite.
  `NFKMLXReferenceParityTests` clears in `tearDown` (`NFKMLXGPU.clearCache()`).
- **Never ASSIGN to a `@ParameterInfo` or `@ModuleInfo` property.** `attention.sink = newValue` aborts
  the process with "please call update() on the array rather than setting it" — not a thrown error, a
  fatal one, which in a test run kills the process and silently truncates the reported test count (a
  suite of 10 reported 6 and still said "0 failures"). Mutate through
  `update(parameters: ModuleParameters.unflattened([...]))` instead. Same family as the numeric-key
  trap below.
- **Never hand `ModuleParameters.unflattened` two entries with the same key.** It recurses to a
  stack overflow — a SIGSEGV process kill, not an error. A Python dict deduplicates the same
  collision silently, so a remap ported from a converter carries the hazard invisibly; a rename that
  deliberately collides two aliases of one shared tensor (RAFT's `norm3`/`downsample.1`) must build
  into a `[String: MLXArray]` first.
- **Never give a `@ModuleInfo` a numeric key** (`@ModuleInfo(key: "0")` to mirror a reference
  `nn.Sequential` position). MLX's `update(parameters:)` parses a numeric key as an **array index**, so
  the unflattened checkpoint arrives as a list where the module tree has a child module, and the update
  aborts the process (`try!` over `UpdateError.incompatibleItems` inside MLXNN). Use semantic keys
  (`conv`, `bn`) and translate the reference's positions in the model's `remapReferenceKey` — RVM's
  covers every form. A genuine `[Module]` array property is fine; that is what numeric keys are for.
- Padding modes: MLX offers `.constant` and `.edge` only. Reflection padding is
  `NFKMLXResample.reflectPadded` — a border approximation is not cosmetic in a network that normalizes
  over whole feature maps (it cost style transfer a 0.049 mean pixel error).
- Download-and-build coverage (`NFKMLXDownloadTests`): the hub/factory happy path is tested **hermetically
  with no network** — a real safetensors is written to the exact cache location the hub resolves, so
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
**Every package carries its examples in BOTH languages**, so a documented snippet cannot rot in
either: core `Examples/` (ObjC) + `SwiftExamples/` (Swift, which also pins what the ObjC importer
renames — `runInference(for:)`, throwing instead of `NSError **`); `InferKitMLX/Examples` (Swift) +
`InferKitMLX/ObjCExamples`; `InferKitFoundationModels/Examples` (Swift) +
`InferKitFoundationModels/ObjCExamples`.
`InferKitMLX/Examples/MLXModelGalleryExamples.swift` is the live example of **every** shipped MLX model:
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

## Safeguards (Anti-Patterns)

Required without exception:

- **NEVER** run `git clone/mv/restore/rm/add/branch/commit/merge/rebase/reset/pull/push/fetch`
  without developer approval first.
- **NEVER** run `rm` on any path without developer approval first.
- **NEVER** erase or overwrite files for the task of unit testing — the changes being tested must be
  preserved.
- **NEVER** delete a file or folder until its associated task is completely finished.
