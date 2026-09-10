<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# Project structure

The full repository tree with per-directory notes.

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
├── Docs/agent-reference/            # Auxiliary maintainer notes behind AGENTS.md / CLAUDE.md, one file
│                                    #   per subject; README.md there is the index
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
│                                    #   See "Packaging InferKitMLX as an XCFramework" in `distribution-and-packaging.md`.
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
