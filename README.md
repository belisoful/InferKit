# InferKit

A small, cross-platform (macOS / iOS / tvOS) inference toolkit for Objective-C, usable from Swift.

One protocol — `NFKInferenceBackend` — covers every engine, so the same request and result types drive
a Core ML model, an OpenAI-compatible or Anthropic endpoint (hosted, or a local runner such as Ollama),
Apple's on-device Foundation Models, or an MLX model.
There is no FxPlug or host-framework dependency, so any Metal or Apple app can use it.

```objc
id<NFKInferenceBackend> backend = [NFKRemoteBackend backendWithEndpointURL:endpoint];
NFKInferenceRequest *request =
    [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Explain diffraction in one sentence." }];
NSString *answer = [backend runInferenceForRequest:request error:&error].text;
```

Swapping engines means swapping the first line. Inference is synchronous and multi-second, so run it
off the main thread — or submit it as a job for progress and cancellation:

```objc
NFKInferenceJob *job = NFKInferenceSubmit(backend, request, NULL);
job.completionHandler = ^(NFKInferenceJob *done) { NSLog(@"%@", done.result.text); };
```

## Install

```swift
.package(url: "https://github.com/belisoful/InferKit.git", from: "0.2.0")
```

or `pod 'InferKit'`. The optional MLX and Foundation Models companions are separate packages in this
repository — see **[Installation](Docs/installation.md)** for those and for the CocoaPods details.

## What's in it

- **`NFKInferenceBackend`** — the swappable-engine protocol; `NFKInferenceRequest` and
  `NFKInferenceResult` carry named inputs, parameters, and outputs; `NFKInferenceJob` runs one
  asynchronously with progress and cancellation.
- **Backends**, on Apple frameworks alone: a passthrough mock, in-process Core ML for images and
  tensors, an on-device Core ML language-model runner, OpenAI-compatible and Anthropic chat clients
  that stream, cancel, call tools, return structured output, and take images, audio, documents, and
  video beside the prompt, remote embeddings / speech / image generation / transcription / video
  generation / reranking / moderation clients, and a submit-poll-fetch base for job-style services.
- **Remote providers** — `NFKRemoteProvider` names thirteen services (OpenAI, Anthropic, xAI, Gemini,
  Groq, Mistral, DeepSeek, Together, OpenRouter, and the local runners Ollama, LM Studio, llama.cpp,
  and vLLM), lists each one's models from the server rather than a constant, and reaches a local
  runner's native API — what is installed and loaded, and Ollama's pull and delete.
- **Subsystems** — RGBA ↔ planar tensor conversion, an `MLMultiArray` bridge, image and video coding
  (`NFKImageCoding`, `NFKVideoSampling`), a tokenizer (BPE / CLIP / WordPiece / Unigram), a Core ML
  compute-plan reader, a hardware profile, and a Hugging Face download and cache layer.
- **Runtime discovery** — `NFKDynamicBackend` activates an optional engine only when it is linked,
  resolving it by name, so the core never references it.

A consumer brings a heavier runtime (MLX, a C or Rust engine) by adopting `NFKInferenceBackend`. Two
companion packages already do: **InferKitMLX** (60-plus models across image, video, audio, and
language — text-to-image with Stable Diffusion, Z-Image, and SANA, text-to-video with LTX-Video and Wan,
on-device language models (Qwen3, Qwen3.5, Gemma 4, DeepSeek V4, and any dense GGUF), text embeddings
and reranking, vision-language, speech recognition and synthesis, neural audio codecs, and MiniMax
Music 3 text-to-music — each validated numerically against its reference implementation) and
**InferKitFoundationModels** (Apple's on-device model, with tool calling and structured output).

## Models (InferKitMLX)

Every row links to its class, configuration, and factory in the [model index](Docs/model-index.md);
the measured parity of each is in [model parity](Docs/model-parity.md).

- **[Image → image](Docs/model-index.md#image-to-image)** — Real-ESRGAN, SwinIR, NAFNet, Zero-DCE, fast style transfer, colorizers (ECCV-16, SIGGRAPH-17), LaMa, CodeFormer, TAESD, SD inpainting
- **[Image → map](Docs/model-index.md#image-to-map)** — Depth Anything V2, Marigold, SegFormer, DeepLabV3, BiSeNet V1 / V2
- **[Matting & faces](Docs/model-index.md#matting-segmentation-and-faces)** — U²-Net, Robust Video Matting, MODNet, SAM, SAM 2, RetinaFace, face alignment
- **[Detection & pose](Docs/model-index.md#detection-and-pose)** — YOLOv8 (n–x), RT-DETR, SimpleBaseline pose
- **[Embeddings & vision-language](Docs/model-index.md#embeddings-reranking-and-vision-language)** — CLIP, SigLIP 2, Qwen3-Embedding, EmbeddingGemma, ModernBERT reranker, SmolVLM2, Qwen3-VL, Gemma 4 vision / audio
- **[Language models](Docs/model-index.md#language-models)** — Qwen3 (dense, MoE), Mixtral, GGUF, Qwen3.5 / 3.8, Gemma 4 (E2B, E4B, 26B-A4B, 12B), Gemma 2, DeepSeek V4, T5 / umT5, chat templates, constrained decoding
- **[Video](Docs/model-index.md#video)** — RIFE HDv3 / v4, RAFT, BasicVSR, SD ×4 upscaler, the clip backend
- **[Audio](Docs/model-index.md#audio)** — Whisper (tiny–large-v3), Parakeet-TDT, Demucs v2 / HT Demucs, speech denoiser, Conv-TasNet, MarbleNet and Silero VAD, PANNs tagger, DAC, SNAC
- **[Speech & music](Docs/model-index.md#text-to-speech-and-music)** — FastSpeech2 + HiFi-GAN voice, Kokoro-82M, Chatterbox voice cloning, phonemizers, MiniMax Music 3
- **[Text → image & video](Docs/model-index.md#text-to-image-and-video)** — Stable Diffusion 1.5 / 2.1 / SDXL-Turbo, IP-Adapter, Z-Image, SANA, LTX-Video, Wan

## Documentation

| | |
| --- | --- |
| **[Inference guide](Docs/inference-guide.md)** | Choosing a backend, running a model locally, using Apple's Foundation Models. Start here. |
| **[Examples](Docs/examples.md)** | Complete examples across every modality, backend, and subsystem — compiled by CI in both Swift and Objective-C. |
| **[Installation](Docs/installation.md)** | Swift Package Manager, CocoaPods, and adding a companion package. |
| **[Core ML language models](Docs/coreml-llm.md)** | Converting a Hugging Face checkpoint and running it on device. |
| **[Companion packages](Docs/companions.md)** | InferKitMLX and InferKitFoundationModels: what each ships, and the full model gallery. |
| **[Model index](Docs/model-index.md)** | Every implemented model: its entry class, network class, the configuration preset or variant for the released weights, registered name, and base backend. |
| **[Model parity](Docs/model-parity.md)** | Every implemented model, the reference it is measured against, and the number from the recorded run; the shared subsystems and which models depend on each. |
| **[Runtime hazards](Docs/mlx-runtime-hazards.md)** | Where MLX, Metal, and Core ML return a wrong answer quietly. Each entry carries an executable probe. |
| **[Changelog](CHANGELOG.md)** | What each release contains. |
| **[Validation](Tools/validation-assets/manifest.json)** | Every model's reference-parity evidence, rebuildable with `Tools/validation-assets/fetch.py`. |

API reference is DocC:

```bash
Tools/docc/build.sh          # the core        (--preview to serve it, --all for the companions)
```

**Do not use Xcode's Product ▸ Build Documentation for the core.** Neither `xcodebuild docbuild` nor
the swift-docc-plugin extracts a symbol graph from a pure Objective-C SwiftPM target, so that path
produces an archive with no symbols in it and reports every ``NFKFoo`` link as
`'NFKFoo' doesn't exist` — 94 symbol pages become 0. `Tools/docc/build.sh` exists for exactly this
reason: it runs `clang -extract-api` over the public headers and hands the result to `docc`. The two
Swift companions have no such problem and build either way.

## Build & test

```bash
Tools/build-all.sh --test                       # all three packages
swift build && swift test                       # just the core
pod lib lint InferKit.podspec --quick           # the CocoaPods spec
```

Building a package produces object files and a module for SwiftPM to link, not a standalone binary.
For a binary to drop into a third-party app, build an XCFramework:

```bash
Tools/xcframework/build.sh                      # -> .xcframework-build/InferKit.xcframework
```

That yields a universal static XCFramework with three slices — macOS (arm64 + x86_64), iOS device,
and iOS simulator — carrying the 44 public headers.

The MLX companion packages too, through its own script:

```bash
Tools/xcframework/build-mlx.sh --verify
```

That emits `InferKitMLX.xcframework` (static) and `InferKitMLXDynamic.xcframework` (dynamic, with the
Metal library inside), arm64, three slices each. Each static slice carries the Metal library a consumer
of that slice ships.
`--verify` links a consumer against each and runs a model on the GPU, because a binary that links can
still fail to find its Metal library. `--variant`, `--slices`, and `--no-swift-interfaces` trim the
build and the output. A consumer's binary grows by about 14 MB plus a 3.6 MB Metal library, of which
MLX is roughly 97% — InferKitMLX and the core together are a few hundred kilobytes of it.

[Docs/installation.md](Docs/installation.md) carries a linking recipe for each case — Xcode app,
framework, plug-in bundle, SwiftPM `binaryTarget`, your own static library — and the release-asset
matrix. The artifacts are not committed; they compress to about 49 MB across three release assets.

The MLX companion's tests need the Metal library only Xcode's build system bundles, so run those
through `xcodebuild test -destination 'platform=macOS' -skipPackagePluginValidation` with each of its
three shared schemes in turn — `InferKitMLXTests`, `InferKitMLXExamples`, and `InferKitMLXObjCExamples`
(the library scheme `InferKitMLX` runs only the first of them). The parity suites read real checkpoints
from `~/.inferkit-validation`, fetched by `Tools/validation-assets/fetch.py`, and skip where a
checkpoint is absent.
`InferKit.xcworkspace` opens all three packages in one window, each still built as its own package.
Its schemes cover every target; the core's suite splits across `InferKitTests` (297),
`InferKitExamples` (20), and `InferKitSwiftExamples` (20) — the same 337 `swift test` runs.

## Consumers

Built for [MetalForge](https://github.com/belisoful/MetalForge), an FCPX/Motion effects suite, but the
toolkit has no host dependency.

## License

MIT. See [LICENSE](LICENSE).
