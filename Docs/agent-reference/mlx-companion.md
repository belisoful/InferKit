<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# InferKitMLX (companion package)

Package overview, the bring-your-own backend seams, and the Objective-C bridging layer.

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

## Backends, factories, and the Objective-C surface

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
- `NFKMLXSpeechBackend` (`@objc`) — a bring-your-own MLX text-to-speech backend: supply a
  `@Sendable (String) -> MLXArray` closure returning a mono waveform in `-1...1`; the backend reads the
  prompt (`NFKInputPrompt` or `NFKInputMessages`), writes a 16-bit PCM WAV via `NFKMLXWaveFile`
  (Foundation-only, unit-tested), and returns an `NFKAudioAsset` under `NFKOutputAudio`.
  `NFKMLXReferenceModels.registerToneSpeech` is the shipped reference (`tone-speech`), so ObjC builds
  the text→audio path by name. This is the first backend for the audio modality.
- **Declared keys (2026-09-18).** Every backend class here implements the core protocol's optional
  `supportedParameterKeys` / `supportedInputKeys`, so a caller reads what an engine acts on: the
  Foundation Models provider bridge derives Apple's capabilities from them, and a router picks the
  engine a request needs. A backend that reads one input and no parameters declares exactly that,
  which is a different answer from declaring nothing. Rules that keep the declarations honest:
  - Declare only what the code reads. `NFKMLXLanguageBackend` declares the core sampling keys,
    `NFKParameterJSONSchema`, `NFKParameterOutputFormat`, `NFKParameterChoices`, and every
    `NFKMLXGenerationParameterKey`; it does not declare `NFKParameterTopK` or `NFKParameterTools`,
    which it does not read.
  - A closure backend cannot know what its closure reads, so the caller names it:
    `NFKMLXModuleBackend` and `NFKMLXMattingBackend` take `forwardInputKeys` / `forwardParameterKeys`
    on the request-aware initializer, and `NFKMLXDiffusionBackend` takes `encodedInputKeys` /
    `encodedParameterKeys`. Each unions them with the keys the class itself reads. AdaIN passes
    `NFKInputControl` and `NFKParameterStrength`, SAM passes `NFKSAMPointKey`, and
    `NFKMLXTextToImage` passes the prompt pair and `NFKParameterWidth` / `NFKParameterHeight`. Those
    two sets are `NFKMLXTextToImage.encodedInputKeys` / `.encodedParameterKeys`, and the lazy
    `NFKMLXBackend` reads the same two: asking the backend it builds would download the release.
  - `NFKMLXTensorBackend` derives its inputs from the configured ports.
  - A new model that reads a request key beyond its class's own adds it at the construction site, or
    the declaration lies. `NFKMLXDeclaredKeysTests` covers the fixed sets, the unions, and the ports.
- **Reasoning and usage (2026-09-19).** The core's three reasoning keys reach the MLX text backends.
  `NFKMLXReasoning.swift` holds the pieces; `NFKMLXReasoningTests` covers them without the runtime,
  since all of it is text work.
  - **The markers are a value, not a constant, because the shipped families disagree.** Qwen3 writes
    `<think>` … `</think>` and continues with the answer. gpt-oss writes the harmony channels:
    `<|channel|>analysis<|message|>` … `<|end|>` and then `<|start|>assistant<|channel|>final<|message|>`,
    so the format carries an `answerPrefix` as well, or the scaffolding would land in the answer.
    `NFKMLXReasoningFormat` is `@objc` with both as presets, and a request names one under
    `NFKMLXGenerationParameterKey.reasoningFormat` by preset name or by the markers themselves.
  - The format is **detected from the release's own chat template**, which is what renders a prior
    chain back into the prompt and so states the markers. Either marker is enough. A run with no
    Jinja template splits nothing, because there is no way to know what the model wraps a chain in;
    it does not guess from the text.
  - **The reasoning effort can be honored, through the template rather than the backend.** The two
    families spell the control differently and neither is a number: Qwen3's template tests
    `enable_thinking` and pre-closes the block with `<think>\n\n</think>` when it is false, while
    gpt-oss's takes a `reasoning_effort` string and writes `Reasoning: <level>` into its system
    message. So the backend binds BOTH variables (`enable_thinking` = the level is not light;
    `reasoning_effort` = light/moderate/deep renamed to low/medium/high, another string as written)
    and lets the release's own Jinja decide. A template that reads neither renders unchanged, which
    is what an unused binding means in Jinja. `NFKMLXChatTemplateRenderer.render` gained a
    `variables:` parameter for this. A request that names a level with no Jinja template is refused,
    and the Gemma backends refuse it outright, since neither has anywhere to put it.
  - Usage: the runtime knows the prompt, the reply, and what a retained prompt cache served, so all
    three are always reported. `NFKMLXPromptCache.sharedPrefixLength` records what `align(to:)`
    matched, which is the only way to recover the cached share after generation. The chain's share
    is reported only where a format applied, because without one the backend cannot say whether the
    text holds a chain; it is found by **bisecting** the decoded prefix rather than re-encoding the
    chain, since a detokenizer's output grows with its input.
  - `NFKMLXLanguage.chatTemplate(inDirectory:)` reads the release's template (`chat_template.jinja`
    beside the weights, else `tokenizer_config.json`, including the 5.x named-template list), shared
    with the Gemma 3 backend, which had the only copy. The release factory still defaults
    `chatTemplate` to `.none`, so a caller sets it; without it the chain is not split and the effort
    is refused.
  - Measured live on the released Qwen3-0.6B: "What is 2 + 2?" costs 19 input and 156 output tokens
    with 147 in the chain, and the answer comes back as "2 + 2 = 4." alone; at
    `NFKReasoningEffortLight` the same question costs 8 output tokens with no chain. A second
    request sharing a prefix reported 5 of its 6 input tokens as cached. gpt-oss is covered by the
    marker and detection tests only, not by a run.
  - Correcting a doc claim found on the way: `NFKMLXLanguageBackend` has no
    `submitInferenceJobForRequest:`, so its jobs go through the core's wrapper and report no
    per-token partials. Its class comment said otherwise; it now says what the code does. The Gemma
    backends do stream.
- `NFKStableDiffusionProvider` (`@objc`) — the bridge that lets the core activate the bundled
  `NFKMLXBackend` (Stable Diffusion) without depending on InferKitMLX. It conforms to the core's
  `NFKDynamicBackendProvider` and is named exactly the default the core tries for its `stable-diffusion`
  capability, so linking InferKitMLX makes `NFKDynamicBackend.stableDiffusionBackend()` return a working
  SD backend, built lazily. It returns SD 1.5: the ungated release, so the capability activates with
  no credential. See "Dynamic backend discovery" in `core-runtime-notes.md`.
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
  `NFKMLXRVMVariant` (mobileNetV3/resNet50), `NFKMLXRTDetrVariant` (r50vd/r18vd/r34vd/r101vd, and the four v2 releases),
  `NFKMLXRFDetrVariant` (base/nano/small/medium/large), `NFKMLXCLIPVariant` (vitB32/vitB16/vitL14/vitL14At336),
  `NFKMLXSigLIP2Variant` (every release), `NFKMLXSNACVariant` (speech24kHz/music32kHz/music44kHz),
  `NFKMLXHTDemucsVariant` (fourStem/sixStem); single-config models omit it. Every size a family's
  authors released is a case on its enum — that was audited against the release lists in September
  2026 and closed (`NFKMLXReleasedSizesTests`), so a missing size is a defect, not a backlog item. Each `register()` delegates to the
  local factory (DRY), so registry/`registerAll()`/`NFKMLXHub` behavior is unchanged. The `*Configuration`
  structs stay Swift-only; the enum is the ObjC knob. The shared `NFKMLXDownload` helper wraps both the
  blocking (`weightsURL`) and async (`backend(…build:completionHandler:)`) download (used by `NFKMLXHub`
  and these factories). A model that builds from a whole release directory downloads through
  `NFKMLXReleaseDownload.directory(repo:revision:cacheDirectoryURL:required:optional:weights:)` (in
  `NFKMLXMarian.swift`): required files fail the download when absent, optional ones are fetched when
  served, and the first weights entry served is fetched with every shard its index names, relative to
  the index's own folder. It returns the snapshot root whatever folders the files sit in, and a
  weights entry that fails reports each candidate's error, so a refused credential reads as one. A
  release laid out in component folders (FLUX, FLUX.2, Music 3) goes through the `components:` overload
  in `NFKMLXReleaseComponents.swift`. Each class keeps its lists as `requiredFiles` / `optionalFiles`
  / `weightFiles`, and the tests prove them complete offline: `NFKMLXReleaseStore.offline` (test
  support) links only the listed files from the validation store into a cache and runs the factory
  through `NFKMLXReleaseDownload.using(NFKOfflineHub)`, which refuses the network, so a list that
  misses a file the model reads fails its build. Releases whose weights exceed about 8 GB are checked
  download-only (configuration and tokenizer loaded, every shard present). One test reaches the network:
  `testKokoroDownloadsItsReleaseLive` (opt-in, `INFERKIT_LIVE_RELEASE_DOWNLOADS=1`, run through the LMC
  lock) fetches `hexgrad/Kokoro-82M` into a fresh cache and speaks; measured 2026-09-23, 18.5 s for the
  first build (about 330 MB) and 0.1 s for the cached rebuild, with only the named voice fetched. Not every directory is a
  release: none is left without a download today, and Kokoro's became one when the native reader
  learned a bare-tensor root (`torch.save(tensor)`, its voicepacks). All
- `NFKMLXModelRegistry` (`@objc`) — lets an Objective-C consumer build and run an MLX model without
  writing Swift: a model author registers a factory by name from Swift (capturing the `MLXArray`
  forward), and ObjC calls `+backendNamed:weightsURL:error:` to get an `id<NFKInferenceBackend>`. The
  bring-your-own-closure backends (`NFKMLXModuleBackend`/`MattingBackend`/`TensorBackend`/`SpeechBackend`)
  stay registry/Swift-only (their init takes a Swift closure); the shipped real models use the direct
  factories above.
  `NFKMLXReferenceModels.registerGreenScreenKeyer` is the shipped reference; a learned keyer
  (CorridorKey's GreenFormer) registers the same way. `InferKitMLXObjCExamples` proves the ObjC path.
  `NFKMLXReferenceModels.registerAll` registers every shipped model at once — the real models
  (`real-esrgan-x4` + `-anime` + `real-esrgan-general-x4v3` + `real-esrgan-anime-video-x4v3`, `depth-anything-v2-small`/`-base`/`-large`, `depth-anything-3-small`/`-base`/`-large`, `lama-inpaint`, `sd-inpaint`,
  `fast-style-transfer`, `adain`, `clip-vit-b-32`/`-b-16`/`-l-14`/`-l-14-336`, `siglip2-base-patch16-224` and every other SigLIP 2 release under its own name, `taesd`, `robust-video-matting` + `-resnet50`, `codeformer`, `zero-dce`, `zero-dce-plus`, `modnet`, `yolo`, every YOLOv9/v10/11/12/26 release under its checkpoint stem (`yolo11n` … `yolo26x`),
  `segformer-b0`, `swinir-x4`, `hat-x4`/`hat-l-x4`/`real-hat-gan-x4`, `colorizer-eccv16`, `ddcolor`/`ddcolor-paper`/`-artistic`, `pose-simplebaseline`, `vitpose-base-simple`/`vitpose-base`, `deeplabv3`, `conv-tasnet`, `denoiser`,
  `vad-marblenet`, `silero-vad`, `dac`, `snac`/`snac-32khz`/`snac-44khz`, `bigvgan-v2-24khz`, `mimi`, `audio-tagger-panns`, `bisenet`, `video-super-resolution`, `cosmos-tokenizer-ci8x8`/`-ci16x16`/`-di8x8`/`-di16x16`/`-cv4x8x8`/`-cv8x8x8`/`-cv8x16x16`/`-dv4x8x8`/`-dv8x8x8`/`-dv8x16x16`, `htdemucs`/`htdemucs-6s`, `rtdetr`/`rtdetr-r18vd`/`-r34vd`/`-r101vd`, `rtdetr-v2-r18vd`/`-r34vd`/`-r50vd`/`-r101vd`, `rf-detr`/`rf-detr-nano`/`-small`/`-medium`/`-large`, `birefnet`, `sam`, `sam2`, `u2net`/`u2netp`, `isnet`, `mpsenet`, `gtcrn`, `sgmse`, `storm`, `mossformer2-se`, `deepfilternet3`, `voicerestore`, `resemble-enhance`, `metricgan-plus`, `cmgan`, `frcrn`, `mossformer2-sr`, `nuwave2`, `apollo`, `basic-pitch`, `allin1`, `muscriptor`/`muscriptor-small`/`muscriptor-large`, `hft-transformer`, `opus-mt`, `m2m100`, `small100`, `madlad400-3b-mt`, `translategemma`)
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
- HF vs MLX: `NFKHFHub` is a download/cache layer, not a runtime. Every model here downloads through
  it, the bundled Stable Diffusion releases included (`NFKMLXBackend.cacheDirectoryURL` chooses where).
  A gated repository needs a credential: `NFKHFHub.accessToken` sends it as a bearer token and falls
  back to the `HF_TOKEN` environment variable, which is where the tooling around the hub keeps one.
  The companion creates a hub per download, so the cache size limit and backup exclusion reach it
  only through the class defaults (`NFKHFHub.defaultCacheSizeLimit`,
  `defaultExcludesCacheFromBackup`); see `core-runtime-notes.md`, "Hugging Face hub cache policy".
- Weights are downloaded at runtime, not bundled at build time (size / redistribution-licensing /
  update reasons). `NFKHFHub.downloadRepo:` resolves `<endpoint>/<repo>/resolve/<revision>/<path>`,
  fetches with `NSURLSession`, and caches at `<cacheDirectoryURL>/<repo>/<revision>/<path>`. The download
  **blocks** (a semaphore wait), so the caller runs it off the main/render thread — or uses the async
  `downloadRepo:…completionHandler:` (Swift `try await`). `cacheDirectoryURL` is host-supplied
  (security-scoped for sandboxed apps); `+defaultCacheDirectoryURL` gives a ready `Application
  Support/InferKit/models` location, and the `NFKMLXDownload`/`NFKMLXHub` factories substitute it when a
  caller passes `nil`. The core hub stays strict (explicit `nil` cache → fails, asserted by a test).

## Residency: staging and paging

`NFKMLXResidency` (`NFKMLXResidency.swift`) is how a model holds its weights against the working set,
and one plan, `NFKMLXResidencyBudget.plan(_:residency:budget:)`, chooses between two mechanisms from
each stage's `NFKMLXStageFootprint` (its bytes, and how many of them are routed experts).

- Staging changes WHEN whole stages are loaded. A model built from stages that run strictly in turn
  loads each for its turn and releases it after. FLUX.1 (CLIP-L and T5-XXL, then transformer and
  autoencoder), FLUX.2 (text encoder, then transformer), Qwen-Image 2.1 (Qwen3-VL 8B, then
  transformer and autoencoder), LTX-Video (T5-XXL, then transformer and autoencoder), Wan (umT5-XXL,
  then transformer and autoencoder) and MiniMax Music 3 (language model and depth decoder, then
  condition encoder and DiT, then vocoder) stage. The stage bytes come from shard headers at the load
  precision (`NFKMLXStageWeights`), not file sizes.
- Paging changes WHERE part of one stage lives. A mixture's routed experts stay in the release and are
  materialized as the router reaches them (`NFKMLXExpertStore.swift`, `NFKMLXExpertPaging.swift`).
  The language family (Qwen2-MoE, Qwen3-MoE, Mixtral, gpt-oss bf16 and MXFP4, a quantized save), the
  Gemma 4 26B-A4B mixture, the Qwen3-VL 30B-A3B decoder (fused `gate_up_proj` split per expert),
  Qwen4-Exp, Granite 4.0-H's mixtures and DeepSeek page.

The cases: `.resident` holds everything and throws where it is known not to fit; `.staged` stages and
never pages; `.paged` stages and pages; `.automatic` holds everything where the total fits (`holds`),
pages where the largest stage is known not to fit whole and has experts (`!admits`), and stages
otherwise. Staging is tried first: it costs a load per run, paging a read per step. A model without
experts reads `.paged` as `.staged`; a single-stage backend (a language model) has nothing to stage and
loads resident under `.staged`.

Staging:

- Each stage is an `NFKMLXStage`, loaded on use; `NFKMLXStagedModel` serializes runs on one lock,
  releases a stage after its use where the model is staged, and clears MLX's buffer cache before the
  next loads.
- A stage's result is EVALUATED before the stage is released. An unevaluated result is a graph over
  the stage's weights and keeps them alive, so releasing the stage frees nothing. The `with` forms
  evaluate arrays for the caller; `use` leaves it to a caller whose result is not an array.
- A model that can be built before its weights exist plans at each run (Music 3); one that reads its
  release at construction plans once (FLUX.1, FLUX.2).

The budget:

- `NFKMLXResidencyBudget` plans against 0.85 of Metal's recommended working set with a 4 GiB reserve,
  never live free memory, which a resident model's own weights would count against.
- A machine that reports no budget gets two answers. It is not held resident, and no larger precision
  is chosen for it (`holds`), because staging always runs and holding everything may not: Music 3's
  "an unreadable budget stages rather than gambling". A single stage is not refused there (`admits`),
  as `NFKMLXReleaseWeights.verifyFits` does not refuse a load it cannot judge. It is never paged.
- A paged load's cache gets half of what the largest stage leaves once its unpaged weights and the
  reserve are counted, capped at that stage's experts; the other half is left to the page cache that
  holds recently read release bytes. The split is unmeasured policy.

Paging:

- The store is an `@objc` class: Objective-C reaches it through `NFKMLXLanguageBackend.expertStore`
  and `NFKMLXGemmaBackend.expertStore` for the budget, the counters and `clearCache`. Filing sources
  takes `MLXArray`s and stays internal.
- The store is family-agnostic. Experts are filed by group (the module path of the expert-holding
  layer, or `layers.N` for DeepSeek) and index, each with named parts (`weight`, `scales`, `biases`, or
  DeepSeek's `w1`/`w2`/`w3`). A source is an `NFKMLXExpertSource`: a held array, a byte range of an
  `NFKMLXMappedFile`, or DeepSeek's stored fp8/fp4 matrix with its decode. The cache is LRU by bytes.
- The loader half, `NFKMLXExpertInventory`, weighs a release from its shard headers alone and
  classifies each tensor into `NFKMLXExpertSlice`s. A family supplies the classifier and an install step
  that swaps its expert layers for paged ones; `NFKMLXExpertInventory.load` runs the rest. A stacked
  `[experts, …]` tensor pages one contiguous byte range per expert.
- Two seams take a paged layer. The stacked `gatherMM` families put an `NFKLMPagedSwitchLinear` (no
  parameters) where an `NFKLMSwitchLinear` was, both `NFKLMExpertLinear`; Gemma 4 and Qwen4-Exp build
  their experts module in its paged form. The paged call stacks the routed experts into a bank,
  renumbers the indices into it, runs the same gathered multiply, and evaluates the result, so a lazy
  forward pass never holds every layer's bank at once.
- A paged model computes the resident model's logits EXACTLY (`NFKMLXExpertPagingTests`, every
  layout at a tiny geometry, and the released gpt-oss-20b under `IK_VAL_GPT_OSS`: max abs difference
  0.0, greedy continuation identical, 407 s for both loads under the LMC lock). Two traps found making that true: gpt-oss's `[experts, in, out]` stack multiplies as stored
  (the resident path transposes twice), and a paged bank rebuilt `[out, in]` selects another kernel and
  accumulates in another order, so such a slice is marked `inputMajor`. And MLX refuses to set a
  `@ParameterInfo` to nil after init ("please call update() on the array"), so a paged module is built
  paged and swapped in through `update(modules:)`.
- A request's missed experts are acquired as one batch: each part's mapped range gets a page-aligned
  `madvise(MADV_WILLNEED)` first, is copied once (`NFKMLXMappedFile.array(at:count:shape:dtype:)`,
  `bytesNoCopy` into `MLXArray`), and the batch is evaluated in a single `eval`. Measured by
  `testPagedGatherThroughput` (`IK_BENCH_EXPERT_PAGING=1`, 64 bf16 experts, two back-to-back repeats
  each, 2026-09-24): uncached 424–429 ms/call (0.95 GB/s) before, 19.6–20.2 ms/call (≈20 GB/s) after;
  cached 8.6–9.3 ms/call both ways. The per-expert materialize-then-eval loop was the cost.
- A paged layer holds no parameters, so the load's coverage check cannot see a missing expert;
  `NFKMLXExpertInventory.verifyComplete` is that check.
- A paged load is inference-only: the paged layers evaluate inside the forward pass and hold nothing
  trainable.
- The key-value cache is not paged. Paged attention would be a separate feature over
  `NFKMLXKeyValueCache`.
- DeepSeek plans through the shared planner as one stage whose pageable bytes are what
  `NFKMLXDeepSeekPaging.fullyMapped` leaves in the release (routed experts and n-gram tables):
  `NFKMLXDeepSeek.paging(for:residency:budget:includesDraftStack:)` turns the plan into a preset.
  `.resident`/`.staged` → `none`; `.paged` → `fullyMapped`; `.automatic` → `none` where the decoded
  weights fit, `mapped` (experts held stored in memory, decoded without a read) where that fits beside
  the plan's cache, `fullyMapped` otherwise. `backend(directoryURL:)` and
  `deepSeekBackendWithDirectoryURL:error:` are `.automatic`; the explicit `paging:` presets and
  `deepSeekPagedBackendWithDirectoryURL:…` stay for a caller choosing groups itself, and an
  `NFKMLXDeepSeekLoadOptions.paging` preset overrides its `residency`.

Precision choice:

- A stage may declare a wider precision it prefers (`NFKMLXStageFootprint.widenedBytes`), and the plan
  reports the stages that take it (`widenedStages` / `widens(_:)`). Residency is decided first, at each
  stage's own precision; then a resident placement widens stages in order while the widened total still
  `holds`, and a staged or paged one widens a stage that `holds` alone. No budget, no widening.
- FLUX.2's text encoder (float32 over its stored bfloat16), Wan's umT5 and SD3's T5-XXL (float32 over
  bfloat16) declare one. FLUX.2's `plan(encoderStoredBytes:…)` is now a wrapper over the shared plan, and
  its planning test pins the placements it had.
- `verifyEachStageLoads(_:plan:budget:names:)` refuses a staged placement with a stage known not to load
  on its own, at the precision the plan chose. It is separate from `plan` because Music 3 plans from one
  total footprint, where "the total does not fit" means stage, not refuse. FLUX.2 calls it.

## Swift and Objective-C gotchas

Swift↔ObjC gotchas seen here: an `MLXArray` is not `Sendable`, so a backend hands its networks to the
generation task through an `@unchecked Sendable` holder rather than capturing them across an isolation
boundary; the `@unchecked Sendable` extensions on the ObjC value types carry `@retroactive`.

An array of a Core Foundation type in an `@objc` signature compiles in Swift and produces an invalid
header. `references: [CGImage]` builds cleanly; the generated `InferKitMLX-Swift.h` then declares
`NSArray<CGImageRef>`, which is not Objective-C, because `CGImageRef` is not an object type, and the
module's precompile fails. The Swift build is not the check for an `@objc` change; the generated
header is. A method that takes several images declares them `[Any]`, which bridges to `NSArray *`,
and reads each through `NFKMLXImageBridge.tensor(from:channels:colorSpace:)`, which already accepts a
`CGImage` or an `MTLTexture` as `Any`. An Objective-C caller passes `@[(__bridge id)image]`. A single
`CGImage` parameter is fine, since a lone `CGImageRef` is valid Objective-C
(`NFKMLXFlux2.cgImage(forPrompt:negativePrompt:references:width:height:seed:)`).
