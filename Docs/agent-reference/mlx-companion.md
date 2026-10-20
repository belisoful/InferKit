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
- **Model catalog (2026-10-15).** `NFKMLXModelDescriptors` (`NFKMLXCatalog.swift`) is the
  `NFKModelDescriptorProvider` that `NFKModelCatalog.sharedCatalog` looks up by name, so linking the
  package lists its models.
  - An entry is an internal `NFKMLXCatalogEntry` whose defaults are the package's: engine `mlx`, macOS 14
    and iOS 17 on Apple silicon, resident weights, introduced in 0.4.0. Its `descriptor()` reads the
    entry through `descriptorWithDictionary:error:`, the reader a plugin's manifest goes through. The
    internal helpers `NFKModelPort.input` / `.output`, `NFKModelOption.option`, `NFKModelLicense.license`
    / `.apache2` / `.mit` / `.inferKit`, `NFKModelSource.hub` / `.noDownload`, and
    `NFKModelVariant.variant` keep an entry short.
  - Entries live per modality. `NFKMLXCatalogReference.swift` holds the six stand-ins: `standIn`, source
    kind `none`, InferKit's MIT license. `NFKMLXCatalogImage`, `…Video`, `…Language`, and `…Audio` each
    hold `entries` and `registerAdditionalModels()`, which `NFKMLXReferenceModels.registerAll` calls. A
    model with a factory and no registry name registers there, under its entry's `registryName`, with
    no edit to `NFKMLXReferenceModels.swift`.
  - Every backend class declares `supportedOutputKeys`: each result key it can write, conditional ones
    included. The language backends share `NFKMLXToolCalling.outputKeys` (text, tool calls, assistant
    turn). A set the configuration fixes follows it: the matting backend adds `NFKOutputMask` when it
    emits the matte, the tensor backend lists its output ports, and the separators list their stems.
    `NFKMLXGenerationBackend.init` takes a required `outputKeys:`, so each closure backend names what
    its closure writes.
  - `NFKMLXCatalogTests` checks that the shared catalog finds the provider and hides the stand-ins, that
    every descriptor round-trips through its dictionary, that identifiers are unique, that every
    registry name is registered after `registerAll()`, and that each stand-in backend declares exactly
    its descriptor's ports and options.
  - The measurements, the snapshot, and the completeness rules are in "Model catalog" below.
  - The language releases list their `model_type` (`modelTypes`), and
    `NFKMLXModelDescriptors.descriptor(for:matching:)` (`NFKMLXRepositoryMatching.swift`) answers the core
    resolver for them: it judges another repository against the loader and measures it the way
    `Tools/catalog/measure.py` does. The judgment's refusals and the measurement are in
    `model-resolver.md`.
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
  `vad-marblenet`, `silero-vad`, `dac`, `snac`/`snac-32khz`/`snac-44khz`, `bigvgan-v2-24khz`, `mimi`, `audio-tagger-panns`, `bisenet`, `video-super-resolution`, `cosmos-tokenizer-ci8x8`/`-ci16x16`/`-di8x8`/`-di16x16`/`-cv4x8x8`/`-cv8x8x8`/`-cv8x16x16`/`-dv4x8x8`/`-dv8x8x8`/`-dv8x16x16`, `htdemucs`/`htdemucs-6s`, `rtdetr`/`rtdetr-r18vd`/`-r34vd`/`-r101vd`, `rtdetr-v2-r18vd`/`-r34vd`/`-r50vd`/`-r101vd`, `rf-detr`/`rf-detr-nano`/`-small`/`-medium`/`-large`, `rf-detr-seg`/`rf-detr-seg-small`/`-preview`/`-medium`/`-large`/`-xlarge`/`-xxlarge`, `birefnet`, `sam`, `sam2`, `u2net`/`u2netp`, `isnet`, `mpsenet`, `gtcrn`, `sgmse`, `storm`, `mossformer2-se`, `deepfilternet3`, `voicerestore`, `resemble-enhance`, `metricgan-plus`, `cmgan`, `frcrn`, `mossformer2-sr`, `nuwave2`, `apollo`, `basic-pitch`, `allin1`, `muscriptor`/`muscriptor-small`/`muscriptor-large`, `hft-transformer`, `opus-mt`, `m2m100`, `small100`, `nllb-200`, `madlad400-3b-mt`, `translategemma`, `real-esrgan-x2`, `nafnet`, `colorizer-siggraph17`, `marigold-depth`, `sd-x4-upscaler`, `retinaface-mobile025`, `bisenet-v2`, `rife`, `rife-v4`, `raft`, `whisper-tiny`, `demucs`, `minimax-music3`, `stable-audio-open-1.0`, `chronos-bolt-base`, the clip backends `rife-clip`, `video-super-resolution-clip`, `robust-video-matting-clip`/`-resnet50-clip`, `sam2-video`, and the release-directory models `parakeet-tdt`, `canary-1b-v2`, `voxtral-mini-3b`, `granite-speech-3.3-2b`, `phi-4-multimodal`, `codestral-mamba`, `granite-4.0-h`, `nemotron-nano-2`, `qwen4-exp`, `qwen3.5`, `ltx-video-0.9.0`, `wan`, `z-image`, `qwen-image-2.1`, `sd3`, `flux.1-schnell`, `flux.2-klein-4b`, `sana`, `flux.1-controlnet`, `sd3-controlnet`, `smolvlm2-500m`, `qwen3-vl-2b`, `pixtral-12b`, `gemma4-multimodal`, `gte-reranker-modernbert-base`, `qwen3-vl-reranker-2b`, `timesfm-2.5-200m`, `sam3`, `fastspeech2-voice`, `clip-probe`, which refuse to build without a directory URL)
  and the reference stand-ins (`green-screen-keyer`, `tone-speech`, and the `diffusion-*` oracle
  pipelines, which are distinct from the real models of the same task). Depth `register` uses the
  `NFKMLXDepthConfiguration.small`/`.base`/`.large` presets; Real-ESRGAN `register` varies `blocks`
  (23 vs 6).
- `NFKMLXHub` (`@objc`) — combines the core's `NFKHFHub` download with the registry:
  `+backendNamed:repo:weightsPath:revision:cacheDirectoryURL:error:` downloads a model's weights from
  Hugging Face and builds the registered backend around them (fail-fast if the name is unregistered).
  `NFKHFHub` stays MLX-free in the core; this companion helper is where HF-download and MLX meet.
  `+backendForModel:variant:cacheDirectoryURL:progressHandler:completionHandler:` is the catalog-level
  call: it downloads a descriptor variant's source with progress and builds its registered backend, and
  `+hubWithCacheDirectoryURL:` returns the hub the factories use, for status and removal. Progress and
  cancellation reach every factory through the core's current `NFKDownload`; see "Release download
  layer" in `mlx-weights-and-formats.md`.
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
  `NFKMLXGPU` adopts the core's `NFKServingRuntimeStatus` for `NFKInferenceServer`'s status route. Its
  report checks `metalLibraryURL` first, because a memory reading without the library aborts the
  process, and it never reads `cacheLimit`: mlx-swift 0.32's getter calls `mlx_set_cache_limit` twice
  on the first read in a process (to learn the value), which trims the cache. `memoryLimit`,
  `activeMemory`, `cacheMemory`, and `peakMemory` are plain reads.
  It also implements the protocol's optional `releaseCachedMemory` (2026-10-13), which the server calls
  after it unloads a model while none of its models runs or builds: `synchronize()` then
  `clearCache()`, skipped without the Metal library. Without it an unloaded model's buffers stay in
  MLX's cache.
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
- `NFKMLXHalfPrecision` / `NFKMLXArithmetic` (`@objc`): the arithmetic of a flash attention's exponentials
  (`softmaxExponential`), the shared softmax (`softmaxProbabilities`: the eager, math, upcast, and sink
  attentions and the routers), a router's kept-weight sum (`routingSum`), a half-precision forward's layer
  norms (`NFKReferenceRounding.layerNorm`, reached by `NFKLayerNorm`, `sd3AffineFreeLayerNorm`, and the float32
  norms of Wan, Wan Animate, and SAM 3), and the float32 exponentials a reference forms with `torch.exp`
  (`NFKReferenceRounding.exponential`). `.fast` is the default. `.reference` runs the Metal kernels in
  `NFKMLXReferenceKernels.swift` (`torchExp`, `torchSoftmax`, `torchHalfSoftmax`, `torchSoftplus`,
  `torchWindowPool`, `torchPackedSoftmax`, and `torchLayerNorm`), each wrapped in a
  `CustomFunction` whose VJP is the gradient of the MLX operations it stands for. A kernel cannot dispatch on
  the CPU stream, so there the exponential takes MLX operations (`torchExpOperations`) and the softmax and
  layer norm take MLX's own. `NFKMLXBFloat16ParityTests` sets `.reference` in `setUp` and `.fast` in
  `tearDown`.
  - `torchExp` is Sleef's `expf` (`Sleef_expf4_u10`), which torch 2.8 computes for every vectorized float32
    exponential on ARM, `exp_u20` included, with `fma` where Sleef fuses. 0 of 1,048,576 results differ from
    torch's. `torchExpOperations` rounds each multiply-add and leaves 1.1% one unit apart.
  - `torchSoftmax` is torch's float32 `softmax`: the row maximum, Sleef's exponential of each
    `x − maximum`, their sum over 4 lanes (each lane's elements in order, a partial last vector into the
    first lanes, the lanes joined as `(a₀ + a₂) + (a₁ + a₃)`, a row shorter than 4 in order), and each
    exponential times `1 / sum`. A row gets a threadgroup of 32 threads, 4 of them carrying the lanes.
  - `torchRowSum` is torch's float32 `sum` over a row of up to 19 values: 4-lane vectors, the values past
    them summed in order and added into the first lane, then the lanes in order. MLX's sums in order and
    differs on 44% of 8-wide rows.
  - `torchLayerNorm` is torch's `RowwiseMoments` and `LayerNormKernelImpl`. A half-precision input is read in
    8-wide vectors, each widened to two 4-lane float32 vectors, and a float32 input in 4-wide vectors. The
    moments take Welford updates over chunks of 16 vectors, a binary cascade, the remainder one element at a
    time, and the lanes' join. A half-precision row's output is `(x · rstd + (−rstd · mean)) · γ + β`, and a
    float32 row's `((x − mean) · rstd) · γ + β`. torch's vector code rounds every operation, and its compiled
    scalar code fuses the lanes' join, each remaining element's second moment, and the output of a
    half-precision row narrower than 8. A row gets a threadgroup that computes up to 32 chunks in parallel and
    replays torch's cascade; a wider row's lanes compute their chunks in order. 0 of 13.7 million
    half-precision outputs differ from torch's, and the float32 cases of
    `testReferenceArithmeticComputesTorchsLayerNorm` (rows 7 to 5,120 wide, with and without the affine) match
    torch's SHA-256. The float32 kernel takes 4.4 to 4.9 times `MLXFast`'s time over Wan's 7,800 tokens at
    widths 1,536 and 5,120. A probe over torch's own `native_layer_norm` mean and rstd separated the two
    differences from the half-precision kernel: the moments match at 4-wide vectors exactly, and only
    `((x − mean) · rstd) · γ + β` of eleven output forms matches torch's output.
  - `NFKReferenceRounding.exponential` takes `torchExp` at the float32 exponentials of a diffusion model's
    timestep frequencies (SD 1.5, 2.1, and SDXL, SD3, FLUX, FLUX.2, SANA, Wan, Wan Animate, LTX-Video, LTX-2,
    Z-Image, Qwen-Image), a state-space layer's `exp(A_log)` (the Mamba-2 layers of Codestral, Nemotron-H, and
    Granite-4.0-H, and the gated-delta layers of Qwen3.5 and Qwen4-Exp), and Wan Animate's masked attention.
    DeepSeek V4's sink attention takes torch's softmax over the scores with the sink appended as the last
    column (`deepSeekSinkWeights`), and its hyper-connections and Gemma 3n's and Gemma 4's audio attention take
    `softmaxProbabilities`. Each torch counterpart is a vectorized float32 `torch.exp` or `softmax` whose
    length is a whole number of 8-wide vectors, where torch computes every element with Sleef.
  - GELU and SiLU at a half precision take `torchGELU` and `torchSiLU` in either mode: torch's vectorized
    `erf` (the Abramowitz–Stegun polynomial with `fma` and Sleef's exponential) and `x / (1 + exp(−x))`. On the
    GPU each matches torch over every float16 and bfloat16 input the GPU keeps normal and runs faster than the
    MLX operations it replaced, which the CPU stream takes: GELU over 2 × 4,096 × 5,120 values in 1.1 ms against
    30.4 ms, SiLU over 2 × 128 × 128 × 640 values in 0.9 ms against 2.3 ms.
  - Softplus at a half precision takes `torchSoftplus` in either mode: `x` above 20, otherwise
    `log1p(exp(x))` with Sleef's `expf` and `log1pf`, formed in float32 and rounded once. The `log1pf` port
    matches torch's `log1p` below 10³⁸. The kernel matches torch over every float16 and bfloat16 input whose
    result is a normal float32 value and runs in 1.4 ms against 2.9 ms for
    MLX's float32 `softplus` over 2 × 4,096 × 5,120 bfloat16 values. A float32 softplus takes it in
    `.reference`: a Mamba-2 step, the gated-delta decay of Qwen3.5 and Qwen4-Exp, DeepSeek V4's router, and
    TimesFM's query scale. Gemma 3n's and Gemma 4's audio query scales take `NFKReferenceRounding.softplus`.
    Over bfloat16 MLX's `softplus` already rounds every normal result as torch does; over float16 it rounds
    471 of 63,488 inputs apart.
  - torch softmaxes half-precision scores without widening them at Whisper's and Voxtral's attention, the
    T5 encoders' under transformers 5 (4.57 widens them), SigLIP 2's pooling head, Granite Speech's
    mid-encoder CTC skip and Q-Former, Phi-4-MM's speech Conformer, Sa2VA's InternViT, and gpt-oss's sink
    attention and router. Its reduced-type kernel widens the row, sums the exponentials of whole 4-value
    vectors over 4 lanes, joins the lanes, adds the remaining exponentials in order with the C library's
    `expf`, and multiplies by the sum's reciprocal. `halfSoftmax(_:fast:)` takes `torchHalfSoftmax` in
    `.reference` and the site's own softmax otherwise; `attention(…halfSoftmax:)` marks the T5 and InternViT
    calls. Sleef's `expf` stands in for the remaining exponentials and leaves 0 to 3 outputs in a million
    apart; MLX's float32 softmax rounded once leaves 1 in 1,000 to 7,000. Apple's `expf` is faithfully
    rounded and leaves 0.6% of float32 results one unit from the correctly rounded value, so no kernel
    reproduces it. The kernel takes 4.55 times MLX's softmax over 8 × 4,096 rows of
    4,096 bfloat16 values.
  - DeepSeek V4's compressor pools each window by `(values · softmax(scores, dim)).sum(dim)` along a
    non-trailing axis, which torch vectorizes over the trailing axis: per column the exponentials summed in
    order, a true division, each product, and torch's cascade sum (blocks of up to 16 rows summed in order,
    the block sums joined in order). `torchWindowPool` computes it in one kernel in either mode, in
    1.0 ms against 2.0 ms for the MLX operations over 1,024 windows of 8 at
    width 1,024.
  - Wan Animate's generation attention in `.reference` follows the reference's eager flex attention: float32
    scores, torch's float32 softmax over each query's row of the packed buffer (the generation span padded to
    a multiple of 128, then one slot of `videoArea` positions a frame), summed over 4 lanes by buffer
    position (`torchPackedSoftmax`), and the probabilities rounded to the operands' type before the product
    with the values. The fast mode adds the row's zero keys to the denominator as a count.
  - The GPU flushes a subnormal float32 result to zero, where torch and bfloat16 keep it: a bfloat16 softmax
    whose scores spread past about 87, and a softplus below −87.
  - Overall, a whole half-precision forward in `.reference` takes 0.97 to 1.30 times the time of the fast mode
    in a diffusion denoiser (SDXL's UNet, at 1.30, under the standing limits) and up to 2.27 times in an
    encoder (`NFKMLXHalfPrecisionArithmeticTimingTests`, `IK_ARITHMETIC_TIMING=1`; the table is in
    `Docs/examples.md`).
  Measured, and kept out of the switch:
  - Sigmoid and swish, over every float16 and bf16 input: Sleef's exponential and MLX's leave the same
    count of rounded outputs apart from torch's.
  - Sleef's `expf` from MLX operations on the GPU: 20 to 25 times MLX's `exp`, and a whole denoiser forward
    2.3 to 4.7 times slower. `compile` of it is 3.2 times faster and 98% of its results are wrong.
  - A Mamba-2 or gated-delta layer's prefill: torch exponentiates its chunked scan's segment sums, an
    operation the port forms another way.
  A half-precision flash attention without a `queryBlock` attends `automaticQueryBlock` rows at a time when
  its float32 scores would pass 1 GiB, each row computed as before: Wan at 480×832 and 33 frames held 9.5 GB
  of scores, and at 81 frames would hold 51 GB.
- `NFKMLXModelDescription` (internal) is what a backend's `modelInfo` reads: it walks
  `parameters().flattened()` and matches each key's owner against `namedModules()` paths, which use
  the same dotted form. A `Quantized` owner's `weight` counts `size × 32 / bits` parameters and its
  `scales`/`biases` count bytes only; a `Linear`'s own `bias` counts as a parameter. Precision is the
  floating type holding the most bytes. `NFKMLXModelInfoCache` computes it once per backend, since a
  status poll would otherwise walk a large net each second. `NFKMLXGemmaBackend`,
  `NFKMLXGraniteBackend`, `NFKMLXMambaBackend`, `NFKMLXNemotronBackend`, and `NFKMLXDecoderBackend`
  run a logits closure, so their init also takes the `modules` the closure runs and the release
  directory; `info(of:releaseDirectoryURL:)` adds the `model_type` and `max_position_embeddings` its
  `config.json` states, at the top level or under `text_config`. A factory that builds one of them
  passes its net, or the walk reports nothing. A paged mixture's experts sit in its
  `NFKMLXExpertStore`, outside `parameters()`, so each paged layer adopts `NFKMLXPagedParameters`
  and states what it stands for from its geometry, which every storage format shares: the paged
  switch linear experts × out × in (and its quantization), Gemma's and Qwen4-Exp's paged experts the
  element count of the stacked tensors they replaced (passed in at the swap), and DeepSeek's paged
  mixture routed experts × 3 × hidden × intermediate (its experts carry no biases), and a DeepSeek
  n-gram table held stored rows × columns. The walker adds these to the parameter count, and to the
  bytes only `pagedHeldBytes`: what a held load keeps in memory (`NFKMLXExpertStore.heldBytes(group:)`,
  the stored table's bytes), nothing for a mapped one. The paging tests hold a paged net's count to
  the resident net's, and DeepSeek's held load's bytes to its parameters plus the stored bytes. Reading the stored tensors instead would
  take one rule per format (stacked matrices, packed quantized words with scales, MXFP4 blocks,
  DeepSeek's FP8/FP4 store). `storageBytes(at:)` measures a release directory, every file under it,
  or a single file (a GGUF), a symbolic link counted at its target (a Python Hugging Face cache links
  snapshot files to blobs). `NFKMLXLanguageBackend` takes `releaseURL` from its directory and GGUF
  factories, `NFKMLXDeepSeekBackend` `releaseDirectoryURL` from its directory factory.
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

## Objective-C reachability gaps

`CLAUDE.md` requires every model to run from Objective-C as an `NFKInferenceBackend`. The 2026-10-13 inventory
found 35 gaps; 33 are closed. A model object wraps itself through `-inferenceBackend`, and its class carries
`backendWith…` factories beside its object factories. The shared closure backend is `NFKMLXGenerationBackend`
(`NFKMLXGenerationBackend.swift`): one request at a time, so a request's settings apply to that request only.
`MLXObjCExample.testEveryModelReachesObjectiveCAsABackend` asserts every factory selector and registered name.

New request and result keys from the closure:

- `NFKMLXRerankKey`: results under `NFKOutputStructured` (`scores`, `rankedIndices`) and the reranker
  `instruction` parameter.
- `NFKMLXForecastKey`: the input `series` (`NSArray<NSNumber>`), the `horizon` parameter, and the forecast fields
  (`median`, `mean`, `quantiles`, `quantileLevels`). `NFKMLXTimesFMParameterKey` carries TimesFM's options.
- `NFKMLXControlNetParameterKey` (`conditioningScale`, `controlMode`), `NFKMLXSAM3ParameterKey`,
  `NFKMLXSAM2ParameterKey.box`, `NFKMLXChatterboxParameterKey.exaggeration`, `NFKMLXFlux2ParameterKey`,
  `NFKMLXTextToImageInputKey.imageEmbedding`.
- `NFKMLXPhonemizing`: the Objective-C phonemizer protocol. `NFKMLXNeuralG2P` adopts it; a caller's own object
  can supply phonemes to the FastSpeech2 voice and the hand-chained TTS.

**Open (2).**

| Model | Blocker |
| --- | --- |
| Wan 2.2 Animate | Its `CLIPVisionModel` image encoder (open-clip XLM-RoBERTa ViT-H/14) and the modular pipeline's segment loop (`diffusers/modular_pipelines/wan_animate_2/denoise.py`: fps resample, letterbox, zigzag padding, per-segment i2v encode, reference pass, per-segment decode with carry-over) are not ported. The pipeline needs about 50 GB. |
| LTX-2 | Only the transformer is ported. The video and audio autoencoders, the text front end, and the vocoder are not. |

**Limits of closed bridges.**

- SAM 3.1: `facebook/sam3.1` ships `sam3.1_multiplex.pt` only, whose names the loader does not read; a 3.1
  checkpoint in the transformers layout loads through the directory factory. The catalog lists the `.pt`
  with `requiresConversion` and a source `note`: the file is the `facebook/sam3` codebase's layout with the
  Object Multiplex tracker, and its key names are behind the gate (the model card states the codebase
  layout and no transformers integration), so a reader change is not a small remap. Preprocessing omits the reference's
  antialiasing when shrinking.
- SANA: no resolution binning and no `clean_caption`. The pipeline loads its Gemma 2 encoder through its own
  loader, because `NFKMLXGemma2Net.load(directoryURL:)` keeps only `model.`-prefixed keys and SANA's are unprefixed.
- Z-Image: the backend is text-to-image; `NFKMLXZImageGenerator` does not expose the pipeline's image-to-image path.
- SAM 2 video: the prompt applies to the first frame. The mask clip comes back under `NFKOutputVideo` with
  per-frame scores under `NFKOutputStructured`, since `NFKOutputMask` holds one image.
- The variant check covered download peers only; each variant factory's internal use of the variant is
  unverified.
- Five download peers are covered by no test (they need a full-size checkpoint or the network): Cosmos,
  SigLIP 2, Depth Anything 3, CodeFormer photo, SD and IP-Adapter.

**Defect found while bridging (not fixed).**

- `NFKMLXGemma4Vision` runs the encoder's attention without a padding mask and pools padding hidden states
  without zeroing them; the reference does both. Padding falls in cell 0, so a padded grid (most real images)
  changes the first real soft token. The parity records use full grids only, so a fix needs a padded-grid
  record.

## Residency: staging, paging, and streaming

`NFKMLXResidency` (`NFKMLXResidency.swift`) is how a model holds its weights against the working set,
and one plan, `NFKMLXResidencyBudget.plan(_:residency:budget:)`, chooses between three mechanisms from
each stage's `NFKMLXStageFootprint` (its bytes, how many of them are routed experts, and its repeated
layers' bytes).

- Staging changes WHEN whole stages are loaded. A model built from stages that run strictly in turn
  loads each for its turn and releases it after. FLUX.1 (CLIP-L and T5-XXL, then transformer and
  autoencoder), the FLUX.1 ControlNet (the same encoders, then transformer, ControlNet, and
  autoencoder), FLUX.2 (text encoder, then transformer), Stable Diffusion 3 (two CLIP encoders and
  T5-XXL, then transformer and autoencoder), the SD3 ControlNet (the same encoders, then transformer,
  ControlNet, and autoencoder), SANA (Gemma 2, then transformer and DC-AE), Z-Image (Qwen3, then
  transformer and autoencoder), Qwen-Image 2.1 (Qwen3-VL 8B, then transformer and autoencoder),
  LTX-Video (T5-XXL, then transformer and autoencoder), Wan (umT5-XXL, then transformer and
  autoencoder) and MiniMax Music 3 (language model and depth decoder, then condition encoder and DiT,
  then vocoder) stage. Each builds an `NFKMLXStagedModel` of `NFKMLXStage`s (`NFKMLXSD3Generator.swift`,
  `NFKMLXSD3ControlNetGenerator.swift`, `NFKMLXSANAGenerator.swift`, `NFKMLXZImageGenerator.swift`,
  `NFKMLXFluxControlNetGenerator.swift`). The stage bytes come from shard headers at the load
  precision (`NFKMLXStageWeights`), not file sizes.
- Paging changes WHERE part of one stage lives. A mixture's routed experts stay in the release and are
  materialized as the router reaches them (`NFKMLXExpertStore.swift`, `NFKMLXExpertPaging.swift`).
  The language family (Qwen2-MoE, Qwen3-MoE, Mixtral, gpt-oss bf16 and MXFP4, a quantized save), the
  Gemma 4 26B-A4B mixture, the Qwen3-VL 30B-A3B decoder (fused `gate_up_proj` split per expert),
  Qwen4-Exp, Granite 4.0-H's mixtures and DeepSeek page.
- Streaming changes WHEN one stage's layers are loaded. A dense stage too large to hold whole holds
  the repeated layers that fit and reads each of the others from the release in its turn, on every pass
  (`NFKMLXLayerStreaming.swift`). Gemma 3, TranslateGemma, and MADLAD-400's T5 encoder-decoder stream.

The cases: `.resident` holds everything and throws where it is known not to fit; `.staged` stages and
never pages; `.paged` stages and pages; `.streamed` stages and streams each stage with layers that is
known not to fit whole (`!admits`), and on a machine that reports no budget streams every such stage
holding none of its layers; `.automatic` holds everything where the total fits (`holds`), pages where
the largest stage is known not to fit whole and has experts (`!admits`), streams where a stage with
layers is known not to fit whole, and stages otherwise. Staging is tried first: it costs a load per run,
paging a read per step, streaming a read of every unheld layer per pass. A model without
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

- `NFKMLXResidencyBudget` plans against 0.85 of Metal's recommended working set with the core's reserve,
  never live free memory, which a resident model's own weights would count against. The reserve is 4 GiB
  (`maximumReserve`) from a budget of 8 GiB and half the budget below it (`reserve(forBudget:)`, since
  2026-10-17); `holds` and `admits` read it per budget. A test that builds a budget from a footprint
  uses `leastBudget(holding:)`, the least budget that holds those bytes beside its own reserve, which
  equals `maximumReserve` plus the bytes from 4 GiB of bytes up; the layer-streaming tests' tiny
  releases need the scaled form, since 4 GiB plus a few kilobytes is a budget below 8 GiB.
- The planner is the core's (`NFKModelResidencyPlan`, `NFKModelMemoryBudget`, 2026-10-15).
  `NFKMLXResidencyBudget` keeps its Swift API and calls through: `NFKMLXStageFootprint.core` converts a
  footprint, `NFKMLXResidencyPlan` reads the core plan it holds, and a refusal arrives as
  `NFKMLXError.unsupportedConfiguration` with the core's sentence. `NFKMLXPlannerDifferentialTests`
  keeps the Swift planner's last body, with the reserve scaled as the core's is, as its reference and
  compares it with the core planner and the call-through over 18 footprint sets × 5 residencies × 34
  budgets: equal plans, held layers, and stage checks, and identical refusal sentences.
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

Streaming:

- A stage declares its repeated layers' bytes (`NFKMLXStageFootprint.layerBytes`, counted inside
  `bytes`). Streamed, it needs everything outside the layers plus the two largest layers, the one
  computing and the one being read (`streamedMinimumBytes`); a stage short of that beside the reserve is
  refused. The plan gives each streamed stage what the budget leaves past that minimum
  (`streamedStages`), and the stage holds its first layers, in order, while they fit (`heldLayers`).
  `verifyEachStageLoads` skips a streamed stage.
- `NFKMLXLayerStream` opens the release's shards with `F_NOCACHE` and reads each tensor of a layer in
  one `pread` into page-aligned memory of its own (`posix_memalign`), which `MLXArray(rawPointer:…)`
  wraps. MLX's Metal allocator wraps a page-aligned pointer with `newBuffer(bytesNoCopy:)` and copies
  once, calling the finalizer at once, where Metal refuses it. A serial `.userInitiated` queue reads the
  next streamed layer while the current one computes, cyclically, so the next pass's first layer is read
  at the end of a pass and held until it starts.
- A streamed layer's module is built, its parameters replaced with zero-length placeholders before
  anything evaluates the random initialization (`vacate`), filled through `NFKMLXWeights.apply` for
  its turn (`install`), and emptied again once its output and the cache arrays it wrote are evaluated
  (`release`). An unevaluated output is a graph over the layer's weights and would keep them alive.
  The held-weight apply includes the placeholders, so its coverage check still sees every layer.
- A read failure skips the layer and is sticky (`readFailure`). `NFKMLXGemma3Net.verifyStream()` throws
  it; `NFKMLXGemma3Model.generate` checks after every pass and empties the prompt cache it was writing.
  The non-throwing forward APIs (`hiddenStates`, `logits`) cannot report it themselves.
- Gemma 3 plans one stage (decoder, and the vision tower and projector where they load) from its shard
  headers at the load precision (`NFKMLXGemma3.decoderFootprint`). `.automatic` holds the release whole
  where `verifyFits` passes, so every load that worked before loads the same way, and plans only where
  it fails. `.resident`, `.staged`, and `.paged` keep `verifyFits`'s refusal. A streamed decoder does
  not fine-tune (`requireHeldLayers`).
- MADLAD-400 (`NFKMLXT5Seq2SeqNet`) streams both T5 stacks through two streams, one a stack, each
  cyclic over its own blocks. The plan is one stage (`NFKMLXMADLAD.footprint`) whose `layerBytes` list
  the decoder's blocks from 1 and then the encoder's, so the decoder, which runs on every token, is held
  before the encoder, which runs once per source. Block 0 of each stack stays fixed: it carries the
  relative-position table every block's bias is computed from before the loop. The fixed bytes also
  count one more block, because each stream keeps its next pass's first block read while the other
  stack runs. `.automatic` holds the network whole where its bytes at the load precision fit Metal's
  working set (`NFKMLXMADLAD.verifyFits`; the release-wide `verifyFits` doubles every stored byte at
  `.float32` and would refuse the 3B). A `half` load streams through `converting: .bfloat16`, the
  conversion its held blocks take. A failed read stops every block of every later pass
  (`streamFailed`): a skipped decoder block leaves no cache entry, and the next pass would attend over
  keys of the wrong length. `NFKMLXTranslationBackend.streamedLayerCount` and `streamedBytesPerPass`
  report any translator's streams through `NFKMLXTranslator.layerStreams`. Measured on the 7B at float32
  on this M1 Max: 38 of 48 decoder blocks and encoder block 0 held, the rest streamed; the parity test
  read 63.4 GB of encoder blocks at 1.76 GB/s and 181.5 GB of decoder blocks at 2.32 GB/s in 332 s, and
  its footprint peaked at 18.0 GB, above the planner's budget of 0.85 of Metal's recommended working
  set. On the 3B with half its decoder streamed, the decoder read 45.6 GB at 1.92 GB/s.
- A draft cuts the passes (`NFKMLXGemma3Model.useDraft(_:)`, `speculate`): a smaller release of the
  same vocabulary, held, proposes `draftTokens` tokens a round and one pass of the streamed decoder scores
  them, through the language family's `verifyGreedily` / `verifyBySampling`. The target's rollback of
  the rejected proposals always succeeds: the sliding cache trims to `window - 1` before it appends, so a
  multi-token pass leaves every row a rollback of up to its length removes. The draft feeds itself one
  token a pass, and past its window a single-token cache keeps too few rows to roll back more than one;
  the draft then prefills again from every token the target holds rows for. The drafted factories load
  the draft's decoder alone (`load(…, vision: false)`), at the release's own precision, quantize it to
  `draftBits` (4 by default), and plan the target against the working set less the draft's held bytes,
  passing those bytes to `verifyFits` as its reserve. Unquantized, a 4B draft (about 9 GB with its
  vision tower) left this machine's planner budget of about 15.4 GB too little for the 12B's streamed
  minimum (3.75 GB plus the 4 GiB reserve) or the 27B's (5.3 GB plus the reserve).
- A streamed pass computes the held pass's values exactly: the same stored tensors, the same exact
  bfloat16 → float32 widening at `.float32`, the same kernels; only the evaluation boundaries move.
- The drive: the store's 4 TB Thunderbolt SSD reads uncached at 2.12 GB/s in 64 MB `pread`s at one
  request in flight and 2.27–2.30 GB/s at two to four (2026-10-03, cold 27B shards); 1 MB reads reach
  0.91 GB/s, and a mapped range faulted in after `MADV_WILLNEED` 0.80 GB/s. Hence `pread` per tensor,
  not the expert store's mapping, which suits small reads that repeat. Decode is read-bound: a layer's
  compute is milliseconds against hundreds of milliseconds of read.
- Inside a run the read rate is set by memory pressure, not the read pattern. The same per-tensor reads
  of 12B layers measured 2.0–2.2 GB/s on a quiet machine; the streamed 12B read 1.02 GB/s at an 18 GB
  footprint with swap at 6–8 GB, and the 27B 0.69 GB/s. A pool reusing released layers' buffers was
  tried and read 0.78 GB/s on the 12B against 1.02 without it, with swap starting higher, so the stream
  allocates fresh memory per tensor. What lowers the pressure is what the run holds beside the stream:
  quantizing a draft left its replaced bfloat16 weights in MLX's buffer cache, 7 GB. `quantize(module:)`
  leaves lazy arrays over the weights it replaces, so the drafted factory evaluates the draft before it
  clears the cache; clearing first freed nothing. With the evaluation the drafted 12B and 27B runs hold
  0.5–0.7 GB of cache, peak in the 18 GB bucket in place of 24.6, and take 93 s and 183 s in place of
  234 s and 467 s.

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

## Model catalog

Where the entries live:

- One file per modality: `NFKMLXCatalogImage.swift`, `NFKMLXCatalogVideo.swift`,
  `NFKMLXCatalogLanguage.swift`, `NFKMLXCatalogAudio.swift`, and the stand-ins in
  `NFKMLXCatalogReference.swift`. `NFKMLXModelDescriptors.entries` concatenates them.
- Each registry name is one entry, or one variant of an entry whose variants share a factory (OPUS-MT's
  pairs, Qwen3.5's sizes). A released size with its own factory gets its own name and entry: Granite
  4.0-H registers `granite-4.0-h` (1B, dense) and `granite-4.0-h-tiny` (routed experts, paged where
  the release does not fit under `.automatic`).
- A file a variant reads from another repository names it (`NFKModelFile.repository` / `revision` /
  `url`, the Swift helper `NFKModelFile.file(_:role:optional:repository:revision:)`): VoiceRestore's
  BigVGAN vocoder, Open-Jev's base model at the commit its adapter was trained over, and the base
  releases the two ControlNets steer. The ControlNets' registry directory holds the base under `base/`
  and the ControlNet under `controlnet/`; the catalog lists each file at its path in its own repository.
- A released LoRA merged into its base release is an entry of its own (`lcm-lora-sdv1-5`,
  `sdxl-lightning-4step-lora`; `NFKMLXCatalogImage.stableDiffusionAdapters`). Its source is the base
  release's files plus the adapter file from the adapter's repository, its options are the base's at
  the steps and guidance the adapter is trained for, and its factory reads the adapter beside the
  release, else from the default Hugging Face cache, downloading it when absent. Decision (2026-10-15):
  an entry per adapter. An option choice carries no files, so the download, its status, and the
  measurement would miss the adapter file. Every declared load option belongs to the memory model, and
  an in-place merge changes no memory figure. The measured stage counts the adapter's tensors beside the
  base, which is what the merge holds while it runs.
- A variant whose weights carry another license names it (`NFKModelVariant.license`): OPUS-MT's en-de
  and zh-en pairs are CC BY 4.0 under an Apache 2.0 entry, as each model card states.
- A decision model's tasks are `decision`, then `classification`: Laya, Open-Jev, and the Open-Jev
  DeBERTa model.
- A source carries a `note` a user should read (SAM 3.1's unconverted `.pt`), and
  `publiclyServed: false` where the publisher withdrew the repository (`stabilityai/stable-diffusion-2-1-base`,
  401 without a token since 2026-10). The entry and its registry name stay, because a local copy builds;
  `NFKModelSource.noting(_:publiclyServed:)` writes both, `inferkit-catalog sources` passes them on, and
  `measure.py` records such a source as unmeasurable.
- A backend reports its registry name as its identifier: each Whisper size (`whisper-base`; a `large`
  backend built outside the registry reports `whisper-large`), each wav2vec 2.0 and HuBERT release (the
  repository's name, or the configuration's `model_type` from a bare directory), and Granite's two names.

The measurement:

- `Tools/catalog/measure.py` writes `NFKMLXCatalogMeasurements.swift` (generated: two arrays, the files
  and the variants). `NFKMLXCatalogMeasurement.swift` holds the row types, the lookup, and
  `NFKMLXCatalogEntry.measured()`, which `NFKMLXModelDescriptors.modelDescriptors()` applies to every
  entry. The figures and the extension fields are documented in `Docs/model-catalog.md`.
- The download the script sizes is the one `NFKMLXReleaseDownload` makes: required files, served
  optional files, the first served weight candidate per directory, and the shards of a shard index. A
  gated repository lists its files without a token, so its sizes and digests are measured; its shard
  index and tensor headers are refused, so the shards come from the file names beside the index and the
  safetensors bytes count as opaque (no parameter count). A repository that refuses the listing is
  `gated`; a release page or a Google Drive folder is `unmeasurable`; a required file the repository
  does not serve is `missing`.
- `residentBytes` follows `NFKMLXLoadPlan.plan(for:)`: the precision each family's registered loader
  holds its weights at, read from the loaders. The default is stored, which is where the decoders held
  to their bfloat16 references load by default: Gemma 3, Gemma 3n, and the `NFKMLXLanguage` families
  (Qwen3, Qwen3-MoE, Qwen2-MoE, Mixtral, Mistral). Float32: BiRefNet, Cosmos tokenizer, DeBERTa
  (Open-Jev), Florence-2, Gemma 4, Laya, M2M-100, Marian, NLLB, ModernBERT, Pixtral, Qwen3-Embedding,
  Qwen3-VL, SmolLM2 (the GGUF), SmolVLM, and TrOCR. Bfloat16: DeepSeek V4, MADLAD-400 (registered `half: true`), and Sa2VA. Phi-4
  multimodal loads its decoder stored and its towers at float32 from the same shards, so it is an
  estimate (`residentEstimate`). The staged generators' plans give each component directory its own
  precision; a component that widens only where the budget holds it (FLUX.2's text encoder, SD3's
  T5-XXL, Wan's umT5) counts at its narrower precision.
- A variant with memory facts takes `residentBytes`, `peakBytes`, and `minimumMemoryBytes` from the
  core's estimate at the default options (`descriptorByDerivingMemoryFiguresInCatalog:`): the plan's
  precisions, a decoder's key-value cache at 4096 positions, and `automatic` residency, which without a
  budget holds the largest stage. A variant without facts keeps `NFKMLXVariantFigures`: the bytes at the
  plan's precisions and `NFKMLXResidencyBudget.leastWorkingSet(holding:)` of the largest stage. A
  model's requirement is the least of its variants' figures, set after the derivation.
- Rerun the script after adding a source or changing a file list. It runs in a few minutes on a fresh
  network and sends `HF_TOKEN` when set, which measures the gated repositories' tensor headers.

Memory facts, load options, and presets (`NFKMLXCatalogOptions.swift`, 2026-10-15):

- `measure.py` also writes, per top-level directory, the repeated layers of the largest layer container
  (`NFKMLXMeasuredLayers`), the `.experts.` bytes, the quantizable elements (rank-2 `.weight`, input
  width divisible by 64, no embedding or norm), and the `config.json` geometry (`NFKMLXMeasuredGeometry`,
  root and `text_config`). The fields default to empty, so a row the script wrote before them still
  compiles. A gated repository refuses `config.json` without a token: Gemma 3, Gemma 3n, TranslateGemma,
  and the gated generators have no geometry; a run with `HF_TOKEN` fills it.
- `NFKMLXLoadProfile` describes what a family's registered factory reads; `familyProfiles` maps a family
  to profiles filtered by the entry's primary task (Qwen3's embedding model and EmbeddingGemma take none),
  and `NFKMLXCatalogEntry.load` overrides it (`gemma4-multimodal` takes `.plain`). The residency option
  comes from the entry's `residencies` when it lists more than one. The option registrations (`NFKMLXOptionRegistrations.swift`) consume
  the same keys; a profile's `declaration(for:…).consumes` lists them.
- `NFKMLXVariantFacts` builds a variant's facts. The recipe (`NFKMLXFactsRecipe`) leaves out a figure the
  loader cannot act on, because the core planner's `automatic` would otherwise plan a residency the
  factory does not offer: experts only for an entry that pages, layers only for one that streams,
  quantizable elements only where the profile quantizes, the key-value cache only with a context option
  (a model without one would be estimated at its maximum context).
- A stage whose directories load at different precisions states `"loaded"` bytes and the memory model's
  `stagePrecisions` names it; a uniform stage names its precision. The precision option, where declared,
  replaces `stagePrecisions` for the stages it governs: every stage, or the profile's `precisionStage`
  alone (Wan's `transformer`), whose siblings keep their plan's precisions.
- `NFKMLXLoadPlan.castDirectories` names the directories a precision option converts. A stage holding
  some converted directories and some not states its `checkpoint`, `float32`, and `bfloat16` figures with
  the converted directories at that precision and the others at the precision the plan gives them, so
  Wan's `transformer` stage at `bfloat16` counts the transformer at bfloat16 and the VAE at float32. A
  stage holding none, or only converted ones, states its figures as every other stage does. A
  directory's figure converts every floating tensor, so Wan's `bfloat16` figure counts the parameters
  diffusers keeps float32 (`NFKMLXWanRelease.keepsFloat32`) at bfloat16, about 55 MB short on the 14B.
  The loaded size counts them at float32, as the generator's residency plan does (`transformerBytes`);
  that figure is the right one.
- `NFKMLXKeyValueRule` gives the 16-bit cache per position: `.dense` is
  `NFKMLXModelSizing.keyValueBytesPerToken(of:precision: .checkpoint)` (the tests compare it on the
  released configurations), `.slidingWindow` bounds Gemma 3's sliding layers to the window (pattern by
  `NFKMLXGemma3Configuration.layerTypes(count:pattern:)` when `layer_types` is absent), `.sharedLayers`
  skips the trailing `num_kv_shared_layers` and widens full layers to `global_head_dim` /
  `num_global_key_value_heads` (Gemma 3n, Gemma 4), `.attentionLayers` counts `full_attention` /
  `attention` kinds (Granite 4.0-H; Qwen3.5's attention). gpt-oss is `.dense` because its sliding layers
  keep every key. Only the `NFKMLXLanguage` decoders declare cache options (2026-10-15): Gemma 3, Gemma
  3n, Gemma 4, Qwen3.5, Granite 4.0-H, and Qwen3-VL read no cache parameter, so their profiles declare
  none and their facts state no cache. The other rules are tested and wait for a backend that reads the
  options; a cache without an option would be estimated at the maximum context.
- Draft choices are computed after every entry is measured (`draftChoices`), from the default variants'
  `vocabularySize` and `parameterCount`, among entries whose profile takes drafts (the `NFKMLXLanguage`
  families); `declaringDrafts` adds the options. The figures derive in a catalog of the package's raw
  descriptors: the shared catalog calls `modelDescriptors()` from inside its `dispatch_once`, so touching
  it there would recurse.
- Presets: `fast` (4-bit weights and 8-bit cache) where both options exist, `fast` (half the steps, as a
  request default) for a generator profile with at least 8 default steps, `precise` (float32) where
  float32 is a choice and not the default. All are `suggested`.
- `NFKMLXCatalogMemoryTests` holds them: facts on every measured variant with figures equal to the
  derived estimate, a figure on every memory choice but `automatic`, every profile's defaults resolving,
  the rules against the built configurations, draft choices, and every preset validating.

The snapshot:

- `Docs/model-catalog.json` is `swift run inferkit-catalog export` (from `InferKitMLX/`): the core's
  Apple engines and the MLX models. `NFKMLXCatalogCompletenessTests.testTheCommittedSnapshotIsTheCatalogsManifest`
  builds the same manifest in process from the two provider classes and fails with the regenerate
  command when the file differs.
- `inferkit-catalog` is an `executableTarget` with no product, so a consumer never builds it. It calls
  `NFKMLXReferenceModels.registerAll()`, builds descriptors, and evaluates no array, so `swift run`
  needs no Metal library and no `Tools/mlx-metallib.sh`.

The completeness rules (`NFKMLXCatalogCompletenessTests`, no MLX runtime):

- After `registerAll()`, the registry's names and the descriptors' and variants' registry names are the
  same set (a test's own registrations use the `test-` prefix and are left out).
- Identifiers are unique, and no MLX identifier is an Apple engine's.
- The stand-ins are exactly the six reference models, `sam`, and `swinir-x4`.
- Every other model has an output, a task, a license, and a default variant with a source.
- Every Hugging Face or URL source has `downloadBytes > 0`, except the sources the test lists by
  `model/variant` with the reason they are not sized.

## Unload, reload, and load measurement

The core protocol's optional `unload`, `isLoaded`, `residentByteCount`, and `peakByteCount`
(2026-10-15) reach every MLX model through one shared mechanism in `NFKMLXLifecycle.swift`.

- `NFKMLXLifecycle<State>` holds a backend's loaded state, the closure that loads it again, the runs in
  flight, and the last measurement. `withState` counts a run (loading first when unloaded, under a
  separate load lock so concurrent runs share one load) and `begin`/`end` do the same for a run that
  outlives a call. `ifLoaded` reads the state without loading it and counts as a run meanwhile, so a
  `modelInfo` walk never races a free. `prepare` loads and withdraws a pending unload.
- An unload during a load is recorded like one during a run (`isLoading`): the load completes, the run
  that asked for it proceeds, and its `end` frees the state. A failed load withdraws the request. An
  unload that finds the state neither loaded nor loading records nothing, so it cannot free a later
  load.
- A load after an unload reads local files only (2026-10-16). The lifecycle runs every load inside
  `NFKMLXReload.reloading(_:)`, a task-local flag. Every package download site fetches through
  `NFKHFHub.fetchFile(repo:revision:path:)` (`NFKMLXHub.swift`), which under the flag returns the
  cached file's URL or throws `NFKMLXReloadError.modelFilesRemoved` without calling `downloadRepo`, so
  no request is possible. Under the flag the release helpers in `NFKMLXMarian.swift` skip the listing,
  read only the cached optional files, throw the same error when no weights candidate is cached, and
  write no completion record. A build from a local source (the registry's `weightsURL`,
  `NFKMLXDownload.backend`'s file) records the model's own files (`NFKMLXLocalFiles`), and the reload
  checks them before it builds: the source itself, the files the first build opened, and, for a
  catalog build (`NFKMLXCatalogBuilding`, `NFKMLXHub.backend(for:variant:…)`), the variant's listed
  files under the folder that were on disk (`NFKModelSource.localFilePaths`, which drops a file fetched
  from another repository). A build records what it opens through `NFKMLXFileReads`, a task-local
  record that `NFKMLXWeights.loadCheckpoint`, `NFKMLXSafetensors.entries`, `NFKMLXMappedFile`,
  `NFKMLXGGUFFormat`, and `fetchFile` note into; reads on another thread, and configuration and
  tokenizer reads through `Data(contentsOf:)`, are not noted. A folder's other files are never
  recorded: a shared snapshot holds other variants' files, and a `*.download` partial goes away when
  its transfer completes, and either would fail a reload whose model is intact (2026-10-16, Commit
  Manager review). A reload's build error that reports a missing file under the source
  (`NSFileReadNoSuchFileError`, `NSFileNoSuchFileError`, or `ENOENT`, directly or underlying) becomes
  the same error naming that file, which covers the unrecorded reads. The error carries
  `NSFilePathErrorKey`; the same backend loads on its next run once a download puts the files back.
  The core's `NFKDownloadErrorInUse` describes a removal refused during a download and does not fit,
  so the error is the package's own `@objc` enum (`NFKMLXReloadErrorDomain`). A reload does not mark
  the snapshot used in the hub's eviction order, since the cache-only path never calls `downloadRepo`.
- Decision: an unload never waits. With no run in flight it frees before returning; otherwise it is
  recorded and the last run's `end` frees. A blocking unload would deadlock a job completion handler
  that unloads, since `NFKMLXReloadableBackend` finishes its job before it ends the run. The free calls the backend's
  free handler (the language backend drops its prompt caches and vocabulary), drops the state inside an
  `autoreleasepool`, then calls `NFKMLXActivity.releaseCachedMemoryWhenIdle`.
- `NFKMLXActivity` counts runs and loads in flight across every lifecycle in the process. The cache
  release (`NFKMLXGPU.releaseCachedMemory`: synchronize, then clear) runs only at a count of zero and
  under the count's lock, so no managed run or load starts during the trim (`mlx-runtime-gotchas.md`).
  MLX work outside a lifecycle (a caller's own forward) is not counted.
- A lifecycle unloads only after `enableReload` gives it a closure. `NFKMLXReload.reloadable(_:)` is the
  one place that happens: `NFKMLXModelRegistry.backend(named:weightsURL:)` with a URL,
  `NFKMLXDownload.backend(…build:completionHandler:)` (every asynchronous `repo:` factory and
  `NFKMLXHub`'s), and `NFKMLXReleaseDownload.async` (every asynchronous release factory that delivers a
  backend), so registry and asynchronous-factory builds are reloadable. A factory that delivers a model
  object or a concrete class goes through `NFKMLXReleaseDownload.asyncValue`, which carries the download
  but cannot wrap the result. Decision (2026-10-15): the synchronous `@objc`
  factories stay as they are, and a backend built by a direct call reloads only when wrapped in
  `NFKMLXReloadableBackend`.
  `reloadable` measures the first build, then:
  - a backend adopting the internal `NFKMLXReloading` (module, matting, speech, generation, language)
    keeps its class and takes the closure. Its reload builds a fresh backend of the same class and takes
    that backend's state (`enableReload(building:as:measurement:state:)`), so the closure the factory
    captured (a holder over the network) moves over and the fresh shell is dropped;
  - any other backend is wrapped in `NFKMLXReloadableBackend`, whose state is the backend itself.
- A registry build without a URL is not made reloadable: random weights would not come back the same.
  `NFKMLXReloadableBackend.init(build:)` does not check this, and its documentation states the rule.
- `NFKMLXReloadableBackend` keeps the first backend's identifier and declared keys, and `responds(to:)`
  answers for the held class on the optional keys, `backendStatus`, and `tokenCountForRequest:error:`,
  so a wrapper does not claim a member the model lacks. It implements the token count itself, inside
  `withState`. An Objective-C message it does not implement goes to the held backend through the
  runtime's full forwarding path, as a run from before the message to its return. Swift cannot declare
  `methodSignatureForSelector:` or `forwardInvocation:` (both `OBJC_SWIFT_UNAVAILABLE`, and a method
  with either selector conflicts with `NSObject`'s), so the class adds both as block implementations
  (`imp_implementationWithBlock`, `class_replaceMethod`) once. The signature comes from the held class's
  method encoding through `+[NSMethodSignature signatureWithObjCTypes:]`, and `NSObject`'s own lookup
  answers for the wrapper's methods. `forwardInvocation:` begins a run (loading the model), calls
  `-[NSInvocation invokeWithTarget:]` on the held backend, and ends the run when that returns. Both
  `NSInvocation` calls go through the method implementations, cast to `@convention(c)` types. A probe
  forwarding integer, object, error-pointer, and `CGRect` messages 100,000 times leaked nothing. A
  backend that cannot be built raises `NSInvalidArgumentException` with the build error's reason. A
  Swift frame an Objective-C exception unwinds runs no cleanup: its `defer` blocks skip, and every
  strong reference it holds stays retained. A raise through the forward would leave the run and
  `NFKMLXActivity` counted, so the model never unloads and no cache release runs again in the process,
  and would keep the held backend and its weights alive past an unload. So no Swift frame of the forward
  is on the stack during a raise: `NFKMLXInvokeCatching` sends the message from Objective-C and catches
  what it raises, restoring the invocation's target so a leaked invocation does not keep the backend;
  `forward(_:)` ends the run and returns the exception; and `forwardInvocation:` is the support
  target's implementation (`NFKMLXForwardInvocationImplementation`, receiver and invocation held
  unretained), which raises the exception after the Swift handler has returned. A Swift `@objc` method
  that raises still leaks its own receiver, since its thunk retains it: measured in a probe, a Swift
  class raising from an `@objc` method stayed alive after its last reference went, and the same class
  raising from an Objective-C method deallocated. The test therefore raises from an Objective-C backend
  (`InferKitMLXTestSupport`, a target only the tests depend on). The helpers live in the package's
  Objective-C target `InferKitMLXSupport` (`Sources/InferKitMLXSupport`, compiled under ARC, which its
  source checks), which `InferKitMLX` imports with `internal import`, so its module stays out of the
  public interface and of a binary consumer's imports; `Tools/xcframework/build-mlx.sh` sweeps its
  objects into the archive with every other target's. The name `InferKitMLXObjC` is taken by the
  xcframework's module over the generated header. A run that
  `forwardingTarget(for:)` begins and an autoreleased object ends outlives the call: a block on a global
  queue drains no autorelease pool per block. Calling the protocol's optional throwing `tokenCount?(for:)` from
  Swift crashes the Swift 6.4 compiler in IRGen (a reabstraction thunk), so the wrapper and its test
  send the message through a private `@objc` protocol after a `responds(to:)` check. Registry and asynchronous results of classes
  other than the five lifecycle classes arrive wrapped, so a Swift cast to the model's class fails
  (decided acceptable, 2026-10-15; no shipped caller casts one). `wrappedBackend` returns the held backend
  for that cast, and is nil while the model is unloaded. Its `submitInferenceJob` runs a held
  backend's own job and counts the run until that job completes, or runs the synchronous method on its
  detached task, so a cancelled synchronous run is still counted until it returns.
- Measurement (`NFKMLXLoadMeasurement.measuring`): active memory before, `resetPeakMemory`, the load,
  then active and peak after; resident = the rise, peak = the peak less the start (at least the rise).
  Without the Metal library the load runs unmeasured, since a memory reading aborts. Arrays a loader
  `residentByteCount` reads 0 while unloaded; `peakByteCount` and the `modelInfo` keys keep the last
  measurement. Accuracy limits:
  - The peak is one reading for the whole process, so overlapping loads and runs fall into each other's
    figures.
  - Arrays a loader leaves lazy for the first run are not counted, since forcing them would defeat the
    streamed and paged loads.
  - The cache release sees only the runs and loads of the lifecycle backends, not MLX work a caller
    runs itself.
- The language backend records `isReady`, `hasDraftModel`, and `pagesExperts` at creation, and
  `expertStore` reads nil while unloaded. `modelInfo` keeps the description read while loaded
  (`NFKMLXModelInfoCache.currentValue`).
- `NFKMLXGPU` adopts the server's optional runtime members (`allocatedByteCount`, `peakAllocatedByteCount`,
  `resetPeakAllocatedByteCount`), so a server measures a provider build that measures nothing itself.
- Tests: `NFKMLXLifecycleTests` (24), on seeded 8 MiB weights behind test registry factories with a
  placeholder URL: a closure backend and a URL-less registry build stay loaded, unload frees within 10%
  (plus 1 MiB) of the measured bytes and a run reloads to identical output, prepare reloads, a generation
  backend reloads, an unload during a gated run frees at its end, a plain backend is wrapped and built
  twice with `wrappedBackend` reaching its class while loaded, the wrapper over a direct build, and the
  server's figures and unload. Twelve more run without weights or MLX: an unload during a gated load
  frees when the run ends, a double unload leaves the next load resident, an unload during a gated
  forwarded message sent from a global-queue block with no autorelease pool frees when the message
  returns while the block still runs, a forwarded message carries its arguments and result and loads an
  unloaded model, an unload during a gated token count frees after it returns, a wrapper over a class
  without a token count neither claims nor answers one, a fetch inside `reloading` makes no request, a
  release removed from a failing-transport hub's cache fails the reload with the error and no request and
  loads again once reseeded, a removed weights file and a removed file the build read inside a release
  folder fail the reload naming the file, a sibling variant's file and a `*.download` partial removed
  from the folder leave the reload working while the removed file the build opened fails it, a listed
  file removed fails the reload while an unlisted one does not, a forwarded message that raises reaches
  the sender through `NFKMLXCatchException` with the activity count back where it was and an unload
  deallocating the held backend (a weak reference reads nil), a release fetched inside
  `NFKMLXBuildHolds.keeping` refuses removal (`NFKDownloadErrorInUse`) until the scope returns, and one
  fetched outside a scope is released when the fetch returns.
- Holds across a build (2026-10-16). `NFKHFHub.holdRelease(repo:revision:)` hands its hold to the
  `NFKMLXBuildHolds` scope open on the calling task, which ends it when the build returns; with no scope
  open it returns the hold, and the helper ends it when the fetch returns. Scopes open at every public
  synchronous factory that takes a `cacheDirectoryURL` (each body runs in `NFKMLXBuildHolds.keeping`),
  at `NFKMLXReleaseDownload.asyncValue` (every asynchronous release factory), and around every load
  after an unload; a nested scope joins the outer one. `NFKMLXDownload.weightsURL` holds the single
  file's snapshot too, `NFKMLXDownload.backend` takes its hold before the asynchronous download and
  ends it after the build, and `NFKMLXHub.backend(for:variant:…)` holds a repository source's snapshot
  the same way. A new public download factory with a `cacheDirectoryURL` parameter wraps its body in
  `NFKMLXBuildHolds.keeping` the same way.

## Load options and the build entry

`NFKModelCatalog`'s build entry (`backendForModelIdentifier:options:error:`, core, 2026-10-15) reaches an
MLX model through the registry. The pieces:

- `NFKMLXModelRegistry` holds an `OptionsFactory` per name, `(URL?, NFKMLXLoadOptions)`, with the set of
  `NFKModelLoadOption*` keys it consumes. `register(name:factory:)` stores `{ url, _ in factory(url) }`
  consuming none, so every existing registration is unchanged. `register(name:consumes:factory:)` takes
  an options factory. `consumedOptionKeysForModelNamed:` reads the set, and
  `backendNamed:weightsURL:options:error:` is the Objective-C entry with a dictionary of resolved values;
  its draft is a file URL, an absolute path, or a descriptor identifier the shared catalog's locator finds.
- `NFKMLXLoadOptions` (`NFKMLXLoadOptions.swift`) reads the resolved dictionary: `residency`
  (`NFKMLXResidency(optionValue:)`, `.automatic` when absent), `precision` (`float32` → `.float32`;
  `checkpoint`, `bfloat16`, `float16` → `.checkpoint`; nil when absent, so a factory keeps its default),
  `halfPrecision`, `quantizationBits` (0 → nil) at `quantizationGroupSize` (64), `contextLength`,
  `cacheBits` (0 → nil) at `cacheGroupSize` (64), `draftModel`, `draftURL`, `draftBits` (0 → nil;
  `draftBits(orDefault:)` keeps a factory's default when the key is absent). `generationOptions(base:)`
  maps the context length to `contextWindow` and the cache bits to `cacheQuantization`.
  `applyWeightQuantization(to:)` is the runtime weight quantization: `NFKMLXQuantization.quantize` at
  the bits and group size, embeddings untouched (a tied head), then `eval` and `NFKMLXGPU.clearCache()`.
  Paged experts are not `NFKLMSwitchLinear`, so a paged mixture packs only its resident layers.
- `NFKMLXModelDescriptors.backend(for:)` (`NFKMLXCatalogBuilding.swift`, `+backendForBuildRequest:error:`)
  is the provider's builder; `registerBackendBuilder()` registers `NFKMLXEngineBuilder` for
  `NFKModelEngineMLX`, so a descriptor a manifest or another provider adds builds the same way. The name
  is the variant's registry name, else the descriptor's, else its identifier; an unregistered name runs
  `registerAll()` once, then fails `NoBuilder`. The draft is located through the request's catalog. The
  build goes through `NFKMLXModelRegistry.backend(named:weightsURL:loadOptions:configure:)`, whose
  `configure` sets the property options, so a reload sets them again.
- Property options: one the factory consumes is skipped (the factory read it as an argument; Stable
  Diffusion's precision). A string value naming an `NFKModelPrecision*` or `NFKModelResidency*` case is
  converted to the enumeration's raw value where the property currently holds an `NSNumber`, so
  `NFKMLXBackend.precision` takes `"float32"`. Key-value coding a string into an `@objc` enum property
  raises otherwise.
- `loadOptionValues`: the requested options with defaults filled, `automatic`
  included, so an app's comparison with the options it would build with matches.
  `NFKMLXLifecycle.setLoadOptions` keeps the options, and the five lifecycle classes and
  `NFKMLXReloadableBackend` report `NFKMLXLoadOptions.values`. The residency the load resolved to goes to
  `modelInfo` under the core's `NFKModelInfoResidency`: a factory reports it (`reportResidency`, the
  staged generators from `holdsStagesResident`), then the registry reads the built backend
  (`pagesExperts` → paged or resident, `streamedLayerCount > 0` → streamed, `holdsStagesResident` →
  resident or staged), and `NFKMLXLifecycle.loadInfo` adds it beside the measured figures. The language
  backend, `NFKMLXGemmaBackend` (`pagesExperts`), and `NFKMLXGemma3Backend` (`streamedLayerCount`) also
  write it into their own description, so a direct build reports it. A measured build whose factory and
  backend say nothing is `resident` (`NFKMLXLifecycle.modelInfo(over:)`): the factory read its weights,
  and every MLX catalog build is measured. MiniMax Music 3 reports `resident` where resident was asked
  for and `staged` otherwise, since each run reads its stages.
- The language backend reads `NFKModelLoadOptionContextLength`, `…CacheBits`, and `…CacheGroupSize` as
  request parameters (`applyLoadDefaults`, before the `NFKMLXGenerationParameterKey` keys, which win), so
  the defaults `NFKDefaultedBackend` fills are honored and a request overrides them.

The registrations (`NFKMLXOptionRegistrations.swift`, called last by `registerAll()`, after the catalog
files' `registerAdditionalModels`, so they replace those names' factories without editing the catalog
files). Names come from `NFKMLXModelDescriptors.entries` by family (and task where a family mixes
kinds). Consumed keys:

| Families | Consumes | Factory |
|---|---|---|
| qwen3, qwen3-moe, qwen2-moe, mixtral, gpt-oss, mistral (text generation) | residency, precision, quantizationBits, quantizationGroupSize, contextLength, cacheBits, cacheGroupSize, draftModel, draftBits | `NFKMLXLanguage.backend(directoryURL:loadOptions:)` |
| smollm2 (GGUF) | contextLength, cacheBits, cacheGroupSize | `backend(ggufURL:options:)` |
| gemma3 (text generation) | residency, precision, draftModel, draftBits (draft default 4 bits) | `NFKMLXGemma3.backend` / `model(…draftDirectoryURL:…draftBits:)` |
| gemma3 (translation: TranslateGemma) | residency, precision, draftModel, draftBits | `NFKMLXTranslateGemma` |
| gemma3n | precision | `NFKMLXGemma3n.backend(directoryURL:precision:)` |
| gemma4 (not the multimodal entry) | residency, precision | `NFKMLXGemmaLanguage.backend(directoryURL:precision:residency:)` |
| qwen3.5 | precision | `NFKMLXHybridLanguage.backend(directoryURL:precision:)` |
| qwen4-exp, granite-4.0-h | residency, precision | their `backend(directoryURL:precision:residency:…)` |
| qwen3-vl (question answering) | residency, precision, quantizationBits, quantizationGroupSize | `NFKMLXQwen3VL.model(directoryURL:loadOptions:)` |
| madlad | residency, precision (`halfPrecision`, default half) | `NFKMLXMADLAD.backend(directoryURL:half:residency:)` |
| deepseek-v4 | residency, paging, expertCacheBytes, speculates, quantizesActivations, computesInFloat32 | key-value coding onto `NFKMLXDeepSeekLoadOptions` |
| stable-diffusion (the five text-to-image releases) | precision | `NFKMLXTextToImage.backend(configuration:directoryURL:precision:)` |
| stable-diffusion (the two released adapters) | precision | `NFKMLXTextToImage.backend(configuration:directoryURL:adapterURL:adapterScale:precision:)` |
| flux, flux2, stable-diffusion-3, sana, z-image, qwen-image, wan, ltx-video, minimax-music | residency | each generator's `generator(directoryURL:residency:)`; `flux.1-dev` takes the `.flux` schedule |
| whisper | precision | `NFKMLXWhisper.backend(variant:…precision:identifier:)` by `whisperVariants` |

Each factory keeps its family's default for an absent key: stored for the `NFKMLXLanguage` families,
Gemma 3, Gemma 3n, Qwen3.5, Qwen4-Exp, Granite, TranslateGemma, Whisper, and Stable Diffusion; float32
for Gemma 4 and Qwen3-VL; half for MADLAD; automatic residency.

Not consumed: the context and cache keys on Gemma 3, Gemma 3n, Gemma 4, Qwen3.5 (no key-value cache),
Granite, and Qwen3-VL, whose backends read no cache parameters, so their descriptors do not declare them;
and the weight quantization outside the language families and Qwen3-VL.
`NFKMLXModelBuilderTests.testEveryDeclaredArgumentLoadOptionIsConsumed` fails for a descriptor that
declares an argument load option its factory does not read.

Tests: `NFKMLXLoadOptionsTests` (13: readings, defaults, generation defaults, the language backend's
request defaults, the draft's location, the reported residency, and the 4-bit and 8-bit quantization of
the tiny random-weight decoder behind `requireMLXRuntime`), `NFKMLXModelBuilderTests` (14: the options
factory, the requested values beside the resolved residency, the Objective-C draft path, a reload that rebuilds with the same options, the catalog's build
entry through the MLX builder with a placeholder location, `NFKMLXBackend.precision` set from
`"float32"` without a download, a consumed property option, `NoBuilder`, the registrations, Whisper's
sizes, DeepSeek's key-value coding, and the declared-versus-consumed check), and
`NFKMLXPlannerDifferentialTests` (3).

## Release tokenizers

`NFKMLXLanguage.readTokenizer(inDirectory:)` builds the tokenizer of a release directory, and
`releaseTokenizer(inDirectory:)` is its optional form. `tokenizerScheme(tokenizerJSON:tokenizerConfiguration:)`
decides what the files state, from parsed JSON, so the resolver judges a repository before downloading it.
The reader accepts only what it reproduces token for token:

- A `Split` followed by a `ByteLevel` that does not split again → byte-level BPE under the named pattern
  the regex equals byte for byte (`qwen2`, `qwen35`, `o200k`, `llama3`, `nemotron`). `Isolated` and
  `Removed` with `invert` cut identically, because every pattern matches every character.
- A lone `ByteLevel` with its own regex, no prefix space → `gpt2` (Granite Speech).
- A normalizer that is none, NFC, or an empty sequence → NFC where stated. `ignore_merges` passes through.
- Mistral's Metaspace form (`▁` prepended to the first segment only, no split, byte fallback, the
  Replace/ByteFallback/Fuse/Strip decoder) → `NFKMLXMetaspaceTokenizer` over `NFKMLXMistralTokenizer`
  (Mixtral). A run that follows an added token takes no `▁`: `<s>[INST]` encodes `[` bare, as
  `tokenizers` does.
- No `tokenizer.json` → read only when `tokenizer_config.json` names `Qwen2Tokenizer`, whose pattern
  and NFC normalization the class fixes.
- Anything else (another regex, Digits before the GPT-2 regex, BPE dropout, subword affixes, byte
  fallback outside Metaspace, a Unigram model, Llama 2's legacy Prepend normalizer) →
  `NFKMLXError.unsupportedConfiguration` naming the part.

The special tokens are the union of `tokenizer.json`'s `added_tokens`, `tokenizer_config.json`'s
`added_tokens_decoder`, and `added_tokens.json`. `NFKMLXReleaseTokenizerTests` holds every shipped family
(Qwen3, Qwen3.5, gpt-oss, Mistral-Small 3, Pixtral, Granite 4.0, Nemotron-H, Granite Speech, Phi-4
multimodal, Mixtral, and Llama 3.2) to the `tokenizers` library's ids on nine strings that separate the
schemes: digit runs, contractions, whitespace runs, decomposed accents, case changes, code, and added
tokens written literally.

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
