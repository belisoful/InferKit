# Changelog

All notable changes to InferKit are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below 1.0.0, a minor bump may change public API. SwiftPM treats a 0.x minor as
breaking, so `from: "0.1.0"` resolves 0.1.x only and a consumer opts into each minor deliberately.

## [0.3.0] — unreleased

`NFKInferKit.version` and `InferKit.podspec` already read `0.3.0`; the `v0.3.0` tag is the release step.

### InferKitMLX (companion)

#### Native PyTorch checkpoint reader

- Every `weightsURL:` factory and every registry build accepts a raw PyTorch checkpoint (`.pth`,
  `.pt`, `.ckpt`, `.th`, HF `.bin`) wherever it accepted a converted safetensors, with no Python
  toolchain. The loader sniffs a file's leading bytes rather than its extension, so an HF torch `.bin`
  and a safetensors `.bin` are told apart by content. Both containers parse: the modern ZIP archive
  (zip64 and deflate) and the pre-1.6 five-pickle stream. The file stays memory-mapped, and the
  restricted pickle machine never executes a global; every construction it does not recognize becomes
  inert data that flattening drops.
- Training wrappers unwrap in the offline converters' own precedence (`state_dict`,
  `model_state_dict`, `params_ema`, `params`, `model`, `generator`, `state`) before the root is
  flattened, non-tensor sidecars drop, a tensor stored as a strided view (Whisper's transposed
  `Linear` weights) is gathered to row-major, and float64 narrows to float32.
- Three checkpoint shapes beyond a plain state dict load: a pickled live `nn.Module` tree (YOLO's
  `ultralytics` DetectionModel, walked through its `_parameters` / `_buffers` / `_modules` state), a
  TorchScript archive (CLIP, walked through its attribute-keyed scripted-module state), and a `.nemo`
  tar (VAD, unwrapped to the checkpoint inside). No class is constructed and no serialized `code/` is
  interpreted.
- `NFKMLXTorchCheckpoint` (`@objc`) is the inspection and conversion API over the same reader:
  `checkpointWithContentsOfURL:error:`, `tensorNames`, `infoForTensor:` (an `NFKMLXTorchTensorInfo`
  with `shape` and `NFKMLXTorchScalarType`), `dataForTensor:error:`, and `writeSafetensorsToURL:error:`,
  which converts on device to the file the model's `Tools` converter produces. Swift adds `arrays()`
  for feeding a model directly.
- Each model's loader carries its converter's renames and transforms, so a raw release loads end to
  end wherever a converted one does: U²-Net's `rebnconvN` index rename, the colorizer's `Sequential`
  table and transposed-convolution axis swap, HiFi-GAN's weight-norm fusion and `vocoder.` / `ups.N`
  renames, the NAFNet / RAFT / RIFE renames, LaMa's `generator.` and FastSpeech2's `model.` prefixes,
  and the pose head's transposed-convolution axis order. Conv-TasNet and the denoiser needed no change.
  `NFKMLXTorchParityTests` holds each raw load to its converted file, parameter for parameter.
- A big-endian save and a storage kind the reader has no counterpart for are refused with an error
  naming the offline converter; a file whose bytes do not parse fails as a malformed checkpoint rather
  than crashing.

#### Generation runtime (dense decoder)

- `NFKMLXGenerationOptions.cacheQuantization` stores the key-value cache affine-packed at 4 or 8 bits
  (`NFKMLXKeyValueCache.Quantization(bits:groupSize:)`; `groupSize` must divide the head dimension).
  It changes storage, not positions, so the window and mask accounting are untouched. Off by default
  because it is lossy; measured, 8-bit decoding tracks the full-precision logits at a cosine above 0.99.
- `NFKMLXGenerationOptions.prefillChunkSize` runs a long prompt through the cache in slices, bounding
  the prefill's attention peak by the chunk rather than the prompt. It is exact: each chunk attends
  through the cache to the keys a single pass would.
- `NFKMLXGenerationOptions.chatTemplate` (`NFKMLXChatTemplate.none` or `.chatML`) renders an
  `NFKInputMessages` list in the `<|im_start|>role … <|im_end|>` format an instruct release is trained
  on, using the release's own special tokens. Off by default; a base model wants the plain text, and a
  raw `NFKInputPrompt` is always used verbatim.
- `NFKMLXLanguage.backend(directoryURL:…)` refuses a release whose weights exceed the memory budget
  before materializing any of them. The stored bytes are compared against Metal's recommended working
  set, doubled for a `.float32` load of a 16-bit release, and the error names the shortfall and the
  remedies (`.checkpoint` precision, quantization, a smaller size). A machine with no readable budget
  does not gate the load.
- The dense loader validates every checkpoint shape against the built module, so a checkpoint whose
  shapes disagree fails at load instead of computing wrong numbers. The check is scoped to the dense
  decoder alone: shape adoption is load-bearing in the Gemma E4B and Conv-TasNet builders.

#### Long-form diffusion output

- `NFKMLXDiffusionBackend.windowedContinuation(totalWidth:height:channels:windowWidth:hop:scheduler:steps:seed:denoiseWindow:progress:)`
  denoises a latent longer than the model's native window by tiling one axis into overlapping windows
  and keeping continuity inside the sampler. Each window's overlap is held to the previous window's
  finished latent, re-noised to the current step's level through `scheduler.addNoise` (the same
  primitive the inpaint path uses, so any `NFKDiffusionScheduler` serves), locked after the loop, and
  the windows are stitched at the hop stride; the final window is pulled back to end exactly at the
  total. Window `i` draws its noise from `seed + i`; `progress` runs once per solver step and cancels
  by returning false. This is the `NFKMLXMusic3` mechanism lifted out for the shared-conditioning case
  (text-to-long-image, an extendable texture). Swift-only: it takes a closure over `MLXArray`.

#### Objective-C parity

- `NFKMLXGenerationParameterKey` carries the MLX generation options that have no core parameter key
  as request-parameter strings, read by `runInference` and overriding the backend's build-time
  options for that request: `contextWindow`, `cacheQuantizationBits` (4 or 8),
  `cacheQuantizationGroupSize` (default 64), `prefillChunkSize`, and `chatTemplate` (`"chatml"`).
- `NFKMLXLanguage.backendWithDirectoryURL:error:` builds the dense language model from a downloaded
  release directory (`config.json`, tokenizer, shards) at the default options.
- Every released size reaches every factory. Whisper: `backendWithVariant:weightsURL:tokenizer:timestamps:error:`,
  `backendWithVariant:weightsURL:error:`, `backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:error:`,
  and its `…completionHandler:` peer, over `NFKMLXWhisperVariant` (tiny / small / medium / large-v3),
  which was declared and reached by no factory. NAFNet and SwinIR gain
  `backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:error:` and the asynchronous peer;
  YOLO gains `backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:labels:error:` and its
  peer.
- RetinaFace: `NFKMLXRetinaFace.backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:`
  is the asynchronous download peer. `detectorWithWeightsURL:confidenceThreshold:suppressionThreshold:error:`
  returns an `NFKMLXRetinaFaceDetector`, now an `@objc` class, whose `facesInImage:error:` hands back
  `NFKFaceObservation` objects. `NFKFaceObservation` is now an `@objc` class carrying `boundingBox`,
  `confidence`, and the five landmarks by name (`leftEye`, `rightEye`, `nose`, `leftMouthCorner`,
  `rightMouthCorner`; the `landmarks` array stays Swift-only).
- `NFKMLXGPU.measuredMemoryBandwidth`, `measuredMemoryBandwidthWithMegabytes:repetitions:`, and
  `resetMeasuredBandwidth` expose the bandwidth probe `NFKMLXModelSizing` measures, beside the other
  machine readings.

### Changed

- `NFKFaceObservation` and `NFKMLXRetinaFaceDetector` are classes rather than structs. Reads are
  source-compatible; a Swift consumer that relied on value semantics now holds a reference.
  `NFKFaceObservation` gains a public `init(boundingBox:landmarks:confidence:)`.
- A dense-decoder release that does not fit the working set now throws at
  `NFKMLXLanguage.backend(directoryURL:…)` instead of proceeding to a process kill. A release that
  loaded at `.float32` with little margin may now be refused; load it at `.checkpoint` precision.
- A dense-decoder checkpoint whose tensor shapes disagree with `config.json` now fails at load.

### Documentation

- `Docs/examples.md`: a "Loading a PyTorch checkpoint directly" section in Swift and Objective-C;
  generation-option examples (cache quantization, chunked prefill, chat template, the fit refusal);
  an Objective-C example configuring generation through `NFKMLXGenerationParameterKey`, building the
  language model from a directory, reaching Whisper's sizes, reading a face's landmarks, and the
  bandwidth probe. Each is mirrored by a compiled example test in `InferKitMLX/Examples` and
  `InferKitMLX/ObjCExamples`.
- The `WeightsAndConversion` DocC article gains "Reading a PyTorch checkpoint natively" and a
  "PyTorch checkpoints" topic group (`NFKMLXTorchCheckpoint`, `NFKMLXTorchTensorInfo`,
  `NFKMLXTorchScalarType`).
- `Docs/mlx-runtime-hazards.md`: duplicate keys crash `ModuleParameters.unflattened` with a stack
  overflow (found by loading the raw RAFT checkpoint, whose `norm3` / `downsample.1` rename collides
  deliberately). Build remapped parameters into a dictionary first.
- `Docs/companions.md`: the quantized Music 3 stack is 7.7 GiB, and `transformerBits: 6` is documented
  as the measured fallback (DiT velocity cosine 0.99844 against 0.99990 at 8-bit).
- `Tools/README.md` (new) records that nothing under `Tools/` ships in any distribution, that a
  consumer needs no Python, and why the converters and the Python reference-parity tooling remain.
  Every `*-to-safetensors/convert.py` docstring now says the converter is optional.
- `CLAUDE.md` / `AGENTS.md`: Objective-C parity is a written, required rule for new and changed code,
  with the fixed list of what legitimately stays Swift-only.

### Tooling

- `Tools/validation-assets/fetch.py` gains two acquisition routes for a raw checkpoint, `gdrive` (a
  Google Drive id, MODNet) and `extract` (a member of a zip at `url`, LaMa's Lightning `best.ckpt`),
  and stamps `IK_RAW_<KEY>` for every asset whose raw download is on disk; the manifest adds the raw
  NAFNet, RIFE, LaMa, MODNet, and YOLO entries. The raw-versus-converted parity tests read those keys
  and skip when absent.
- `NFKMLXZipArchiveTests`, `NFKMLXPickleTests`, and `NFKMLXTorchCheckpointTests` run under
  `swift test`: the parsing layers are pure Foundation below the MLX materialization.

#### Language-model runtime, continued

- `NFKMLXKeyValueCache.rollback(by:)` discards the newest positions by moving cursors; it returns
  false, changing nothing, where a window has already dropped what it would reach.
- `NFKMLXPromptCache` keeps a key-value cache and its token ids between generations and rolls back to
  the longest prefix a new prompt shares, so a conversation's next turn prefills only what it adds.
  `save(to:)` / `load(from:)` persist it, quantized rows included.
  `NFKMLXGenerationOptions.reusesPromptCache` (request key `NFKMLXGenerationParameterKey.reusesPromptCache`)
  makes the backend keep one; `NFKMLXLanguageBackend.resetPromptCache` and `promptCacheLength` reach it.
- Speculative decoding: `NFKMLXLanguageNet.generate(prompt:options:draft:promptCache:report:onToken:)`
  verifies a draft model's proposals in one cached pass and keeps the leading run that agrees, so the
  output is the model's own greedy run token for token (measured on Qwen3-1.7B drafted by 0.6B:
  identical, 73% of proposals accepted, no wall-clock gain at float32 on an M1 Max, where a step is
  launch-bound). `NFKMLXLanguage.backend(directoryURL:draftDirectoryURL:options:)`, the `@objc`
  `backendWithDirectoryURL:draftDirectoryURL:error:`, `NFKMLXGenerationOptions.draftTokens`, the
  request key `draftTokens`, and `NFKMLXSpeculativeReport`.
- Mixture of experts: the dense decoder reads Qwen3-MoE (`qwen3_moe`) and Mixtral (`mixtral`)
  releases through the same factories. `NFKMLXLanguageConfiguration` gains `expertCount`,
  `activeExpertCount`, `expertIntermediateSize`, and `normalizesExpertWeights`; the loader stacks the
  released per-expert tensors into one tensor per projection and runs the chosen experts through a
  gathered matrix multiplication; `NFKMLXQuantization` packs the stacked experts and the checkpoint
  contract carries them. Both families match `transformers`' arithmetic layer by layer at a tiny
  configuration, and every released tensor of Qwen3-30B-A3B is accounted for by shape. Configs that
  interleave dense layers, use a sliding window, or name another expert family are refused.
- Constrained decoding: `NFKMLXTokenConstraint` masks the logits before sampling. `NFKMLXJSONConstraint`
  admits only well-formed JSON (root `.container` / `.object` / `.array` / `.any`, whitespace runs
  capped) and ends the run when the document closes; `NFKMLXChoiceConstraint` admits exactly one of a
  fixed set of strings; `NFKMLXByteConstraint` is the base for any byte-level grammar with a hashable
  state, with the admissible mask cached per state. `NFKMLXVocabulary` reads every token's bytes from
  the tokenizer. `NFKMLXGenerationOptions.constraint`, `jsonOutput`, `jsonRoot`, and `choices`; the
  request keys `outputFormat` (`"json"`, `"json-object"`, `"json-array"`) and `choices`.

### Core (`InferKit`)

- `NFKTokenizer.bytesForTokenId:` returns the bytes one id contributes to decoded text (a fragment of
  a multi-byte character as that fragment, a special token as its literal), implemented by the
  byte-level BPE, unigram, and word-piece tokenizers.

### Changed

- `NFKMLXLanguage.backend(directoryURL:)` now reads the release's special tokens and `eos_token` from
  `tokenizer_config.json`. A ChatML marker encodes to its single id where it previously encoded as
  plain text, and generation stops at the release's end-of-sequence token when a request names no
  stop tokens where it previously ran to `maxTokens`.
- `NFKMLXLanguageBackend` is exported to Objective-C under that name (`@objc(NFKMLXLanguageBackend)`).

### Tooling

- `Tools/validation-assets/shapes.py` fetches a release's `config.json` and every tensor's shape and
  dtype by HTTP range request over the safetensors headers, without downloading weights.
- `run_reference.py` gains the `qwen3_moe` and `mixtral` tiny-configuration modes, and parses under
  the LLM oracle environment's Python 3.9 again (a backslash inside an f-string expression, legal
  only from 3.12, had made every mode there unrunnable).

### Documentation

- `Docs/inference-guide.md`, deleted in an earlier commit, is restored and rewritten for the current
  backends, with a roadmap. `Docs/examples.md` gains prompt-cache, speculative-decoding,
  mixture-of-experts, and constrained-decoding sections in Swift and Objective-C, each mirrored by a
  compiled example; `Docs/companions.md` lists the language backend.
- `AGENTS.md` is again a byte-identical copy of `CLAUDE.md`. The two had drifted since v0.2.0, with
  `AGENTS.md` taking abbreviated versions of the same edits; `CLAUDE.md` is the one to edit.


## [0.2.0] — 2026-08-29

### Core (`InferKit`)

- `NFKInferKit.version` — a class property returning the core's semantic-version string
  (`NFKInferKit.h`, in the umbrella).
- `NFKComputePlan` reports where Core ML plans to run each operation of a compiled model, without
  running it: `planForCompiledModelAtURL:…` (synchronous and completion-handler forms),
  `neuralEngineOperationCount` / `gpuOperationCount` / `cpuOperationCount` / `unknownOperationCount`,
  `neuralEngineFraction`, `runsEntirelyOnNeuralEngine`, `operatorNamesOffNeuralEngine`, and
  `describedPlacement`. `MLComputeUnits` is a request, and Core ML says nothing about moving an
  operation the Neural Engine cannot run; this is the answer. Needs macOS 14.4 / iOS 17.4 / tvOS 17.4
  (`isAvailable`); an older system fails with `kNFKError_InferenceUnsupported` rather than returning
  an empty plan.
- `NFKHardwareProfile` reports what the machine is and how much of it is left: `currentProfile` with
  `chipName`, `modelIdentifier`, `graphicsArchitecture`, `performanceCoreCount`,
  `efficiencyCoreCount`, `hasUnifiedMemory`, and three distinct ceilings (`physicalMemory`,
  `recommendedWorkingSetSize`, `maximumBufferLength`); `+availableMemory` is a live reading. Every
  reading degrades to zero or an empty string rather than throwing.
- `NFKCoreMLBackend.computeUnits` selects the compute units a model loads with (default
  `MLComputeUnitsAll`; set before `loadModelFromURL:error:`).
- `NFKInputLyrics` — the lyrics a music-generation backend sings, distinct from `NFKInputPrompt`.
- `NFKByteLevelBPETokenizer` takes a `pretokenization:` name (`"gpt2"`, the default, or `"qwen2"`),
  and `NFKTokenizer.tokenizerForManifest:` reads it from the manifest's `tokenizer.pretokenizer`
  key. An unknown name is refused rather than defaulted.

### InferKitMLX (companion)

- MiniMax Music 3: `NFKMLXMusic3` and `NFKMLXMusicBackend` (`@objc`, `backendWithDirectoryURL:error:`,
  registry name `minimax-music3`) generate stereo 44.1 kHz music from a description
  (`NFKInputPrompt`) and lyrics (`NFKInputLyrics`), honoring `NFKParameterDurationSeconds`, `Seed`,
  `Steps`, and `GuidanceScale`. The five networks are each at measured reference parity against
  diffusers' own implementation, and the prompt contract is token-exact against the release
  tokenizer. The weights are under the MiniMax-Music3 Community License, which is not permissive.
- Runtime MLX quantization. `NFKMLXMusic3.quantizeRelease(at:to:bits:transformerBits:groupSize:)`
  writes a quantized copy of the release in the release's own layout (4-bit language model including
  its untied input embedding, 8-bit DiT, the split measured), which brings the stack from 27 GB to
  7.7 GiB and lets it stay resident between runs. A quantized checkpoint records its bits and group
  size in metadata (`inferkit.quantization`), the language-model and music loaders rebuild the
  matching packed structure before applying it, and it loads at its stored types whatever precision
  the caller requests.
- `NFKMLXGenerationOptions.contextWindow` bounds the key-value cache (`NFKMLXKeyValueCache(layerCount:window:)`):
  the oldest positions are dropped, so memory stops growing with the conversation, while rotary
  offsets stay absolute. The cache holds its rows in a buffer that grows in blocks rather than
  concatenating per token. Off by default; for a model whose attention is not natively windowed it
  is exact while the conversation fits and an approximation past that.
- `NFKMLXModelSizing` / `NFKMLXModelFit` decide whether a language model fits before anything is
  allocated: `parameterCount(of:)` counts a dense decoder from its geometry, `keyValueBytesPerToken`
  counts the cache at the key-value head count, `fit(of:tokens:precision:budget:)` returns `fits` /
  `fitsWithinWindow(n)` / `tooLarge(shortfall:)`, and `options(for:requesting:…)` derives a
  `contextWindow` from the machine. `measuredMemoryBandwidth` times a large read rather than
  tabulating per chip; `decodeCeiling` and `achievedFraction` turn it into tokens per second.
  Swift-only (it takes `NFKMLXLanguageConfiguration`).
- `NFKMLXGPU` reports the machine beside MLX's counters: `physicalMemory`,
  `recommendedWorkingSetSize`, `reclaimableMemory`, `memoryPressure`, `deviceArchitecture`, and
  `applyStandingLimits` / `applyStandingLimitsWithCacheBytes:fractionOfRecommendedWorkingSet:`
  (a standing cache cap, default 256 MB, plus a soft memory limit). Swift adds the scoped async
  `withWiredLimit(_:_:)`; there is deliberately no persistent wired-limit setter.
- `NFKMLXRoPEScaling` applies a release's `rope_scaling` (`linear` and `yarn`) in the dense decoder
  and DeepSeek, at reference parity against transformers' own frequency functions. `dynamic`,
  `llama3`, and `longrope` are refused rather than approximated.
- `NFKMLXLanguage.configuration(fromHuggingFace:)` reads transformers-5.x-shaped configs: a
  `layer_types` list naming only `full_attention` is dense (a mixed stack is still rejected), and
  `rope_theta` is read from `rope_parameters`. A release directory in the diffusers spelling
  (`diffusion_pytorch_model.safetensors`, single or sharded) also resolves.
- `NFKDiffusionLatentPreview` gives a diffusion job's progress callback an image: a per-channel map
  from latent to RGB, reported as the job's `partialResult` each step through
  `NFKDiffusionConfiguration.latentPreview` and thinned by `previewEverySteps`. `.passthrough`,
  `.stableDiffusion`, and `.stableDiffusionXL` are the published factors (the SD map tracks a real
  decode at a mean-removed correlation of 0.93); `fitted(latentChannels:decode:sample:)` derives a
  map for any decoder by least squares.
- RAFT's correlation lookup runs as one fused Metal kernel per pyramid level on the GPU
  (`MLXFast.metalKernel`, compiled and cached by MLX on first use): 3.98 ms against 2753 ms at the
  model's eighth-resolution geometry. The gather path remains for the CPU stream and is what the
  kernel is held to, bit for bit.
- `NFKMLXTrainer.clipGradientNorm` sanitizes before it clips: non-finite entries are zeroed and the
  norm is taken relative to the largest magnitude, so a finite gradient set whose sum of squares
  overflows is scaled to the norm rather than to zero.
- `NFKMLXLoRA.apply` and `merge` refuse a quantized model with `loRANotApplicable`: a
  `QuantizedLinear` satisfies a `Linear` type test while holding packed integers, and merging a delta
  into a quantized base then requantizing rounds the training away.
- `NFKMLXReferenceModels.registerAll` registers `minimax-music3`.

### Changed

- The Qwen text path (`NFKMLXLanguage.backend(directoryURL:)`) tokenizes with the `qwen2`
  pre-tokenization pattern rather than the byte-level BPE GPT-2 default, which encodes the same prompt
  to different, valid-looking token ids. A consumer that cached Qwen-encoded text from an earlier
  build re-encodes it.
- Every asynchronous path in InferKitMLX runs at user-initiated quality of service (`Task.detached(priority: .userInitiated)`).
  An unprioritized task inherits the default, which Apple Silicon schedules on the efficiency cores; a
  decode loop there ran several times slower.

### Documentation

- `Docs/mlx-runtime-hazards.md` (new, linked from the README): where MLX, Metal, and Core ML return a
  wrong answer quietly, each entry with an executable probe in `NFKMLXRuntimeHazardTests`. Four
  hazards reported elsewhere against mlx-swift 0.31.6 were reduced and do not reproduce here.
- `Docs/companions.md`: the `NFKMLXMusicBackend` entry and a "Model weight licenses" section
  recording the MiniMax-Music3 Community License obligations.
- `Docs/examples.md`: the MLX language backend in Swift, bounding the cache, sizing a model to the
  machine, extended context, `NFKComputePlan`, `NFKHardwareProfile`, and the latent preview, each
  mirrored by a compiled example.
- DocC pages for `NFKMLXMusic3` and `NFKMLXMusicBackend`, and a "Text → music" topic group.

### Tooling

- `Tools/ane-placement/`: paired Core ML models differing in exactly one property, to measure what
  lands on the Neural Engine. The measurement isolated why a converted language model runs on the
  GPU: a single-token forward is not placed on the Neural Engine, and a multifunction package takes
  one placement decision for both functions.
- `Tools/reference-parity/run_reference.py` gains the `music_vocoder`, `music_depth`,
  `music_condition`, `music_dit`, `music_ar`, `music_tokenizer`, and `rope_scaling` modes; the
  validation manifest gains the `MUSIC3` entry and records the music oracle's own interpreter under
  `oracle_environments`. `fetch.py` lands an `input` asset under `inputs/` as-is.

## [0.1.0] — 2026-08-22

First public release.

### Core (`InferKit`)

- `NFKInferenceBackend`, `NFKInferenceRequest`, `NFKInferenceResult`, and the thread-safe
  `NFKInferenceJob` handle, with a shared input/parameter/output key vocabulary.
- Backends with no third-party dependency: the passthrough mock, in-process Core ML (image and tensor),
  an on-device Core ML language model, OpenAI-compatible chat and transcription clients, and a
  submit-poll-fetch base for generation services.
- `NFKDynamicBackend` — runtime discovery, so an optional engine activates only when its classes are
  linked, with no build dependency on it.
- `NFKRemoteProvider` — named presets for the services a consumer is likely to call: OpenAI, xAI Grok,
  Google Gemini, Groq, Mistral, DeepSeek, Together, OpenRouter, and the local servers Ollama, LM
  Studio, llama.cpp, and vLLM, all OpenAI-compatible. `NFKAnthropicBackend` covers Anthropic's Messages
  API, whose authentication, required fields, and response envelope differ.
- Value types for detections, pose keypoints, classifications, audio segments, video, and audio.
- Tokenizers as a class cluster: byte-level BPE, CLIP, WordPiece, and Unigram, built from tokenizer
  files.
- `NFKHFHub` — Hugging Face resolve, download, checksum, and cache, with bearer-token support for a
  gated repository through `accessToken` or `HF_TOKEN`.
- RGBA-interleaved ↔ planar tensor conversion and an `MLMultiArray` bridge.
- A DocC catalog, and examples compiled in both Objective-C and Swift so a documented snippet cannot
  drift from the API.

### InferKitMLX (companion)

- Around thirty-five models implemented in `MLXNN` and validated at measured reference parity against
  each model's own reference implementation, spanning image, video, audio, and text.
- Stable Diffusion text-to-image built entirely from this package's own components — CLIP text tower,
  UNet, DDIM sampler, autoencoder — at end-to-end parity for SD 1.5, SD 2.1, and SDXL-Turbo.
- On-device text generation: `NFKMLXLanguage`, a dense decoder (grouped-query attention, rotary
  embeddings, SwiGLU) with a key-value cache and sampling, at measured parity against Qwen3 0.6B, 1.7B,
  and 4B, including sharded checkpoints.
- `NFKMLXHybridLanguage` — the Qwen3.5 / 3.6 / 3.8 hybrid decoder (gated delta-rule recurrence with
  full attention every fourth layer). Built and verified structurally against the released Qwen3.8-27B
  checkpoint; its numerics are unmeasured, because the smallest release does not fit on the machine
  that built it.
- `NFKMLXDeepSeek` — the DeepSeek V4 decoder (multi-head latent attention over a mixture of experts),
  with `NFKMLXDeepSeekQuantization` decoding the block-scaled fp8 and 4-bit formats the release is
  stored in.
  Built and verified structurally against the released Flash checkpoint; unmeasured numerically, and
  verified more weakly than the hybrid because that release is quantized.
- `NFKMLXGemmaLanguage` — the Gemma 4 text decoder (per-layer input embeddings, sliding and full
  attention at different head widths, soft-capped logits). Built and verified structurally against the
  released E2B checkpoint; unmeasured numerically, for want of an oracle on this machine.
- DeepSeek V4 arithmetic measured against transformers' own implementation at a tiny all-sliding
  configuration — every layer exact, logits 0.9999999999 — which found and fixed the head collapse
  being built as the wrong class. The release's fp8/fp4 storage decodes exactly; DeepSeek V4 Pro is
  accounted structurally.
- A complete trained text-to-speech voice: the espnet FastSpeech2 conformer acoustic model at
  reference parity on its released LJSpeech weights (durations exact, mel 0.9999999999), chained
  through its paired HiFi-GAN and the release's own phoneme vocabulary by `NFKMLXVoice`. The
  end-to-end test synthesizes "hello world" and the package's own Whisper transcribes it back.
- HiFi-GAN vocoder at reference parity on the released UNIVERSAL_V1 weights (0.9999999999); the
  converter fuses the release's weight normalization and reads the espnet-paired layout too.
- Whisper reads transformers-layout checkpoints (`model.encoder.layers.N.self_attn.q_proj` …) as
  well as OpenAI-layout ones.
- Whisper timestamped decoding — `transcribeWithTimestamps`, and `emitsTimestamps` on the backend,
  which adds `NSArray<NFKAudioSegment *>` under `NFKOutputSegments`. Off by default: it is a
  different decode, not a different reading of one.
- Additional sizes in ported families, each at measured parity: Whisper small / medium / large-v3,
  YOLOv8 l / x, SwinIR ×8 and the lightweight ×2 (a different upsampler), SAM 2 base_plus and large,
  and Gemma 4 E4B (measured at the bf16 the release ships in). DeepSeek V4 Pro is verified
  structurally — its 149782-tensor index accounted for exactly, never run.
- Bring-your-own-model backends: module, matting, tensor, diffusion, speech, and video.
- Video as a produced modality: `NFKMLXVideoBackend` reads and writes `NFKVideoAsset`, with clip-level
  frame interpolation (RIFE) and super-resolution (BasicVSR) over whole sequences.
- Face detection and alignment, so `NFKMLXCodeFormer` restores a photograph rather than only a
  pre-aligned crop. Two detectors: `NFKMLXRetinaFace` (the reference pipeline's own, at measured
  parity, 1.7 MB) and `NFKMLXVisionFaceDetector` (no weights, no download).
- On-device fine-tuning: `NFKMLXTrainer`, LoRA, training data helpers, and recipes for Zero-DCE,
  SegFormer, CLIP probes, and Whisper.
- `NFKMLXModelRegistry`, `NFKMLXHub`, and direct `@objc` factories, so an Objective-C consumer builds
  and runs a model without writing Swift.
- `NFKMLXRandom`, `NFKMLXGPU`, and `NFKMLXDevice` expose MLX's global runtime knobs to Objective-C.
- Every model's weight loader reads through `NFKMLXWeights.loadCheckpoint`, so a checkpoint written by
  fine-tuning reloads through the model's existing factory without being transposed twice.

### InferKitFoundationModels (companion)

- `NFKFoundationModelsBackend` over Apple's on-device system language model, with streaming, multi-turn
  transcripts, runtime-schema tool calling, and structured output.

### Distribution

- Source through Swift Package Manager and CocoaPods, from the same files.
- Prebuilt XCFrameworks for the MLX companion in static and dynamic variants, arm64, with macOS, iOS
  device, and iOS simulator slices. The static variant carries the Metal library each slice's consumer
  ships. Linking recipes per consumer shape are in `Docs/installation.md`.
