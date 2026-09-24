<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# Xcode 27, Foundation Models 27, and the M5 / M6 neural accelerators

An audit of the toolkit against the macOS 27 SDK, written 2026-09-16. It records what the SDK
ships, what the toolkit maps today, and the ordered work that follows. Model training changes are
out of scope here and go to `mlx-training.md`.

## What was measured, and what could not be

- Xcode 27.0 (27A266a), macOS 27.0 SDK, Swift 6.4 (swiftlang-6.4.0.34.1). `xcode-select` points at it.
- Host: macOS 26.6.2 on an Apple M1 Max (`applegpu_g13s`, GPU families Apple7 and Metal4).
- `InferKitFoundationModels` builds against the 27 SDK with zero warnings and its floor unchanged.
- Not measurable on this machine: any macOS 27 runtime behavior (the SDK is present, the OS is not),
  any neural-accelerator kernel (M1 Max has none), and Private Cloud Compute. Every claim below
  about those three comes from the SDK interfaces and the mlx sources, not from a run.

The interface read for this audit is
`MacOSX27.0.sdk/System/Library/Frameworks/FoundationModels.framework/Modules/FoundationModels.swiftmodule/arm64e-apple-macos.swiftinterface`
(module version 2.0.68.1.402). Re-read it before adopting an API named here, since a beta SDK moves.

## Foundation Models: what the 27 SDK adds

Availability column: the OS that introduces the API. Everything at 27 needs
`if #available(macOS 27, iOS 27, *)` inside the companion, whose floor stays macOS 26 / iOS 26.

| Area | API | Availability |
|---|---|---|
| Provider protocols | `LanguageModel` (`capabilities`, `executorConfiguration`), `LanguageModelExecutor` (`respond(to:model:streamingInto:)`, `prewarm`), `LanguageModelExecutorGenerationRequest` (transcript, enabled tool definitions, schema, generation and context options, metadata), `LanguageModelExecutorGenerationChannel` (events `.response`, `.reasoning`, `.toolCalls`; actions `appendText`, `replaceTextSegment`, `addAttachmentSegment`, `updateUsage`, `toolCall(id:name:action:)`, `appendArguments`) | 27 |
| Capabilities | `LanguageModelCapabilities.Capability`: `.vision`, `.guidedGeneration`, `.reasoning`, `.toolCalling` | 27 |
| Session over any model | `LanguageModelSession(model: some LanguageModel, tools:, instructions: / transcript:)` | 27 |
| Private Cloud Compute | `PrivateCloudComputeLanguageModel` (`availability` with `.deviceNotEligible` / `.systemNotReady`, `quotaUsage` with `.belowLimit(isApproachingLimit:)` / `.limitReached`, `resetDate`, `LimitIncreaseSuggestion.show()`, `contextSize`, `supportedLanguages`; errors `networkFailure`, `quotaLimitReached`, `serviceUnavailable`) | 27 |
| System model variants | `SystemLanguageModel.variant` with `.core3` and `.coreAdvanced3` (`displayName`) | 27 |
| Images in | `Attachment<ImageAttachmentContent>` from `CGImage`, `CIImage`, `CVPixelBuffer`, or a file URL, with `.label(_:)`; `Transcript.Segment.attachment`, `Transcript.ImageAttachment`; `ImageReference` (a `Generable` the model uses to point at an attachment) | 27 |
| Reasoning | `ContextOptions.reasoningLevel` (`.light`, `.moderate`, `.deep`, `.custom`), `Transcript.Entry.reasoning` (segments, `signature`) | 27 |
| Usage | `LanguageModelSession.usage`, `Response.usage`, `Snapshot.usage`: input `totalTokenCount` / `cachedTokenCount`, output `totalTokenCount` / `reasoningTokenCount` | 27 |
| Tool calling mode | `GenerationOptions.toolCallingMode`: `.allowed`, `.required`, `.disallowed` | 27 |
| Errors | `LanguageModelError` (`contextSizeExceeded` with counts, `rateLimited` with `resetDate`, `guardrailViolation`, `refusal` with `explanation`, `unsupportedCapability`, `unsupportedTranscriptContent`, `unsupportedGenerationGuide`, `unsupportedLanguageOrLocale`, `timeout`), `SystemLanguageModel.Error.assetsUnavailable`, `LanguageModelSession.Error.concurrentRequests` / `.transcriptMutationWhileResponding` | 27 |
| Transcript editing | `Transcript` is `MutableCollection` + `RangeReplaceableCollection`; `transcript.history`; `session.transcript` is settable; `transcriptErrorHandlingPolicy` (`.revertTranscript` / `.preserveTranscript`) | 27 |
| Dynamic instructions | `DynamicInstructions` result builder, `LanguageModelSession.Profile` / `DynamicProfile`, `.model(_:)` modifier, `SessionPropertyKey` / `SessionPropertyValues` | 27 |
| Metadata | `metadata: [String: any ConvertibleToGeneratedContent]` on every `respond` / `streamResponse`, on prompts, responses, and tool calls | 27 |
| Token counting | `SystemLanguageModel.tokenCount(for:)` over a prompt, instructions, tools, a schema, or transcript entries; `contextSize` (back-deployed, 4096 below 27) | 26.4 |
| Schema | `DynamicGenerationSchema.null`, `representNilExplicitlyInGeneratedContent` | 26.4 |
| Sampling | `GenerationOptions.samplingMode`: `.greedy`, `.random(top:seed:)`, `.random(probabilityThreshold:seed:)` | 26 |
| Refusal explanation | `GenerationError.Refusal.explanation` / `explanationStream` | 26 |
| Prewarm | `session.prewarm(promptPrefix:)` | 26 |
| Feedback | `session.logFeedbackAttachment(sentiment:issues:desiredOutput:)` | 26 |
| Locale | `supportedLanguages`, `supportsLocale(_:)` | 26 |

Deprecations and removals that touch the toolkit's dependencies:

- `LanguageModelSession.GenerationError` → deprecated at 27, replaced case by case by `LanguageModelError`,
  `SystemLanguageModel.Error`, `LanguageModelSession.Error`, and `GeneratedContent.ParsingError`.
- `GenerationOptions.sampling` → renamed `samplingMode`; `init(sampling:…)` deprecated.
- `Transcript.StructuredSegment.source` → renamed `schemaName`.
- `Transcript.Response(assetIDs:segments:)` → still available; 27 adds `init(metadata:segments:)`.
- `SystemLanguageModel.Adapter` and `init(adapter:)` → deprecated at 26.4, **obsoleted at 27**. Custom
  adapters for the system model have no public API in the 27 SDK. A training plan for the system model
  is therefore not a toolkit feature.

## The companion today, against that list

As of 2026-09-16 `NFKFoundationModelsBackend` maps `NFKInputPrompt` / `NFKInputMessages` (including
assistant `tool_calls` and `tool` messages), `NFKParameterTemperature`, `NFKParameterMaxTokens`,
`NFKParameterTopK` / `TopP` / `Seed`, `NFKParameterJSONSchema`, `NFKParameterChoices`,
`NFKParameterTools` with handlers from `backend.tools`, and `NFKOutputToolCalls`; it streams text,
seeds a transcript, preflights the token count on 26.4+, prewarms once, and reports `contextSize` and
availability. As of 2026-09-17 it also selects the model: the on-device model with `useCase` and
`guardrails`, or Private Cloud Compute with its quota and the variant name (item 2). The rest of the
table at 27 is unmapped. Nothing it uses is deprecated in 27. The two
places that change shape under 27 are the error `catch` (it rethrows whatever the session throws, so
a consumer sees `LanguageModelError` on 27 and `GenerationError` on 26) and
`Transcript.Response(assetIDs:)`. `NFKInputImage` (→ `Attachment`, 27, when
`capabilities.contains(.vision)`) is the one core input key still ignored.

## Companion work, in order

Rule for all of it: the build baseline is the 27 SDK, the package floor stays 26, every 27 symbol sits
behind `#available`, and the 26 behavior is unchanged when the check fails. Objective-C parity applies
to each item (an `@objc` key, enum, or `NSObject` wrapper), per the CLAUDE.md rule.

1. **Honor the core keys that need no gating. SHIPPED 2026-09-16.** `NFKParameterTopK` / `TopP` /
   `Seed` → `samplingMode`; `NFKParameterJSONSchema` → `DynamicGenerationSchema`; `NFKParameterChoices`
   → an `anyOf` of strings; `NFKParameterTools` (`{name, description, parameters}`) → `NFKToolAdapter`
   over a JSON-Schema-built `GenerationSchema`, handlers from the registered `NFKFoundationTool`s by
   name, and an unhandled call ends the turn under `NFKOutputToolCalls` in the remote shape; `tool`
   and assistant `tool_calls` messages seed the transcript. The private `responseSchema` and the typed
   `NFKFoundationToolParameter` were removed (pre-1.0, the core keys replace them). `tokenCount(for:)`
   preflights against `contextSize` (26.4), and `prepare()` prewarms once. Details and the live
   measurements: `foundation-models-companion.md`.
2. **Model selection. Shipped 2026-09-17.** `model` (`NFKFoundationModel`: on-device or Private
   Cloud Compute), `useCase`, and `guardrails` on the backend, all `@objc`; `isReady` and `prepare()`
   consult the chosen model; `privateCloudComputeQuota` (`NFKFoundationModelQuota`: limit reached /
   approaching, reset date, limit-increase suggestion) and `variantDisplayName` are
   `@available(macOS 27, iOS 27, *)`. Below 27 a Private Cloud Compute request fails with
   `kNFKError_InferenceUnsupported`. The Private Cloud Compute path is unmeasured (host is 26.6.2);
   the content-tagging model is measured live. Details: `foundation-models-companion.md`.
3. **The provider bridge. Shipped 2026-09-18.** `NFKInferKitLanguageModel: LanguageModel` wraps any
   `NFKInferenceBackend`, with `NFKInferKitLanguageModelExecutor: LanguageModelExecutor`, so a
   consumer writes `LanguageModelSession(model: NFKInferKitLanguageModel(backend:))`. The mapping is
   as planned: transcript → `NFKInputMessages` (instructions → system, prompt → user, response →
   assistant, toolCalls / toolOutput → the remote tool-message shapes, attachments →
   `NFKInputImage`); `enabledToolDefinitions` → `NFKParameterTools`; `schema` →
   `NFKParameterJSONSchema`; `GenerationOptions` → temperature, max tokens, and the sampling mode's
   top-k / top-p / seed; the job's `partialResult` → `.response(action: .appendText(delta,
   tokenCount:))`; the result's `NFKOutputToolCalls` → `.toolCalls(action: .toolCall(id:name:action:
   .appendArguments))`. No `updateUsage` yet: the core reports no token counts, so every count on the
   channel is 0 until item 4 adds `NFKOutputUsage`. The core protocol gained the optional
   `supportedParameterKeys` / `supportedInputKeys` the capabilities derive from
   (`NFKInferKitLanguageModelCapabilities`, `@objc`, readable below 27), and
   `NFKInferencePrepare(backend, &error)`. The remote, Anthropic, Core ML language, and Foundation
   Models backends declare their keys, and every InferKitMLX backend followed on 2026-09-18, so an
   MLX language model reports guided generation through the bridge and a Gemma 3 model reports
   vision. Unmeasured on this host: the bridge needs macOS 27 at run time, so the model, the
   executor, and the streaming are compile-verified only; the capability reading and the transcript,
   option, and schema mappings run in the test bundle on 26. Details:
   `foundation-models-companion.md` and `mlx-companion.md`.
4. **Images, reasoning, usage. Shipped 2026-09-19.** The three keys are in the core's
   `NFKInferenceKeys.h`: `NFKParameterReasoningEffort` (a string, with `NFKReasoningEffortLight` /
   `Moderate` / `Deep` as the named levels and any other string passing through),
   `NFKOutputReasoning`, and `NFKOutputUsage` (a dictionary keyed by `NFKUsageInputTokens`,
   `NFKUsageCachedTokens`, `NFKUsageOutputTokens`, `NFKUsageReasoningTokens`, with an unreported
   count absent rather than zero). In the companion, all three are `#if compiler(>=6.4)` plus
   `#available`: `NFKInputImage` / `NFKInputImages` become prompt attachments, the effort becomes
   `ContextOptions.reasoningLevel`, and the reasoning entries and `Response.usage` / `Snapshot.usage`
   come back under the two output keys, on the partial results as well as the final one. Below 27
   the backend leaves the keys out of what it declares and refuses a request that carries one. The
   provider bridge carries the same three the other way, and its `.updateUsage` closes the turn, so
   the channel's appends still carry 0 because the counts are totals rather than per-token readings.
   The remote backends fill the same keys: `NFKRemoteBackend` renames the effort's value to
   `low` / `medium` / `high` under `reasoning_effort` and reads `reasoning_content` and `usage`;
   `NFKAnthropicBackend` turns it into a `thinking` budget, drops the sampling the API forbids
   beside it, and reads thinking blocks and the message events' counts. Unmeasured on this host:
   every companion path needs OS 27, so the tests assert the refusal and the declarations instead.
   Details: `foundation-models-companion.md`, `remote-providers.md`, and `core-runtime-notes.md`.
5. **Errors. Shipped 2026-09-21.** The core gained `kNFKError_InferenceRefused` and
   `kNFKError_InferenceRateLimited`, because the distinction the framework draws is the one an app
   acts on and the core had nowhere to put it. `NFKFoundationModelsFailure` maps both error families
   and the Private Cloud Compute, system-model, and session errors besides. Details:
   `foundation-models-companion.md`. The original plan: catch `LanguageModelError` on 27 and
   `GenerationError` below it; map
   `contextSizeExceeded` → `NFKInferenceError` with the counts in `userInfo`, `rateLimited` → the
   remote backend's rate-limit error, `refusal` / `guardrailViolation` → a distinct code so an app does
   not retry them, `unsupportedCapability` → invalid request.
6. **Stays Swift-only. Shipped 2026-09-21.** The companion DocC article `SwiftOnly.md` names each
   one, what it does, and the contract's path instead: dynamic instructions and profiles → a system
   message with the tools and sampling keys; `@Generable(name:)` → `NFKParameterJSONSchema`;
   `ImageReference` → `NFKInputImage`; the session's properties → the job's `partialResult`, the
   caller's messages, `NFKOutputUsage`, and `prepare()`; `transcriptErrorHandlingPolicy` → the caller
   owning its own conversation. The list is closed, and the rule for a future SDK is that anything
   expressible as a key gets one.

**The companion work above is complete.** Items 1 through 6 shipped between 2026-09-16 and
2026-09-21. What remains from this audit is outside the companion: the Apple-framework backends in
the core (shipped 2026-09-21, see `apple-framework-backends.md`), re-pinning mlx-swift when a 0.32.x
tag appears, and measuring the neural accelerators on real M5 hardware.

`Docs/inference-guide.md`, `InferKitFoundationModels/README.md`, `Docs/companions.md`,
`Docs/examples.md`, the companion `Package.swift` header, and the companion DocC now describe the
bridge as shipped (2026-09-18) and the three new keys as shipped (2026-09-19).

## What Apple now implements that overlaps the toolkit

**Acted on 2026-09-21.** The core now ships seven Apple-framework engines from this survey:
`NFKVisionTextBackend`, `NFKVisionSegmentationBackend`, `NFKVisionPoseBackend`,
`NFKVisionFaceBackend`, `NFKVisionFeaturePrintBackend`, `NFKVideoToolboxBackend`, and
`NFKSpeechRecognitionBackend`. Nothing was deprecated. The survey below also missed two overlaps
that VideoToolbox covers, super resolution and frame interpolation, which the frame processors have
shipped since macOS 15.4 and 26; both are in the new backend. What each engine costs and what it
cannot do: `apple-framework-backends.md`.

Nothing in the core or the companions is superseded outright. Each Apple framework below covers one
fixed model on one OS floor; the toolkit's version covers chosen weights on the core's floor. The
right move is an additional zero-dependency backend per framework, discoverable through
`NFKDynamicBackend`, not a deprecation. Ordered by overlap:

- **Speech (26): `SpeechAnalyzer`, `SpeechTranscriber`, `DictationTranscriber`, `SpeechDetector`,
  `AssetInventory`.** Overlaps Whisper and Parakeet for the `transcription` capability. A core
  `NFKSpeechAnalyzerBackend` (availability-gated to 26, `Speech` is a system framework) returning
  `NFKOutputText` and `NFKOutputSegments` fits `NFKCapabilityTranscription`; the dynamic default
  stays `NFKMLXWhisperProvider` and a consumer picks Apple's when it prefers no download. Whisper keeps
  translation and word timestamps; Parakeet keeps its token-level timestamps.
- **Vision (27): `GenerateIterativeSegmentationRequest`** (a `DownloadableAssetsRequest` with
  `assetStatus` / `downloadAssets(progress:)`, result a `PixelBufferObservation`). Overlaps SAM and SAM 2
  for prompted segmentation. A core backend returning `NFKOutputMask` is the same shape as the MLX one.
  Vision 27 also gives `RecognizeAnimalsRequest.Identifier` an enumerated identifier set.
- **ImagePlayground (26; 27 adds the `.animation` style, `creationStrategy`, `creationVariety`,
  `sizeSpecification`).** Overlaps `NFKCapabilityStableDiffusion` for text-to-image on Apple
  Intelligence hardware. Lowest priority: the output is a system style, not a chosen checkpoint.
- **Foundation Models 26.4 token counting** covers only the system model's tokenizer. The core
  tokenizers stay.
- **Foundation Models guided generation** covers only the system model. `NFKMLXJSONSchemaConstraint`
  and the Core ML grammar path stay.
- **`NFKCoreMLLanguageBackend` and `NFKMLXLanguage`** run a chosen checkpoint. The system model is
  one size (`core3`) or two (`coreAdvanced3`). No deprecation.
- **`NFKRemoteBackend`** and Private Cloud Compute both leave the device. Item 2 above puts PCC on the
  Foundation Models backend rather than on the remote one, because it has no HTTP surface.

Not in the 27 SDK: any change to `NaturalLanguage`, `Translation`, or `SoundAnalysis`;
`MetalPerformanceShadersGraph` carries three headers with 27 availability but no new
machine-learning entry point the toolkit would call.

## Core ML

The 27 SDK adds no Core ML API. Since 26.0 the only addition is `MLMultiArrayDataTypeInt8`. What the
core still leaves unused from macOS 15:

- `MLOptimizationHints` (`reshapeFrequency`, `specializationStrategy`) on `MLModelConfiguration`.
  `NFKCoreMLLanguageBackend` runs a prefill shape and a decode shape through the same compiled
  model; `reshapeFrequency = .frequent` on the prefill function and `.infrequent` on the decode
  function is the documented way to keep the decode graph specialized. Measure it on
  `coreml-compute-plan.md`'s model before making it the default.
- `MLModelConfiguration.allowLowPrecisionAccumulationOnGPU` for the fp16 LM path.
- `MLModelAsset` (load from memory) and the async `MLModel.load`, for a consumer that streams a
  model in without a file.
- `MLTensor` and `MLComputePolicy` are Swift-only and stay out of the Objective-C core.

Core ML dispatches to the neural accelerators on its own when it places an op on the GPU. Nothing
in the toolkit's compute-unit policy changes for M5 / M6. The placement numbers in
`coreml-compute-plan.md` were measured on M1 Max and say nothing about M5 placement.

## Neural accelerators (M5, M6)

**MLX. Superseded 2026-09-21; read this first.** The companion no longer pins 0.31.6. It pins the
`mlx-swift` main revision `901941965d82e4a216d4d117231d847d194c563d` (2026-09-17), which vendors mlx
core **0.32.2**, so every NAX fix this section was waiting for is already in the tree: 0.32.0's
kernel-name and edge-tile fixes, 0.32.1's Metal 4.1 address-space fix, its `qmv` batch limit on
M5-class GPUs and NVFP4 `qmv` on M5 Max, and 0.32.2's fused full-attention path and quantized MoE
matmul on NAX. The re-pin was made for a different defect: core 0.31.1 computes the wrong GPU
gradient after a fine-tune's first step. What is still outstanding is the **tag**, and it is a
release blocker rather than a nicety, because SwiftPM accepts a revision requirement only in a root
package: see `distribution-and-packaging.md`. As of 2026-09-21 the newest `mlx-swift` tag is still
0.31.6 (2026-07-02). The version numbers below describe the older pin and are kept for the history.

**MLX (as audited 2026-09-16).** The companion pins `mlx-swift` 0.31.6 (2026-07-02), which carries
mlx core **0.31.1** (`Source/Cmlx/mlx/mlx/version.h`). Neural-accelerator ("NAX") support in mlx core:

- 0.30.0 (2025-11-19) → NAX kernels; 0.30.1 → JIT-built so mlx-swift can use them; 0.31.1 → initial
  M5 Pro / Max tuning. All three are in the pinned core.
- The runtime gate (`is_nax_available()` in `backend/metal/device.h`): the OS is 26.2 or newer,
  and the GPU architecture generation parsed from `MTLDevice.architecture.name` is ≥ 17, or ≥ 18
  when the name ends in `p` (phone). An M1 Max reads `applegpu_g13s` and is excluded.
- Where the pinned core uses NAX: `steel_matmul` (fused and split-K, non-complex), quantized matmul
  (transposed, `K % 64 == 0`), and scaled-dot-product attention (head dim ≠ 80). The NAX gemm and
  gather kernels are JIT-compiled at run time with Metal language 4.0 when the device offers it; the
  quantized NAX kernels ship as `.metal` sources. Whether SwiftPM's `CompileMetalFile` step includes
  the quantized ones in `default.metallib` under Xcode 27 is unverified here (no device to run them).
- Later cores not in any mlx-swift release yet (mlx-swift 0.31.6 is the newest tag): 0.31.2 (segmented
  NAX matmul, split-K tuning, NAX addmm fix, int16 overflow fix for SDPA masks over 32K positions),
  0.32.0 (NAX kernel-name and edge-tile fixes, the note that the AOT build needs
  `MACOSX_DEPLOYMENT_TARGET=26.2`), 0.32.1 (a **Metal 4.1** address-space fix, `qmv` batch limit on
  M5-class GPUs, NVFP4 `qmv` on M5 Max), 0.32.2 (fused full-attention path for head dim 256, quantized
  MoE matmul on NAX). Watch for an mlx-swift tag carrying 0.32.x and re-pin; the Metal 4.1 fix is the
  one that matters on an OS 27 runtime, because the JIT path compiles with the OS's Metal compiler.
- `mlx-c` does not expose the NAX check, so the companion cannot ask MLX. A reading has to mirror the
  gate: parse `architecture.name` and check the OS.

**What NAX changes for the sizing model.** `NFKMLXModelFit` sizes decode by measured bandwidth.
Decode stays bandwidth-bound on NAX hardware, so `decodeCeiling` and `achievedFraction` keep their
meaning. Prefill, vision towers, diffusion, and the restoration models are matmul-bound and are where
NAX pays; the toolkit has no prefill-rate model to update.

**Hardware profile. Shipped 2026-09-21.** `NFKHardwareProfile` now reports `graphicsGeneration` and
`hasNeuralAccelerators`, mlx's gate verbatim, with `graphicsGenerationForArchitecture:` and
`architectureHasNeuralAccelerators:` taking a name so the M5 and M6 answers are pinned by tests on
hardware that has neither. One correction to the description below: mlx parses the generation
**positionally**, from the two characters before the last, not as "the integer after `g`". The
measurement procedure and the M1 Max baseline are in `hardware-and-model-sizing.md`. The original
note: two readings close the gap, both cheap and both
degrade to zero / NO on older OS or Intel: `graphicsGeneration` (the integer after `g`) and
`hasNeuralAccelerators` (generation ≥ 17, or ≥ 18 with the `p` suffix, and OS ≥ 26.2, the mlx gate
verbatim so a test on M5 can assert the two agree). The 27 SDK also defines `MTLGPUFamilyApple10` and
`MTLGPUFamilyApple11`; `supportsFamily` on those is the Metal-side reading, and which family maps to
M5 versus M6 is a measurement to take on the hardware, not a table to copy. Record it in
`hardware-and-model-sizing.md` when an M5 is available.

**Metal 4.** `MTLTensor` and `MTL4MachineLearningCommandEncoder` (26.0) are the direct route to the
accelerators for a consumer's own Metal 4 pipeline; 26.4 adds `MTLTensorDataTypeInt4` / `UInt4` and
27.0 adds `MTLTensorDataTypeMetalFloat8UE8M0`. `NFKTensorConversion` converts textures and
`MLMultiArray`s. An `MTLTexture` ↔ `MTLTensor` path, availability-gated to 26, is the readiness item
for those consumers. Nothing in the core needs it today.

**Smaller weights on NAX.** mlx 0.31.0 adds tensor-scale NVFP4 and 0.32.1 tunes it on M5 Max. The
companion reads `mxfp4` (gpt-oss) and quantizes at run time with affine modes; an `nvfp4` mode on
`NFKMLXQuantization` is a small add once the pinned core has the tuned kernels.

**Faster loads, already there.** The wired-memory limit (`NFKMLXRuntime.setWiredLimit`, mlx-swift
0.30.6) and memory-mapped safetensors are in place. Foundation Models `prewarm` (item 1) is the
remaining load-time win on the Apple model.

## Deployment floors, unchanged

- Core: macOS 11 / iOS 14 / tvOS 14. Every Apple-framework backend above is availability-gated; the
  umbrella header keeps compiling on the floor.
- InferKitMLX: macOS 14 / iOS 17 (mlx-swift's floor). NAX is a run-time gate inside mlx, so the floor
  does not move for it. mlx-swift 0.31.5 and later require `swift-tools-version` 6.3 in their own
  manifest; the companion's 5.9 manifest resolves them under Xcode 27.
- InferKitFoundationModels: macOS 26 / iOS 26, spelled `.macOS("26.0")` under tools 5.9. The 27
  APIs are gated, not a floor bump.
- CI: the hosted runners build the 27 SDK items only once their image carries Xcode 27; the MLX job's
  Metal-component step is already in `.github/workflows/ci.yml` (see `build-and-verification.md`).
