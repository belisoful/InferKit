<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# InferKitFoundationModels (companion package)

`InferKitFoundationModels/` is a separate SwiftPM package (macOS 26 / iOS 26 — the Foundation
Models floor; the model itself needs Apple Intelligence enabled). It depends only on the core.

- `NFKFoundationModelsBackend` adopts `NFKInferenceBackend` and wraps `LanguageModelSession`:
  `NFKInputPrompt` or `NFKInputMessages` (a system message becomes the session instructions),
  `NFKParameterTemperature` / `NFKParameterMaxTokens` map to `GenerationOptions`, and
  `streamResponse` feeds the job's `partialResult`. `isReady` mirrors
  `SystemLanguageModel.default.availability`; `prepare()` also prewarms once. Multi-turn history seeds
  a `Transcript` (system → `.instructions`, user → `.prompt`, assistant → `.response`, assistant
  `tool_calls` → `.toolCalls`, `tool` → `.toolOutput`); the last user turn is the prompt, and tool turns
  after it stay in the history so the question is asked again over the tool result (a transcript
  cannot be continued without a prompt). `Transcript.TextSegment` / `Transcript.Prompt` /
  `Transcript.Response` / `Transcript.ToolCall(id:toolName:arguments:)` / `Transcript.ToolOutput` are
  the entry constructors; `GeneratedContent(json:)` builds a call's arguments.
- Every option is a core request key (2026-09-16; the earlier `responseSchema` and typed
  `NFKFoundationToolParameter` were removed rather than deprecated, pre-1.0): `NFKParameterTopK` /
  `TopP` / `Seed` → `GenerationOptions.samplingMode` (temperature 0 → `.greedy`; a seed alone →
  `.random(probabilityThreshold: 1, seed:)`); `NFKParameterJSONSchema` → `NFKSchema.generationSchema`
  (JSON Schema → `DynamicGenerationSchema`: object / array / string with pattern, const, enum / integer
  and number with bounds as `GenerationGuide`s / boolean / anyOf / `$ref` into `$defs` as dependencies;
  nested schemas are named by property path, since names must be unique in the graph); an unsupported
  keyword throws `NFKSchemaError` by path. `NFKParameterChoices` → `DynamicGenerationSchema(anyOf:
  [String])`, the reply read with `content.value(String.self)`. `NFKParameterOutputFormat` is refused
  (`kNFKError_InferenceUnsupported`): guided generation has no "any JSON" schema. Results are read back
  through `GeneratedContent.jsonString` + `JSONSerialization` (`.fragmentsAllowed`), which keeps integers
  integral.
- Tool calling: `NFKFoundationTool` (name, description, JSON-Schema `parameters`, handler) registers on
  `backend.tools`. `NFKParameterTools` on a request declares the offered set and takes handlers from
  the registered tools by name; without the key every registered tool is offered. `NFKToolAdapter`
  adapts each to Apple's `Tool` with the runtime schema. A declared tool with no handler records the
  call in an `NFKToolCallRecorder` and throws `NFKUnhandledToolCall`; the session surfaces it as
  `LanguageModelSession.ToolCallError`, which the backend catches and turns into a finished job carrying
  `NFKOutputToolCalls` in the remote shape (`{id, name, arguments, argumentsJSON}`; the id is minted
  here, since `Tool.call` does not receive the transcript's). Executed calls are not reported under
  that key, so generic code that acts on `result.toolCalls` never runs a tool twice.
- Model selection (2026-09-17): `model` (`NFKFoundationModel`: `.onDevice` / `.privateCloudCompute`),
  `useCase`, and `guardrails` are `@objc` enums on the backend; `NFKFoundationModelConfiguration`
  snapshots them when a request is submitted, and `checkAvailability()` / `makeSession(tools:entries:)`
  dispatch on it. `SystemLanguageModel.default` is a computed static that builds a new instance per
  read, so never compare it by identity. Private Cloud Compute is `#available(macOS 27, iOS 27, *)`
  throughout: below it `isReady` is false and a request fails with `kNFKError_InferenceUnsupported`
  (never a silent fall-through to the device). `PrivateCloudComputeLanguageModel` has no
  `tokenCount(for:)`, so the context preflight is on-device only, and its `contextSize` is
  `async throws`, so `prepare()` reads it through a semaphore and `contextSize` is 0 before then. A
  reached quota (`quotaUsage.status == .limitReached`) makes the backend not ready with
  `NFKFoundationModelsErrorKey.resetDate` in `userInfo`. `privateCloudComputeQuota` and
  `variantDisplayName` are `@available(macOS 27, iOS 27, *)` `@objc` members, which ObjC reaches
  under `if (@available(macOS 27, *))`; the quota is optional and nil in a build with an SDK before 27. Live (M1 Max, macOS 26.6.2): the content-tagging model
  answers "photography, emotion, nature"; the Private Cloud Compute tests skip below 27 and are
  unmeasured.
- Gotcha, load-time crash: an `@objc` class is realized when the binary loads, which lays out its
  stored properties and needs their types' metadata. A stored property whose type exists only in the
  27 SDK (`PrivateCloudComputeLanguageModel.QuotaUsage`) crashes every process on macOS 26 at
  `realizeAllClasses` → `type metadata completion function`, before any test runs, even with the
  class marked `@available(macOS 27, *)`. `NFKFoundationModelQuota` stores it as `Any` and casts in
  the one method that uses it.
- Context preflight: on 26.4+ the backend sums `tokenCount(for:)` over the prompt, transcript entries,
  tools, and schema against `contextSize` and throws `kNFKError_InferenceUnsupported` with
  `NFKFoundationModelsErrorKey.tokenCount` / `.contextSize` in `userInfo`; below 26.4 the session's own
  error stands. `contextSize` is `@objc` on the backend (back-deployed, 4096 below 26.4). The
  preflight runs for the on-device model only.
- Live measurements (M1 Max, macOS 26.6.2, 2026-09-16): a name / age / traits schema returns typed
  fields with the integer inside its bounds; `["yes", "no", "unsure"]` returns `yes`; a declared
  vault-code tool without a handler returns the call with `{"vault": "Orion"}`; the same call handed
  back as a `tool` message yields `7391`; greedy decoding repeats itself across two runs.
- Generation tests skip (`XCTSkipUnless`) where the model is unavailable, so CI stays green.
- `NFKFoundationModelsProvider` (`@objc`) conforms to the core's `NFKDynamicBackendProvider` and is
  named the default the core tries for `NFKCapabilityTextGeneration`, so linking this package activates
  on-device LLM through `NFKDynamicBackend.backendForCapability:` with no registration (mirrors
  InferKitMLX's `NFKStableDiffusionProvider` / `NFKMLXWhisperProvider`).
- The reverse bridge ships (2026-09-18). `NFKInferKitLanguageModel` adopts `LanguageModel` and
  `NFKInferKitLanguageModelExecutor` adopts `LanguageModelExecutor`, so
  `LanguageModelSession(model:)` runs any `NFKInferenceBackend`. Design points worth keeping:
  - `LanguageModel` requires `Self == Executor.Model`, and `Executor.Configuration` is
    `Hashable & Sendable`. The backend travels in the configuration, which is a struct holding the
    existential, `@unchecked Sendable`, hashed and compared by object identity. The framework keeps
    one executor per distinct configuration, so identity is the right equality.
  - Capabilities come from the core protocol's new `supportedParameterKeys` / `supportedInputKeys`
    (`NFKParameterJSONSchema` → `.guidedGeneration`, `NFKParameterTools` → `.toolCalling`,
    `NFKInputImage` → `.vision`), read by `NFKInferKitLanguageModelCapabilities`. That class is
    `@objc` and ungated, so the capability reading is testable on macOS 26 and reachable from
    Objective-C, which cannot use `LanguageModelSession` at all.
  - The mapping functions that name only macOS 26 types (`messages(for:)`, `parameters(for:)`,
    `schemaJSON(for:)`) stay outside the compiler gate, so the test bundle exercises them on this
    host. Only the model, the executor, the attachment reading, and the tool declarations are gated.
  - Streaming: `NFKInferenceSubmit` covers sync and async backends, the job's `progressHandler`
    feeds an `AsyncStream`, and each reading contributes its new suffix through `appendix(sent:text:)`
    (a reading that does not extend what was sent is a rewrite, which an append cannot express).
    `withTaskCancellationHandler` cancels the job. Each reading is a `Reading` (text plus reasoning),
    so the two grow apart into the channel's response and reasoning through the same `appendix`.
    Every `tokenCount` on an append is 0: the counts are the turn's totals, which arrive once the
    turn is over, as one `.updateUsage` built from the result's `NFKOutputUsage`.
  - **Compiler crash to avoid:** calling an `@optional` Objective-C protocol method as a value from
    Swift (`backend.prepare?()`) crashed swift-frontend 6.4 in IRGen, emitting the reabstraction
    thunk for the imported throwing function. The core's `NFKInferencePrepare(backend, &error)`
    replaces it and is the shape to use for any other optional member.
- Images, reasoning, and usage (2026-09-19, item 4 of the Xcode 27 audit). All three are 27-only, so
  all three take both gates.
  - Images: `NFKInputImage` / `NFKInputImages` → `Attachment(cgImage)` appended to the prompt. The
    prompt stops being a `String` and becomes a `Prompt` built through
    `Prompt(@PromptBuilder _:)`, whose builder has `buildArray`, so a `for` over the images works
    inside it. `NFKImageCoding.cgImage(forImage:)` is the core's converter for the three
    representations. `tokenCount(for:)` takes `some PromptRepresentable`, so the preflight counts the
    attachments rather than the text alone.
  - Reasoning in: `NFKParameterReasoningEffort` → `ContextOptions.reasoningLevel`. The level is a
    27-only enum, so what crosses the gate is the **level's name as a String**, not a boxed enum, and
    each call site builds its own `ContextOptions`. That matters because the 27 schema overloads
    default to `ContextOptions(includeSchemaInPrompt: true)`: passing a bare `ContextOptions` would
    silently drop the schema from the prompt.
  - The `contextOptions:` parameter exists only on the 27 overloads, so `textStream(from:…)`,
    `schemaStream(from:…)`, and `choiceResponse(from:…)` each wrap one call with the compiler check
    inside, which keeps the three format branches from doubling.
  - Reasoning and usage out: `Response.usage` / `Snapshot.usage` (27) → `NFKOutputUsage`, and the
    `.reasoning` transcript entries → `NFKOutputReasoning`. `runOutputs(of:)` is generic over the
    snapshot and the response and answers an empty dictionary below 27, so the branches read one
    line each. A `guard case .reasoning(let r) = entry` inside a 27-available function needs no
    `@unknown default`, unlike a `switch`.
  - Below 27 a request that carries an image or an effort is **refused** with
    `kNFKError_InferenceUnsupported` rather than answered without it, which matches how
    `NFKParameterOutputFormat` is handled in the same file, and `supportedParameterKeys` /
    `supportedInputKeys` leave the keys out so a caller can ask first.
  - Unmeasured on this host (macOS 26.6.2): every one of these paths needs OS 27, so the tests
    assert the refusal and the declaration instead, through a `reachesOS27` helper that is true only
    when both gates pass.
- Errors map to the contract in `NFKFoundationModelsErrors.swift` (2026-09-21).
  `NFKFoundationModelsFailure.coreError(for:)` is the one seam, called from the job's catch and from
  `prepare()`'s cloud read; an error already in `NFKInferenceErrorDomain` passes through untouched, so
  the backend's own refusals are not double-wrapped. Mapping: guardrail violation and refusal →
  `kNFKError_InferenceRefused`; rate limit and a reached Private Cloud Compute quota →
  `kNFKError_InferenceRateLimited` with `NFKFoundationModelsErrorKey.resetDate`; context overflow and
  the four unsupported cases → `kNFKError_InferenceUnsupported`; missing assets →
  `kNFKError_InferenceNotReady`; Private Cloud Compute network failure and service unavailable →
  `kNFKError_RemoteUnreachable`; everything else → `kNFKError_InferenceBackendFailure`. Two facts
  worth keeping:
  - **The 26 error carries no counts.** `LanguageModelSession.GenerationError.exceededContextWindowSize`
    has only a `Context` with a debug description, where 27's `LanguageModelError.ContextSizeExceeded`
    has `tokenCount` and `contextSize`. That asymmetry is why the backend's own token preflight
    (item 1) still earns its place below 27.
  - **The 26 mapping is testable and the 27 mapping is not, for a reason that is not the OS.**
    `GenerationError.Context` has a public initializer, so every 26 case is constructed in
    `NFKFoundationModelsErrorTests`. The 27 payloads have initializers too, but the types need
    macOS 27 at run time, so those tests skip on a 26 host and the mapping is compile-verified.
- What stays Swift is a closed list, documented in the companion DocC article `SwiftOnly.md`
  (2026-09-21): `DynamicInstructions` and `LanguageModelSession.Profile` (result-builder protocols
  with an associated `Body`), `@Generable(name:)` (a macro on a declaration), `ImageReference` (a
  generable type), the session's own properties (`isResponding`, `transcript`, `usage`,
  `prewarm(promptPrefix:)`), and `transcriptErrorHandlingPolicy`. None can be a request key, and each
  has a contract path named in the article. The rule for a future SDK: if a new API can be expressed
  as a key, it gets one; the article grows only when the framework adds something that genuinely
  cannot bridge.
- Gotchas: SwiftPM tools 5.9 spells the platform `.macOS("26.0")` (`.v26` needs newer tools); the
  `NFKInferenceError` cases import into Swift as `.error_InferenceNotReady` style.
- Two SDKs, one source: CI's `macos-latest` image builds this package with an Xcode 26 SDK while the
  host builds with 27. Every 27-only symbol (`PrivateCloudComputeLanguageModel`,
  `SystemLanguageModel.variant`, `LanguageModel` / `LanguageModelExecutor`, `LanguageModelError`)
  needs BOTH gates:
  `#if compiler(>=6.4)` so the 26 SDK never sees the name (Xcode 27 is the first toolchain with
  Swift 6.4; `#available` alone fails CI with "cannot find type in scope"), and `#available(macOS 27,
  iOS 27, *)` inside it for the run-time check. The `#else` branch behaves as "below 27": the
  unsupported error, or nil (`privateCloudComputeQuota`, `variantDisplayName`). The API surface stays
  the same on both SDKs so the ObjC example target compiles against either. The 27 SDK also renames
  `GenerationOptions.sampling` to `samplingMode` and deprecates the old spelling; the 26 SDKs have only
  `sampling`. `setSamplingMode(_:on:)` / `samplingMode(of:)` select the spelling under the same
  compiler check, and tests read the mode through the accessor.
  `GenerationOptions(temperature:maximumResponseTokens:)` resolves without a warning on both.
- A test that calls `runInferenceForRequest:` assumes a model the CI runner does not have. The
  runners carry no Apple Intelligence, so `NFKFoundationModelConfiguration.checkAvailability()`
  refuses first and every generation path returns `kNFKError_InferenceNotReady` before it reads
  anything about the request. A test asserting a more specific refusal therefore passes on a
  developer's Mac and fails on CI, which is what
  `FoundationModelsObjCExample.testObjectiveCAsksForReasoningAndReadsWhatTheTurnCost` did from
  2026-09-19 until it keyed its expected code on `backend.isReady`. `isReady` is
  `(try? checkAvailability()) != nil`, so it answers exactly the question of which refusal comes
  back. The Swift examples avoid the trap a second way, by asserting on a static helper such as
  `NFKFoundationModelsBackend.reasoningEffort(for:)` that needs no model and then skipping. Prefer
  the static helper where one exists, and gate on `isReady` where the example has to run inference.
