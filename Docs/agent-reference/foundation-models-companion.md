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
- Context preflight: on 26.4+ the backend sums `tokenCount(for:)` over the prompt, transcript entries,
  tools, and schema against `contextSize` and throws `kNFKError_InferenceUnsupported` with
  `NFKFoundationModelsErrorKey.tokenCount` / `.contextSize` in `userInfo`; below 26.4 the session's own
  error stands. `contextSize` is `@objc` on the backend (back-deployed, 4096 below 26.4).
- Live measurements (M1 Max, macOS 26.6.2, 2026-09-16): a name / age / traits schema returns typed
  fields with the integer inside its bounds; `["yes", "no", "unsure"]` returns `yes`; a declared
  vault-code tool without a handler returns the call with `{"vault": "Orion"}`; the same call handed
  back as a `tool` message yields `7391`; greedy decoding repeats itself across two runs.
- Generation tests skip (`XCTSkipUnless`) where the model is unavailable, so CI stays green.
- `NFKFoundationModelsProvider` (`@objc`) conforms to the core's `NFKDynamicBackendProvider` and is
  named the default the core tries for `NFKCapabilityTextGeneration`, so linking this package activates
  on-device LLM through `NFKDynamicBackend.backendForCapability:` with no registration (mirrors
  InferKitMLX's `NFKStableDiffusionProvider` / `NFKMLXWhisperProvider`).
- The reverse bridge (Apple's `LanguageModel` / `LanguageModelExecutor` provider protocols, WWDC26)
  needs the macOS 27 / iOS 27 SDK; it is documented in the package README, not built. Xcode 27's SDK
  carries the protocols (verified 2026-09-16); the API inventory, the mapping, and the ordered work
  are in `xcode27-foundation-models-and-neural-accelerators.md`.
- Gotchas: SwiftPM tools 5.9 spells the platform `.macOS("26.0")` (`.v26` needs newer tools); the
  `NFKInferenceError` cases import into Swift as `.error_InferenceNotReady` style.
- Two SDKs, one source: CI's `macos-latest` image builds this package with an Xcode 26 SDK while the
  host builds with 27. The 27 SDK renames `GenerationOptions.sampling` to `samplingMode` and
  deprecates the old spelling; the 26 SDKs have only `sampling`. `setSamplingMode(_:on:)` /
  `samplingMode(of:)` in the backend select the spelling under `#if compiler(>=6.4)` (Xcode 27 is the
  first toolchain with Swift 6.4), and tests read the mode through the accessor. A 27-SDK API that is
  a rename rather than an addition cannot be gated with `#available`; it needs this compile-time
  check. `GenerationOptions(temperature:maximumResponseTokens:)` resolves without a warning on both.
