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
