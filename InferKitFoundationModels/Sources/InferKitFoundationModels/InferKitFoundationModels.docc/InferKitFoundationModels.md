# ``InferKitFoundationModels``

Runs Apple's language models behind the InferKit contract, and InferKit's backends behind Apple's
session API: the on-device system model or Private Cloud Compute, with text generation, streaming,
tool calling, and structured output.

@Metadata {
    @DisplayName("InferKitFoundationModels")
}

## Overview

This companion wraps Apple's Foundation Models `LanguageModelSession` as an `NFKInferenceBackend`, so an
InferKit consumer swaps on-device generation in like any other engine. It needs Apple Intelligence and
macOS 26 / iOS 26, so the package is opt-in and never raises the core's platform floor.

```swift
import InferKit
import InferKitFoundationModels

let backend = NFKFoundationModelsBackend()
guard backend.isReady else { return }   // SystemLanguageModel availability

let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "Name one color."])
let reply = try backend.runInference(for: request).text
```

- **Model** — `model` picks the on-device system model or Private Cloud Compute (macOS 27 / iOS 27);
  `useCase` and `guardrails` specialize the on-device model. See ``NFKFoundationModel``.
- **Multi-turn** — pass `NFKInputMessages` (an OpenAI-style array); a system message becomes the
  session instructions and the history seeds a `Transcript`.
- **Streaming & cancellation** — read `NFKInferenceJob`'s `partialResult` in its `progressHandler`.
- **Sampling** — `NFKParameterTemperature`, `NFKParameterMaxTokens`, `NFKParameterTopK`,
  `NFKParameterTopP`, and `NFKParameterSeed`; a temperature of zero is greedy.
- **Images, reasoning, usage** (macOS 27 / iOS 27) — `NFKInputImage` and `NFKInputImages` attach to
  the prompt, `NFKParameterReasoningEffort` becomes the context's reasoning level, and the result
  adds `NFKOutputReasoning` and `NFKOutputUsage`. Below 27 the backend declares none of the three
  and refuses a request that asks for one.
- **Tool calling** and **structured output** — through the core's `NFKParameterTools`,
  `NFKParameterJSONSchema`, and `NFKParameterChoices`, no compile-time `@Generable` type.
  See <doc:ToolsAndStructuredOutput>.
- **The provider bridge** — ``NFKInferKitLanguageModel`` presents any `NFKInferenceBackend` to
  `LanguageModelSession` (macOS 27 / iOS 27). See <doc:ProviderBridge>.
- **Failures** — the framework's errors become the contract's, so `kNFKError_InferenceRefused` means
  change the request and `kNFKError_InferenceRateLimited` means wait. The framework's own error stays
  under `NSUnderlyingErrorKey`.
- **What stays Swift** — dynamic instructions, profiles, `@Generable`, `ImageReference`, and the
  session's own properties are result builders, macros, and generics, so they carry no request key.
  What each one does and what the contract offers instead: <doc:SwiftOnly>.

### Activates the core's text-generation capability

Linking this package ships ``NFKFoundationModelsProvider``, named the default the core tries for its
`text-generation` capability. So on-device LLM lights up through `NFKDynamicBackend` with no
registration:

```swift
if NFKDynamicBackend.isCapabilityAvailable(NFKCapabilityTextGeneration) {
    let llm = try NFKDynamicBackend.backend(forCapability: NFKCapabilityTextGeneration)
}
```

## Topics

### Essentials

- ``NFKFoundationModelsBackend``

### Concepts

- <doc:ToolsAndStructuredOutput>
- <doc:ProviderBridge>
- <doc:SwiftOnly>

### Choosing the model

- ``NFKFoundationModel``
- ``NFKFoundationModelUseCase``
- ``NFKFoundationModelGuardrails``
- ``NFKFoundationModelQuota``

### Tools & structured output

- ``NFKFoundationTool``
- ``NFKFoundationModelsErrorKey``

### The provider bridge

- ``NFKInferKitLanguageModel``
- ``NFKInferKitLanguageModelExecutor``
- ``NFKInferKitLanguageModelCapabilities``

### Dynamic discovery

- ``NFKFoundationModelsProvider``
