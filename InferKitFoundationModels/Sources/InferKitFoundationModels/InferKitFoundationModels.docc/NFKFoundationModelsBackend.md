# ``NFKFoundationModelsBackend``

## Overview

The backend maps the InferKit request onto a `LanguageModelSession`. A single `NFKInputPrompt` runs a
one-shot generation; an `NFKInputMessages` array replays a conversation, where a system message becomes
the session instructions and the prior turns seed a `Transcript`. `NFKParameterTemperature` and
`NFKParameterMaxTokens` map to `GenerationOptions`, and `NFKParameterTopK`, `NFKParameterTopP`, and
`NFKParameterSeed` choose the sampling mode (a temperature of zero is greedy). `isReady` mirrors the
chosen model's availability; an on-device request that needs more tokens than ``contextSize`` fails
before the session runs, with both counts under ``NFKFoundationModelsErrorKey``.

### Choosing the model

``model`` picks the on-device system model (the default) or Apple's larger model on Private Cloud
Compute (macOS 27 / iOS 27). ``useCase`` and ``guardrails`` specialize the on-device model. Below
macOS 27 a Private Cloud Compute backend is not ready, and a request fails with
`kNFKError_InferenceUnsupported`.

```swift
let backend = NFKFoundationModelsBackend()
backend.useCase = .contentTagging
if #available(macOS 27, iOS 27, *), !backend.privateCloudComputeQuota.isLimitReached {
    backend.model = .privateCloudCompute
}
```

``privateCloudComputeQuota`` reads the quota whatever `model` is set to, so an app decides before
switching; a reached quota makes the backend not ready. ``variantDisplayName`` names the on-device
model's variant. A request captures the model when it is submitted.

### Multi-turn conversations

Pass an OpenAI-style message array under `NFKInputMessages`. The backend splits it into the session
instructions and the live prompt: a leading system message becomes the instructions, the trailing user
turn becomes the prompt, and the turns in between seed a `Transcript` so the model replays the real
conversation.

```swift
let messages: [[String: String]] = [
    ["role": "system",    "content": "You are a terse assistant."],   // → session instructions
    ["role": "user",      "content": "What is the capital of France?"],
    ["role": "assistant", "content": "Paris."],                       // → seeded transcript
    ["role": "user",      "content": "And of Japan?"],                // → the live prompt
]
let request = NFKInferenceRequest(inputs: [NFKInputMessages: messages])
let reply = try backend.runInference(for: request).text   // "Tokyo."
```

### Streaming

For interactive use, submit a job and read partial text as it arrives; `streamResponse` feeds each
partial to the job's `partialResult`, and the job is cancellable mid-generation.

```swift
let job = backend.submitInferenceJob(for: request)
job.progressHandler = { job in print(job.partialResult?.text ?? "") }
```

## Topics

### Choosing the model

- ``NFKFoundationModelsBackend/model``
- ``NFKFoundationModelsBackend/useCase``
- ``NFKFoundationModelsBackend/guardrails``
- ``NFKFoundationModelsBackend/privateCloudComputeQuota``
- ``NFKFoundationModelsBackend/variantDisplayName``

### Runtime configuration

- ``NFKFoundationModelsBackend/tools``
- ``NFKFoundationModelsBackend/contextSize``

### Related

- <doc:ToolsAndStructuredOutput>
