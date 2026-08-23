# ``NFKFoundationModelsBackend``

## Overview

The backend maps the InferKit request onto a `LanguageModelSession`. A single `NFKInputPrompt` runs a
one-shot generation; an `NFKInputMessages` array replays a conversation, where a system message becomes
the session instructions and the prior turns seed a `Transcript`. `NFKParameterTemperature` and
`NFKParameterMaxTokens` map to `GenerationOptions`; `isReady` mirrors `SystemLanguageModel.default.availability`.

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

### Runtime configuration

- ``NFKFoundationModelsBackend/tools``
- ``NFKFoundationModelsBackend/responseSchema``

### Related

- <doc:ToolsAndStructuredOutput>
