# The provider bridge

Run any InferKit backend behind `LanguageModelSession`, so an app written against Apple's session
API reaches a remote endpoint, a converted Core ML model, or an MLX model.

## Overview

``NFKFoundationModelsBackend`` presents Apple's models to an InferKit consumer.
``NFKInferKitLanguageModel`` runs the bridge the other way: it adopts Apple's provider protocols,
`LanguageModel` and `LanguageModelExecutor`, so a backend stands behind the session API an app
already uses.

```swift
let backend = NFKRemoteBackend(endpointURL: url)
backend.modelName = "qwen3:8b"

let session = LanguageModelSession(model: NFKInferKitLanguageModel(backend: backend))
let reply = try await session.respond(to: "Name three sea birds.")
```

The protocols are in the macOS 27 / iOS 27 SDK, so the model needs that OS and a build with that
SDK. The package floor stays at macOS 26 / iOS 26.

### What the session becomes

| Foundation Models | InferKit |
|---|---|
| `Transcript` instructions | a `system` message in `NFKInputMessages` |
| a prompt, a response | a `user` message, an `assistant` message |
| tool calls, a tool output | an `assistant` message carrying `tool_calls`, a `tool` message |
| an image attachment | `NFKInputImage`, or `NFKInputImages` for several |
| `enabledToolDefinitions` | `NFKParameterTools` (`{name, description, parameters}`) |
| `schema` | `NFKParameterJSONSchema` |
| `GenerationOptions.temperature`, `maximumResponseTokens` | `NFKParameterTemperature`, `NFKParameterMaxTokens` |
| the sampling mode | `NFKParameterTopK` or `NFKParameterTopP` with `NFKParameterSeed`; greedy is a temperature of zero |
| the executor's streaming channel | the job's `partialResult`, appended as it grows |
| the channel's tool calls | the result's `NFKOutputToolCalls` |

`.disallowed` tool calling drops the declarations. `.required` has no core key, so the declarations
go out and the backend decides. The core reports no token counts, so the counts on the channel are
zero.

### What the backend declares

The model reports the capabilities the backend declares through the core protocol's
`supportedParameterKeys` and `supportedInputKeys`:

| Declared key | Capability |
|---|---|
| `NFKParameterJSONSchema` | `.guidedGeneration` |
| `NFKParameterTools` | `.toolCalling` |
| `NFKInputImage` or `NFKInputImages` | `.vision` |

A session refuses what the model does not report, so a backend that takes no schema fails the
request rather than returning unconstrained text. A backend that declares no keys takes its
capabilities from the caller:

```swift
let capabilities = NFKInferKitLanguageModelCapabilities(guidedGeneration: true,
                                                        toolCalling: false,
                                                        vision: false)
let model = NFKInferKitLanguageModel(backend: backend, capabilities: capabilities)
```

`LanguageModelSession` is a Swift API, so the bridge is reached from Swift.
``NFKInferKitLanguageModelCapabilities`` is `@objc`, so an Objective-C caller reads what a backend
offers the Swift side of the same app.

## Topics

### The bridge

- ``NFKInferKitLanguageModel``
- ``NFKInferKitLanguageModelExecutor``
- ``NFKInferKitLanguageModelCapabilities``
