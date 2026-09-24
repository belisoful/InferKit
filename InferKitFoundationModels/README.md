# InferKitFoundationModels

A companion SwiftPM package bridging InferKit and Apple's Foundation Models framework
(macOS 26 / iOS 26; the model itself needs Apple Intelligence enabled on supported hardware).

## NFKFoundationModelsBackend (shipped)

Wraps the on-device system language model (`LanguageModelSession`) as an `NFKInferenceBackend`, so
an InferKit consumer swaps it in like any other engine. The same request runs against
`NFKCoreMLLanguageBackend` (a converted local model), `NFKRemoteBackend` (an OpenAI-compatible
endpoint), or this backend (Apple's model).

- `NFKInputPrompt` (string) or `NFKInputMessages` (OpenAI-style array); a system message becomes
  the session's instructions.
- `NFKParameterTemperature` and `NFKParameterMaxTokens` map to `GenerationOptions`. `NFKParameterTopK`,
  `NFKParameterTopP`, and `NFKParameterSeed` choose the sampling mode, and a temperature of zero is
  greedy decoding.
- On macOS 27 / iOS 27: `NFKInputImage` and `NFKInputImages` attach to the prompt, and
  `NFKParameterReasoningEffort` (`NFKReasoningEffortLight`, `…Moderate`, `…Deep`) becomes
  `ContextOptions.reasoningLevel`. Below 27 the framework has neither, so the backend leaves the
  keys out of what it declares and refuses a request that carries one.
- The result carries text under `NFKOutputText`, and on macOS 27 / iOS 27 the reasoning the model
  showed under `NFKOutputReasoning` and what the turn cost under `NFKOutputUsage`;
  `submitInferenceJob(for:)` streams partial text through the job's `partialResult` and honors
  cancellation.
- `isReady` mirrors the chosen model's availability; `prepare()` reports the reason when the model
  is unavailable (Apple Intelligence off, unsupported hardware, model not downloaded, Private Cloud
  Compute quota reached) and warms the model up once. `contextSize` reports the tokens the context
  holds, and an on-device request that needs more fails before the session runs, with both counts in
  the error's `userInfo`.

### Choosing the model

`model` picks the Apple model a request runs on. `.onDevice` (the default) is the system language
model; `useCase` (`.general`, `.contentTagging`) and `guardrails` (`.default`,
`.permissiveContentTransformations`) specialize it. `.privateCloudCompute` is Apple's larger model on
Private Cloud Compute (macOS 27 / iOS 27): the request leaves the device, and usage counts against a
quota. Below macOS 27 a Private Cloud Compute backend is not ready, and a request fails with
`kNFKError_InferenceUnsupported` rather than running on the device unasked.

```swift
let backend = NFKFoundationModelsBackend()
backend.useCase = .contentTagging
if #available(macOS 27, iOS 27, *), let quota = backend.privateCloudComputeQuota, !quota.isLimitReached {
    backend.model = .privateCloudCompute
}
```

`privateCloudComputeQuota` (macOS 27 / iOS 27) reads the quota whatever `model` is set to:
`isLimitReached`, `isApproachingLimit`, `resetDate`, and `showLimitIncreaseSuggestion()`; it is nil
when the package was built with an SDK before macOS 27, which has no Private Cloud Compute. A reached
quota makes the backend not ready, and `prepare()` throws with the reset date under
`NFKFoundationModelsErrorKey.resetDate`. `variantDisplayName` (macOS 27 / iOS 27) names the on-device
model's variant. A request captures the model when it is submitted, so changing `model` does not move
a running request. Verified live: the content-tagging model answers "photography, emotion, nature"
to a sentence about a hiker at a glacier.

Multi-turn: prior turns seed a Foundation Models `Transcript` (system → `.instructions`, user →
`.prompt`, assistant → `.response`, assistant `tool_calls` → `.toolCalls`, `tool` → `.toolOutput`),
and the last user turn is the prompt, so the model sees a real conversation rather than a flattened
string. Verified live: given "my favorite color is teal" earlier in the history, the model answers
"Teal." to "what is my favorite color?".

Every option is a core request key, so the request that runs against `NFKRemoteBackend` or the MLX
language backend runs here unchanged.

### Tool calling

A tool is declared the way a remote backend takes it under `NFKParameterTools`: a name, a description
the model reads to decide relevance, and a JSON Schema object for its arguments. Registering an
`NFKFoundationTool` on `backend.tools` adds the handler, which receives the parsed arguments and
returns text the model reads before continuing:

```swift
let backend = NFKFoundationModelsBackend()
backend.tools = [
    NFKFoundationTool(
        name: "get_temperature",
        description: "Get the current temperature for a city.",
        parameters: ["type": "object",
                     "properties": ["city": ["type": "string", "description": "the city"]],
                     "required": ["city"]],
        handler: { arguments in
            let city = arguments["city"] as? String ?? ""
            return "It is 21°C in \(city)."
        })
]
let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "How warm is it in Paris?"], parameters: nil)
let reply = try backend.runInference(for: request).output(forKey: NFKOutputText)
```

A request without `NFKParameterTools` offers every registered tool. A request with the key offers
its own declarations and takes handlers from the registered tools by name. A declared tool with no
handler is the remote backend's contract: when the model calls it, the turn ends with the call under
`NFKOutputToolCalls` (`{id, name, arguments, argumentsJSON}`), the caller runs the tool, and the next
request carries the assistant `tool_calls` message and a `tool` message with the result. Verified
live in both forms: the model calls a registered tool and folds its result into the reply, and a
`tool` message with "7391" in it produces "7391".

Each tool becomes an Apple `Tool` with a runtime `GenerationSchema` built from the JSON Schema; no
compile-time `@Generable` type is needed. Objective-C callers use the synchronous `syncHandler:`
initializer.

### Structured output

`NFKParameterJSONSchema` constrains generation to a JSON Schema object, again with no compile-time
`@Generable` type:

```swift
let request = NFKInferenceRequest(
    inputs: [NFKInputPrompt: "Invent a fictional character."],
    parameters: [NFKParameterJSONSchema: [
        "type": "object",
        "properties": ["name": ["type": "string", "description": "the character's full name"],
                       "age": ["type": "integer", "description": "the character's age", "minimum": 20, "maximum": 40]],
        "required": ["name", "age"],
    ]])
let result = try backend.runInference(for: request)
let fields = result.structured   // ["name": "Elara Windrider", "age": 28]
let json = result.text           // the same as JSON
```

The supported JSON Schema subset: object (`properties`, `required`), array (`items`, `minItems`,
`maxItems`), string (`pattern`, `const`, `enum`), integer and number (`minimum`, `maximum`), boolean,
`anyOf` / `oneOf`, `$ref` into `$defs`, and `description`. A keyword outside it is refused by path
rather than dropped. `NFKParameterChoices` constrains the reply to exactly one of its strings.
`NFKParameterOutputFormat` (JSON of no particular shape) is refused, since guided generation needs a
schema. Verified live: a name / age / traits schema returns `["name": "Elara Windrider", "age": 28,
"traits": […]]`, and `["yes", "no", "unsure"]` returns `yes`.

## What a failure means

Every failure arrives as an `NSError` in `NFKInferenceErrorDomain`, the way a remote endpoint's does,
with the framework's own error under `NSUnderlyingErrorKey`. The code carries the decision:

| Foundation Models | Code | What an app does |
|---|---|---|
| `guardrailViolation`, `refusal` | `kNFKError_InferenceRefused` | Change the request. Retrying gives the same answer. |
| `rateLimited`, a reached Private Cloud Compute quota | `kNFKError_InferenceRateLimited` | Back off. The reset date is under `NFKFoundationModelsErrorKey.resetDate` when the service names one. |
| `contextSizeExceeded`, `unsupportedCapability`, `unsupportedGenerationGuide`, `unsupportedLanguageOrLocale` | `kNFKError_InferenceUnsupported` | Shorten or simplify the request. |
| `assetsUnavailable` | `kNFKError_InferenceNotReady` | Wait for the model, and ask `prepare()` why. |
| A Private Cloud Compute network failure | `kNFKError_RemoteUnreachable` | Retry, or fall back to the on-device model. |
| Anything else | `kNFKError_InferenceBackendFailure` | Report it. |

On macOS 27 a context overflow carries the tokens the request needed and the tokens the context holds
under `NFKFoundationModelsErrorKey`. The macOS 26 error reports neither, which is why the backend
counts tokens itself before the session runs.

## Provider bridge (macOS 27 / iOS 27)

`NFKInferKitLanguageModel` runs the bridge the other way: an InferKit backend stands behind
`LanguageModelSession`, so an app written against Apple's session API reaches a remote endpoint, a
converted Core ML model, or any other `NFKInferenceBackend`.

```swift
let backend = NFKRemoteBackend(endpointURL: url)
let session = LanguageModelSession(model: NFKInferKitLanguageModel(backend: backend))
let reply = try await session.respond(to: "Name three sea birds.")
```

The mapping: transcript entries → `NFKInputMessages` (instructions to a system message, tool calls
and outputs to the `tool_calls` and `tool` shapes, attached images to `NFKInputImage`);
`GenerationOptions` → `NFKParameterTemperature`, `NFKParameterMaxTokens`, and the sampling keys;
`enabledToolDefinitions` → `NFKParameterTools`; `schema` → `NFKParameterJSONSchema`;
`ContextOptions.reasoningLevel` → `NFKParameterReasoningEffort`; the job's `partialResult` → the
executor's streaming channel, its `NFKOutputReasoning` → the channel's reasoning, its
`NFKOutputToolCalls` → the channel's tool calls, and its `NFKOutputUsage` → the channel's token
counts at the end of the turn.

The model reports the capabilities the backend declares through the core protocol's
`supportedParameterKeys` and `supportedInputKeys`: `NFKParameterJSONSchema` is guided generation,
`NFKParameterTools` is tool calling, `NFKInputImage` is vision. A session refuses what the backend
does not declare. A backend that declares no keys takes them from the caller:
`NFKInferKitLanguageModel(backend:capabilities:)`, whose `NFKInferKitLanguageModelCapabilities` an
Objective-C caller reads as well.

The provider protocols are in the macOS 27 / iOS 27 SDK and not in 26, so the type needs that OS
and a build with the macOS 27 SDK. The package floor stays at 26. The token counts are the turn's
totals, which arrive when the turn is over, so each appended fragment carries a count of zero and a
backend that reports no counts sends none.

## What stays Swift

Every option the backend supports is a core request key, so an Objective-C app configures the model
the way it configures any other engine. Six parts of the framework cannot become a key, because a
result builder is a closure over types, a macro runs at compile time, and a generic needs a type the
caller names:

| Swift-only | What the contract offers instead |
|---|---|
| `DynamicInstructions`, `LanguageModelSession.Profile` | A `system` message in `NFKInputMessages`, with `NFKParameterTools` and the sampling keys |
| `@Generable(name:)` | `NFKParameterJSONSchema`, named in the JSON, parsed under `NFKOutputStructured` |
| `ImageReference` | `NFKInputImage` and `NFKInputImages` carry the pictures; naming one in a reply needs Swift |
| Session properties (`isResponding`, `transcript`, `usage`, `prewarm(promptPrefix:)`) | The job's `partialResult`, the messages the caller holds, `NFKOutputUsage`, and `prepare()` |
| `transcriptErrorHandlingPolicy` | The caller owns the conversation and decides what to resend after a failure |

The list is closed: anything else the framework offers has a key. A Swift app that wants these uses
`LanguageModelSession` directly and loses nothing, because the provider bridge puts an InferKit
backend behind that same session.

## Build & test

```bash
cd InferKitFoundationModels
swift build
swift test    # generation tests skip where the system model is unavailable
```
