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
- The result carries text under `NFKOutputText`; `submitInferenceJob(for:)` streams partial text
  through the job's `partialResult` and honors cancellation.
- `isReady` mirrors `SystemLanguageModel.default.availability`; `prepare()` reports the reason when
  the model is unavailable (Apple Intelligence off, unsupported hardware, model not downloaded) and
  warms the model up once. `contextSize` reports the tokens the context holds, and a request that
  needs more fails before the session runs, with both counts in the error's `userInfo`.

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

## Provider bridge (planned, in the macOS 27 / iOS 27 SDK)

WWDC26 introduced public provider protocols — `LanguageModel` (capabilities + executor
configuration) and `LanguageModelExecutor` (transcript in, streamed response out) — that let a
third-party model stand behind `LanguageModelSession`. Adopting them here would let InferKit's
backends (a converted Core ML model, a remote endpoint) serve any app written against Apple's
session API:

```swift
let session = LanguageModelSession(model: NFKInferKitLanguageModel(backend: coreMLBackend))
```

The protocols are in the macOS 27 / iOS 27 SDK (Xcode 27) and not in 26, so the bridge lands gated
to that OS with the package floor at 26. The planned mapping: transcript entries → `NFKInputMessages`;
`GenerationOptions` → the standard `NFKParameter*` keys; `enabledToolDefinitions` → `NFKParameterTools`;
`schema` → `NFKParameterJSONSchema`; the executor's streaming channel ← the job's `partialResult`.

## Build & test

```bash
cd InferKitFoundationModels
swift build
swift test    # generation tests skip where the system model is unavailable
```
