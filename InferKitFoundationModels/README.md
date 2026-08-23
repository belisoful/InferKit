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
- `NFKParameterTemperature` and `NFKParameterMaxTokens` map to `GenerationOptions`.
- The result carries text under `NFKOutputText`; `submitInferenceJob(for:)` streams partial text
  through the job's `partialResult` and honors cancellation.
- `isReady` mirrors `SystemLanguageModel.default.availability`; `prepare()` reports the reason when
  the model is unavailable (Apple Intelligence off, unsupported hardware, model not downloaded).

Multi-turn: prior turns seed a Foundation Models `Transcript` (system → `.instructions`, user →
`.prompt`, assistant → `.response`), and the final turn is the prompt, so the model sees a real
conversation rather than a flattened string. Verified live: given "my favorite color is teal" earlier
in the history, the model answers "Teal." to "what is my favorite color?".

### Tool calling

Register tools on `backend.tools` and the model calls them during generation when they fit the
question. A tool is defined at runtime — no compile-time `@Generable` type — with a name, a
description, typed parameters, and a handler that receives the model's arguments and returns text:

```swift
let backend = NFKFoundationModelsBackend()
backend.tools = [
    NFKFoundationTool(
        name: "get_temperature",
        description: "Get the current temperature for a city.",
        parameters: [
            NFKFoundationToolParameter(name: "city", description: "the city", type: .string, required: true),
        ],
        handler: { arguments in
            let city = arguments["city"] as? String ?? ""
            return "It is 21°C in \(city)."
        })
]
let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "How warm is it in Paris?"], parameters: nil)
let reply = try backend.runInference(for: request).output(forKey: NFKOutputText)
```

Each tool becomes an Apple `Tool` with a runtime `GenerationSchema` (`DynamicGenerationSchema` per
parameter); the model's arguments arrive as `GeneratedContent`, are read into a `[String: Any]`
(values are `String`, `Int`, `Double`, or `Bool`), and passed to the handler. Objective-C callers use
the synchronous `syncHandler:` initializer. Verified live: the model calls a registered tool and folds
its result into the reply.

### Structured output

Set `backend.responseSchema` to a list of typed fields and the model generates a structured result
matching them instead of free text — again with no compile-time `@Generable` type:

```swift
backend.responseSchema = [
    NFKFoundationToolParameter(name: "name", description: "the character's full name", type: .string, required: true),
    NFKFoundationToolParameter(name: "age", description: "the character's age in whole years", type: .integer, required: true),
]
let result = try backend.runInference(for: request)
let fields = result.output(forKey: NFKOutputStructured) as? [String: Any]  // ["name": "Aria Thompson", "age": 27]
let json = result.output(forKey: NFKOutputText) as? String
```

The result carries the parsed fields under `NFKOutputStructured` (a dictionary of `String` / `Int` /
`Double` / `Bool` values) and their JSON under `NFKOutputText`. `NFKFoundationToolParameter` doubles as
a schema field (name, description, type, required). Verified live: a name/age schema returns
`["name": "Aria Thompson", "age": 27]`.

## Provider bridge (planned, needs the macOS 27 / iOS 27 SDK)

WWDC26 introduced public provider protocols — `LanguageModel` (capabilities + executor
configuration) and `LanguageModelExecutor` (transcript in, streamed response out) — that let a
third-party model stand behind `LanguageModelSession`. Adopting them here would let InferKit's
backends (a converted Core ML model, a remote endpoint) serve any app written against Apple's
session API:

```swift
let session = LanguageModelSession(model: NFKInferKitLanguageModel(backend: coreMLBackend))
```

Those protocols are not in the macOS 26 SDK (verified against Xcode 26.6: `LanguageModel` is not a
resolvable type), so this direction lands when the macOS 27 / iOS 27 SDK is the build baseline. The
planned mapping: transcript entries → `NFKInputMessages`; `GenerationOptions` → the standard
`NFKParameter*` keys; the executor's streaming channel ← the job's `partialResult`.

## Build & test

```bash
cd InferKitFoundationModels
swift build
swift test    # generation tests skip where the system model is unavailable
```
