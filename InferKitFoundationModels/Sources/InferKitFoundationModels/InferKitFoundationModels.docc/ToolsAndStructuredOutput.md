# Tools & structured output

Extend on-device generation with your own functions and constrain it to a JSON Schema, both through
the core's request keys, without a compile-time `@Generable` type.

## Overview

The backend covers plain text generation on its own. Two additions let the model reach into your code
and return machine-readable results, and both use the keys a remote backend or the MLX language
backend reads, so one request runs against any of them:

- **Tool calling** — `NFKParameterTools` declares the tools a request offers; an ``NFKFoundationTool``
  registered on ``NFKFoundationModelsBackend/tools`` supplies the handler. The model decides when to
  call one, the handler runs and returns a value, and the model folds that value into its answer.
- **Structured output** — `NFKParameterJSONSchema` switches generation to a schema-constrained mode,
  and the parsed object arrives under `NFKOutputStructured`. `NFKParameterChoices` constrains the reply
  to one of its strings.

![The tool-calling loop: the model calls your tool, the handler returns a value, and the model finishes the answer.](tool-calling)

### Registering a tool

A tool carries a name, a description the model reads to decide relevance, a JSON Schema object for
its arguments (the same dictionary an `NFKParameterTools` entry carries), and a handler that receives
the parsed arguments and returns a string.

```swift
let backend = NFKFoundationModelsBackend()
backend.tools = [
    NFKFoundationTool(
        name: "get_temperature",
        description: "Return the current temperature for a city.",
        parameters: ["type": "object",
                     "properties": ["city": ["type": "string", "description": "the city"]],
                     "required": ["city"]],
        handler: { arguments in
            let city = arguments["city"] as? String ?? "unknown"
            return "\(city): 21°C"
        }
    )
]
```

A request without `NFKParameterTools` offers every registered tool. A request with the key offers its
own declarations and takes handlers from the registered tools by name. A declared tool with no handler
keeps the remote contract: when the model calls it, the turn ends with the call under
`NFKOutputToolCalls`, the caller runs the tool, and the next request carries the assistant `tool_calls`
message and a `tool` message with the result, which seed the transcript as `.toolCalls` and
`.toolOutput` entries.

Under the hood the backend builds a runtime `GenerationSchema` per tool from the JSON Schema, so no
`@Generable` Swift type is declared. The model's arguments arrive already parsed in the handler's
dictionary.

### Constraining the output

`NFKParameterJSONSchema` switches a run to `session.streamResponse(to:schema:)`. The result carries the
parsed object under `NFKOutputStructured` (reachable through `result.structured`) and the JSON text under
`NFKOutputText`.

```swift
let request = NFKInferenceRequest(
    inputs: [NFKInputPrompt: "Alan Turing, mathematician, was 41."],
    parameters: [NFKParameterJSONSchema: [
        "type": "object",
        "properties": ["name": ["type": "string", "description": "the person's name"],
                       "age": ["type": "integer", "description": "the person's age"]],
        "required": ["name", "age"],
    ]])
let fields = try backend.runInference(for: request).structured
```

The supported subset: object (`properties`, `required`), array (`items`, `minItems`, `maxItems`),
string (`pattern`, `const`, `enum`), integer and number (`minimum`, `maximum`), boolean, `anyOf` /
`oneOf`, `$ref` into `$defs`, and `description`. A keyword outside it is refused by path rather than
dropped, because a dropped keyword changes what the model may produce. `NFKParameterOutputFormat`
(JSON of no particular shape) is refused for the same reason.

## Topics

### Building blocks

- ``NFKFoundationTool``
