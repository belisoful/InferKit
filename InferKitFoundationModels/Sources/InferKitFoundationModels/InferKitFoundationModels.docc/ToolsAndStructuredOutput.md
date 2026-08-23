# Tools & structured output

Extend on-device generation with your own functions and constrain it to a typed shape — both defined at
runtime, without a compile-time `@Generable` type.

## Overview

The backend covers plain text generation on its own. Two additions let the model reach into your code
and return machine-readable results:

- **Tool calling** — you register ``NFKFoundationTool``s. The model decides when to call one, your
  handler runs and returns a value, and the model folds that value into its answer.
- **Structured output** — you set ``NFKFoundationModelsBackend/responseSchema``. Generation switches to a
  schema-constrained mode and the parsed fields arrive under `NFKOutputStructured`.

Both describe their shape with the same building block, ``NFKFoundationToolParameter`` (a name, a
description, a ``NFKToolParameterType``, and a required flag), so there is one vocabulary to learn.

![The tool-calling loop: the model calls your tool, the handler returns a value, and the model finishes the answer.](tool-calling)

### Registering a tool

A tool carries a name, a human-readable description the model reads to decide relevance, its parameters,
and a handler that receives the parsed arguments and returns a string.

```swift
let backend = NFKFoundationModelsBackend()
backend.tools = [
    NFKFoundationTool(
        name: "get_temperature",
        description: "Return the current temperature for a city.",
        parameters: [
            NFKFoundationToolParameter(name: "city", description: "the city", type: .string, required: true)
        ],
        handler: { arguments in
            let city = arguments["city"] as? String ?? "unknown"
            return "\(city): 21°C"
        }
    )
]
```

Under the hood the backend builds a runtime `GenerationSchema` per tool, so no `@Generable` Swift
type is declared. The model's arguments arrive already parsed in the handler's dictionary.

### Constraining the output

Setting `responseSchema` switches a run to `session.respond(to:schema:)`. The result carries the parsed
dictionary under `NFKOutputStructured` (reachable through `result.structured`) and the JSON text under
`NFKOutputText`.

```swift
backend.responseSchema = [
    NFKFoundationToolParameter(name: "name", description: "the person's name", type: .string, required: true),
    NFKFoundationToolParameter(name: "age", description: "the person's age", type: .integer, required: true),
]

let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "Alan Turing, mathematician, was 41."])
let fields = try backend.runInference(for: request).structured
```

## Topics

### Building blocks

- ``NFKFoundationTool``
- ``NFKFoundationToolParameter``
- ``NFKToolParameterType``
