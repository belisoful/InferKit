# On-device and remote inference with InferKit

InferKit is a hub for inference on Apple platforms. One backend protocol,
[`NFKInferenceBackend`](../Sources/InferKit/include/InferKit/NFKInferenceBackend.h), fronts several
engines: a converted local Core ML language model, Apple's on-device Foundation Model, an
OpenAI-compatible remote endpoint, and MLX image models. A caller writes one request and swaps the
engine underneath.

This guide covers the contract, each engine, and how to use Apple's Foundation Models framework
(streaming, multi-turn, tool calling, structured output) both directly and through InferKit.

## Contents

- [The one contract](#the-one-contract)
- [Backends at a glance](#backends-at-a-glance)
- [Running a language model locally (Core ML)](#running-a-language-model-locally-core-ml)
- [Apple's Foundation Models](#apples-foundation-models)
  - [Availability](#availability)
  - [Text generation and options](#text-generation-and-options)
  - [Streaming](#streaming)
  - [Multi-turn conversation](#multi-turn-conversation)
  - [Tool calling: app tools for the model](#tool-calling-app-tools-for-the-model)
  - [Structured output](#structured-output)
- [Remote endpoints](#remote-endpoints)
- [Choosing and combining engines](#choosing-and-combining-engines)
- [Roadmap](#roadmap)

## The one contract

Every engine consumes an [`NFKInferenceRequest`](../Sources/InferKit/include/InferKit/NFKInferenceRequest.h)
and returns an [`NFKInferenceResult`](../Sources/InferKit/include/InferKit/NFKInferenceResult.h):

- **inputs** — the conditioning content and media, keyed by name. Text engines read `NFKInputPrompt`
  (a string) or `NFKInputMessages` (an OpenAI-style `[{role, content}]` array).
- **parameters** — scalar controls. Text engines read `NFKParameterTemperature`, `NFKParameterTopP`,
  `NFKParameterTopK`, `NFKParameterMaxTokens`, `NFKParameterRepetitionPenalty`,
  `NFKParameterStopSequences`, and `NFKParameterSeed`.
- **outputs** — the result, keyed by name: `NFKOutputText` (a string), `NFKOutputStructured` (a
  dictionary), `NFKOutputImage`, `NFKOutputVideo`.

The vocabulary lives in [`NFKInferenceKeys.h`](../Sources/InferKit/include/InferKit/NFKInferenceKeys.h).
A backend maps the shared keys to its provider's own names and ignores what it does not use.

Inference is multi-second, so run it off the render thread. `runInferenceForRequest:error:` blocks;
`submitInferenceJobForRequest:` returns an [`NFKInferenceJob`](../Sources/InferKit/include/InferKit/NFKInferenceJob.h)
that reports progress, streams partial results (`partialResult`), and cancels.

```objc
NFKInferenceRequest *request =
    [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Explain diffraction in one sentence." }
                                parameters:@{ NFKParameterMaxTokens: @64 }
                            outputModality:NFKModalityText];
NFKInferenceJob *job = NFKInferenceSubmit(backend, request, NULL);
job.progressHandler = ^(NFKInferenceJob *j) {
    NSString *sofar = [j.partialResult outputForKey:NFKOutputText];   // streamed text
    // update UI
};
job.completionHandler = ^(NFKInferenceJob *j) {
    NSString *text = [j.result outputForKey:NFKOutputText];
};
```

## Backends at a glance

| Backend | Where | Runs | Notes |
| --- | --- | --- | --- |
| `NFKCoreMLLanguageBackend` | core | A converted Core ML LLM, on device | 8 architectures, streaming, chat templates, batched prefill, int8. macOS 15 / iOS 18. |
| `NFKFoundationModelsBackend` | `InferKitFoundationModels/` | Apple's on-device system model | Streaming, multi-turn, tool calling, structured output. macOS 26 / iOS 26, Apple Intelligence. |
| `NFKRemoteBackend` | core | An OpenAI-compatible endpoint | Local server (Ollama, `mlx_lm`) or a hosted API. Foundation-only. |
| `NFKCoreMLBackend` | core | A Core ML image/tensor model | Image and MLMultiArray I/O. |
| `NFKMLXBackend` | `InferKitMLX/` | A bundled Stable Diffusion release | Text-to-image / image-to-image, on Apple Silicon (macOS and iOS). SD 1.5, SD 2.1 base, or SDXL-Turbo. |
| `NFKMLXModuleBackend` | `InferKitMLX/` | A bring-your-own MLX image model | RGB `CGImage ↔ MLXArray` around a forward closure. |
| `NFKMLXMattingBackend` | `InferKitMLX/` | A bring-your-own MLX matting model | Plate + optional hint in, foreground + alpha matte out. Matte key, premultiply, color space, tiling, texture output. |
| `NFKMLXTensorBackend` | `InferKitMLX/` | A bring-your-own MLX model over named tensors | Several image inputs in, several out. |

The MLX image backends share `NFKMLXImageBridge`, which converts a `CGImage` **or an `MTLTexture`** to
and from `MLXArray` in either direction (alpha-preserving), so a Metal render pipeline passes textures
straight through.
| `NFKPassthroughBackend` | core | Nothing — returns inputs | Keeps builds and tests green with no weights. |

## Running a language model locally (Core ML)

Two steps: convert a model once, then run it.

**Convert** a Hugging Face causal-LM checkpoint to a Core ML model directory with the offline tool in
[`Tools/inferkit-convert`](../Tools/inferkit-convert). It writes a stateful multifunction
`model.mlpackage` (KV cache as Core ML state; a `decode` function and a `prefill` function sharing
the weights), the tokenizer files, and a `manifest.json`. Validated architectures: `llama`, `qwen2`,
`mistral`, `gemma2`, `phi3`, `stablelm`, `starcoder2`, `gpt2`.

```bash
cd Tools/inferkit-convert
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python convert.py --model Qwen/Qwen2.5-0.5B-Instruct --output ../../Local/models/qwen --quantize int8
```

**Run** the directory through `NFKCoreMLLanguageBackend`:

```objc
NSURL *dir = [NSURL fileURLWithPath:@"…/qwen"];
NFKCoreMLLanguageBackend *backend = [NFKCoreMLLanguageBackend backendWithModelDirectoryURL:dir];
NSError *error = nil;
[backend prepareWithError:&error];

NSArray *messages = @[ @{ @"role": @"user", @"content": @"Name one color." } ];
NFKInferenceRequest *request =
    [NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: messages }
                                parameters:@{ NFKParameterMaxTokens: @64, NFKParameterTemperature: @0 }
                            outputModality:NFKModalityText];
NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
NSString *reply = [result outputForKey:NFKOutputText];   // "One color is blue."
```

The backend owns everything Core ML does not: tokenization (byte-level BPE, SentencePiece unigram, or
WordPiece per the manifest), the autoregressive sampling loop, chat-template rendering for
`NFKInputMessages`, and turn-stop at the assistant boundary. It prefills a prompt in chunks through
the `prefill` function and decodes token by token. `computeUnits` selects the Core ML compute units
(`MLComputeUnitsAll` by default and fastest; the Neural Engine cannot compile the stateful KV-cache
graph, so avoid `MLComputeUnitsCPUAndNeuralEngine`).

See [`Tools/inferkit-convert/README.md`](../Tools/inferkit-convert/README.md) for conversion detail.

## Apple's Foundation Models

Apple's Foundation Models framework (macOS 26 / iOS 26) exposes an on-device large language model
through Swift. `InferKitFoundationModels/` wraps it as `NFKFoundationModelsBackend`, so the same
InferKit request that runs against a converted model or a remote endpoint runs against Apple's model.
The sections below show the framework API directly and the InferKit mapping.

### Availability

The model needs Apple Intelligence enabled on supported hardware and the assets downloaded. Check
before use:

```swift
switch SystemLanguageModel.default.availability {
case .available:            break                       // ready
case .unavailable(let why): // .deviceNotEligible, .appleIntelligenceNotEnabled, .modelNotReady
    print(why)
}
```

`NFKFoundationModelsBackend.isReady` mirrors this, and `prepare()` throws
`NFKInferenceError.error_InferenceNotReady` with the reason when the model is unavailable.

### Text generation and options

Directly:

```swift
let session = LanguageModelSession(instructions: "Answer in one word.")
let response = try await session.respond(to: "Name a primary color.",
                                         options: GenerationOptions(temperature: 0.5, maximumResponseTokens: 16))
print(response.content)
```

Through InferKit: `NFKInputPrompt` becomes the prompt, a system message in `NFKInputMessages` becomes
the session instructions, and `NFKParameterTemperature` / `NFKParameterMaxTokens` map to
`GenerationOptions`. The reply arrives under `NFKOutputText`.

### Streaming

The session streams a growing snapshot; read `.content` off each partial:

```swift
for try await partial in session.streamResponse(to: prompt) {
    render(partial.content)          // the text so far
}
```

Through InferKit, `submitInferenceJobForRequest:` runs the stream and reports each snapshot as the
job's `partialResult` (text under `NFKOutputText`); cancelling the job cancels the task.

### Multi-turn conversation

A stateless InferKit request carries the whole conversation in `NFKInputMessages`. The backend seeds
a Foundation Models `Transcript` so the model sees a real conversation, not a flattened string:
a system message becomes an `.instructions` entry, prior user turns `.prompt` entries, prior
assistant turns `.response` entries, and the final turn is the prompt.

```swift
let transcript = Transcript(entries: [
    .instructions(Transcript.Instructions(segments: [.text(Transcript.TextSegment(content: "Be terse."))],
                                          toolDefinitions: [])),
    .prompt(Transcript.Prompt(segments: [.text(Transcript.TextSegment(content: "My favorite color is teal."))])),
    .response(Transcript.Response(assetIDs: [], segments: [.text(Transcript.TextSegment(content: "Got it."))])),
])
let session = LanguageModelSession(transcript: transcript)
let answer = try await session.respond(to: "In one word, my favorite color?")   // "Teal."
```

### Tool calling: app tools for the model

The model can call app-provided functions mid-generation. Apple's `Tool` protocol pairs a
description with a `@Generable` argument type; the framework runs the call-and-continue loop.
InferKit registers tools at **runtime** — no compile-time type — by building the schema dynamically:

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
// "How warm is it in Paris?" → the model calls get_temperature(city: "Paris") and folds the result in.
```

Under the hood each tool becomes an Apple `Tool` whose parameters are a runtime `GenerationSchema`
(a `DynamicGenerationSchema` per field); the model's arguments arrive as `GeneratedContent`, are read
into a `[String: Any]`, and passed to the handler. Objective-C callers use the synchronous
`syncHandler:` initializer.

### Structured output

Ask the model for typed fields instead of free text. Apple's `@Generable` marks a type the model
fills; InferKit does the runtime equivalent with `responseSchema`:

```swift
backend.responseSchema = [
    NFKFoundationToolParameter(name: "name", description: "the character's full name", type: .string, required: true),
    NFKFoundationToolParameter(name: "age", description: "the character's age in years", type: .integer, required: true),
]
let result = try backend.runInference(for: request)
let fields = result.output(forKey: NFKOutputStructured) as? [String: Any]   // ["name": "Aria Thompson", "age": 27]
let json = result.output(forKey: NFKOutputText) as? String
```

This calls `session.respond(to:schema:)` with a runtime `GenerationSchema` and reads the result out
of `GeneratedContent`. The parsed fields land under `NFKOutputStructured`; the JSON under
`NFKOutputText`.

## Remote endpoints

`NFKRemoteBackend` calls an OpenAI-compatible `/chat/completions` endpoint — a localhost server
(Ollama, `mlx_lm`, a BaseRT server) or a hosted API. One path serves both; the difference is the URL
and the key. A request supplies its prompt through `NFKInputMessages` (used as-is) or `NFKInputPrompt`
(wrapped as one user message); parameters fold into the request body; the reply arrives under the
backend's text key. It speaks the synchronous text protocol and blocks, so run it off the render
thread.

## Choosing and combining engines

- **Private, offline, no download infrastructure, latest OS** → `NFKFoundationModelsBackend` (Apple's
  model, zero weights to ship).
- **Private, offline, a specific open model or older OS** → `NFKCoreMLLanguageBackend` (convert the
  model you want; runs on macOS 15 / iOS 18).
- **A hosted or local server model** → `NFKRemoteBackend`.

Because all three adopt `NFKInferenceBackend`, a caller can pick at runtime — prefer Apple's model
when `isReady`, fall back to a converted local model, fall back to a remote endpoint — without
changing request-building code. Tool calling and structured output are Foundation-Models features
today; the text, message, streaming, and parameter contract is shared across all three.

## Roadmap

- **Provider bridge (WWDC26).** Apple's `LanguageModel` / `LanguageModelExecutor` protocols let a
  third-party model stand *behind* `LanguageModelSession`. Adopting them would let InferKit's local
  and remote backends serve any app written against Apple's session API. Those protocols are not in
  the macOS 26 SDK; this direction lands when macOS 27 / iOS 27 is the build baseline.
- **Exact-type structured output** through Apple's `@Generable` macro for compile-time result types
  (the runtime-schema path here covers the dynamic case).
