# On-device and remote inference with InferKit

InferKit is a hub for inference on Apple platforms. One backend protocol,
[`NFKInferenceBackend`](../Sources/InferKit/include/InferKit/NFKInferenceBackend.h), fronts several
engines: a converted local Core ML language model, an MLX language model read straight from its
Hugging Face release, Apple's on-device Foundation Model, a named remote provider (OpenAI-compatible or
Anthropic), and the MLX image, audio, video, and music models. A caller writes one request and swaps
the engine underneath.

This guide covers the contract, each text engine, how to use Apple's Foundation Models framework
(streaming, multi-turn, tool calling, structured output) both directly and through InferKit, and where
the toolkit is going. [Examples](examples.md) has a compiled snippet for every backend and modality;
[Companion packages](companions.md) has the full MLX model gallery.

## Contents

- [The one contract](#the-one-contract)
- [Backends at a glance](#backends-at-a-glance)
- [Running a language model locally (Core ML)](#running-a-language-model-locally-core-ml)
- [Running a language model locally (MLX)](#running-a-language-model-locally-mlx)
- [Apple's Foundation Models](#apples-foundation-models)
  - [Availability](#availability)
  - [Text generation and options](#text-generation-and-options)
  - [Streaming](#streaming)
  - [Multi-turn conversation](#multi-turn-conversation)
  - [Tool calling: app tools for the model](#tool-calling-app-tools-for-the-model)
  - [Structured output](#structured-output)
- [Remote providers](#remote-providers)
- [Speech in, text out](#speech-in-text-out)
- [Optional engines discovered at runtime](#optional-engines-discovered-at-runtime)
- [Choosing and combining engines](#choosing-and-combining-engines)
- [Roadmap](#roadmap)

## The one contract

Every engine consumes an [`NFKInferenceRequest`](../Sources/InferKit/include/InferKit/NFKInferenceRequest.h)
and returns an [`NFKInferenceResult`](../Sources/InferKit/include/InferKit/NFKInferenceResult.h):

- **inputs** — the conditioning content and media, keyed by name. Text engines read `NFKInputPrompt`
  (a string) or `NFKInputMessages` (an OpenAI-style `[{role, content}]` array).
- **parameters** — scalar controls. Text engines read `NFKParameterTemperature`, `NFKParameterTopP`,
  `NFKParameterTopK`, `NFKParameterMaxTokens`, `NFKParameterRepetitionPenalty`,
  `NFKParameterStopSequences`, and `NFKParameterSeed`. Each engine honors the subset it can and
  ignores the rest.
- **outputs** — the result, keyed by name: `NFKOutputText` (a string), `NFKOutputStructured` (a
  dictionary), `NFKOutputEmbedding`, `NFKOutputImage`, `NFKOutputAudio`, `NFKOutputVideo`, and the
  typed lists (`NFKOutputDetections`, `NFKOutputPose`, `NFKOutputClassifications`,
  `NFKOutputSegments`).

The vocabulary lives in [`NFKInferenceKeys.h`](../Sources/InferKit/include/InferKit/NFKInferenceKeys.h).
A backend maps the shared keys to its provider's own names and ignores what it does not use. The keys
with a single natural type have typed accessors (`result.text`, `result.structured`,
`result.embedding`, `result.toolCalls`, `request.prompt`, `request.messages`), each returning nil on
a type mismatch rather than crashing.

Inference is multi-second, so run it off the render thread. `runInferenceForRequest:error:` blocks;
`submitInferenceJobForRequest:` returns an [`NFKInferenceJob`](../Sources/InferKit/include/InferKit/NFKInferenceJob.h)
that reports progress, streams partial results (`partialResult`), and cancels. The asynchronous path
runs at user-initiated quality of service, so a decode loop is not scheduled onto the efficiency cores.

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

The core ships only backends built on Apple frameworks, with no third-party dependency. The two
companion packages add heavier engines without raising the core's platform floor.

| Backend | Where | Runs | Notes |
| --- | --- | --- | --- |
| `NFKCoreMLLanguageBackend` | core | A converted Core ML LLM, on device | Streaming, chat templates, chunked prefill, int8. macOS 15 / iOS 18. |
| `NFKMLXLanguageBackend` | `InferKitMLX/` | A Hugging Face LLM release, on device through MLX | Qwen3 / Llama dense decoders, Gemma 4, the Qwen3.5 hybrid, each at reference parity. Cache window and quantization, chunked prefill, ChatML. Apple Silicon, macOS 14 / iOS 17. |
| `NFKFoundationModelsBackend` | `InferKitFoundationModels/` | Apple's on-device system model | Streaming, multi-turn, tool calling, structured output. macOS 26 / iOS 26, Apple Intelligence. |
| `NFKRemoteBackend` | core | An OpenAI-compatible endpoint | Twelve named presets, hosted and local (Ollama, LM Studio, llama.cpp, vLLM). Foundation-only. |
| `NFKAnthropicBackend` | core | Anthropic's Messages API | Its own backend, because the protocol differs; the same request shape. |
| `NFKRemoteTranscriptionBackend` | core | An OpenAI-compatible audio→text (Whisper) endpoint | `NFKInputAudio` in, `NFKOutputText` out. |
| `NFKRemoteEmbeddingBackend` | core | An OpenAI-compatible embeddings endpoint | `NFKInputPrompt` in, `NFKOutputEmbedding` out; batches through `embeddingsForTexts:error:`. |
| `NFKRemoteSpeechBackend` | core | An OpenAI-compatible text→speech endpoint | `NFKInputPrompt` in, an `NFKAudioAsset` (WAV by default) under `NFKOutputAudio`. A voice is required. |
| `NFKRemoteImageBackend` | core | OpenAI-compatible image generation and edits | Prompt → image; `NFKInputImage` → edit; `+ NFKInputMask` → inpaint. 32BGRA `CVPixelBuffer` under `NFKOutputImage`. |
| `NFKRemoteVideoBackend` | core | OpenAI's videos API (job-style) | Prompt, or prompt + reference image → an `.mp4` `NFKVideoAsset` under `NFKOutputVideo`; submit, poll, download on `NFKAsyncGenerationBackend`. |
| `NFKRemoteModerationBackend` | core | An OpenAI-compatible moderation endpoint | Text (and an image) → per-category `NFKClassification`s and the verdict under `NFKOutputStructured`. |
| `NFKRemoteReranker` | core | A hosted rerank endpoint | Query + documents → scores; the same shape as the on-device reranker. A scoring object, not a backend. |
| `NFKAsyncGenerationBackend` | core | A submit → poll → fetch generation service | Subclass and fill in the five request-shape methods. |
| `NFKCoreMLBackend` | core | A Core ML image/tensor model | Image and MLMultiArray I/O; `NFKComputePlan` reports where each operation lands. |
| `NFKMLXBackend` | `InferKitMLX/` | A bundled Stable Diffusion release | Text-to-image / image-to-image. SD 1.5, SD 2.1 base, or SDXL-Turbo, at end-to-end parity. |
| `NFKMLXWhisperBackend` | `InferKitMLX/` | Whisper, on device | Tiny through large-v3, optional timestamps under `NFKOutputSegments`. |
| `NFKMLXMusicBackend` | `InferKitMLX/` | MiniMax Music 3 | `NFKInputPrompt` + `NFKInputLyrics` → stereo 44.1 kHz audio. Restrictive weight license. |
| `NFKMLXModuleBackend` / `MattingBackend` / `TensorBackend` / `DiffusionBackend` / `SpeechBackend` / `VideoBackend` | `InferKitMLX/` | A bring-your-own MLX model | The InferKit contract around a forward closure; the shipped model gallery is built on these. |
| `NFKPassthroughBackend` | core | Nothing — returns inputs | Keeps builds and tests green with no weights. |

The MLX image backends share `NFKMLXImageBridge`, which converts a `CGImage` **or an `MTLTexture`** to
and from `MLXArray` in either direction (alpha-preserving), so a Metal render pipeline passes textures
straight through.

## Running a language model locally (Core ML)

Two steps: convert a model once, then run it. [Core ML language models](coreml-llm.md) is the
full walkthrough.

**Convert** a Hugging Face causal-LM checkpoint to a Core ML model directory with the offline tool in
[`Tools/inferkit-convert`](../Tools/inferkit-convert). It writes a stateful multifunction
`model.mlpackage` (KV cache as Core ML state; a `decode` function and a `prefill` function sharing
the weights), the tokenizer files, and a `manifest.json`. Validated architectures: `llama`, `qwen2`,
`mistral`, `gemma2`, `phi3`, `stablelm`, `starcoder2`, `gpt2`. Mixture-of-experts models do not
convert; see the MLX path for where that is heading.

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
the `prefill` function and decodes token by token. `computeUnits` selects the Core ML compute units.
`MLComputeUnitsAll` is the default and the fastest measured; a compute plan shows the converted
model placed on the GPU rather than the Neural Engine, because a multifunction package takes one
placement decision and the single-token decode function is not Neural Engine eligible.
[`NFKComputePlan`](../Sources/InferKit/include/InferKit/NFKComputePlan.h) is how to measure that for
any model rather than assume it.

## Running a language model locally (MLX)

`InferKitMLX/` reads a released Hugging Face directory directly: its `config.json`, tokenizer files,
and safetensors shards. The module's parameter names are the checkpoint's, so nothing is remapped, and
a raw PyTorch `.pth`/`.bin` loads through the native checkpoint reader with no Python toolchain.

Four decoder families are implemented, each measured against `transformers`' own implementation on
released weights: the dense decoder Qwen3 and Llama share (`NFKMLXLanguage`), which also reads the
Qwen3-MoE and Mixtral mixtures of experts through a routed feed-forward, Gemma 4
(`NFKMLXGemmaLanguage`), the Qwen3.5 / 3.6 / 3.8 hybrid with its gated delta-rule recurrence
(`NFKMLXHybridLanguage`), and DeepSeek V4's latent attention over a mixture of experts
(`NFKMLXDeepSeek`, implemented and measured at a small configuration; the released weights do not fit
a 32 GB machine).

```swift
let backend = try NFKMLXLanguage.backend(directoryURL: releaseDirectory)
let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "Explain diffraction in one sentence."])
let text = try backend.runInference(for: request).text
```

```objc
id<NFKInferenceBackend> backend = [NFKMLXLanguage backendWithDirectoryURL:releaseDirectory error:&error];
NFKInferenceRequest *request =
    [NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: messages }
                                parameters:@{ NFKParameterMaxTokens: @256,
                                              NFKMLXGenerationParameterKey.chatTemplate: @"chatml",
                                              NFKMLXGenerationParameterKey.contextWindow: @4096 }
                            outputModality:NFKModalityText];
```

The generation options beyond the core's sampling parameters are `NFKMLXGenerationOptions` in Swift
and the `NFKMLXGenerationParameterKey` request keys in Objective-C, the same set either way:

- **`contextWindow`** bounds the key-value cache by dropping the oldest positions. Exact while the
  conversation fits inside the window; an approximation past it, so off by default.
- **`cacheQuantization`** stores the cache at 8 or 4 bits instead of a float per element. 8-bit was
  measured to track the full-precision logits at cosine above 0.99.
- **`prefillChunkSize`** runs a long prompt through the cache in slices, bounding the attention peak.
  Exact, pinned by a test against the single-pass result.
- **`chatTemplate`** renders `NFKInputMessages` in ChatML with the release's own special tokens, which
  is what an instruct release was trained on. Off by default, because a base model wants plain text.
- **`reusesPromptCache`** keeps the key-value cache between requests, so a conversation's next turn
  prefills only what it adds. Exact, since the cache rolls back to where the prompts diverge.
- **`draftTokens`**, with a backend built from a main release and a draft release, decodes
  speculatively: the draft proposes, the model verifies in one pass, and the output is the model's own.
- **`jsonOutput`** and **`choices`** constrain sampling through a grammar mask, so the reply is
  well-formed JSON or exactly one of a fixed set of answers.

Generation stops at the release's end-of-sequence token unless the request names its own stop tokens.

Two things are refused rather than approximated, because both load cleanly and produce fluent
nonsense: a mixture-of-experts or hybrid config handed to the dense loader, and a `rope_scaling` kind
the port does not implement (`dynamic`, `longrope`; `linear`, `yarn`, and `llama3` are implemented).
A release whose weights exceed the memory budget is refused before any are materialized
(`NFKMLXReleaseWeights.verifyFits`), and `NFKMLXModelSizing` derives the context window from what the
machine can hold and reports the decode rate memory bandwidth allows. `NFKMLXQuantization` packs a
model's linear layers to 4- or 8-bit for a release that would not otherwise fit.

The [MLX language section of the examples](examples.md#local-on-device-through-mlx-nfkmlxlanguagebackend-swift)
shows each option in use.

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

## Remote providers

[`NFKRemoteProvider`](../Sources/InferKit/include/InferKit/NFKRemoteProvider.h) names the services a
consumer is likely to call, so pointing at one is an identifier rather than a hand-typed URL. Every
preset carries its API base and protocol, derives each operation's URL from the base (`endpointURL`
for chat, `modelsURL` for the list, `URLForPath:` for anything else), and deliberately carries
**no default model name**: model identifiers change faster than a release does.
`modelsWithAPIKey:error:` (and its completion-handler form) asks the provider for its current list as
`NFKRemoteModel`s, so an app populates a picker from the server rather than from a constant; every
preset answers the same envelope, and Anthropic's pagination is followed to the end. A local runner
that is not running fails with `kNFKError_RemoteUnreachable`, which is a different answer from an
empty list. `providerWithBaseURL:` re-points a preset at another port or another machine, keeping its
identity and protocol.

- **OpenAI-compatible**, served by `NFKRemoteBackend`: `openai`, `xai`, `gemini`, `groq`, `mistral`,
  `deepseek`, `together`, `openrouter`, and the local servers `ollama`, `lmstudio`, `llamacpp`,
  `vllm`. A local server needs no key, and llama.cpp needs no model name either, since it answers for
  whatever it has loaded.
- **`anthropic`**, served by `NFKAnthropicBackend`: the key travels as `x-api-key`, an
  `anthropic-version` header is required, `max_tokens` is required, and a system prompt is a top-level
  field. The backend lifts a leading system turn out of `NFKInputMessages` into that field, so the
  request is written the same way for both.

```objc
NFKRemoteProvider *provider = [NFKRemoteProvider providerWithIdentifier:@"ollama"];
id<NFKInferenceBackend> backend = [NFKRemoteProvider backendForProvider:provider
                                                                apiKey:nil
                                                             modelName:@"llama3.2"];
```

`backendForProvider:apiKey:modelName:` returns whichever backend the provider's protocol needs, so
switching from a local server to Anthropic changes one argument. Both backends speak the synchronous
text protocol and block, so run them off the render thread. `NFKRemoteTranscriptionBackend` is the
audio→text counterpart for an OpenAI-compatible transcription endpoint, and
`NFKAsyncGenerationBackend` is the base for a service that answers with a job identifier to poll.
`NFKRemoteEmbeddingBackend` is the embeddings counterpart (`POST /embeddings`, which the hosted
providers and every local runner serve): `NFKInputPrompt` in, the vector under `NFKOutputEmbedding`
out, the same key the on-device embedders in InferKitMLX answer with, so search or clustering code does
not change with the engine; `embeddingsForTexts:error:` embeds a batch in one call.

**Local runners have a second surface.** The OpenAI-compatible endpoints say nothing about what is
installed, what is loaded, how large a model is, or how to get one. `-[NFKRemoteProvider localRunner]`
hands back an adapter over the runner's native API (`NFKOllamaRunner`, `NFKLMStudioRunner`; llama.cpp
and vLLM have nothing beyond the OpenAI surface, so they answer nil), each adopting `NFKLocalModelRunner`:
`isRunning`, `installedModelsWithError:` (size, quantization, context length, and capabilities per
model), `loadedModelsWithError:`, and `detailsForModel:error:`. The optional actions change the machine
and are offered only where the runner has them, so a caller checks `respondsToSelector:` before showing
a button: Ollama's `pullModel:` streams a download as an `NFKInferenceJob` (progress, a status line
under `partialResult`, cancellation), and `deleteModel:error:` removes weights. Shapes were measured
against Ollama 0.33.

**The other directions have remote backends too.** `NFKRemoteSpeechBackend` (text → an `NFKAudioAsset`
under `NFKOutputAudio`, `POST /audio/speech`) and `NFKRemoteImageBackend` (a prompt → image, an image →
an edit of it, an image and mask → an inpaint; `POST /images/generations` and `/images/edits`) answer
with the keys the on-device speech and Stable Diffusion backends use, so a feature built on one engine
falls back to a hosted one without leaving the contract. Which presets serve each path was verified by
probe at release and is recorded on each factory. An image beside a prompt is a vision question through
the ordinary chat backends: `NFKRemoteBackend` sends it as an inline `image_url` content part, which the
local runners' vision models read too (measured against Ollama with `qwen3.5:27b`), and
`NFKAnthropicBackend` as an `image` block. `NFKImageCoding` is the public codec under all of it —
`CGImage`, `CVPixelBuffer`, or `MTLTexture` to PNG, and any ImageIO-readable bytes back to a 32BGRA
pixel buffer.

**The chat backends stream, and a cancelled job cancels the request.** `submitInferenceJobForRequest:`
on both sends the request with streaming on and reads server-sent events into the job's
`partialResult` token by token, so a chat interface on a remote provider fills in exactly as it does
on the on-device engines; `[job cancel]` closes the connection, so an abandoned completion stops
costing then. Tools go under `NFKParameterTools` (`{name, description, parameters}`, translated into
each provider's shape) and what the model called comes back under `NFKOutputToolCalls`
(`result.toolCalls`, arguments parsed); a JSON Schema under `NFKParameterJSONSchema` comes back parsed
under `NFKOutputStructured` — a `json_schema` response format on the OpenAI shape, a forced tool on
Anthropic's, which has none. Several images go under `NFKInputImages`. Every blocking remote call
retries a rate limit or gateway error after the provider's `Retry-After` or an exponential delay,
bounded by `NFKRemoteTransport.retryAttempts` and `maximumRetryDelay`; a refused connection is not
retried. Streaming and the tool call are measured against a live Ollama.

**Audio, documents, and video in; speech out.** Beside images, the chat backends take `NFKInputAudio`
(an `input_audio` part; the Messages API refuses audio rather than dropping it), `NFKInputDocument` /
`NFKInputDocuments` (PDFs, as `file` parts or `document` blocks), and `NFKInputVideo`, which
`NFKVideoSampling` turns into `NFKParameterVideoFrameCount` evenly spaced frames for a vision model —
measured against a live Ollama, which names the colours of a sampled clip. `NFKParameterAudioOutput`
asks an OpenAI-compatible chat model to speak its reply, which arrives as an `NFKAudioAsset` under
`NFKOutputAudio` beside the text, streamed or not. The transcription backend gains `emitsTimestamps`
(segments under `NFKOutputSegments`, as the on-device Whisper backend emits) and `translates`. Three
more services complete the surface: `NFKRemoteVideoBackend` (OpenAI's job-style videos API on
`NFKAsyncGenerationBackend`, the on-device LTX pipeline's counterpart), `NFKRemoteReranker` (Together,
OpenRouter; the on-device reranker's shape), and `NFKRemoteModerationBackend` (OpenAI, Mistral).

## Speech in, text out

Transcription follows the same contract with `NFKInputAudio` (an `NFKAudioAsset`) in and
`NFKOutputText` out. Two engines serve it: `NFKRemoteTranscriptionBackend` for a hosted Whisper
endpoint, and `NFKMLXWhisperBackend` for Whisper on device, every released size at an exact token
match against the reference decoder, with `emitsTimestamps` adding per-segment times under
`NFKOutputSegments`. `NFKMLXVADBackend` (voice activity) and `NFKMLXAudioTaggerBackend` (audio
tagging) answer the questions around a transcript; the text→speech direction is `NFKMLXVoice`.
[Companion packages](companions.md) lists them all.

## Optional engines discovered at runtime

[`NFKDynamicBackend`](../Sources/InferKit/include/InferKit/NFKDynamicBackend.h) activates a heavier
engine only when its classes are linked into the consumer's build, with no build dependency on it.
The core resolves a provider class by name through `NSClassFromString`; when the companion is absent,
resolution returns nil and the feature is unavailable, with no link error. Each capability has a
default provider the companions ship:

- `NFKCapabilityStableDiffusion` → InferKitMLX's `NFKStableDiffusionProvider` (SD 1.5, the ungated release).
- `NFKCapabilityTranscription` → InferKitMLX's `NFKMLXWhisperProvider`.
- `NFKCapabilityTextGeneration` → InferKitFoundationModels' `NFKFoundationModelsProvider`.
- `NFKCapabilityControlNet` → no shipped default; a consumer registers their own.

A consumer's own engine registers under a capability with `registerProviderClassName:forCapability:`,
and the most recently registered present provider wins.

## Choosing and combining engines

- **Private, offline, no download infrastructure, latest OS** → `NFKFoundationModelsBackend` (Apple's
  model, zero weights to ship).
- **Private, offline, a specific open model, Apple Silicon** → `NFKMLXLanguageBackend` (the release
  directory as published, no conversion step; Qwen3, Llama, Gemma 4, Qwen3.5).
- **Private, offline, Intel Macs or a Core ML deployment** → `NFKCoreMLLanguageBackend` (convert
  once; macOS 15 / iOS 18).
- **A hosted or local server model** → `NFKRemoteProvider` and whichever backend it returns.

Because every engine adopts `NFKInferenceBackend`, a caller can pick at runtime: prefer Apple's model
when `isReady`, fall back to a local model, fall back to a remote endpoint, without changing
request-building code. Tool calling and structured output are Foundation Models features today; the
text, message, streaming, and parameter contract is shared across all of them.

## Roadmap

What the toolkit does not do yet, in the order the work is likely to pay off. Each entry names the
constraint that gates it. Kept beside the code so a stale entry is a diff rather than a memory.

### Foundational gaps

Each of these unlocks a category rather than a model.

- **Text embeddings and reranking — SHIPPED.** Two embedders and a reranker. `Qwen3-Embedding-0.6B`
  (`NFKMLXQwen3Embedding`) is the dense decoder read one layer earlier, last-token pooled over an
  appended `<|endoftext|>` and L2-normalized. `EmbeddingGemma-300M` (`NFKMLXEmbeddingGemma`) is the
  bidirectional Gemma 3 encoder mean-pooled through a Dense bottleneck, with its own byte-fallback BPE
  tokenizer. `NFKMLXModernBERTReranker` (`gte-reranker-modernbert-base`) is a cross-encoder that reads a
  query and document together and predicts a relevance score, which reorders an embedder's shortlist. All
  three are at reference parity against their own recipes.
- **A vision-language model — SHIPPED.** `SmolVLM2-500M` (`NFKMLXSmolVLM`): a SigLIP vision encoder, a
  pixel-shuffle connector, and a Llama decoder the dense stack already runs, with the projected vision
  tokens spliced into the decoder's embeddings at the image-token positions. At reference parity against
  transformers' own SmolVLMForConditionalGeneration (vision, connector, and fused logits exact, greedy
  continuation token for token), with a CoreGraphics image processor and token-exact prompt expansion
  for the consumer `answer(image:question:)` path. **`Qwen3-VL-2B`'s vision tower also ships**
  (`NFKMLXQwen3VLVisionNet`): a 2D-rotary ViT whose patches are laid out in 2×2 merge blocks, a
  bilinearly interpolated position embedding, a merger that folds each block to the decoder width, and
  the three-layer "deepstack", all at reference parity. Qwen3-VL turned out far more than "reuses the
  Qwen3 decoder": the decoder adds interleaved M-RoPE (3D positions) and deepstack injection at its first
  layers, so the decoder integration is the remaining wiring on top of the shipped Qwen3 stack.
- **A native GGUF reader — SHIPPED.** `NFKMLXGGUF` reads a GGUF model natively — the container's typed
  metadata and tensor table — and dequantizes the block-quant formats a real model uses (`Q4_K`, `Q6_K`,
  `Q8_0`, `Q5_0`, `Q4_0`, `F16`, `F32`) into `MLXArray`s, with no Python and no llama.cpp. Bit-exact
  against the `gguf` package on a real Q4_K_M model (worst |difference| 0.0 across every dequantizer). A
  type it does not implement is refused per-tensor rather than failing the file. **It is now wired into
  the language-model loader**, so a GGUF release generates text end to end (`backend(ggufURL:)`): the
  metadata becomes a configuration, the llama.cpp tensor names are remapped and the query/key
  projections un-permuted for the decoder's rotary, and the embedded tokenizer is rebuilt. Reference
  parity against transformers loading the same GGUF (logit cosine 0.9999999999). Only the dense
  `llama`/`qwen2`/`qwen3` families are read.
- **A chat-template engine — SHIPPED.** `NFKMLXChatTemplateRenderer` renders the Jinja `chat_template`
  an instruct release ships, so the backend reproduces the model's trained input instead of the ChatML
  approximation — a wrong render is a wrong input of the same class as the Qwen2 pre-tokenization defect.
  A compact interpreter for the subset chat templates use (for / if / set, `namespace`, slicing, the
  `loop` variable, `is` tests, string methods, `tojson`/`trim`, and the `trim_blocks`/`lstrip_blocks`
  whitespace model), pure Foundation so it runs under `swift test`. Reference parity against
  transformers' own `apply_chat_template` over six cases (Qwen3 with its tool-call/tool-role branches,
  Llama-3, Gemma). Exposed as `NFKMLXChatTemplate.jinja(template:…)`, and from Objective-C a
  `chatTemplate` request parameter carrying Jinja delimiters.
- **Gemma 4 and its variants.** The Gemma 4 TEXT decoder (`gemma4_text`) ships at parity for E2B and
  E4B (`NFKMLXGemmaLanguage`). Three more of the family now ship at parity, each against transformers'
  own Gemma 4 at a tiny configuration under the gemma oracle interpreter:
  - the **26B-A4B mixture** (`enable_moe_block`) — every layer's dense feed-forward gains a parallel
    routed-expert branch (`NFKGemmaRouter`/`NFKGemmaExperts`), summed (`gemma4_moe`, logits
    0.9999999999996). Two geometry facts were load-bearing: the per-layer input embedding is a fixed
    262144 rows (`vocab_size_per_layer_input`), and a full-attention layer runs a 512-wide head.
  - the **12B `gemma4_unified_text`** decoder (`NFKMLXGemma4UnifiedNet`) — a different architecture (no
    per-layer input embeddings, no mixture), reusing the E-series attention directly (`gemma4_unified`,
    every layer exact, logits 0.9999999999995).
  - the **vision encoder** (`NFKMLXGemma4VisionNet`) — patch embedder plus bidirectional sandwich
    encoder, no rotary (`gemma4_vision`, encoder cosine 0.9999999850).
  - the **audio Conformer** (`NFKMLXGemma4AudioNet`) — a 2-D convolutional subsampler and Conformer
    layers (blocked relative-position attention, a causal light conv, macaron feed-forwards), running
    end to end from the mel features with its own sliding-window mask (`gemma4_audio`, subsampler
    0.99999999999998, full tower 0.999999999999996).
  **All four Gemma 4 architectures are at reference parity**, both towers run their full forward to soft
  tokens (vision through its pooler, audio through its mask), the text decoders generate through
  `NFKMLXGemmaBackend` ("The capital of France is" → " Paris." on E2B), and the tri-modal input adapters
  and fusion are in: `NFKMLXGemma4ImageProcessor` (`CGImage` → patches, resize a documented
  approximation), `NFKMLXGemma4AudioFeatureExtractor` (audio → log-mel, `gemma4_mel` cosine
  0.9999999999928), `NFKMLXGemma4MultimodalEmbedder` (soft tokens → decoder space, `gemma4_embedder`
  cosine 0.99999999999999), and `NFKMLXGemma4Fusion.fuse` (the placeholder splice).
  `NFKMLXGemma4ConditionalGeneration` wires the whole chain end to end (image/audio + a
  placeholder-carrying prompt → generated continuation). **The E2B release is the full tri-modal
  `Gemma4ForConditionalGeneration`** (its checkpoint carries the vision tower, audio Conformer, both
  embedders, and the decoder), so the whole family is now at reference parity **on the released
  weights**: the vision path (`gemma4_vision_real`, ≥ 0.99999999999), the audio path
  (`gemma4_audio_real`, ≥ 0.99999999999), and the full conditional chain end to end
  (`gemma4_conditional_real`, an image → four soft tokens → fused logits at cosine 0.999999999959,
  argmax 8/8). The real weights exposed two bugs the tiny tests could not: the vision attention's 2-D
  rope, and the finite `use_clipped_linears` clamp bounds. The only thing left is the optional vision
  standardization.

### Language-model runtime

Shipped since the roadmap was written: cache rollback and the prompt cache, speculative decoding, the
Qwen3-MoE and Mixtral mixture-of-experts feed-forward, and JSON and fixed-choice constrained decoding.
What remains of each:

- **Speculative decoding that pays.** Measured on Qwen3-1.7B drafted by 0.6B at float32, 73% of
  proposals are accepted and the wall clock is unchanged: a 28-layer step here is bound by kernel
  launches, not memory traffic, so the draft step costs nearly what the target step does. The gain
  needs a target whose step is bandwidth-bound (large or quantized) and a draft with far fewer layers,
  and a draft that runs its proposals as one asynchronous chain rather than a synchronized step each.
- **More expert families.** gpt-oss needs alternating sliding-window layers, its own gated activation,
  an attention sink in the dense attention, and its MXFP4 experts kept packed (the pinned mlx-swift
  has the `mxfp4` mode); Qwen2-MoE adds a shared expert. Each is a configuration flag on the routed
  feed-forward once its attention exists.
- **Schema-constrained decoding.** The JSON grammar guarantees syntax; a JSON-schema grammar would
  guarantee the keys and types too. The byte-level engine takes any grammar with a hashable state, so
  this is a grammar, not a new engine. The Core ML language backend has no constraint path yet.
- **4-bit cache measurement.** 8-bit cache quantization shipped and was measured; 4-bit has not been,
  and if it degrades, quantizing keys per channel (the KIVI axis) is the change to measure next.

### Foundation Models

- **Provider bridge (WWDC26).** Apple's `LanguageModel` / `LanguageModelExecutor` protocols let a
  third-party model stand *behind* `LanguageModelSession`. Adopting them would let InferKit's local
  and remote backends serve any app written against Apple's session API. Those protocols are not in
  the macOS 26 SDK; this direction lands when macOS 27 / iOS 27 is the build baseline.
- **Exact-type structured output** through Apple's `@Generable` macro for compile-time result types
  (the runtime-schema path here covers the dynamic case).

### Model additions

The standout pick per modality, each entry naming what makes it worth the port ahead of its
alternatives.

- **Audio.**
  - `Kokoro-82M` TTS (Apache) — an 82M-parameter voice, small enough to ship as a default.
  - `Silero VAD v6` (~1 MB) — **SHIPPED** (`NFKMLXSileroVAD`). A streaming STFT + conv encoder + LSTM
    decoder scoring one speech probability per 512-sample chunk, beside the existing MarbleNet VAD.
    At reference parity against the released snakers4 JIT (silero_vad 6.2.1): per-chunk cosine
    0.9999999999998, max |difference| 6.9e-7, threshold agreement 32/32. Loads the released `.jit`
    through its own `_model.*` weights (the converter keeps them; the 8 kHz branch is dropped).
  - `Parakeet` ASR — **SHIPPED** (`NFKMLXParakeet`): Parakeet-TDT 0.6B v2 (CC-BY-4.0), the FastConformer
    encoder and the token-and-duration transducer at reference parity against NeMo on the released
    weights and the validation clip (encoder 0.99999999999, tokens and frame timestamps exact), reading
    the unpacked `.nemo` through the native torch reader. A faster second speech-to-text engine beside
    Whisper, with a timestamp per token.
  - the `SNAC` and `DAC` neural codecs — the codec class the toolkit had no path for. Codec tokens are
    what a speech-LLM TTS generates, so this unlocks that whole approach. **`DAC` is SHIPPED**
    (`NFKMLXDAC`): the 44.1 kHz model (encoder + residual vector quantizer + decoder, the decoder reusing
    the Music 3 Snake blocks), at reference parity against `descript-audio-codec` — codebook tokens
    matching exactly (783/783) and the decoder reconstructing at cosine 0.99999999999986. `encode`
    returns the tokens; `decode` and the backend reconstruct. **`SNAC` is SHIPPED too** (`NFKMLXSNAC`,
    the 24 kHz speech model): a multi-scale codec whose codebooks code at different temporal rates (strides
    `[4,2,1]`), with depthwise convolutions and a decoder noise block, at reference parity against the
    `snac` package — per-codebook tokens matching exactly (42/42) and the decoder reconstructing at cosine
    0.9999999999998 (noise disabled, its expected contribution zero).
  - `pyannote` diarization — who-spoke-when over a recording. BLOCKED: `pyannote/segmentation-3.0`
    is a gated repository and no token is available here.
  - `Chatterbox` voice cloning (MIT) — **SHIPPED** (`NFKMLXChatterbox`): all five networks (VoiceEncoder,
    S3 speech tokenizer, the T3 Llama with llama3 rope scaling, S3Gen's flow-matching decoder, the HiFT
    vocoder) at reference parity on the released weights, stage by stage; a clip synthesized in the
    validation clip's voice transcribes back through the package's own Parakeet exactly.
  - The four dereverberation candidates (SGMSE+, Resemble Enhance, VoiceFixer, DeepFilterNet) stay on
    the list beneath these.
- **Image.**
  - `Z-Image Turbo` (6B, Apache) — the leading open text-to-image model, it fits a Mac. **The DiT stage
    is SHIPPED** (`NFKMLXZImageTransformerNet`): the single-stream S3-DiT — image and caption tokens
    concatenated into one self-attention sequence, a `noise_refiner` / `context_refiner` / unified
    `layers` structure, sandwich RMS norms, 4-chunk adaptive modulation, SwiGLU FFN, and a 3-axis complex
    rotary — at reference parity against diffusers on the first numeric run (velocity cosine
    0.9999999999999653, pad-token path exercised), verified in isolation with recorded Qwen3 captions.
    The Flux VAE (a diffusers `AutoencoderKL`, 16 channels) and the Qwen3-4B encoder are the remaining
    stages; the flow sampler is already shipped.
  - `SANA` (0.6B) — the only new text-to-image model small enough for a phone. **The DiT stage is
    SHIPPED** (`NFKMLXSANATransformerNet`): the linear-attention DiT — ReLU O(N) self-attention, GLUMBConv
    gated-depthwise Mix-FFN, softmax cross-attention to the Gemma text, PixArt-α shared-timestep
    modulation — at reference parity against diffusers on the first numeric run (velocity cosine
    0.9999999999999959). **The DC-AE stage is SHIPPED too** (`NFKMLXDCAutoencoderNet`): the 32×
    deep-compression autoencoder — ResBlocks + EfficientViTBlocks (multiscale ReLU linear attention +
    GLUMBConv) + pixel-shuffle up/down blocks — at reference parity against diffusers (latent cosine
    0.99999999999993, decode 0.99999999999998). **`NFKMLXSANAPipeline` chains it end to end** (DiT + flow
    → DC-AE decode; caller supplies the Gemma embedding). The released DPM-Solver sampler is now ported
    (`NFKMLXDPMSolverScheduler`, at reference parity), and the Gemma-2 text encoder is ported
    too (`NFKMLXGemma2Net`, at parity) — SANA runs end to end from a raw prompt.
  - `IP-Adapter` — image conditioning, a cheap gap to close.
  - `TAESD` — **SHIPPED** (`NFKMLXTAESD`): the tiny SD autoencoder for an instant latent preview, at
    reference parity against madebyollin's own taesd (latent and decode cosine 0.9999999999, mean
    |difference| 1.9e-7). Encoder + decoder modeled as `[Module]` arrays so the numeric Sequential keys
    load with no remap.
  - `SigLIP 2` — **SHIPPED** (`NFKMLXSigLIP2`, base-patch16-224): the CLIP upgrade that doubles as a VLM
    vision tower. Vision ViT (reusing the SigLIP encoder) + attention-pooling head + a 256k-vocab text
    tower + sigmoid logit scale/bias, at reference parity against transformers (image and text embedding
    cosine 0.999999999999, logits to 1e-5).
- **Vision.**
  - `Depth Anything V3` — a drop-in on the V2 port.
  - `BiRefNet` (MIT) — matting.
  - `SAM 3.1`.
  - `RT-DETR` / `RF-DETR` — a detector that avoids YOLO's AGPL.
- **Video generation**, a new capability the toolkit lacks (it only interpolates and upscales today).
  - `LTX-Video` / `LTX-2` — the first video generator here, with MLX ports available as parity
    references. **The VAE stage is SHIPPED** (`NFKMLXLTXVideoVAE`): the causal 3D autoencoder
    (`AutoencoderKLLTXVideo`) — causal Conv3d, RMSNorm resnets, stride-2 downsamples, a 3D pixel-shuffle
    upsampler, patchify/unpatchify — at reference parity against diffusers on the first numeric run
    (latent cosine 0.99999999999, decode cosine 0.99999999996; every encoder seam exact). **The DiT stage
    is SHIPPED too** (`NFKMLXLTXTransformer`): the 2B denoising transformer — 28 adaLN blocks with 3D
    rotary self-attention, cross-attention to text, RMS qk-norm, gelu-approx feed-forward — at reference
    parity against diffusers (velocity cosine 0.99999999999, every seam exact), verified in isolation with
    recorded text embeddings. **The full text-to-video pipeline is now COMPLETE**: the T5-XXL text encoder
    (`NFKMLXT5Encoder`, text-embedding cosine 0.99999999998 against transformers), the rectified-flow
    sampler (`NFKMLXFlowMatchScheduler`, `FlowMatchEulerDiscreteScheduler` with dynamic shifting, schedule
    exact to the reference), and the pipeline glue (`NFKMLXLTXPipeline`: T5 → DiT+flow loop with
    classifier-free guidance → VAE decode) all ship, so LTX-Video runs end to end. The T5 encoder and the
    flow sampler are reusable across the other flow-matching models (Wan 2.2 shares the T5 family; Z-Image
    / SANA / Flux share the flow sampler).
  - `Wan 2.2` — for quality. **The DiT stage is SHIPPED** (`NFKMLXWanTransformerNet`): the text-to-video
    transformer — a `Conv3d`-patchified video latent, a 3-axis interleaved rotary (the Z-Image rope
    reused), self + cross attention with an across-heads RMS q/k norm, gelu FFN, and PixArt-α adaptive
    norms — at reference parity against diffusers on the first numeric run (velocity cosine
    0.9999999999999767). **The 3D causal VAE is SHIPPED too** (`NFKMLXWanVideoVAENet`, the hardest port
    of the batch): a stateful feat_cache STREAMING autoencoder (chunked encode, per-frame decode) at
    reference parity (encoder 0.9999999999999997, decode 0.999999999999892). **`NFKMLXWanPipeline` chains
    it end to end** (DiT + UniPC flow sampler → 3D VAE decode; caller supplies the umT5 embedding).
    The released UniPC sampler is now ported (`NFKMLXUniPCScheduler`, at reference parity), and umT5
    is verified (`NFKMLXT5Encoder` per-layer-bias mode) — Wan runs end to end from a raw prompt.

Licensing gates what ships as a default: permissive weights ship, research-only and vendor-licensed
weights (FLUX.1-dev, F5-TTS, DMD2, Apple's FastVLM and Depth Pro, and Parakeet under NVIDIA/NeMo) are
gated the way Music 3's are, and a permissively licensed detector is preferred over an AGPL one. `ml-explore/mlx-lm` and `Blaizzy/mlx-audio` are the
standing parity oracles for the language and audio work.
