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
`result.embedding`, `request.prompt`, `request.messages`), each returning nil on a type mismatch
rather than crashing.

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
the port does not implement (`dynamic`, `llama3`, `longrope`; `linear` and `yarn` are implemented).
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
preset carries its endpoint and protocol and deliberately **no default model name**: model identifiers
change faster than a release does, and each preset's `modelsURL` is where the current list lives.

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

- **Text embeddings and reranking.** No text embedder means no retrieval or semantic search.
  `Qwen3-Embedding-0.6B` reuses the dense decoder with last-token pooling, which makes it the cheapest
  entry here; `EmbeddingGemma-300M` and a ModernBERT cross-encoder reranker follow.
- **A vision-language model.** None ships. `SmolVLM2-500M` first (a SigLIP-family encoder over a
  decoder already covered, with `mlx-vlm` as the oracle), then `Qwen3-VL-2B`, which reuses the Qwen3
  decoder at parity.
- **A native GGUF reader.** The sequel to the native `.pth` reader: the dominant quantized-LLM format
  is currently unreadable. The work is the block-quant dequantizers (`Q4_K`, `Q6_K`, `Q8_0`) into MLX.
- **A chat-template engine.** Instruct releases ship a Jinja `chat_template`, and a wrong render is a
  wrong input of the same class as the Qwen2 pre-tokenization defect. Today only ChatML is rendered;
  the fix is a minimal native renderer held to `apply_chat_template` output.

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

- **Audio.** Kokoro TTS, Silero VAD v6, Parakeet ASR, the SNAC and DAC codecs, pyannote diarization,
  and the four dereverberation candidates (SGMSE+, Resemble Enhance, VoiceFixer, DeepFilterNet).
- **Image.** Z-Image Turbo, SANA, IP-Adapter, TAESD, SigLIP 2 as the CLIP upgrade.
- **Vision.** Depth Anything V3 (a drop-in), BiRefNet, SAM 3.1, RT-DETR / RF-DETR.
- **Video generation**, a new capability: LTX-Video / LTX-2, Wan 2.2.

Licensing gates what ships as a default: permissive weights ship, research-only weights (FLUX.1-dev,
F5-TTS, DMD2, Apple's FastVLM and Depth Pro) are gated the way Music 3's are, and a permissively
licensed detector is preferred over an AGPL one. `ml-explore/mlx-lm` and `Blaizzy/mlx-audio` are the
standing parity oracles for the language and audio work.
