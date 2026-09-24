# InferKit examples

Complete examples for InferKit v0.1.0: the input/output modality matrix, every backend (local,
remote, Apple on-device, MLX), and each subsystem. Objective-C for the core, Swift for the companion
packages; the two interoperate through the same `NFKInferenceBackend` protocol.

See also the [inference guide](inference-guide.md) for the concepts behind these examples.

> The snippets here are mirrored by compiled example targets that `swift test` builds and runs so this
> document cannot silently drift: `InferKitExamples` ([`Examples/NFKExamples.m`](../Examples/NFKExamples.m))
> for the core, `InferKitMLXExamples` and `InferKitMLXObjCExamples` for MLX, and
> `InferKitFoundationModelsExamples` for Apple. **Changing an example in one place may require changing
> it in the other** — keep the doc snippet and its example method in sync.

## Contents

- [Setup](#setup)
- [The modality matrix](#the-modality-matrix)
- [The shared contract](#the-shared-contract)
- [Text → text](#text--text) — local, remote, Apple
- [Text → image and image → image](#text--image-and-image--image) — MLX Stable Diffusion
- [Image → image](#image--image) — bring-your-own MLX, Core ML
- [Image → image + mask](#image--image--mask) — matting
- [Many tensors in and out](#many-tensors-in-and-out)
- [Diffusion: upscale, depth, inpaint](#diffusion-upscale-depth-inpaint) — bring-your-own MLX diffusion
- [Text → video](#text--video) — LTX-Video and Wan, staged
- [Structured output and tools](#structured-output-and-tools) — Apple
- [Audio → text](#audio--text-transcription) — remote transcription (Whisper)
- [Text → audio](#text--audio-speech) — bring-your-own MLX speech
- [Loading a PyTorch checkpoint directly](#loading-a-pytorch-checkpoint-directly) — no Python toolchain
- [Choosing a backend at runtime](#choosing-a-backend-at-runtime)
- [Subsystems](#subsystems) — jobs, tokenizers, tensor conversion, Hugging Face hub, conversion tool
- [Testing without weights](#testing-without-weights)

## Setup

Swift Package Manager:

```swift
.package(url: "https://github.com/belisoful/InferKit.git", from: "0.1.0")
```

Add `"InferKit"` to a target. The companions are separate packages in the same repository:
`InferKitMLX` (Apple Silicon, macOS 14 / iOS 17) and `InferKitFoundationModels` (macOS 26 / iOS 26).
CocoaPods installs the core with `pod 'InferKit'`; the companions are SwiftPM only.

## The modality matrix

InferKit's modalities are text, image, video, and audio (`NFKModality`). The shipped backends cover
the following cells; each links to its example.

| Input | Output | Backend(s) | Example |
| --- | --- | --- | --- |
| text | text | `NFKCoreMLLanguageBackend`, `NFKRemoteBackend`, `NFKFoundationModelsBackend` | [Text → text](#text--text) |
| text | structured | `NFKFoundationModelsBackend`, `NFKRemoteBackend` | [Structured output](#structured-output-and-tools) |
| text | image | `NFKMLXBackend` (text-to-image) | [Text → image](#text--image-and-image--image) |
| text + image | image | `NFKMLXBackend` (image-to-image) | [Image → image](#text--image-and-image--image) |
| image | image | `NFKMLXModuleBackend`, `NFKMLXRealESRGAN` (upscale), `NFKMLXDepthAnything` (depth), `NFKMLXNAFNet` (restore), `NFKCoreMLBackend` | [Image → image](#image--image) |
| image (+ hint) | image + mask | `NFKMLXMattingBackend`, `NFKMLXU2Net` (bg removal), `NFKMLXSAM` (segment) | [Image → image + mask](#image--image--mask) |
| image + mask | image | `NFKMLXLaMa` (inpaint) | [LaMa inpainting](#lama-inpainting-mtimlxlama-a-shipped-mlx-model) |
| image(s) | image(s) | `NFKMLXTensorBackend`, `NFKMLXRIFE` (frame interpolation), `NFKMLXRAFT` (optical flow) | [Many tensors](#many-tensors-in-and-out) |
| image (+ mask) | image | `NFKMLXDiffusionBackend` (upscale, depth, inpaint) | [Diffusion](#diffusion-upscale-depth-inpaint) |
| audio | text | `NFKRemoteTranscriptionBackend` (remote), `NFKMLXWhisper` (local) | [Audio → text](#audio--text-transcription) |
| text | text (translation) | `NFKMLXMarian`, `NFKMLXM2M100`, `NFKMLXMADLAD`, `NFKMLXTranslateGemma` (local) | [Translation](#translation-nfkmlxmarian-nfkmlxm2m100-nfkmlxmadlad-swift-and-objective-c) |
| text | audio | `NFKMLXSpeechBackend` | [Text → audio](#text--audio-speech) |
| audio | audio(s) | `NFKMLXDemucsBackend` / `NFKMLXHTDemucsBackend` (stem separation) | [Audio → stems](#audio--stems-demucs) |
| any | unchanged | `NFKPassthroughBackend` | [Testing](#testing-without-weights) |

Audio uses `NFKModalityAudio` and `NFKAudioAsset` (under `NFKInputAudio` / `NFKOutputAudio`), with the
`sampleRate` / `channelCount` / `durationSeconds` parameters. `NFKAudioAsset` is a file-based value
type like the video asset; a backend working with in-memory samples uses `NSData` PCM or an
`AVAudioPCMBuffer` under the audio key instead.

Video uses `NFKVideoAsset` under `NFKInputVideo` / `NFKOutputVideo`, with the `frameCount` /
`framesPerSecond` / `durationSeconds` / `motionScale` parameters. InferKitMLX's `NFKMLXVideoBackend`
produces it; see [Video](#video-clip--clip) below.

## The shared contract

Every backend consumes an `NFKInferenceRequest` and returns an `NFKInferenceResult`. Inference is
multi-second, so submit it as a job and keep it off the render thread.

```objc
#import <InferKit/InferKit.h>

NFKInferenceRequest *request =
    [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Explain diffraction in one sentence." }
                                parameters:@{ NFKParameterMaxTokens: @64 }
                            outputModality:NFKModalityText];

// Synchronous (already off the main thread):
NSError *error = nil;
NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
NSString *text = result.text;                       // convenience for [result outputForKey:NFKOutputText]

// Or as a job, uniformly across sync and async backends:
NFKInferenceJob *job = NFKInferenceSubmit(backend, request, NULL);
job.progressHandler = ^(NFKInferenceJob *j) {
    NSString *partial = [j.partialResult outputForKey:NFKOutputText];   // streamed text, when supported
};
job.completionHandler = ^(NFKInferenceJob *j) {
    if (j.result) { /* j.result outputs */ } else { /* j.error */ }
};
// [job cancel];  // cooperative cancellation
```

For the keys with a single natural type there are typed convenience accessors — `result.text`,
`result.structured`, `request.prompt`, `request.negativePrompt`, `request.messages` — each returning
nil when the value is absent or the wrong type. Image, mask, and video values stay on `outputForKey:`
/ `inputForKey:` because the backend or caller chooses their representation (a `CVPixelBuffer`, a
texture, a `CGImage`).

## Text → text

The same request runs against a converted local model, a remote endpoint, or Apple's on-device
model — only the backend differs.

### Local, on device (`NFKCoreMLLanguageBackend`)

A model directory produced by [`Tools/inferkit-convert`](../Tools/inferkit-convert). macOS 15 / iOS 18.

```objc
NSURL *directory = [NSURL fileURLWithPath:@"…/qwen2.5-0.5b-instruct"];
NFKCoreMLLanguageBackend *backend = [NFKCoreMLLanguageBackend backendWithModelDirectoryURL:directory];
backend.computeUnits = MLComputeUnitsAll;                 // default; avoid CPUAndNeuralEngine for this graph
NSError *error = nil;
[backend prepareWithError:&error];                        // slow; do once, off the render thread

NSArray *messages = @[ @{ @"role": @"user", @"content": @"Name one color." } ];
NFKInferenceRequest *request =
    [NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: messages }
                                parameters:@{ NFKParameterMaxTokens: @64, NFKParameterTemperature: @0 }
                            outputModality:NFKModalityText];
NSString *reply = [backend runInferenceForRequest:request error:&error].text;
// "One color is blue."
```

**Constraining its output.** The core carries the byte-level JSON and fixed-choice grammars
(`NFKJSONConstraint`, `NFKChoiceConstraint` over an `NFKTokenVocabulary`), and the backend masks its
logits with one before sampling when a request asks: `NFKParameterOutputFormat` (`"json"`,
`"json-object"`, `"json-array"`) or `NFKParameterChoices`. JSON that was asked for comes back parsed under
`NFKOutputStructured` beside the text. The same keys are honored by the MLX backend, so the request is
engine-agnostic; the schema grammar (`NFKMLXJSONSchemaConstraint`) is MLX-only.

```objc
NFKInferenceRequest *request =
    [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Describe Paris as JSON." }
                                parameters:@{ NFKParameterOutputFormat: @"json-object",
                                              NFKParameterMaxTokens: @96, NFKParameterTemperature: @0 }
                            outputModality:NFKModalityText];
NSDictionary *object = [backend runInferenceForRequest:request error:&error].structured;

// A classification: the answer is exactly one of the choices.
NFKInferenceRequest *pick =
    [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Is the sky blue? Answer yes or no." }
                                parameters:@{ NFKParameterChoices: @[ @"yes", @"no" ] }];

// The grammar can be inspected on its own, over any byte-level vocabulary:
NFKJSONConstraint *json = [[NFKJSONConstraint alloc] initWithVocabulary:vocabulary root:NFKJSONRootObject];
[json acceptsText:@"{\"city\": \"Par"];                     // YES: a prefix the grammar can complete
[json isCompleteText:@"{\"city\": \"Paris\"}"];             // YES
```

### Local, on device through MLX (`NFKMLXLanguageBackend`, Swift)

The dense decoder — Qwen3 and Llama are the same structure at different settings — reading a released
Hugging Face directory. The module's keys are the checkpoint's, so nothing is remapped.

```swift
let backend = try NFKMLXLanguage.backend(directoryURL: releaseDirectory)
let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "Explain diffraction in one sentence."])
let text = try backend.runInference(for: request).text
```

The Mamba-2 state-space decoder (Codestral-Mamba) loads the same way through
`NFKMLXMamba.backend(directoryURL:)` (`mambaBackendWithDirectoryURL:error:`); every layer is a selective
scan rather than attention, so it carries a fixed-size state and runs prefill-only.

The Granite 4.0-H hybrid decoder loads through `NFKMLXGraniteHybrid.backend(directoryURL:)`
(`graniteBackendWithDirectoryURL:error:`); most layers are the Mamba-2 scan and the few its
`layer_types` names are grouped-query attention, so it too runs prefill-only. The dense sizes carry a shared MLP and the MoE
sizes add a routed mixture of experts.

The Nemotron Nano 2 hybrid decoder loads through `NFKMLXNemotronH.backend(directoryURL:)`
(`nemotronBackendWithDirectoryURL:error:`); its layers interleave the Mamba-2 scan, a ReLU-squared
feed-forward, and grouped-query attention with no positional embedding, one mixer per block from the
release's `hybrid_override_pattern`, so it too runs prefill-only.

Granite Speech 3.3-2b transcribes audio through `NFKMLXGraniteSpeech.backend(directoryURL:)`
(`graniteSpeechBackendWithDirectoryURL:error:`): a Conformer encoder and a BLIP-2 Q-former projector
turn the audio into embeddings that scatter into a dense Granite decoder's prompt. Pass the clip under
`NFKInputAudio` and an optional instruction under `NFKInputPrompt`; the transcript comes back under
`NFKOutputText`.

Voxtral-Mini 3B transcribes through `NFKMLXVoxtral.backend(directoryURL:)` (`voxtralBackendWithDirectoryURL:error:`): the Whisper encoder and a projector feed a Llama decoder. Pass the clip under `NFKInputAudio` and a language code (default `en`) under `NFKInputPrompt`; the transcript comes back under `NFKOutputText`.

Canary-1B-v2 transcribes and translates through `NFKMLXCanary.backend(directoryURL:)` (`backendWithDirectoryURL:error:`): the biased FastConformer encoder feeds a Transformer attention encoder-decoder. Pass the clip under `NFKInputAudio` and, under `NFKInputPrompt`, a language code (default `en`, transcribe) or a `src>tgt` pair (`en>de`, translate); the transcript comes back under `NFKOutputText`.

**Bounding the cache.** A key-value cache that keeps every position grows with the conversation, and
past a certain length that growth is what ends the run. `contextWindow` drops the oldest positions
instead:

```swift
var options = NFKMLXGenerationOptions()
options.maxTokens = 512
// Retain at most 4096 positions. Size it from what the machine can hold.
options.contextWindow = 4096
let backend = try NFKMLXLanguage.backend(directoryURL: releaseDirectory, options: options)
```

It is off by default, and turning it on is a decision rather than a tuning. While the conversation is
shorter than the window nothing has been dropped and the result is bit-identical; past the window the
model stops seeing its beginning. For a model whose attention is not natively windowed that is an
approximation, and it is the caller's to make.

`NFKMLXGPU.recommendedWorkingSetSize` is the budget to divide when choosing the number — Metal's own
recommendation, which sits well below the machine's physical memory.

**Quantizing the cache, chunking the prefill, applying a chat template.** Three more options, each off
by default so an existing caller is unchanged:

```swift
var options = NFKMLXGenerationOptions()
// Store the key-value cache 8-bit instead of a float per element, so a long conversation reaches
// further before the cache is the memory ceiling. Lossy but close; the group size divides the head
// dimension (64 or 128 divide by the default 64).
options.cacheQuantization = .init(bits: 8, groupSize: 64)
// Run a long prompt through the cache in slices, bounding the prefill's attention peak. Exact — each
// chunk attends to the same prefix a single pass would — so only the memory differs.
options.prefillChunkSize = 512
// Render a message list in the ChatML format an instruct release is trained on, with its own special
// tokens. A raw NFKInputPrompt is always used verbatim; a base model wants the default plain text.
options.chatTemplate = .chatML
let backend = try NFKMLXLanguage.backend(directoryURL: releaseDirectory, options: options)
```

**Rendering the release's own chat template.** `.chatML` approximates; the faithful path renders the
Jinja `chat_template` the release ships (from its `tokenizer_config.json`), so the model reads exactly
what it was trained on. `NFKMLXChatTemplateRenderer` is a compact Jinja interpreter held to
transformers' `apply_chat_template`, and it is pure Foundation — no runtime, no MLX:

```swift
// The template text comes from the release's tokenizer_config.json ("chat_template").
options.chatTemplate = .jinja(template: releaseTemplate, bosToken: "<|begin_of_text|>")

// Or render it directly, without a backend:
let prompt = try NFKMLXChatTemplateRenderer.render(
    releaseTemplate,
    messages: [["role": "system", "content": "You are terse."],
               ["role": "user", "content": "What is 2+2?"]],
    addGenerationPrompt: true)
```

**A reasoning model's chain, and what the turn cost.** A reasoning release writes its chain before its
answer. The backend splits the two, so `NFKOutputText` holds the answer alone and the chain rides under
`NFKOutputReasoning`. The markers come from the release's own template, which is also what takes
`NFKParameterReasoningEffort`:

```swift
var options = NFKMLXGenerationOptions()
// The template states the markers and takes the level, so it is what the backend renders with.
options.chatTemplate = .jinja(template: NFKMLXLanguage.chatTemplate(inDirectory: releaseDirectory)!)
let backend = try NFKMLXLanguage.backend(directoryURL: releaseDirectory, options: options)

let request = NFKInferenceRequest(
    inputs: [NFKInputMessages: [["role": "user", "content": "What is 2 + 2? Answer briefly."]]],
    parameters: [NFKParameterReasoningEffort: NFKReasoningEffortDeep])   // light, moderate, or deep
let result = try backend.runInference(for: request)

result.text                                       // "2 + 2 = 4."
result.output(forKey: NFKOutputReasoning)         // the chain, apart from the answer
result.output(forKey: NFKOutputUsage)             // ["inputTokens": 19, "cachedTokens": 0,
                                                  //  "outputTokens": 156, "reasoningTokens": 147]
```

Measured on the released Qwen3-0.6B. `NFKReasoningEffortLight` closes the block through the release's
own template, which took the same question to 8 output tokens and no chain. The level reaches the
model only through a Jinja template, so a request that names one without a template is refused. A
caller who renders its own prompt names the markers instead, through
`NFKMLXGenerationParameterKey.reasoningFormat` (`"think"`, `"harmony"`, or the markers themselves).
`NFKUsageCachedTokens` counts what a retained prompt cache served, so it is above zero only on a
request that shares a prefix with the last one.

**Refusing a release that will not fit.** The dense loader checks a release's weight bytes against the
memory budget before materializing any, so a load that would kill the process is an error instead:

```swift
// Throws "needs about N GiB resident, but the machine's working set is M GiB" rather than OOM-killing.
try NFKMLXReleaseWeights.verifyFits(inDirectory: releaseDirectory, precision: .checkpoint)
```

**Sizing it to the machine.** The window above is a number someone has to choose, and choosing it by
hand is a guess about a machine the author was not using. `NFKMLXModelSizing` derives it:

```swift
// Throws if the weights alone do not fit — no window helps then, and failing here beats failing
// at the load, where it is a process kill rather than an error.
let options = try NFKMLXModelSizing.options(for: configuration, requesting: 32_768,
                                            precision: .checkpoint)
// contextWindow is left unset when the requested length already fits.
let backend = try NFKMLXLanguage.backend(directoryURL: releaseDirectory, options: options)
```

The verdict on its own, for a report or a model picker:

```swift
let fit = NFKMLXModelSizing.fit(of: .qwen3_4B, tokens: 32_768, precision: .checkpoint)
print(fit.describedFit)
// "7.49 GB weights + 4.50 GB cache at 32768 tokens, against 13.41 GB — fits"
```

And what it could possibly decode at, which is bounded by memory traffic rather than by arithmetic:

```swift
let ceiling = NFKMLXModelSizing.decodeCeiling(for: .qwen3_4B, contextLength: 4096,
                                              precision: .checkpoint)   // ~35 tok/s on an M1 Max
// Inverted, it becomes a diagnostic: a rate above the ceiling means the model is not reading every
// parameter, which is what a sparse model doing its job looks like.
let reached = NFKMLXModelSizing.achievedFraction(tokensPerSecond: measured, for: .qwen3_4B)
```

The bandwidth is **measured**, not tabulated — no sysctl reports it, and a per-chip table would be
numbers copied from somewhere rather than a property of the machine running the code. The probe reads
a large array and times it; it is sized from the working-set budget because a small array measures
launch overhead instead (on an M1 Max: 40 GB/s at 16 MB, 158 at 256 MB, settling near 330 from 1 GB).

**Extended context.** A release that was trained short and extended long says so in its config's
`rope_scaling`, and `NFKMLXLanguage.configuration(fromHuggingFace:)` reads it — there is nothing to
switch on. `linear`, `yarn`, and `llama3` are implemented and measured against `transformers`' own
initializers. A kind that is not implemented throws rather than loading:

```swift
// Throws NFKMLXError.unsupportedConfiguration for `dynamic` or `longrope`.
let configuration = try NFKMLXLanguage.configuration(fromHuggingFace: configURL)
```

Refusing is the point. Those kinds compute different frequencies, and a model loaded under a rotary it
was not trained with runs perfectly well and produces fluent nonsense — there is no error to notice.

**Continuing a conversation without re-reading it.** A chat turn's prompt is the previous prompt plus
the reply plus the new message, and prefilling that prefix again is the largest cost in a
conversation. A prompt cache keeps the key-value rows between generations and rolls back to the
point where a new prompt diverges, so only the new tokens run through the model. The result is exact:

```swift
var options = NFKMLXGenerationOptions()
options.reusesPromptCache = true                       // the backend keeps a cache between requests
let backend = try NFKMLXLanguage.backend(directoryURL: releaseDirectory, options: options)

// Or hold one yourself, and persist a long system prompt's prefill for the next launch:
let cache = NFKMLXPromptCache(layerCount: net.configuration.layerCount)
let reply = net.generate(prompt: turn, options: options, promptCache: cache)
let next = net.generate(prompt: turn + reply + newMessage, options: options, promptCache: cache)
try cache.save(to: url)                                // NFKMLXPromptCache.load(from:) restores it
```

**Speculative decoding.** A smaller release of the same family proposes a few tokens, the model
verifies them in one cached pass, and the rejected tail is rolled back. The output is the model's own
greedy run, token for token; above temperature 0 the standard rejection test keeps the distribution
the model's. Measured on Qwen3-1.7B drafted by 0.6B, both float32: 73% of proposals accepted and no
wall-clock gain, because a 28-layer step on this machine is bound by kernel launches rather than by
memory traffic, so the draft's step costs nearly what the target's does. The lever is real where a
decode step is bandwidth-bound: a large or quantized target and a draft with far fewer layers.

```swift
let backend = try NFKMLXLanguage.backend(directoryURL: qwen4B, draftDirectoryURL: qwen06B)
var options = NFKMLXGenerationOptions()
options.draftTokens = 4                                // proposals per round; 0 turns it off

// At the network level, with a report of what the draft achieved:
var report = NFKMLXSpeculativeReport()
let tokens = target.generate(prompt: prompt, options: options, draft: draft, report: &report)
print(report.acceptanceRate)
```

**A mixture of experts.** Qwen3-MoE, Qwen2-MoE, Mixtral, and gpt-oss are this decoder with a routed
feed-forward, and the same factory reads them: the config names the family, the loader stacks the
released per-expert tensors into one `[experts, out, in]` tensor per projection, and a step runs one
gathered matrix multiplication over the experts each token chose. Qwen2-MoE adds a shared expert every
token runs beside the routed ones, gated by a sigmoid; gpt-oss alternates sliding-window and full
attention, carries a learned sink per head, and runs fused clamped experts whose released MXFP4 blocks
stay packed. All four families match `transformers`' arithmetic layer by layer at a tiny configuration, and every one of Qwen3-30B-A3B's 18,867 released tensors is
accounted for by shape. Quantizing packs the stacked experts too. The released sizes need quantized
weights to fit a 32 GB machine.

```swift
let configuration = try NFKMLXLanguage.configuration(fromHuggingFace: configURL)   // Qwen3MoeForCausalLM
configuration.isMixtureOfExperts        // true; expertCount 128, activeExpertCount 8
let backend = try NFKMLXLanguage.backend(directoryURL: releaseDirectory)
```

**Paging a mixture's routed experts.** A token reaches a few of a mixture's experts, so most of what
a resident load holds is not read on any one step. `NFKMLXResidency.paged` leaves the routed experts
in the release and reads each one as the router reaches it, through a bounded cache of recently used
experts; every other tensor loads as usual. `.automatic`, the default, pages only where the release is
known not to fit the machine's working set whole. A paged model computes the resident model's logits
exactly, and runs slower by what each step reads from the release. The release directory stays in
place for the life of the backend. The same residency reaches Qwen3-MoE, Qwen2-MoE, Mixtral, gpt-oss
(bf16 or MXFP4), a quantized checkpoint this package saved, the Gemma 4 26B-A4B mixture, the Qwen3-VL
30B-A3B decoder, Qwen4-Exp, and Granite 4.0-H's mixture sizes. A dense release has nothing to page and loads resident.

```swift
let paged = try NFKMLXLanguage.backend(directoryURL: releaseDirectory, residency: .paged)
if let language = paged as? NFKMLXLanguageBackend, let store = language.expertStore {
    store.cacheByteBudget = 8 << 30          // what the cache may hold in materialized experts
    print(language.pagesExperts, store.mappedBytes, store.materializeCount, store.cacheHitCount)
}
```

```objc
id<NFKInferenceBackend> paged = [NFKMLXLanguage backendWithDirectoryURL:releaseDirectory
                                                              residency:NFKMLXResidencyPaged error:&error];
NFKMLXLanguageBackend *language = (NFKMLXLanguageBackend *)paged;
if (language.pagesExperts) {
    language.expertStore.cacheByteBudget = 8LL << 30;
    NSLog(@"%ld mapped bytes, %ld materialized", (long)language.expertStore.mappedBytes,
          (long)language.expertStore.materializeCount);
}
```

**Gemma 4's mixture.** The Gemma 4 26B-A4B release (`enable_moe_block`) is a different shape of
mixture: each layer keeps its dense feed-forward and adds a routed-expert branch beside it, and the two
are summed. `NFKMLXGemmaLanguage.configuration(fromHuggingFace:)` reads the flag from the release
directory and turns the branch on; the decoder matches `transformers`' own `Gemma4ForCausalLM` layer by
layer at a tiny configuration (logit cosine 0.9999999999996).

```swift
let gemma = try NFKMLXGemmaLanguage.configuration(fromHuggingFace: configURL)   // gemma4 + enable_moe_block
gemma.isMixtureOfExperts                 // true for the 26B-A4B, false for the dense E-series
```

**Gemma 3.** `NFKMLXGemma3` runs the Gemma 3 releases (270M, 1B, 4B) from their directories: the
decoder generates through a hybrid key-value cache (the full-attention layers keep everything, the
sliding-window layers keep the window), a message list is rendered through the release's own chat
template, a submitted job streams each token, and the multimodal 4B takes an image beside the text.
The same factory sits behind `NFKMLXGemmaLanguage.backend(directoryURL:)`, which reads the model
type and routes a Gemma 3 release here.

```swift
let backend = try NFKMLXGemma3.backend(directoryURL: releaseDirectory)      // unsloth/gemma-3-1b-it
let request = NFKInferenceRequest(inputs: [NFKInputMessages: [["role": "user", "content": "Name one colour of the rainbow."]]],
                                  parameters: [NFKParameterMaxTokens: 32, NFKParameterTemperature: 0])
let reply = try backend.runInference(for: request).text
```

```objc
id<NFKInferenceBackend> backend = [NFKMLXGemma3 backendWithDirectoryURL:releaseDirectory error:&error];
NFKInferenceRequest *request =
    [NFKInferenceRequest requestWithInputs:@{ NFKInputImage: (__bridge id)cgImage,       // the 4B only
                                              NFKInputMessages: @[ @{ @"role": @"user", @"content": @"Describe this image." } ] }
                                parameters:@{ NFKParameterMaxTokens: @64 }
                            outputModality:NFKModalityText];
NSString *answer = [backend runInferenceForRequest:request error:&error].text;
```

**Gemma 3n.** `NFKMLXGemma3n` runs the tri-modal E2B and E4B releases: a picture through
MobileNetV5-300M, a clip through the Universal Speech Model Conformer, and the decoder over the fused
prompt. It is a separate architecture from Gemma 3 — AltUp's four parallel residual copies, the LAuReL
detour, per-layer embeddings, activation sparsity, and a trailing run of layers that computes no keys
or values — so it has its own factory rather than being routed through `NFKMLXGemmaLanguage`.

```swift
let gemma = try NFKMLXGemma3n.load(directoryURL: releaseDirectory)   // unsloth/gemma-3n-E2B-it
let answer = try gemma.answer(image: cgImage, question: "Describe this image.")

// Or through the contract, which also takes audio:
let backend = try NFKMLXGemma3n.backend(directoryURL: releaseDirectory)
let request = NFKInferenceRequest(inputs: [NFKInputAudio: clip,
                                           NFKInputPrompt: "Transcribe this."],
                                  parameters: [NFKParameterMaxTokens: 64])
let reply = try backend.runInference(for: request).text
```

```objc
NFKMLXGemma3n *gemma = [NFKMLXGemma3n gemma3nWithDirectoryURL:releaseDirectory error:&error];
NSString *answer = [gemma answerForImage:cgImage question:@"Describe this image." error:&error];
```

**DeepSeek V4.1 Flash.** `NFKMLXDeepSeek` runs the V4 and V4.1 decoders. Generation is incremental
through `NFKMLXDeepSeekCache`, which carries what a step cannot recompute: each layer's sliding
window, the compressed key-value its four source layers publish, the index keys those sources
publish, the group a compressor has pooled but not emitted, and the n-gram memory's id history. The
factory derives the collapsed token map that memory addresses through from the release's own
tokenizer.

A load computes in bf16, the dtype the release declares and its own inference code runs in, and
matches that code bit for bit. The release stores its weights fp8 and fp4, so a load needs two to four
times what the directory measures: V4.1 Flash is 510 GB stored and 1.39 TiB decoded to bf16. The
directory factory holds the release as an `NFKMLXResidency` says. Under the default `.automatic` a
release that fits loads resident, one that does not is paged, holding its experts stored where that fits
and mapping every paged group otherwise, and one that does not fit even mapped is refused before any
weight is read. `.resident` loads decoded or refuses.

```swift
let automatic = try NFKMLXDeepSeek.backend(directoryURL: releaseDirectory)
let decoded = try NFKMLXDeepSeek.backend(directoryURL: releaseDirectory, residency: .resident)
```

The `paging:` presets choose the groups explicitly.

`NFKMLXDeepSeekPaging` holds a group in the form the release stores it and decodes only what a step
reads: the routed experts an expert at a time as the router reaches them, and the n-gram tables a
row at a time as they are looked up. The release stores those tables with one scale per row per 32
channels precisely so a row can decode on its own.

```swift
let paged = try NFKMLXDeepSeek.backend(directoryURL: releaseDirectory, paging: .all)
```

`.mapped` and `.fullyMapped` go further: a mapped group is left in the release and only the bytes a
step reads are copied out, so its cost becomes the operating system's page cache. What the V4.1
Flash decoder allocates goes 1396.4 GiB resident, 475.5 with both groups held stored, 286.6 with the
n-gram tables mapped, and 17.7 GiB with every group mapped. The release has to stay where it was
loaded from, and a mapped decoder produces the same logits as a held one.

```swift
let mapped = try NFKMLXDeepSeek.backend(directoryURL: releaseDirectory, paging: .fullyMapped)
```

`quantizesActivations` adds the release's own activation rounding: activations and GEMM inputs
rounded where its inference code rounds them.
`computesInFloat32` opts out of bf16 for the float32 model the release's arithmetic approximates,
at twice the bytes a step reads (a fully mapped decoder goes from 17.7 GiB to 32.7).

```swift
let served = try NFKMLXDeepSeek.backend(directoryURL: releaseDirectory, paging: .fullyMapped,
                                        quantizesActivations: true)
let wide = try NFKMLXDeepSeek.backend(directoryURL: releaseDirectory, paging: .fullyMapped,
                                      computesInFloat32: true)
```

A long prompt prefills in chunks through the same cache, which bounds the peak by the chunk rather
than by the prompt, and a release carrying an image tower answers about a picture.

```swift
let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "What is in this picture?",
                                           NFKInputImage: photograph],
                                  parameters: [NFKMLXGenerationParameterKey.prefillChunkSize: 512])
let answer = try paged.runInference(for: request).output(forKey: NFKOutputText)
```

```swift
let backend = try NFKMLXDeepSeek.backend(directoryURL: releaseDirectory)
let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "Explain latent attention in one sentence."],
                                  parameters: [NFKParameterMaxTokens: 64])
let text = try backend.runInference(for: request).text

// What a load would need, without reading the release:
let bytes = NFKMLXDeepSeek.residentBytes(for: .v41Flash)
try NFKMLXDeepSeek.verifyFits(.v41Flash)               // throws, naming the shortfall
```

```objc
id<NFKInferenceBackend> deepSeek =
	[NFKMLXDeepSeek deepSeekBackendWithDirectoryURL:releaseDirectory error:&error];
id<NFKInferenceBackend> decoded =
	[NFKMLXDeepSeek deepSeekBackendWithDirectoryURL:releaseDirectory residency:NFKMLXResidencyResident error:&error];

// Every load choice sits on one options object; a paging preset overrides its residency.
NFKMLXDeepSeekLoadOptions *options = [NFKMLXDeepSeekLoadOptions new];
options.paging = NFKMLXDeepSeekPagingModeFullyMapped;
options.quantizesActivations = YES;
id<NFKInferenceBackend> served =
	[NFKMLXDeepSeek deepSeekBackendWithDirectoryURL:releaseDirectory options:options error:&error];
```

**Constraining the output.** A grammar mask over the logits admits only the tokens that keep the
output inside the grammar, so structured output needs no model change and no retry. JSON syntax and
a fixed set of choices ship; a custom constraint adopts `NFKMLXTokenConstraint`. The mask is applied
before temperature and nucleus filtering, so sampling stays inside the grammar too.

```swift
var options = NFKMLXGenerationOptions()
options.jsonOutput = true                              // well-formed JSON, ended when it closes
options.jsonRoot = .object                             // an object rather than an array
// options.choices = ["yes", "no", "unsure"]           // exactly one of these

// A JSON Schema guarantees the keys, the types, the enumerations, and the array bounds as well:
options.jsonSchema = try NFKMLXJSONSchema(jsonText: """
    {"type": "object",
     "properties": {"city": {"type": "string"}, "population": {"type": "integer"},
                    "mood": {"enum": ["sunny", "rainy"]}},
     "required": ["city", "population"], "additionalProperties": false}
    """)

// Or build one against the release's own token bytes:
let vocabulary = NFKMLXVocabulary(tokenizer: tokenizer, size: configuration.vocabularySize)
options.constraint = NFKMLXJSONConstraint(vocabulary: vocabulary, root: .object)
```

Through a request, the schema is the core's own `NFKParameterJSONSchema` (a dictionary, the key the
remote backends read too), so the same request runs against a hosted provider or the on-device model;
the reply comes back parsed under `NFKOutputStructured` beside the text. A schema the grammar cannot
enforce (`allOf`, `not`, a `required` name not under `properties`) is an error, not an unconstrained run.

Two things the grammar cannot do for the model. JSON admits unbounded whitespace, and a model whose
preferred next token is forbidden takes the whitespace it is offered indefinitely, so the grammar
caps a whitespace run (eight bytes by default). And a thinking model opens with a `<think>` block the
grammar forbids; the prompt closes that block itself (`<think>\n\n</think>\n\n` after the assistant
marker, as Qwen3's own template does for its no-think mode) so the answer starts at the document.

### Text embeddings (`NFKMLXQwen3Embedding`)

The decoder read one layer earlier is a text embedder: the post-final-norm hidden states are pooled to
one vector per text and L2-normalized, so a dot product between two embeddings is their cosine
similarity. Qwen3-Embedding-0.6B is the Qwen3-0.6B backbone this package already runs, pooled at the
last token over an appended `<|endoftext|>`. A query carries a one-sentence task instruction and a
document carries none; the two are asymmetric on purpose.

```objc
id<NFKInferenceBackend> embedder =
    [NFKMLXQwen3Embedding backendWithDirectoryURL:releaseDirectory error:&error];

NSString *query = [NFKMLXQwen3Embedding instructWithTask:@"Retrieve passages that answer the query"
                                                   query:@"What is the capital of France?"];
NFKInferenceResult *result =
    [embedder runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: query }]
                               error:&error];
NSArray<NSNumber *> *embedding = result.embedding;                 // 1024 floats, unit length
```

`backendWithDirectoryURL:outputDimensions:error:` truncates each embedding to a smaller Matryoshka
width before normalizing, trading accuracy for storage with no second model. A Swift caller that has
token ids already reads them straight through `embedding(forTokens:)`; the request path needs the
tokenizer the backend was built with.

Reference parity against the model card's own transformers recipe on the released 0.6B weights: query
embedding cosine 0.99999999999, document 0.99999999999, and the retrieval score reproduced to 1e-6 end
to end.

**A second embedder, a second architecture.** `NFKMLXEmbeddingGemma` is EmbeddingGemma-300M: the
bidirectional Gemma 3 encoder (no causal mask, sandwich normalization, dual RoPE), mean-pooled over
every token, run through a Dense bottleneck (768 → 3072 → 768), and L2-normalized. It reads Gemma's
`tokenizer.json` directly, so the text path needs no conversion.

```objc
id<NFKInferenceBackend> embedder =
    [NFKMLXEmbeddingGemma backendWithDirectoryURL:releaseDirectory error:&error];       // unsloth mirror
NSString *query = [NFKMLXEmbeddingGemma query:@"What is the capital of France?"];       // task prompt
NSArray<NSNumber *> *embedding =
    [embedder runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: query }]
                               error:&error].embedding;                                  // 768 floats
```

The backbone is Gemma 3 (`gemma3_text`), not the causal Gemma 4 the language model runs; the two
differ in their normalization, rotary, and per-layer embeddings, so the encoder is its own
implementation. Reference parity against the sentence-transformers pipeline on the released 300M
weights: every one of the 24 layers exact, query and document embedding cosine 0.99999999999, and the
Gemma byte-fallback BPE tokenizer reproduces the reference's ids token for token.

### Reranking (`NFKMLXModernBERTReranker`)

An embedder scores a query and a document independently; a **cross-encoder** reads the pair together —
`[CLS] query [SEP] document [SEP]` — through one bidirectional pass and predicts a single relevance
score, which is more accurate and is what reorders an embedder's shortlist. `NFKMLXModernBERTReranker`
is the released `gte-reranker-modernbert-base`.

```objc
NFKMLXModernBERTReranker *reranker =
    [NFKMLXModernBERTReranker rerankerWithDirectoryURL:releaseDirectory error:&error];
NSArray<NSNumber *> *order =                                    // documents, most relevant first
    [reranker rankedIndicesForQuery:@"What is the capital of France?"
                          documents:@[ @"The Great Barrier Reef is off Australia.",
                                       @"Paris is the capital of France." ]];   // → [1, 0]
double score = [reranker scoreForQuery:@"What is the capital of France?"
                              document:@"Paris is the capital of France."];      // a relevance logit
```

ModernBERT is a modernized BERT encoder: rotary position embeddings (a global base every third layer, a
smaller local base with a 128-token sliding window elsewhere), a GeGLU feed-forward, LayerNorm without
biases, and no absolute position embeddings. Reference parity against transformers' own
`ModernBertForSequenceClassification` on the released weights: every one of the 22 layers exact, and both
the relevant and irrelevant scores reproduced to within 5e-3. The tokenizer is GPT-2-family byte-level
BPE, read from the release's `tokenizer.json`.

### Typed decisions on device (`NFKMLXLaya`)

Laya (`convaiinnovations/laya`, Apache-2.0) is the open reproduction of TypeSafe's Jev: it answers the
same three question types about a state in one bidirectional pass, without generating text. It takes the
same `NFKDecisionQuestion`s and returns the same `NFKDecisionAnswer`s as `NFKTypeSafeBackend`, so a
feature written against the hosted model runs on device by swapping the object. Every option is scored
at its own mask token; a softmax over the question's markers is the answer distribution.

```objc
NFKMLXLaya *laya = [NFKMLXLaya layaWithDirectoryURL:releaseDirectory error:&error];   // the root, typed-decisions, or multilingual folder
NSDictionary<NSString *, NFKDecisionAnswer *> *answers =
    [laya answersForState:@"Help! My payouts have been failing for 3 days."
                questions:@{ @"department": [NFKDecisionQuestion choiceQuestionWithInstructions:@"Which team should handle this?"
                                                                                       options:@[ @"billing", @"technical", @"sales" ]],
                             @"urgent":     [NFKDecisionQuestion noulQuestionWithInstructions:@"The customer needs an answer today."] }];
answers[@"department"].choice;          // "technical"
answers[@"urgent"].probability;         // 0.9

// The same through the contract, the request NFKTypeSafeBackend reads:
NFKMLXLayaBackend *backend = [NFKMLXLaya backendWithDirectoryURL:releaseDirectory error:&error];
NFKInferenceResult *decided = [backend runInferenceForRequest:
    [NFKInferenceRequest requestWithInputs:@{ NFKInputState: record, NFKInputQuestions: questions }] error:&error];
decided.answers[@"urgent"].probability;
```

```swift
let laya = try NFKMLXLaya.laya(directoryURL: releaseDirectory)
let answers = laya.decide(state: "Help! My payouts have been failing for 3 days.", questions: [
    "department": .choiceQuestion(withInstructions: "Which team should handle this?", options: ["billing", "technical", "sales"]),
    "urgent": .noulQuestion(withInstructions: "The customer needs an answer today."),
])
```

The network is the ModernBERT encoder plus a decision head: a question-type embedding, two pre-norm
transformer layers, a scorer read at each marker, and an act-or-escalate head whose probability rides
under `raw["rl_agent"]["act_probability"]`. Three variants ship: the root (ModernBERT-large, 421M),
`typed-decisions` (the same geometry fine-tuned on that benchmark), and `multilingual` (mmBERT-base
with Gemma's tokenizer, 100-plus languages). Reference parity against the release's own inference code
on all three: the prompt token for token, the raw logits, the calibrated probabilities, and the act
probability. The release's README says the base checkpoints sit near chance on a new decision task
and that the capability comes from fine-tuning, which is the recipe under "Customizing a model".

#### Downloading and setting up Laya

The weights are on Hugging Face as `convaiinnovations/laya` (Apache-2.0, ungated, no token). Each
variant is five files in its own folder; `NFKMLXLayaVariant` names the three, and one call downloads a
variant into the `NFKHFHub` cache and builds it. A cached file is read from disk, so later launches make
no network request.

| Variant | Folder | Download | Memory once loaded |
| --- | --- | --- | --- |
| `NFKMLXLayaVariantRoot` | repository root | 846 MB | about 1.7 GB |
| `NFKMLXLayaVariantTypedDecisions` | `typed-decisions/` | 846 MB | about 1.7 GB |
| `NFKMLXLayaVariantMultilingual` | `multilingual/` | 678 MB | about 1.3 GB |

```objc
// Blocks on the network and the load: call it off the main thread.
NFKMLXLaya *laya = [NFKMLXLaya layaWithVariant:NFKMLXLayaVariantTypedDecisions
                                      revision:NFKMLXLaya.measuredRevision   // the commit parity was measured at
                             cacheDirectoryURL:nil                          // Application Support/InferKit/models
                                         error:&error];

// Or on a background queue, delivered to the handler there:
[NFKMLXLaya backendWithVariant:NFKMLXLayaVariantMultilingual revision:NFKMLXLaya.measuredRevision
             cacheDirectoryURL:nil completionHandler:^(NFKMLXLayaBackend *backend, NSError *error) { /* keep it */ }];
```

```swift
let laya = try NFKMLXLaya.laya(variant: .typedDecisions, revision: NFKMLXLaya.measuredRevision,
                               cacheDirectoryURL: nil)

// Fetch now, build later (an onboarding download); the folder holds the five files.
let folder = try NFKMLXLaya.download(variant: .root, revision: NFKMLXLaya.measuredRevision, cacheDirectoryURL: nil)
let later = try NFKMLXLaya.laya(directoryURL: folder)

// Keep the release through a cache size limit; eviction removes a whole repo revision at once.
try NFKHFHub(cacheDirectoryURL: NFKHFHub.defaultCacheDirectoryURL())
    .pinCachedRepo(NFKMLXLaya.repository, revision: NFKMLXLaya.measuredRevision)
```

A nil revision caches `main` and never re-checks it, so pin `measuredRevision` when two devices must
answer identically. The cache folder is excluded from backup by default. An app that bundles the
release skips the download and passes the variant folder to `laya(directoryURL:)`. The weights load at
float32, twice their download size, and the factory refuses a release the machine cannot hold. The
DocC article for `NFKMLXLaya` covers the cache, sandboxed folders, and the tuned-weights reload.

### Community Jev reproductions (`NFKMLXOpenJevDeBERTa`, `NFKMLXOpenJev`)

Two more open reproductions of Jev take the same `NFKDecisionQuestion`s and return the same
`NFKDecisionAnswer`s as Laya and `NFKTypeSafeBackend`. Each is a different design, so the choice is a
trade between cost and how each question is read:

| Model | Design | Download | A question costs |
| --- | --- | --- | --- |
| `NFKMLXOpenJevDeBERTa` (`com-kotobalabs/open-jev-deberta-v3-large`, Apache-2.0) | DeBERTa-v3-large reads the state and every question in one pass; a head scores each option from the mean of its text and its question's text | 1.76 GB | a share of one 512-token pass |
| `NFKMLXOpenJev`, `.twoB` (`ZefanCai/Open-Jev-2B`) | a LoRA adapter on Qwen3.5-2B scores each candidate as its own Yes/No prompt | 10 MB adapter + the 4.6 GB base | one prompt per option |
| `NFKMLXOpenJev`, `.nineB` (`ZefanCai/Open-Jev-9B`) | the same recipe on Qwen3.5-9B | 24 MB adapter + the 19.3 GB base | one prompt per option |
| `NFKMLXOpenJev`, `.twentySevenB` (`ZefanCai/Open-Jev-27B-v1.1`) | the same recipe on Qwen3.8-27B | 62 MB adapter + the base, about 54 GB | one prompt per option |

```objc
// One DeBERTa pass answers every question; the list form keeps your order, which the model reads by.
NFKMLXOpenJevDeBERTa *deberta = [NFKMLXOpenJevDeBERTa openJevWithRevision:NFKMLXOpenJevDeBERTa.measuredRevision
                                                         cacheDirectoryURL:nil error:&error];
NSArray<NFKDecisionAnswer *> *ordered = [deberta answersForState:@"I was charged twice." questionList:questions error:&error];

// Open-Jev downloads the adapter and the Qwen3.5 base revision the adapter names.
NFKMLXOpenJev *openJev = [NFKMLXOpenJev openJevWithVariant:NFKMLXOpenJevVariantTwoB revision:nil
                                         cacheDirectoryURL:nil error:&error];
NSDictionary<NSString *, NFKDecisionAnswer *> *answers =
    [openJev answersForState:record questions:@{ @"intent": intent, @"refund": refund } error:&error];
```

```swift
let deberta = try NFKMLXOpenJevDeBERTa.openJev(revision: NFKMLXOpenJevDeBERTa.measuredRevision, cacheDirectoryURL: nil)
let ordered = try deberta.decide(state: "I was charged twice.", questions: [intent, refund])

let openJev = try NFKMLXOpenJev.openJev(variant: .twoB, revision: nil, cacheDirectoryURL: nil)
let backend = openJev.makeBackend()        // NFKInputState + NFKInputQuestions in, NFKOutputAnswers out
```

Both download through the `NFKHFHub` cache the way Laya does, and both build from local folders too:
`openJev(directoryURL:)` for the DeBERTa release, `openJev(checkpointDirectoryURL:baseDirectoryURL:)`
for an Open-Jev `package/checkpoint` folder and its base. Open-Jev loads the base at its own bfloat16 by
default; `precision: .float32` doubles the memory and is what a fine-tune needs.

- **open-jev-deberta** refuses questions that cannot fit in 512 tokens beside the state, and cuts the
  state to 256 tokens or to whatever room the questions leave. The confidence is the largest
  probability, at the release's fitted temperature of 1.05.
- **Open-Jev** refuses a candidate prompt longer than 4,096 tokens rather than truncating it. A noul
  takes both meanings or neither. The confidence follows the loader's formulas: a choice's is how far
  its top probability sits above uniform, a score's is how concentrated it is around its mode.
- A described option reads as `name: description` in both, as in Laya.

Reference parity against each release's own code: open-jev-deberta through its bundled
`typed_decisions` package (every encoder layer, the logits, the answers, a padded batch), and
Open-Jev-2B through the loader's `DecisionModel` at float32 (every candidate's tokens, every hidden
state of the adapted text model, the logits, the answers). The 9B release does not fit at float32 on a
32 GB machine, so it is compared at bfloat16 against the loader at bfloat16: the same decisions, with
probabilities within 0.004.

### Time-series forecasting (`NFKMLXChronos`)

Chronos-Bolt reads a numeric context window and forecasts a horizon as quantiles (0.1 … 0.9), so the
median row is the point forecast and the outer rows form a prediction interval. It is an object, not a
backend — a numeric series has no core input key.

```swift
let chronos = try NFKMLXChronos.chronos(weightsURL: weightsURL)    // amazon/chronos-bolt-base
let rows = chronos.forecast(context: history, horizon: 64)          // [9 quantiles][64 steps]
let median = rows[4]                                                // the 0.5-quantile point forecast
```

Chronos-Bolt is a patched T5 encoder-decoder: it standardizes the series, splits it into 16-sample
patches, runs the encoder and a single-token decoder, and maps that vector to `quantiles × horizon`.
Reference parity against the `chronos` package's own `ChronosBoltPipeline`: every seam ~1.0 and all nine
quantile rows matching.

### Multimodal retrieval (`NFKMLXQwen3VLEmbedder`, `NFKMLXQwen3VLReranker`)

The two embedders above read text. `NFKMLXQwen3VLEmbedder` is the released `Qwen3-VL-Embedding-2B`,
which embeds a text, an image, or both into one space, so a text query retrieves an image and an image
query retrieves a document. `NFKMLXQwen3VLReranker` is `Qwen3-VL-Reranker-2B`, the cross-encoder over
the same backbone. An instruction conditions both.

```objc
NFKMLXQwen3VLEmbedder *embedder =
    [NFKMLXQwen3VLEmbedder embedderWithDirectoryURL:releaseDirectory error:&error];
NSArray<NSNumber *> *query = [embedder embeddingForText:@"a red bicycle leaning on a wall"];
NSArray<NSNumber *> *picture = [embedder embeddingForImage:cgImage text:@"" instruction:@""];
// Both are L2-normalized, so their dot product is a cosine similarity.

NFKMLXQwen3VLReranker *reranker =
    [NFKMLXQwen3VLReranker rerankerWithDirectoryURL:rerankerDirectory error:&error];
NSArray<NSNumber *> *order =                                    // documents, most relevant first
    [reranker rankedIndicesForQuery:@"How tall is the Eiffel Tower?"
                          documents:@[ @"Sourdough needs a starter kept at room temperature.",
                                       @"The Eiffel Tower stands 330 metres tall." ]];   // → [1, 0]
```

The embedder pools the last position of the prompt and normalizes it; the reranker reads the same
position through the output projection and takes the difference between the "yes" and "no" logits
through a sigmoid, which is a relevance between 0 and 1. The pooled position differs between the two
releases because their tokenizer files differ: the embedding release appends `<|endoftext|>` to every
encoding and the reranker release does not. Reference parity against each release's own script on the
released 2B weights: the prompts tokenize to the reference's ids exactly, the text embedding cosine is
0.9999999999866735, the image embedding 0.9999999999305262, and the reranker's scores agree to 2.4e-6.

### Vision-language (`NFKMLXSmolVLM`)

SmolVLM2-500M answers a question about an image. A SigLIP vision encoder turns each tile into patch
features, a pixel-shuffle connector projects them to the decoder width, and a Llama decoder reads the
text with the projected vision tokens spliced in at the image-token positions.

```objc
NFKMLXSmolVLM *model = [NFKMLXSmolVLM smolVLMWithDirectoryURL:releaseDirectory error:&error];
NSString *answer = [model answerForImage:cgImage question:@"What is in this image?"];   // a caption
```

The prompt expansion (`User:` + the tiled `<image>` structure + the question) is token-exact against the
processor, and the network is at reference parity against transformers'
`SmolVLMForConditionalGeneration`: the vision encoder, the connector, and the fused decoder logits are
exact, and the greedy continuation matches token for token. The image processor is CoreGraphics-based, so
a caption is not token-identical to the reference's PIL pipeline; it is coherent and accurate.

### Vision-language (`NFKMLXPixtral`)

Pixtral 12B answers a question about an image. A from-scratch 2D-rotary vision tower reads the picture at
its native aspect ratio, a two-layer GELU connector projects the patch features to the decoder width, and
a Mistral-Nemo decoder reads the text with the projected vision tokens spliced in at the `[IMG]`
positions.

```objc
NFKMLXPixtral *model = [NFKMLXPixtral modelWithDirectoryURL:releaseDirectory error:&error];   // mistral-experimental/pixtral-12b
NSString *answer = [model answerForImage:cgImage question:@"Describe this image in detail." maxTokens:64];
```

The vision tower and connector are at reference parity against transformers'
`LlavaForConditionalGeneration` (patch embedding, ln_pre, blocks, and connector measured in float32), and
the whole fused pipeline matches a tiny float32 oracle with the reference's argmax at every position. The
image processor is CoreGraphics-based, so a caption is not token-identical to the reference's PIL
pipeline; it is coherent and accurate.

### Unified vision (`NFKMLXFlorence2`)

Florence-2 (base or large, whichever release directory it is given) reads one image and a task token and writes text: a caption, detected objects, or
grounded regions. A DaViT vision tower (windowed spatial attention paired with grouped channel attention
in every block) feeds a projector, whose image tokens concatenate before the prompt for a BART
encoder-decoder; the localization tasks come back as `NFKDetection`s with boxes.

```objc
id<NFKInferenceBackend> florence = [NFKMLXFlorence2 backendWithDirectoryURL:releaseDirectory error:&error];
NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{
    NFKInputImage: (__bridge id)cgImage,
    NFKInputPrompt: @"<OD>"                                   // or <CAPTION>, <DENSE_REGION_CAPTION>, …
}];
NFKInferenceResult *result = [florence runInferenceForRequest:request error:&error];
NSArray<NFKDetection *> *objects = result.detections;        // for the localization tasks
```

The DaViT tower, projector, BART encoder, and first-step logits are at reference parity against the
release's own implementation. Generation follows the release's settings (three beams, no repeated
3-gram) and matches the release's own `generate` token for token on captioning, OCR, and detection.
`NFKParameterMaxTokens` and `NFKMLXTranslationParameterKey.beamCount` override them per request.

### Handwriting reading (`NFKMLXTrOCR`)

TrOCR reads a line of handwriting into text. A plain `google/vit` image encoder patchifies the 384-square
image; its patch tokens are the memory a BART-style decoder cross-attends while generating the
transcription.

```objc
id<NFKInferenceBackend> trocr = [NFKMLXTrOCR backendWithDirectoryURL:releaseDirectory error:&error];
NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{
    NFKInputImage: (__bridge id)cgImage
}];
NFKInferenceResult *result = [trocr runInferenceForRequest:request error:&error];
NSString *transcription = [result outputForKey:NFKOutputText];
```

The ViT encoder and the decoder's first-step logits are at reference parity against transformers' own
`VisionEncoderDecoderModel`, and greedy generation matches its transcription token for token.

### Referring segmentation (`NFKMLXSa2VA`)

Sa2VA reads an image and a referring prompt, answers in text, and segments the object it refers to. An
InternViT-300M encoder and a pixel-shuffle projector feed a Qwen2.5-3B decoder; when the decoder emits a
`[SEG]` token, its hidden state becomes a prompt for a SAM 2 grounding encoder, which returns the mask
under `NFKOutputMask`.

```objc
id<NFKInferenceBackend> sa2va = [NFKMLXSa2VA backendWithDirectoryURL:releaseDirectory error:&error];
NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{
    NFKInputImage: (__bridge id)cgImage,
    NFKInputPrompt: @"<image>Please segment the person on the left."
}];
NFKInferenceResult *result = [sa2va runInferenceForRequest:request error:&error];
NSString *answer = [result outputForKey:NFKOutputText];
CGImageRef mask = (__bridge CGImageRef)[result outputForKey:NFKOutputMask];
```

Every seam is at reference parity against the model's own code — the InternViT tower, the projector, the
image/text fusion, the `[SEG]` bridge, and the mask itself (IoU 0.9998) — and generation is token-exact.

### Image, speech, and text in one model (`NFKMLXPhi4MM`)

Phi-4-multimodal answers a prompt or a conversation that may carry pictures, clips, or both. The inputs
choose the mode: a picture runs the decoder's vision adapter, audio alone runs its speech adapter, and text
alone runs the base decoder. Audio with no prompt is transcribed. WAV audio at any rate from 8 kHz up is
read the way the release's own processor reads it.

```objc
id<NFKInferenceBackend> phi = [NFKMLXPhi4MM backendWithDirectoryURL:releaseDirectory error:&error];

// Caption or question over a picture.
NFKInferenceRequest *describe = [NFKInferenceRequest requestWithInputs:@{
    NFKInputImage: (__bridge id)cgImage,
    NFKInputPrompt: @"Describe the image in one sentence."
}];
NSString *caption = [[phi runInferenceForRequest:describe error:&error] outputForKey:NFKOutputText];

// A spoken question about the same picture: both inputs in one request.
NFKInferenceRequest *ask = [NFKInferenceRequest requestWithInputs:@{
    NFKInputImage: (__bridge id)cgImage,
    NFKInputAudio: wavData
}];
NSString *answer = [[phi runInferenceForRequest:ask error:&error] outputForKey:NFKOutputText];

// A conversation over two pictures, sampled. Pictures and clips open the first user turn unless the text
// places them with <|image_1|>… and <|audio_1|>…; further ones go under NFKInputImages and NFKInputAudios.
NFKInferenceRequest *chat = [NFKInferenceRequest requestWithInputs:@{
    NFKInputMessages: @[
        @{ @"role": @"system", @"content": @"You answer in one short sentence." },
        @{ @"role": @"user", @"content": @"<|image_1|><|image_2|>How many pictures are there?" },
        @{ @"role": @"assistant", @"content": @"There are two pictures." },
        @{ @"role": @"user", @"content": @"What differs between them?" },
    ],
    NFKInputImage: (__bridge id)firstImage,
    NFKInputImages: @[ (__bridge id)secondImage ],
} parameters:@{ NFKParameterTemperature: @0.7, NFKParameterTopP: @0.95, NFKParameterSeed: @7 }];
NSString *reply = [[phi runInferenceForRequest:chat error:&error] outputForKey:NFKOutputText];
```

`backendWithDirectoryURL:precision:error:` loads the decoder at float32 instead of the released bfloat16,
and `backendWithRepo:revision:cacheDirectoryURL:precision:error:` downloads the release
(`NFKMLXPhi4MM.releaseRepo`) first. Every mode is at reference parity against the release's own code, with
token-exact answers from raw inputs, including a multi-turn conversation over two pictures and two clips
and a clip past 40 seconds; both preprocessors match the reference processor. The compiled weight-free
counterpart is `testPhi4Multimodal` in `InferKitMLX/Examples/MLXModelGalleryExamples.swift`, which runs both
preprocessors, tiny towers (two clips as one batch), and a tiny LongRoPE decoder fusing a picture and the
clips.

### Table structure recognition (`NFKMLXTableTransformer`)

Table Transformer recognizes the structure of a table crop. A ResNet-18 backbone reads the image, and a
DETR encoder-decoder predicts one box per structural element: the table, its rows, its columns, and its
headers. The backend reads the release's `config.json` for the class names, so the detections arrive
labeled.

```objc
id<NFKInferenceBackend> tableTransformer = [NFKMLXTableTransformer backendWithDirectoryURL:releaseDirectory error:&error];
NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{
    NFKInputImage: (__bridge id)cgImage
}];
NFKInferenceResult *result = [tableTransformer runInferenceForRequest:request error:&error];
NSArray<NFKDetection *> *structure = [result outputForKey:NFKOutputDetections];   // "table", "table row", "table column", …
```

Every seam is at reference parity against transformers' own `TableTransformerForObjectDetection` on the
released weights, and the backend recognizes a clean grid end to end as a table with its rows and columns.

### Vision-language (`NFKMLXGemma3`, the 4B)

Gemma 3 4B answers a question about an image. The SigLIP so400m tower reads the picture at 896×896, the
projector average-pools its 4096 patch features to 256 soft tokens, and the decoder reads them in front
of the question with the image's tokens attending to each other in both directions.

```objc
NFKMLXGemma3 *gemma = [NFKMLXGemma3 gemma3WithDirectoryURL:releaseDirectory error:&error];   // unsloth/gemma-3-4b-it
NSString *answer = [gemma answerForImage:cgImage question:@"Describe this image in one sentence." error:&error];
```

The prompt (the release's chat template, then the processor's `\n\n<start_of_image>` + 256 soft tokens +
`<end_of_image>\n\n` expansion) is token-exact against the processor, and the network is at reference
parity against transformers' `Gemma3ForConditionalGeneration` on the released weights: the vision tower,
the projector, the fused decoder's argmax at every position, and the greedy continuation. The image
processor is CoreGraphics-based, so an answer to a real photograph is not token-identical to the
reference's PIL pipeline; it is coherent and accurate.

### Remote, OpenAI-compatible (`NFKRemoteBackend`)

A localhost server (Ollama, `mlx_lm`) or a hosted API.

```objc
NSURL *endpoint = [NSURL URLWithString:@"http://localhost:11434/v1/chat/completions"];
NFKRemoteBackend *backend = [NFKRemoteBackend backendWithEndpointURL:endpoint];
backend.modelName = @"llama3.2";
backend.apiKey = nil;                                     // set for a hosted API

NFKInferenceRequest *request =
    [NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: @[ @{ @"role": @"user", @"content": @"Hi" } ] }
                                parameters:@{ @"temperature": @0.7 }];   // parameters fold into the request body
NSError *error = nil;
NSString *reply = [backend runInferenceForRequest:request error:&error].text;
```

`NFKRemoteBackendPromptKey` / `NFKRemoteBackendMessagesKey` / `NFKRemoteBackendTextKey` are the same
strings as `NFKInputPrompt` / `NFKInputMessages` / `NFKOutputText`, so the shared keys work here too.

### Async generation service (`NFKAsyncGenerationBackend`, submit → poll → fetch)

For a service that returns a job id and is polled for completion (many image/video generation APIs),
subclass `NFKAsyncGenerationBackend` and map the service's JSON through its template methods; the base
owns the submit/poll/fetch loop and the `NFKInferenceJob` (progress, cancellation).

```objc
@interface MyGenerationBackend : NFKAsyncGenerationBackend @end
@implementation MyGenerationBackend
- (NSDictionary<NSString *, id> *)submitBodyForRequest:(NFKInferenceRequest *)request {
    return @{ @"prompt": request.prompt ?: @"", @"model": self.modelName ?: @"" };
}
- (nullable NSString *)jobIdentifierFromResponse:(NSDictionary *)response { return response[@"id"]; }
- (nullable NSURL *)statusURLForJobIdentifier:(NSString *)jobID { return [self.submitURL URLByAppendingPathComponent:jobID]; }
- (BOOL)isSucceededStatusResponse:(NSDictionary *)response { return [response[@"status"] isEqual:@"succeeded"]; }
- (nullable NFKInferenceResult *)resultFromStatusResponse:(NSDictionary *)response error:(NSError **)error {
    return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: response[@"output"] ?: @"" }];
}
@end

MyGenerationBackend *backend = [[MyGenerationBackend alloc] init];
backend.submitURL = [NSURL URLWithString:@"https://api.example.com/v1/generations"];
backend.apiKey = @"…"; backend.modelName = @"my-generator"; backend.pollInterval = 1.0;
// submitInferenceJobForRequest: submits, polls at pollInterval, and fetches the result off-thread.
```

### Apple on-device (`NFKFoundationModelsBackend`, Swift)

macOS 26 / iOS 26 with Apple Intelligence. Streams and cancels through the job.

```swift
import InferKit
import InferKitFoundationModels

let backend = NFKFoundationModelsBackend()
guard backend.isReady else { /* SystemLanguageModel unavailable */ return }

let request = NFKInferenceRequest(
    inputs: [NFKInputMessages: [["role": "system", "content": "Answer in one word."],
                                ["role": "user", "content": "Name a primary color."]]],
    parameters: [NFKParameterMaxTokens: 16])
let reply = try backend.runInference(for: request).text
```

Choosing the model: on-device by default, specialized by `useCase` and `guardrails`; Private Cloud
Compute on macOS 27 / iOS 27, with the quota read before switching to it.

```swift
let backend = NFKFoundationModelsBackend()
backend.useCase = .contentTagging                    // the on-device tagging specialization
backend.guardrails = .permissiveContentTransformations
if #available(macOS 27, iOS 27, *) {
    if let quota = backend.privateCloudComputeQuota, !quota.isLimitReached {
        backend.model = .privateCloudCompute         // Apple's larger model; leaves the device
    }
}
```

The provider bridge, the other direction: an InferKit backend behind `LanguageModelSession`
(macOS 27 / iOS 27, built with the macOS 27 SDK).

```swift
let backend = NFKRemoteBackend(endpointURL: url)
backend.modelName = "qwen3:8b"
let model = NFKInferKitLanguageModel(backend: backend)   // capabilities from the keys it declares
let session = LanguageModelSession(model: model)
let reply = try await session.respond(to: "Name three sea birds.")
```

What the backend offers the bridge is readable from Objective-C, where `LanguageModelSession` is
not:

```objc
NFKInferKitLanguageModelCapabilities *capabilities =
    [[NFKInferKitLanguageModelCapabilities alloc] initWithBackend:remote];
BOOL tools = capabilities.toolCalling;                          // NFKParameterTools
BOOL images = capabilities.vision;                              // NFKInputImage
BOOL declared = [remote.supportedParameterKeys containsObject:NFKParameterJSONSchema];
```

### Translation (`NFKMLXMarian`, `NFKMLXM2M100`, `NFKMLXMADLAD`, Swift and Objective-C)

Three open-weight translators answer the same contract Apple's translator does in
`InferKitAppleSwift`: text under `NFKInputPrompt`, the target under `NFKParameterTargetLanguage`
(BCP-47, required), the source under `NFKParameterSourceLanguage` (optional; M2M-100 detects it), and
the translation under `NFKOutputText`. OPUS-MT is one small model per language pair, named by two
tags; M2M-100 covers 100 languages in one release; MADLAD-400 covers 400+ through a `<2xx>` marker;
TranslateGemma (`NFKMLXTranslateGemma`, gated on Hugging Face) is Gemma 3 driven by its translation
template, the strongest of the four and the largest.

```swift
let translator = try NFKMLXMarian.backend(sourceLanguage: "en", targetLanguage: "de", cacheDirectoryURL: nil)
let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "The quick brown fox jumps over the lazy dog."],
                                  parameters: [NFKParameterTargetLanguage: "de",
                                               NFKMLXTranslationParameterKey.beamCount: 4])
let result = try translator.runInference(for: request)
print(result.text ?? "")                      // Der schnelle Braunfuchs springt über den faulen Hund.

let manyToMany = try NFKMLXM2M100.backend(variant: .m418M, revision: nil, cacheDirectoryURL: nil)
let japanese = try manyToMany.runInference(for: NFKInferenceRequest(
    inputs: [NFKInputPrompt: "Where is the station?"],
    parameters: [NFKParameterTargetLanguage: "ja"]))   // the source is detected
```

```objc
NSError *error = nil;
id<NFKInferenceBackend> translator = [NFKMLXMADLAD backendWithRepo:@"google/madlad400-3b-mt" revision:nil
                                                  cacheDirectoryURL:nil halfPrecision:YES error:&error];
NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{NFKInputPrompt: @"Good morning."}
                                                           parameters:@{NFKParameterTargetLanguage: @"zh-Hant",
                                                                        NFKMLXTranslationParameterKey.beamCount: @4}];
NFKInferenceResult *result = [translator runInferenceForRequest:request error:&error];
```

The decode follows each release's generation config (Marian 4 beams, M2M-100 5, MADLAD greedy);
`NFKMLXTranslationParameterKey.beamCount`, `.lengthPenalty`, and `NFKParameterMaxTokens` override it.
Input is translated paragraph by paragraph; `NFKMLXTranslationParameterKey.splitsSentences` splits each
paragraph into sentences first. Linking InferKitMLX also registers `NFKMLXTranslationProvider` for the
core's `translation` capability (M2M-100 when its release is cached; Apple's translator otherwise).

### Streaming and cancellation

```swift
let job = backend.submitInferenceJob(for: request)
job.progressHandler = { j in
    if let text = j.partialResult?.output(forKey: NFKOutputText) as? String { render(text) }
}
job.completionHandler = { j in /* j.result or j.error */ }
// job.cancel()
```

### Reasoning and what the turn cost

`NFKParameterReasoningEffort` asks a reasoning model how hard to think. The three levels the contract
names are `NFKReasoningEffortLight`, `NFKReasoningEffortModerate`, and `NFKReasoningEffortDeep`, and
each backend maps them to its provider's control: `reasoning_effort` on an OpenAI-compatible endpoint,
a thinking budget on the Messages API, `ContextOptions.reasoningLevel` on Apple's model.

```objc
NFKInferenceRequest *request =
    [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Why is the sky blue?" }
                                parameters:@{ NFKParameterReasoningEffort: NFKReasoningEffortDeep }];
NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];

NSString *chain = [result outputForKey:NFKOutputReasoning];      // what the model showed, where it does
NSDictionary *usage = [result outputForKey:NFKOutputUsage];      // nil where the provider reports none
NSNumber *inputTokens = usage[NFKUsageInputTokens];              // beside NFKUsageCachedTokens,
NSNumber *outputTokens = usage[NFKUsageOutputTokens];            // NFKUsageOutputTokens, NFKUsageReasoningTokens
```

A count the provider leaves out is absent from the dictionary rather than zero, so read the key you
need and treat a missing one as unreported. A streamed reply carries the chain as it grows under
`NFKOutputReasoning` on the partial result, beside the text.

Reading a failure. The code is the decision, and it is the same code from any engine.

```objc
NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
if (result == nil) {
    switch (error.code) {
        case kNFKError_InferenceRefused:      /* change the request; retrying gives the same answer */ break;
        case kNFKError_InferenceRateLimited: {
            NSDate *reset = error.userInfo[NFKFoundationModelsErrorKey.resetDate];   // back off until then
            break;
        }
        case kNFKError_InferenceNotReady:     /* the model is unavailable; prepare() says why */ break;
        default: break;
    }
}
```

## Apple's own engines

The core wraps the Apple frameworks that overlap the model gallery, so a consumer reaches them
through the same contract with nothing to download. Each is an alternative to a shipped model rather
than a replacement for it: the models keep chosen weights, finer mattes, translation, and training.

Reading the text in an image. Vision does this on device at the core's floor, and the toolkit ships
no text-recognition model, so this is the one capability that arrives only here.

```objc
NFKVisionTextBackend *reader = [NFKVisionTextBackend backend];
reader.languages = @[ @"en-US" ];
NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputImage: (__bridge id)image }];
NFKInferenceResult *result = [reader runInferenceForRequest:request error:&error];
NSString *text = result.text;                       // the lines, in reading order
NFKDetection *first = result.detections.firstObject; // the first line, boxed
```

A mask, a pose, a face, or an image embedding, each from the same shape of request:

```objc
NFKVisionSegmentationBackend *subject =
    [NFKVisionSegmentationBackend backendWithKind:NFKVisionSegmentationKindForegroundInstance];
CVPixelBufferRef matte = (__bridge CVPixelBufferRef)[[subject runInferenceForRequest:request error:NULL]
                                                     outputForKey:NFKOutputMask];

NSArray<NFKKeypoint *> *joints = [[[NFKVisionPoseBackend backend] runInferenceForRequest:request error:NULL]
                                  outputForKey:NFKOutputPose];
NSArray<NFKDetection *> *faces = [[[NFKVisionFaceBackend backend] runInferenceForRequest:request error:NULL]
                                  outputForKey:NFKOutputDetections];
NSArray<NSNumber *> *print = [[[NFKVisionFeaturePrintBackend backend] runInferenceForRequest:request error:NULL] embedding];
```

Every box and joint is normalized 0...1 with the origin at the top left, the contract's geometry,
which is not Vision's; the backend converts.

A shape a box cannot describe arrives as four corners. A barcode's payload is its label, so reading
one is a detection like any other:

```objc
NFKVisionRectangleBackend *codes = [NFKVisionRectangleBackend backendWithKind:NFKVisionRectangleKindBarcode];
NFKDetection *code = [[codes runInferenceForRequest:request error:NULL] detections].firstObject;
NSString *payload = code.label;
NFKQuadrilateral *corners = code.quadrilateral;     // nil from an engine that reports no corners
CGPoint topLeft = corners.topLeft;                  // normalized, origin top left, like the box
```

Following a region across frames is the one stateful backend in the toolkit. It takes the region
once and a frame per run:

```objc
NFKVisionTrackingBackend *tracker = [NFKVisionTrackingBackend backend];
[tracker startTrackingBoundingBox:CGRectMake(0.4, 0.4, 0.2, 0.2)];
for (id frame in frames) {
    NFKInferenceRequest *step = [NFKInferenceRequest requestWithInputs:@{ NFKInputImage: frame }];
    NFKDetection *now = [[tracker runInferenceForRequest:step error:NULL] detections].firstObject;
}
[tracker reset];                                    // ends the sequence
```

The readings Vision puts a name to arrive under `NFKOutputStructured`: the aesthetics score, whether
a photograph is a utility shot, the horizon, the contours it traced, and the alignment between two
frames.

```objc
NSDictionary *reading = [[[NFKVisionMeasurementBackend backendWithKind:NFKVisionMeasurementKindAesthetics]
                          runInferenceForRequest:request error:NULL] structured];
NSNumber *score = reading[@"overallScore"];
```

A Core ML model can run through Vision instead of through `NFKCoreMLBackend`, which is the choice
between letting Vision resize and crop the image the model's way and handing the model tensors the
caller built:

```objc
NFKVisionCoreMLBackend *backend = [NFKVisionCoreMLBackend backendWithCompiledModelURL:url error:&error];
NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
```

Three of Apple's APIs ship only in Swift, so `InferKitAppleSwift/` hosts them and every type there
is `@objc`. A photographed page becomes a transcript and a structure:

```swift
let read = try NFKVisionDocumentBackend()
    .runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: page]))
read.text                                   // the whole transcript
read.structured?["tables"]                  // rows of cells, as strings
read.structured?["paragraphs"]              // and the paragraphs and lists beside them

let speech = NFKSpeechAnalyzerBackend(locale: Locale(identifier: "en-US"))
try speech.prepare()                        // reserves the locale and installs its assets
let words = try speech.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: asset]))
```

From Objective-C, which is why the package exists:

```objc
NFKVisionDocumentBackend *reader = [[NFKVisionDocumentBackend alloc] init];
NFKInferenceResult *read = [reader runInferenceForRequest:request error:&error];
NSDictionary *structure = read.structured;   // paragraphs, lists, tables
```

Apple's neural video processors. Upscaling and interpolation need macOS 26 on Apple silicon, and
upscaling runs a model the system downloads once, which `prepare` asks for.

```swift
let upscaler = NFKVideoToolboxBackend(task: .superResolution)
guard upscaler.isReady else { return }
try upscaler.prepare()                              // starts the model download, or throws notReady
let larger = try upscaler.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: frame]))

let flow = NFKVideoToolboxBackend(task: .opticalFlow)
let field = try flow.runInference(for: NFKInferenceRequest(inputs: [NFKInputImages: [previous, next]]))
// Packed as NFKMLXRAFT packs it: decode a channel with (value - 0.5) * 2 * flowScale.
```

Apple's field is smaller than the frame, where RAFT's is frame-sized, and how much smaller is the
processor's business rather than a fixed ratio. The values decode the same way from either, so read
the map's own dimensions, sample it in normalized coordinates, and scale to the destination rather
than pairing a pixel of the map with a pixel of the frame.

Transcribing with Apple's recognizer, which needs the user's consent once and keeps the audio on the
machine:

```swift
NFKSpeechRecognitionBackend.requestAuthorization { authorized in
    guard authorized else { return }
    let backend = NFKSpeechRecognitionBackend()
    backend.requiresOnDeviceRecognition = true
    let job = NFKInferenceSubmit(backend, NFKInferenceRequest(inputs: [NFKInputAudio: asset]), nil)
    job.completionHandler = { finished in
        let text = finished.result?.text
        let words = finished.result?.segments        // one per word, with its time range
    }
}
```

Apple classifies several hundred everyday sounds with nothing to download. Each analysis window
becomes a segment, and the clip's best guesses arrive beside them:

```objc
NFKSoundClassificationBackend *sounds = [NFKSoundClassificationBackend backend];
sounds.minimumConfidence = 0.05;                    // 0.3 by default, which a short clip may not clear
NFKInferenceResult *heard = [sounds runInferenceForRequest:
    [NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: asset }] error:&error];
NSArray<NFKAudioSegment *> *windows = heard.segments;          // labeled, with time ranges
NSArray<NFKClassification *> *overall = heard.classifications; // the whole clip
```

Speaking text answers the same contract as the remote and MLX voices, from voices already on the
machine:

```objc
NFKSpeechSynthesisBackend *voice = [NFKSpeechSynthesisBackend backend];
voice.language = @"en-US";                          // or voiceIdentifier, from +availableVoices
NFKInferenceResult *spoken = [voice runInferenceForRequest:
    [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"The plate is ready." }] error:&error];
NFKAudioAsset *audio = [spoken outputForKey:NFKOutputAudio];   // a WAV file on disk
```

The utterance is written on the main run loop, which is where Apple delivers its buffers. A caller
on the main thread has its own timers and sources run during the write; a caller off it, which is
what the contract asks for, needs the program's main run loop to be running.

Word and sentence vectors come from NaturalLanguage, with no model to ship:

```objc
NFKTextEmbeddingBackend *embedder = [NFKTextEmbeddingBackend backend];
NSArray<NSNumber *> *vector = [[embedder runInferenceForRequest:
    [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"a lighthouse at dawn" }]
    error:&error] embedding];
```

Translating on device is Swift-only in Apple's framework, so it lives in `InferKitAppleSwift/` and
reads the same keys the MLX translators read:

```swift
let backend = NFKTranslationBackend(sourceLanguage: "en", targetLanguage: "es")
backend.responseTimeout = 30                        // every wait on the framework is bounded
let request = NFKInferenceRequest(
    inputs: [NFKInputPrompt: "Good morning."],
    parameters: [NFKParameterTargetLanguage: "es"]) // the request's pair wins over the backend's
let translated = try backend.runInference(for: request).text
```

A pair Apple does not translate reports `kNFKError_InferenceUnsupported`. A pair whose model is not
installed reports `kNFKError_InferenceNotReady`, and so does a framework that does not answer, which
is why the wait has a bound rather than hanging the calling thread.

## Text → image and image → image

`NFKMLXBackend` runs a bundled Stable Diffusion release (Swift; Apple Silicon, macOS and iOS). No
image input runs text-to-image; a `CGImage` under `NFKInputImage` runs image-to-image. Three releases
are bundled: `.stableDiffusion15`, `.stableDiffusion21Base`, and `.sdxlTurbo`. The release's files
download from Hugging Face on first use; Stable Diffusion 2.1 base is a gated repository, so it needs
an access token: set `NFKHFHub.defaultAccessToken` before the first download (the backend makes its own
hub), or `HF_TOKEN` in the environment.

```swift
import InferKit
import InferKitMLX

let backend = NFKMLXBackend(model: .sdxlTurbo)

// text → image
let toImage = NFKInferenceRequest(
    inputs: [NFKInputPrompt: "a watercolor lighthouse at dawn"],
    parameters: [NFKParameterSteps: 2, NFKParameterSeed: 42],
    outputModality: .image)
let job = backend.submitInferenceJob(for: toImage)         // GPU work: prefer the job
job.completionHandler = { j in let image = j.result?.output(forKey: NFKOutputImage) }  // CGImage

// text + image → image
let img2img = NFKInferenceRequest(
    inputs: [NFKInputPrompt: "make it snowy", NFKInputImage: sourceCGImage],
    parameters: [NFKParameterStrength: 0.6],               // how much of the source survives
    outputModality: .image)
```

A caller holding a release already — downloaded, converted, or fine-tuned — builds the same model from
its directory, and chooses the precision it runs at:

```swift
let backend = try NFKMLXTextToImage.backend(configuration: .stableDiffusion15,
                                            directoryURL: releaseDirectory,
                                            precision: .checkpoint)
```

The directory is the release's own tree: `unet/`, `vae/`, `text_encoder/`, `tokenizer/`, plus
`text_encoder_2/` and `tokenizer_2/` for SDXL. `.float32` converts a half-precision release to the
module's own precision, which is what the parity records were measured at; `.checkpoint` runs it as
published.

### FLUX.2 [klein] text-to-image, editing and inpainting

A diffusers FLUX.2 [klein] release directory in, an image out. The facade renders the release's own
chat template, reads three intermediate layers of its Qwen3 text encoder, denoises over FLUX.2's
empirical sigma schedule, and decodes through the autoencoder and its latent codec.

```swift
let flux2 = try NFKMLXFlux2.flux2(directoryURL: releaseDirectory)
flux2.steps = 28
let image = try flux2.image(forPrompt: "a red fox in the snow", width: 1024, height: 1024, seed: 0)
```

```objc
NSError *error = nil;
NFKMLXFlux2 *flux2 = [NFKMLXFlux2 flux2WithDirectoryURL:releaseDirectory error:&error];
CGImageRef image = [flux2 imageForPrompt:@"a red fox in the snow"
                          negativePrompt:nil width:1024 height:1024 seed:0 error:&error];
```

Editing conditions on reference images, and inpainting repaints the white part of a mask and keeps
the rest. `strength` is how far into the schedule inpainting starts from the image: 1 regenerates the
masked region from noise, and lower values keep more of it. It is a `Double`, because the start step
is computed in double precision and a `Float` strength can start one step off. Guidance applies only
where the release is not step distilled (`isDistilled`, read from its `model_index.json`).

```swift
let edited = try flux2.image(forPrompt: "the same fox, at night", references: [photo])
let repainted = try flux2.inpaint(prompt: "a snowman", image: photo, mask: mask, strength: 0.8)
```

<!-- objc-check: continues -->
```objc
CGImageRef edited = [flux2 imageForPrompt:@"the same fox, at night" negativePrompt:nil
                               references:@[(__bridge id)photo] width:1024 height:1024 seed:0
                                    error:&error];
CGImageRef repainted = [flux2 inpaintImage:photo mask:mask prompt:@"a snowman" negativePrompt:nil
                                  strength:0.8 seed:0 error:&error];
```

A `CGImageRef` goes into the references array as `(__bridge id)`, because an array of a Core
Foundation type is not an Objective-C collection.

A release that does not fit the machine whole is staged: the text encoder loads, encodes, and is
released before the transformer loads, for every image. `.automatic` decides from the machine's
working set; FLUX.2 [klein] 9B stages on a 32 GB machine.

```swift
let nine = try NFKMLXFlux2.flux2(directoryURL: klein9BDirectory, residency: .automatic)
print(nine.holdsStagesResident, nine.encodesInFloat32)
```

<!-- objc-check: given NSURL *klein9BDirectory = nil; -->
```objc
NFKMLXFlux2 *nine = [NFKMLXFlux2 flux2WithDirectoryURL:klein9BDirectory
                                             residency:NFKMLXResidencyAutomatic error:&error];
```

FLUX.2 [klein] 9B KV runs its references once and reuses their keys and values on every later step.
Its files do not mark it, so the caller says so; it has no guidance and is meant for few steps.

```swift
let kv = try NFKMLXFlux2.flux2(directoryURL: kleinKVDirectory)
kv.cachesReferences = true
kv.steps = 4
let edited = try kv.image(forPrompt: "the same fox, at night", references: [photo])
```

FLUX.2 [dev] is gated, so the end-to-end path is [klein]'s. Its text front end ships: [dev] conditions
on Mistral-Small 3, which `NFKMLXLanguageConfiguration.mistralSmall3` carries and
`NFKMLXFlux2TextEncoder` drives at [dev]'s own layer spacing.

### Qwen-Image 2.1 (`NFKMLXQwenImagePipeline`)

Qwen-Image 2.1 is a 7.1B block-causal DiT with a vision-language text encoder. The weights are under
the Qwen Research License, which is non-commercial. `NFKMLXQwenImageGenerator` assembles the whole model
from the release directory and holds it as an `NFKMLXResidency` says. The 16 GB text encoder and the
14 GB transformer do not fit a 32 GB machine together, so `.automatic` stages them there: the encoder
loads, encodes, and is released before the transformer loads. A staged 256×256 image at 4 steps took
29 s on a 32 GB M-series machine.

```swift
let qwen = try NFKMLXQwenImageGenerator.generator(directoryURL: release, residency: .automatic)
let rgba = try qwen.image(forPrompt: "a calico cat asleep on a stack of books",
                          width: 1024, height: 1024, seed: 0)          // [1024, 1024, 4] in 0…1
```

```objc
NFKMLXQwenImageGenerator *qwen = [NFKMLXQwenImageGenerator generatorWithDirectoryURL:releaseDirectory
                                                                            residency:NFKMLXResidencyAutomatic
                                                                                error:&error];
CGImageRef cat = [qwen imageForPrompt:@"a calico cat asleep on a stack of books" negativePrompt:nil
                                width:1024 height:1024 seed:0 error:&error];
```

`guidance` is 1 by default, the reference's: the release is meant to be sampled without guidance, and a
negative prompt guides only above 1. Each stage also loads on its own, and the pipeline chains them:

```swift
let release = URL(fileURLWithPath: "…/Qwen-Image-2.1")

// The transformer is 7.1B in bfloat16, which is the precision it runs at.
let transformer = NFKMLXQwenImage.makeNet(try NFKMLXQwenImage.configuration(
    fromHuggingFace: release.appending(path: "transformer/config.json")))
try NFKMLXQwenImage.loadWeights(into: transformer,
                                fromDirectory: release.appending(path: "transformer"))

let vaeDirectory = release.appending(path: "vae")
let vae = try NFKMLXQwenImageVAE.net(directoryURL: vaeDirectory)
let (mean, deviation) = try NFKMLXQwenImageVAE.latentStatistics(
    fromHuggingFace: vaeDirectory.appending(path: "config.json"))

let pipeline = NFKMLXQwenImagePipeline(transformer: transformer, vae: vae, latentMean: mean,
                                       latentStandardDeviation: deviation)

// The text encoder is Qwen3-VL at the 8B geometry; its tokenizer lives in the release's processor.
let decoder = try NFKMLXQwen3VL.decoder(directoryURL: release.appending(path: "text_encoder"),
                                        precision: .checkpoint)
let tokenizer = NFKMLXLanguage.releaseTokenizer(inDirectory: release.appending(path: "processor"))!
let embeddings = NFKMLXQwenImagePipeline.promptEmbeddings(
    "a calico cat asleep on a stack of books", decoder: decoder, tokenizer: tokenizer)

let image = pipeline.generate(promptEmbeddings: embeddings, height: 1024, width: 1024, steps: 40)
```

The image comes back `[height, width, 4]` in −1…1, and the fourth channel is the release's own: its
autoencoder reads and writes four channels rather than three. Generation is multi-second per step at
this size; run it off the render thread.

### Z-Image (`NFKMLXZImageGenerator`)

Z-Image is a 6B single-stream DiT conditioned on Qwen3-4B, under the Apache 2.0 license.
`NFKMLXZImageGenerator` assembles it from a diffusers release directory (`Tongyi-MAI/Z-Image-Turbo` or
`Tongyi-MAI/Z-Image`) and holds it as an `NFKMLXResidency` says. The text encoder loads at its stored
bfloat16, about 8 GB, and the transformer at bfloat16, about 12 GB. A 32 GB machine runs the two staged.
The schedule is read from the release's `scheduler_config.json`.

```swift
let zImage = try NFKMLXZImageGenerator.generator(directoryURL: release, residency: .automatic)
let rgb = try zImage.image(forPrompt: "a red fox walking through fresh snow",
                           width: 1024, height: 1024, seed: 0)          // [1024, 1024, 3] in 0…1
```

```objc
NFKMLXZImageGenerator *zImage = [NFKMLXZImageGenerator generatorWithDirectoryURL:releaseDirectory
                                                                        residency:NFKMLXResidencyAutomatic
                                                                            error:&error];
CGImageRef fox = [zImage imageForPrompt:@"a red fox walking through fresh snow" negativePrompt:nil
                                  width:1024 height:1024 seed:0 error:&error];
```

The defaults are Turbo's published settings: 9 steps, the last of which lands on sigma 0, and a
`guidance` of 0. The base release samples at the reference pipeline's 50 steps and a guidance of 5; above
1 the image guides against the negative prompt, or against an empty one. The sides are multiples of 16.

### Stable Diffusion 3 and 3.5 (`NFKMLXSD3Generator`)

`NFKMLXSD3Generator` assembles SD3 Medium or SD3.5 Medium or Large from a diffusers release directory
and holds it as an `NFKMLXResidency` says. The text stage is CLIP-L, OpenCLIP bigG and T5-XXL; T5 runs at
float32 where it fits the working set on its own and at bfloat16 otherwise, and a release without
`text_encoder_3/` conditions on zeros in its place. The transformer loads at bfloat16. The repositories
are gated: the license is accepted once on the Hub.

```swift
let sd3 = try NFKMLXSD3Generator.generator(directoryURL: release, residency: .automatic)
let rgb = try sd3.image(forPrompt: "a red fox walking through fresh snow",
                        width: 1024, height: 1024, seed: 0)             // [1024, 1024, 3] in 0…1
```

```objc
NFKMLXSD3Generator *sd3 = [NFKMLXSD3Generator generatorWithDirectoryURL:releaseDirectory
                                                               residency:NFKMLXResidencyAutomatic
                                                                   error:&error];
CGImageRef fox = [sd3 imageForPrompt:@"a red fox walking through fresh snow" negativePrompt:nil
                               width:1024 height:1024 seed:0 error:&error];
```

`steps` and `guidance` default to the reference pipeline's 28 and 7. Above a guidance of 1 the image
guides against the negative prompt, or against an empty one. The sides are multiples of 16.

### FLUX.1 [schnell] (`NFKMLXFlux`)

FLUX.1 [schnell] turns a prompt into an image in four steps. `NFKMLXFlux` assembles the whole model
from a diffusers release directory — the 12B transformer, the autoencoder, and the two text encoders
(CLIP-L for the pooled projection, T5-XXL for the sequence) — and `image(forPrompt:)` runs the text
encoding, the flow-match sampler, and the decode.

```objc
NFKMLXFlux *flux = [NFKMLXFlux fluxWithDirectoryURL:releaseDirectory error:&error];   // black-forest-labs/FLUX.1-schnell
CGImageRef image = [flux imageForPrompt:@"a photograph of an astronaut riding a horse on the moon"
                                  width:1024 height:1024 seed:0 error:&error];
```

The text front end is at reference parity against transformers' `CLIPTextModel` and `T5EncoderModel`
(CLIP-L pooled 0.99997, T5-XXL sequence 0.9995), and the transformer at released-weight parity against
diffusers' `FluxTransformer2DModel`.

A release that does not fit the machine whole is staged as FLUX.2 is: CLIP-L and T5-XXL (about 19 GB at
the float32 T5 runs at) load, encode, and are released before the 12B transformer (about 24 GB) loads,
for every image. `.automatic` decides from the machine's working set.

```swift
let staged = try NFKMLXFlux.flux(directoryURL: releaseDirectory, residency: .staged)
print(staged.holdsStagesResident)                       // false
let image = try staged.image(forPrompt: "a red fox in the snow")
```

```objc
NFKMLXFlux *staged = [NFKMLXFlux fluxWithDirectoryURL:releaseDirectory
                                            residency:NFKMLXResidencyStaged error:&error];
```

## Image → image

### Bring-your-own MLX image model (`NFKMLXModuleBackend`, Swift)

Supply a `(MLXArray) -> MLXArray` forward closure; the backend bridges the image (a `CGImage` or an
`MTLTexture`) in and out.

```swift
import InferKit
import InferKitMLX
import MLX

let superResolution = NFKMLXModuleBackend(identifier: "sr", isReady: true) { input in
    myModel(input)                                          // [H, W, 3] in 0...1 -> [H, W, 3]
}
let request = NFKInferenceRequest(inputs: [NFKInputImage: sourceCGImage])
let out = try superResolution.runInference(for: request).output(forKey: NFKOutputImage)  // CGImage
```

### Real-ESRGAN ×4 upscaling (`NFKMLXRealESRGAN`, a shipped MLX model)

`NFKMLXRealESRGAN` is a real single-forward model, not a bring-your-own closure: it implements the
Real-ESRGAN generator (RRDBNet) in MLXNN and runs it through `NFKMLXModuleBackend`. Register it once,
then build it by name — from Swift or Objective-C. A downloaded **safetensors** checkpoint makes the
output a true ×4 upscale (with random weights the pipeline runs but the output is not meaningful).

```swift
NFKMLXRealESRGAN.register()                                 // once, at launch
let upscaler = try NFKMLXModelRegistry.backend(named: NFKMLXRealESRGAN.modelName, weightsURL: checkpointURL)
let out = try upscaler.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: plate])).output(forKey: NFKOutputImage)
```

Downloading the checkpoint and building in one call (the Objective-C path MetalForge uses):

```objc
[NFKMLXRealESRGAN register];
NSError *error = nil;
id<NFKInferenceBackend> upscaler =
    [NFKMLXHub backendNamed:@"real-esrgan-x4"
                       repo:@"org/real-esrgan"
                weightsPath:@"RealESRGAN_x4plus.pth"           // .pth or safetensors — both load
                   revision:nil
          cacheDirectoryURL:nil
                      error:&error];
```

The loader matches the reference RRDBNet parameter names (`conv_first.*`, `body.N.rdbM.convK.*`,
`conv_last.*`) and transposes 4-D convolution weights from PyTorch's `[out, in, kH, kW]` to MLX's
`[out, kH, kW, in]`. A PyTorch `.pth` release loads directly (see
[Loading a PyTorch checkpoint directly](#loading-a-pytorch-checkpoint-directly));
`Tools/realesrgan-to-safetensors/convert.py` remains the offline path. `register` adds `real-esrgan-x4` (23 blocks),
`real-esrgan-x4-anime` (6 blocks), and `real-esrgan-x2` (pixel-unshuffle front-end for ×2).

### Depth Anything V2 (`NFKMLXDepthAnything`, a shipped MLX model)

`NFKMLXDepthAnything` is a real single-forward depth model: a DINOv2 ViT encoder and a DPT head in
MLXNN, run through `NFKMLXModuleBackend`. An RGB image in, a grayscale depth map out under
`NFKOutputImage` (near = bright). Register once, build by name, and load a **safetensors** checkpoint
for meaningful depth.

```swift
NFKMLXDepthAnything.register()
let depth = try NFKMLXModelRegistry.backend(named: NFKMLXDepthAnything.modelName, weightsURL: checkpointURL)
let map = try depth.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: frame])).output(forKey: NFKOutputImage)
```

The DINOv2 + DPT key layout is intricate, so `Tools/depth-anything-to-safetensors/convert.py` is
self-validating: it matches every checkpoint key against the layout the module expects and reports
mismatches (adjust with the `remap` on `NFKMLXDepthAnything.loadWeights`). `register` adds all three
sizes — `depth-anything-v2-small` / `-base` / `-large` — via the `NFKMLXDepthConfiguration.small`,
`.base`, and `.large` presets (which set `embedDimensions`, `depth`, `heads`, and the DPT widths).

### Depth Anything 3 (`NFKMLXDepthAnything3`, a shipped MLX model)

`NFKMLXDepthAnything3` predicts depth, a ray map, and the camera. The depth path is an ordinary
backend: an RGB image in, a grayscale depth map out under `NFKOutputImage`, at one of the three
released sizes.

```swift
let depth = try NFKMLXDepthAnything3.backend(variant: .small, weightsURL: checkpointURL)
let map = try depth.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: frame])).output(forKey: NFKOutputImage)
```

The camera and the ray map do not fit a single-image backend, so `NFKMLXDepth3Estimator` carries them.
The focal lengths come back in pixels at the image's own size, so a caller can build an intrinsic
matrix directly.

```swift
let estimator = try NFKMLXDepth3Estimator.estimator(variant: .small, weightsURL: checkpointURL)
let camera = try estimator.camera(for: frame)
let (ray, confidence) = try estimator.rays(for: frame)          // Swift-only: both are MLXArrays
```

A caller who already knows the camera conditions the model on it instead of letting it predict one,
which is what the release's camera encoder is for.

```swift
let known = try estimator.camera(for: frame, knownRotation: rotation, translation: translation,
                                 focalLengthX: 320, focalLengthY: 300)
```

### NAFNet restoration (`NFKMLXNAFNet`, a shipped MLX model)

`NFKMLXNAFNet` is a real single-forward restoration network (denoise / deblur) — a U-shaped stack of
NAFBlocks (SimpleGate + Simplified Channel Attention). A degraded image in, the restored image out
under `NFKOutputImage`, at the input resolution (the net pads to a multiple of `2^levels` and crops
back).

```swift
NFKMLXNAFNet.register()
let restore = try NFKMLXModelRegistry.backend(named: NFKMLXNAFNet.modelName, weightsURL: checkpointURL)
let clean = try restore.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: noisy])).output(forKey: NFKOutputImage)
```

`NFKMLXNAFNetConfiguration` sets `width` and the encoder/middle/decoder block counts (the default is
the SIDD denoiser, width 32). `Tools/nafnet-to-safetensors/convert.py` converts the release and renames
`middle_blks` / `ups.N.0` / `sca.1` to the module's keys so it loads directly.

### In-process Core ML image model (`NFKCoreMLBackend`, Objective-C)

Runs any `.mlpackage` / `.mlmodelc`, keyed by the model's own input/output feature names. An
`id<MTLTexture>` input becomes a `CVPixelBuffer`; an image output returns as an `id<MTLTexture>`.

```objc
NSURL *modelURL = [NSURL fileURLWithPath:@"…/StyleTransfer.mlpackage"];
NFKCoreMLBackend *backend = [NFKCoreMLBackend backendWithModelURL:modelURL];
NSError *error = nil;
[backend prepareWithError:&error];

// "image" here is the model's own input feature name.
NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ @"image": inputTexture }];
id<MTLTexture> styled = [[backend runInferenceForRequest:request error:&error] outputForKey:@"stylized"];
```

### Where Core ML actually runs (`NFKComputePlan`, Objective-C)

`MLComputeUnits` is a request. Core ML places an operation the Neural Engine cannot run somewhere else
and reports nothing about having done so, so a model asked for the Neural Engine can run entirely on
the CPU and behave exactly as if it had not. `NFKComputePlan` reads the placement per operation from a
compiled model, without running it.

```objc
NFKCoreMLBackend *backend = [NFKCoreMLBackend backendWithModelURL:modelURL];
backend.computeUnits = MLComputeUnitsAll;     // the default; CPUOnly is zero, so it is set explicitly

// The plan takes an .mlmodelc and the compute units the backend will load with.
NSError *error = nil;
NFKComputePlan *plan = [NFKComputePlan planForCompiledModelAtURL:compiledURL
                                                    computeUnits:backend.computeUnits
                                                           error:&error];
if (plan != nil) {
    NSLog(@"%@", plan.describedPlacement);     // "142 operations: 138 Neural Engine, 4 GPU, 0 CPU, …"
    if (!plan.runsEntirelyOnNeuralEngine) {
        // The operators to work on when tuning a conversion, most frequent first.
        NSLog(@"off the ANE: %@", [plan.operatorNamesOffNeuralEngine componentsJoinedByString:@", "]);
    }
}
```

`neuralEngineFraction` is the single number to watch; one unsupported operator in the middle of a
network splits it and costs more than its own share of the time, because the intermediate results
cross devices.

This needs macOS 14.4 / iOS 17.4 / tvOS 17.4, which is where Core ML began publishing the information.
Check `NFKComputePlan.isAvailable` first: an older system fails with `kNFKError_InferenceUnsupported`
rather than reporting an empty plan, because zero operations on the Neural Engine and "cannot tell" are
different answers. Where the API is unavailable, `powermetrics --samplers ane_power` is the runtime
cross-check, and it needs elevated privileges.

The same in Swift:

```swift
// The ObjC factory imports as a throwing initializer, so a Swift caller writes it as a constructor.
let plan = try NFKComputePlan(forCompiledModelAt: compiledURL, computeUnits: .all)
print(plan.describedPlacement, plan.neuralEngineFraction)
```

### Will this model fit? (`NFKHardwareProfile`, Objective-C)

Loading a model that does not fit is not a polite failure: the process is killed, or the system pages
until the run is useless. Both are decidable first.

```objc
NFKHardwareProfile *machine = NFKHardwareProfile.currentProfile;
NSLog(@"%@", machine.describedMachine);
// "Apple M1 Max (MacBookPro18,2), 8P+2E, 32.0 GB physical, 25.0 GB recommended working set"

// Three different ceilings, and they are not interchangeable:
machine.physicalMemory;             // what is installed
machine.recommendedWorkingSetSize;  // what Metal expects to stay resident — size against THIS
machine.maximumBufferLength;        // the largest single allocation, whatever else is free

// Live, and the one that decides whether a load succeeds right now.
NSInteger free = [NFKHardwareProfile availableMemory];
```

On macOS `availableMemory` counts the free, inactive and purgeable pages the kernel reports, all of
which are reclaimable under pressure. On iOS and tvOS it is the process's own remaining allowance
before the system terminates it, which is the ceiling that actually applies there.

Every reading degrades rather than throwing: an unknown chip reports an empty name and zero counts, so
a profile is still usable on a machine this was never run on.

## Image → image + mask

`NFKMLXMattingBackend` runs a bring-your-own MLX matting model (a keyer, a background remover). The
plate under `NFKInputImage` and an optional hint under `NFKInputMask` become tensors; the forward
returns `[H, W, 4]` (straight foreground + alpha matte).

```swift
import InferKit
import InferKitMLX
import MLX

var configuration = NFKMattingConfiguration()
configuration.emitsMatte = true                            // also return the matte on its own
configuration.tileSize = 1024                              // process a large plate in tiles
configuration.imageOptions.premultiply = false             // straight foreground (default)
configuration.outputsTexture = false                       // CGImage out (true for MTLTexture)

let keyer = NFKMLXMattingBackend(identifier: "keyer", configuration: configuration) { plate, hint in
    greenFormer(plate, hint)                               // -> [H, W, 4]
}

let request = NFKInferenceRequest(inputs: [NFKInputImage: plateCGImage,
                                           NFKInputMask: trimapCGImage])
let result = try keyer.runInference(for: request)
let composited = result.output(forKey: NFKOutputImage)     // RGBA: foreground + alpha
let matte = result.output(forKey: NFKOutputMask)           // gray matte on its own
```

The plate and hint may equally be `MTLTexture`s from a Metal pipeline, and `outputsTexture` returns
`MTLTexture`s — no CGImage detour.

### U²-Net background removal (`NFKMLXU2Net`, a shipped MLX model)

`NFKMLXU2Net` is a real salient-object / background-removal network (a nested U of Residual U-blocks),
run through the matting backend: the plate stays as the straight foreground and the saliency map
becomes the alpha, so it produces a cutout with no hint needed. `register` adds the full `u2net` and
the light `u2netp`.

```objc
[NFKMLXU2Net register];
NSError *error = nil;
id<NFKInferenceBackend> cutout = [NFKMLXModelRegistry backendNamed:@"u2net" weightsURL:weightsURL error:&error];
NFKInferenceResult *result = [cutout runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputImage: plate }] error:&error];
id foreground = [result outputForKey:NFKOutputImage];      // RGBA cutout
id matte = [result outputForKey:NFKOutputMask];            // the alpha on its own
```

`Tools/u2net-to-safetensors/convert.py` converts `u2net.pth` / `u2netp.pth` and renames each RSU
block's `rebnconvN` convolutions to the module's `enc`/`dec` keys, so the file loads directly.

`NFKMLXISNet` is the successor the same authors published: the same Residual U-blocks behind a
stride-2 stem, wider stages, and six separate side maps instead of a fused one. It loads a released
`.pth` directly and runs the same matting contract under the name `isnet`.

```objc
[NFKMLXISNet register];
id<NFKInferenceBackend> dichotomous = [NFKMLXISNet backendWithWeightsURL:weightsURL error:&error];
NFKInferenceResult *cut = [dichotomous runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputImage: plate }] error:&error];
id alpha = [cut outputForKey:NFKOutputMask];
```

### Segment Anything (`NFKMLXSAM`, a shipped MLX model)

`NFKMLXSAM` is promptable segmentation: a ViT image encoder, a prompt encoder, and a two-way-transformer
mask decoder. Run through the matting backend — the plate under `NFKInputImage` and a click point under
the `NFKSAMPointKey` parameter (pixels, defaults to the center) → the mask as alpha under `NFKOutputImage`
and on its own under `NFKOutputMask`.

```swift
NFKMLXSAM.register()
let sam = try NFKMLXModelRegistry.backend(named: NFKMLXSAM.modelName, weightsURL: checkpointURL)
let result = try sam.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: plate]))
let mask = result.output(forKey: NFKOutputMask)
```

The ViT encoder uses real windowed attention (with global-attention layers) and decomposed
relative-position embeddings, matching the reference; `Tools/sam-to-safetensors/convert.py --list-keys`
covers the remaining block/neck key remap.

SAM 2 and SAM 2.1 ship as `NFKMLXSAM2`, registered as `sam2`, over the same plate-and-click contract.
A clip is tracked rather than segmented frame by frame: one session carries the memory of what came
before, and a frame without clicks reads it.

```swift
let net = NFKMLXSAM2.makeTracker(variant: .tiny, release: .sam21)
try NFKMLXSAM2.loadWeights(into: net, from: checkpointURL)

let session = NFKMLXSAM2TrackerSession(frameCount: frames.count)
for (index, frame) in frames.enumerated() {
    // The click is in pixels of the model's 1024-square input, on the frame that starts the track.
    let prediction = net.track(image: frame, frameIndex: index,
                               points: index == 0 ? [(512, 512, 1)] : nil, session: session)
    masks.append(sigmoid(prediction.maskLogits))
}
```

SAM 3 takes a prompt in words instead of a click, and returns every instance the words name. Its
three networks load from one released checkpoint, and `detect` returns masks, boxes, a logit per
query, and a single presence logit saying whether the prompt names anything in the plate at all.
Box prompts and video tracking are not ported.

```swift
let sam3 = try NFKMLXSAM3.makeImageModel(fromHuggingFace: configURL)
try NFKMLXSAM3.loadWeights(into: sam3, from: checkpointURL)

// `ids` is the prompt through a CLIP tokenizer, padded to the trained 32-position context, and
// `valid` marks the real tokens.
let found = sam3.detect(image: plate, tokens: ids, valid: valid)
let keep = (0 ..< found.logits.dim(1)).filter { found.logits[0, $0].item(Float.self) > 0 }
```

### Arbitrary style transfer (`NFKMLXAdaIN`, a shipped MLX model)

`NFKMLXAdaIN` stylizes a photograph with any style image, where `NFKMLXStyleTransfer` bakes one style
into each checkpoint. A normalized VGG-19 encodes both images, adaptive instance normalization moves
the content features onto the style's per-channel statistics, and a mirrored decoder inverts the
result. The reference publishes the encoder and the decoder as separate files, so the factory takes
both. The content image goes under `NFKInputImage`, the style image under `NFKInputControl`, and
`NFKParameterStrength` blends between the content reconstruction and the full transfer.

```swift
let stylizer = try NFKMLXAdaIN.backend(encoderURL: vggURL, decoderURL: decoderURL)
let request = NFKInferenceRequest(inputs: [NFKInputImage: photo, NFKInputControl: painting],
                                  parameters: [NFKParameterStrength: NSNumber(value: 0.8)])
let stylized = try stylizer.runInference(for: request).output(forKey: NFKOutputImage)
```

### LaMa inpainting (`NFKMLXLaMa`, a shipped MLX model)

`NFKMLXLaMa` is a real single-forward inpainter built on Fast Fourier Convolutions (each layer runs a
spatial branch and an FFT-based spectral branch). The plate goes under `NFKInputImage`, the mask under
`NFKInputMask` (white regenerates), and the inpainted image returns under `NFKOutputImage`.

```swift
NFKMLXLaMa.register()
let inpainter = try NFKMLXModelRegistry.backend(named: NFKMLXLaMa.modelName, weightsURL: checkpointURL)
let filled = try inpainter.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: plate, NFKInputMask: hole]))
    .output(forKey: NFKOutputImage)
```

The big-lama checkpoint stores the generator in a flat `model.N` sequence; `Tools/lama-to-safetensors/convert.py`
extracts and converts it, and the reference names map to the module's names with the `remap` on
`NFKMLXLaMa.loadWeights` (a validation-sweep task). The FFT uses orthogonal normalization; reflection
padding is approximated with edge padding.

## Many tensors in and out

`NFKMLXTensorBackend` is the general case: several named image inputs in, several out (a compositing
model reading a foreground and a background, a model returning both an image and a mask).

```swift
import InferKit
import InferKitMLX
import MLX

let configuration = NFKMLXTensorConfiguration(
    inputs: [
        NFKMLXTensorPort(key: NFKInputImage, tensorName: "foreground", channels: 4),
        NFKMLXTensorPort(key: NFKInputMask, tensorName: "background", channels: 3),
    ],
    outputs: [
        NFKMLXTensorPort(key: NFKOutputImage, tensorName: "composite"),
        NFKMLXTensorPort(key: NFKOutputMask, tensorName: "matte"),
    ])

let backend = NFKMLXTensorBackend(identifier: "compositor", configuration: configuration) { inputs in
    let (composite, matte) = compose(inputs["foreground"]!, inputs["background"]!)
    return ["composite": composite, "matte": matte]
}
```

### RIFE frame interpolation (`NFKMLXRIFE`, a shipped MLX model)

`NFKMLXRIFE` interpolates a frame between two inputs (slow-motion, retiming). It runs through
`NFKMLXTensorBackend`: two frames in under the keys `frame0` / `frame1`, the middle frame out under
`NFKOutputImage`. The IFNet estimates bidirectional flow coarse-to-fine and blends the backward-warped
frames with a learned mask.

```objc
[NFKMLXRIFE register];
NSError *error = nil;
id<NFKInferenceBackend> rife = [NFKMLXModelRegistry backendNamed:@"rife" weightsURL:weightsURL error:&error];
NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ @"frame0": frameA, @"frame1": frameB }];
CGImageRef middle = (__bridge CGImageRef)[[rife runInferenceForRequest:request error:&error] outputForKey:NFKOutputImage];
```

`Tools/rife-to-safetensors/convert.py` converts an HDv3 `flownet.pkl` and renames the nested
`conv0.i.0`/`.1` (Conv / PReLU) to the module keys. RIFE has several incompatible versions; confirm the
block count / channels (this targets HDv3, `c` 240/150/90).

### RAFT optical flow (`NFKMLXRAFT`, a shipped MLX model)

`NFKMLXRAFT` estimates dense optical flow between two frames (motion vectors for warping, retiming,
temporal consistency). It runs through `NFKMLXTensorBackend`: two frames under keys `frame0` / `frame1`,
a packed flow map under `NFKOutputImage` (`R = 0.5 + fx/scale`, `G = 0.5 + fy/scale`, mid-gray = no
motion). The raw flow `[H, W, 2]` is available from `NFKMLXRAFTNet.flow`. The pipeline is a shared
feature encoder, an all-pairs correlation pyramid, a context encoder, and an iterative ConvGRU.

```objc
[NFKMLXRAFT register];
id<NFKInferenceBackend> raft = [NFKMLXModelRegistry backendNamed:@"raft" weightsURL:weightsURL error:&error];
NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ @"frame0": frameA, @"frame1": frameB }];
CGImageRef flowMap = (__bridge CGImageRef)[[raft runInferenceForRequest:request error:&error] outputForKey:NFKOutputImage];
```

Faithful to RAFT-large (feature 256, 4 correlation levels, radius 4) with two documented
simplifications: bilinear ×8 upsampling instead of the learned convex mask (the mask head still loads),
and a low default iteration count. `Tools/raft-to-safetensors/convert.py` converts the release and
renames the nested `update_block` / `downsample` / `flow_head` / `mask` keys.

## Diffusion: upscale, depth, inpaint

The single-forward backends above run a model once. A diffusion model runs an iterative sampler:
encode conditioning, start from noise, denoise over N steps through a scheduler, decode.
`NFKMLXDiffusionBackend` owns that loop and the InferKit contract (progress per step, cancellation
between steps); the consumer supplies three closures and a scheduler.

- `encode`: request + bridged input image + optional mask → an `NFKDiffusionContext` (conditioning,
  output size, and — for image-to-image / inpainting — a source latent and a mask).
- `denoise`: latent + timestep + context + guidance → the model's prediction for that step (per the
  scheduler's prediction type: `epsilon`, `vPrediction`, or `sample`).
- `decode`: the final latent → an image tensor `[H, W, C]` in `0...1` (identity, or a VAE decode).
- `scheduler`: the sampler. `NFKDDIMScheduler` is provided; a flow-matching model (FLUX, SD3) adopts
  `NFKDiffusionScheduler`.

```swift
import InferKit
import InferKitMLX
import MLX

let backend = NFKMLXDiffusionBackend(
    identifier: "my-diffusion",
    configuration: NFKDiffusionConfiguration(steps: 20, guidanceScale: 7.5),
    scheduler: NFKDDIMScheduler(predictionType: .epsilon),
    encode: { request, image, mask in
        // Encode text/image conditioning into MLXArrays; return the latent size and any source.
        NFKDiffusionContext(conditioning: ["text": encodePrompt(request.prompt)], width: 512, height: 512)
    },
    denoise: { latent, timestep, context, guidance in
        unet(latent, timestep.train, context.conditioning["text"]!, guidance)
    },
    decode: { latent in vaeDecode(latent) })
```

Text-to-image runs when `encode` returns no source latent. A source latent starts image-to-image,
with `NFKParameterStrength` controlling how much of the source survives. A source latent and a mask
run inpainting: the kept region (mask `0`) is held to the source each step, the masked region (mask
`1`) is generated.

InferKitMLX ships three reference pipelines, registered by name for the Objective-C path. Each wires
the backend for a real task and I/O shape; a real integration keeps the wiring and swaps the reference
forward for a trained UNet (and a VAE in `decode`):

```swift
NFKMLXReferenceModels.registerDiffusionUpscaler()      // "diffusion-upscaler": image → 2× image
NFKMLXReferenceModels.registerDiffusionDepth()         // "diffusion-depth": image → grayscale depth
NFKMLXReferenceModels.registerDiffusionInpainter()     // "diffusion-inpaint": plate + mask → filled
```

```objc
// Objective-C, after the Swift side registered the reference (or a real model under the same name):
id<NFKInferenceBackend> depth = [NFKMLXModelRegistry backendNamed:@"diffusion-depth" weightsURL:nil error:&error];
NFKInferenceResult *result = [depth runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputImage: frame }] error:&error];
CGImageRef map = (__bridge CGImageRef)[result outputForKey:NFKOutputImage];
```

### A live preview of each step (`NFKDiffusionLatentPreview`, Swift)

A sampler takes tens of seconds and a step count says nothing about what is being made. Decoding the
latent through the autoencoder every step costs more than the sampling does, so the preview is a 1×1
convolution over the channel axis — twelve weights and three biases for a four-channel latent. It
arrives as the job's `partialResult`, the same mechanism a streaming text backend uses.

```swift
var configuration = NFKDiffusionConfiguration(steps: 30, latentChannels: 4)
configuration.latentPreview = .stableDiffusion   // or .stableDiffusionXL, or .passthrough
configuration.previewEverySteps = 2              // thin it; 1 previews every step

let job = backend.submitInferenceJob(for: request)
job.progressHandler = { job in
    // The preview is a CGImage under the configuration's own output key.
    if let preview = job.partialResult?.output(forKey: NFKOutputImage) {
        display(preview)
    }
}
```

`partialResult` holds the **last** non-nil value, so a step that reports no preview still reads as
having one. Compare identity, not presence, if you need the preview rate.

The shipped coefficients approximate the decode and make no parity claim: against the released SD 1.5
autoencoder, on a latent encoded from a real photograph, they reproduce the decode's structure at a
mean-removed correlation of 0.93. Note the map is applied to the **sampler's** latent, which is the
scaled one — handing it a latent at the autoencoder's own scale washes the preview toward flat grey.

For a model with no published factors, derive one from its own decoder:

```swift
let map = NFKDiffusionLatentPreview.fitted(
    latentChannels: 4,
    decode: { latent in myAutoencoder.decode(latent) },
    sample: { index in MLXRandom.normal([32, 32, 4], key: MLXRandom.key(UInt64(index))) })
```

A preview is a progress indicator, so a map whose channel count does not match the latent returns nil
rather than failing the run it is only reporting on.

### Stable Diffusion inpainting (`NFKMLXStableDiffusionInpaint`)

`NFKMLXStableDiffusionInpaint` is a latent-diffusion inpainter on top of `NFKMLXDiffusionBackend`: a
VAE encodes the plate and the masked plate to latents, the UNet denoises a 9-channel input, and the
VAE decodes the result. The backend runs the DDIM loop and holds the kept region to the source latent
each step.

```swift
NFKMLXStableDiffusionInpaint.register()
let inpaint = try NFKMLXModelRegistry.backend(named: NFKMLXStableDiffusionInpaint.modelName, weightsURL: checkpointURL)
let filled = try inpaint.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: plate, NFKInputMask: hole]))
    .output(forKey: NFKOutputImage)
```

The VAE and UNet are the real released networks (`NFKMLXSDUNet` / `NFKMLXSDAutoencoder`, both at
reference parity against diffusers), sized by `NFKMLXSDInpaintConfiguration` — the default is the
released SD 1.5 inpainting geometry. Loading the released two-checkpoint layout and the text
conditioning is shown under "Latent diffusion with real weights" below.

### ControlNet and LCM — without reimplementing Stable Diffusion

ControlNet and LCM plug into the existing diffusion seam; neither needs a from-scratch SD UNet+VAE.
**LCM is a scheduler** — swap `NFKDDIMScheduler` for `NFKLCMScheduler` and drop the step count. Whoever
supplies the UNet (your own `denoise`, or a dynamically linked SD engine via `NFKDynamicBackend`) keeps
the same wiring:

```swift
let fast = NFKMLXDiffusionBackend(
    configuration: NFKDiffusionConfiguration(steps: 4),   // few-step
    scheduler: NFKLCMScheduler(predictionType: .epsilon), // LCM consistency sampling, seed-repeatable
    encode: myEncode, denoise: myUNet, decode: myVAEDecode)
```

**ControlNet is a `denoise` closure** plus a control map. The control image arrives under the core key
`NFKInputControl`; `encode` bridges it into `context.conditioning["control"]`; a real `denoise` runs the
control network on it and adds the residuals to its UNet blocks. The `diffusion-controlnet` reference
shows the wiring (its oracle steers the output toward the control map):

```swift
NFKMLXReferenceModels.registerControlNet()
let controlNet = try NFKMLXModelRegistry.backend(named: "diffusion-controlnet", weightsURL: nil)
let image = try controlNet.runInference(for: NFKInferenceRequest(inputs: [NFKInputControl: edgeMap]))
    .output(forKey: NFKOutputImage)
```

The backend still owns the sampler loop, guidance, per-step progress, and the image bridge — so real
ControlNet/LCM is a `denoise` + scheduler choice, not a model reimplementation.

## Running MLX models from Objective-C

### Direct construction — the primary path for shipped models

Every shipped real model has an Objective-C factory, so a consumer (an FCPX plugin, an app) builds and
runs it directly — no registration, no name lookup. Each returns `id<NFKInferenceBackend>` (or nil with
an `NSError`), takes an optional local `weightsURL` (nil builds random weights, `isReady` true), and —
for models with variants — an `@objc` variant enum. A companion factory downloads the checkpoint from
Hugging Face and builds in one call.

```objc
@import InferKitMLX;
NSError *error = nil;

// Local weights (or nil), no registry:
id<NFKInferenceBackend> depth = [NFKMLXDepthAnything backendWithVariant:NFKMLXDepthVariantBase
															 weightsURL:localURL error:&error];
id<NFKInferenceBackend> restore = [NFKMLXNAFNet backendWithWeightsURL:localURL error:&error];

// Download from Hugging Face, then build (blocking — run off the render thread):
id<NFKInferenceBackend> depthDL =
	[NFKMLXDepthAnything backendWithVariant:NFKMLXDepthVariantBase
									   repo:@"org/dav2"
								weightsPath:@"model.safetensors"
								   revision:nil
						  cacheDirectoryURL:nil            // nil = default cache
									  error:&error];

// Text-to-image takes a whole release rather than one checkpoint: either the bundled backend, which
// downloads its release on first use, or a release directory already on disk.
NFKMLXBackend *diffusion = [[NFKMLXBackend alloc] initWithModel:NFKMLXStableDiffusionModelStableDiffusion15];
id<NFKInferenceBackend> fromDisk =
	[NFKMLXTextToImage backendWithModel:NFKMLXStableDiffusionModelSdxlTurbo
						   directoryURL:releaseDirectory
								  error:&error];

// The language model builds from a downloaded release directory (config.json + tokenizer + shards):
id<NFKInferenceBackend> llm = [NFKMLXLanguage backendWithDirectoryURL:releaseDirectory error:&error];
```

Generation options that have no core parameter key — the cache bound and quantization, prefill
chunking, the chat template — are set on the request through `NFKMLXGenerationParameterKey`, the same
way `NFKParameterTemperature` is, so Objective-C reaches every option the Swift `NFKMLXGenerationOptions`
struct carries:

```objc
NFKInferenceRequest *request = [[NFKInferenceRequest alloc]
	initWithInputs:@{ NFKInputMessages: messages }        // or NFKInputPrompt for raw text
	parameters:@{
		NFKParameterTemperature: @0.7,
		NFKMLXGenerationParameterKey.contextWindow: @4096,          // bound the cache
		NFKMLXGenerationParameterKey.cacheQuantizationBits: @8,     // store it 8-bit
		NFKMLXGenerationParameterKey.prefillChunkSize: @512,        // chunk a long prompt
		NFKMLXGenerationParameterKey.chatTemplate: @"chatml",       // instruct format
	}];
NSString *text = [llm runInferenceForRequest:request error:&error].text;
```

The generation runtime's later additions take the same route — a draft model for speculative
decoding, a cache kept between the turns of a conversation, and a JSON or fixed-choice constraint on
the output:

```objc
// Built with a smaller release of the same family as the draft:
id<NFKInferenceBackend> llm =
	[NFKMLXLanguage backendWithDirectoryURL:qwen4B draftDirectoryURL:qwen06B error:&error];

NFKInferenceRequest *request = [[NFKInferenceRequest alloc]
	initWithInputs:@{ NFKInputMessages: messages }
	parameters:@{
		NFKMLXGenerationParameterKey.chatTemplate: @"chatml",
		NFKMLXGenerationParameterKey.draftTokens: @4,               // proposals per round; 0 disables
		NFKMLXGenerationParameterKey.reusesPromptCache: @YES,       // prefill only what this turn adds
		NFKMLXGenerationParameterKey.outputFormat: @"json-object",  // "json", "json-object", "json-array"
	}];
// A classification: the answer is exactly one of the choices.
NFKInferenceRequest *pick = [[NFKInferenceRequest alloc]
	initWithInputs:@{ NFKInputPrompt: @"Is the sky blue? Answer yes or no." }
	parameters:@{ NFKMLXGenerationParameterKey.choices: @[ @"yes", @"no" ] }];
// When a conversation ends, drop the retained cache:
[(NFKMLXLanguageBackend *)llm resetPromptCache];
```

The two checks a Swift caller reaches through `NFKMLXReleaseWeights` and `contextWindow` are on the
same request path from Objective-C. A release larger than the machine is refused by the directory
factory before any weight is read, with an error naming the shortfall, and the window is a request key:

```objc
NSError *error = nil;
id<NFKInferenceBackend> llm = [NFKMLXLanguage backendWithDirectoryURL:hugeRelease error:&error];
// llm == nil; error: "… needs about 54.0 GiB resident, but the machine's working set is 25.0 GiB;
//                     load at .checkpoint precision, quantize, or use a smaller size"

NFKInferenceRequest *bounded = [[NFKInferenceRequest alloc]
	initWithInputs:@{ NFKInputPrompt: prompt }
	parameters:@{ NFKMLXGenerationParameterKey.contextWindow: @4096,     // retain at most this many positions
				  NFKMLXGenerationParameterKey.prefillChunkSize: @512 }];  // prefill a long prompt in slices
```

The compiled Objective-C example writes a tiny release on the fly (config, vocabulary, random weights
in a hand-written safetensors) and runs both through the public factory with no download.

Every model with released sizes exposes a variant enum to *all* its factories — local, download, and
async — so an ObjC caller reaches every size, and a face detector hands back the five landmarks, not
only a box:

```objc
// Whisper at any released size (small/medium/large-v3), not only tiny:
id<NFKInferenceBackend> whisper =
	[NFKMLXWhisper backendWithVariant:NFKMLXWhisperVariantSmall weightsURL:localURL error:&error];

// A reusable face detector; each NFKFaceObservation carries its box, confidence, and five landmarks:
NFKMLXRetinaFaceDetector *detector =
	[NFKMLXRetinaFace detectorWithWeightsURL:localURL confidenceThreshold:0.8 suppressionThreshold:0.4 error:&error];
for (NFKFaceObservation *face in [detector facesInImage:image error:&error]) {
	CGPoint leftEye = face.leftEye, nose = face.nose;   // image pixels, top-left origin
}

// The machine's measured memory bandwidth, beside the other NFKMLXGPU machine properties:
double bytesPerSecond = [NFKMLXGPU measuredMemoryBandwidthWithMegabytes:0 repetitions:4];
```

Every backend declares the request keys it acts on, so a caller sets what the engine honors and a
router picks the engine a request needs:

```objc
id<NFKInferenceBackend> style = [NFKMLXAdaIN backendWithEncoderURL:nil decoderURL:nil error:&error];
BOOL takesStyle = [style.supportedInputKeys containsObject:NFKInputControl];       // YES
BOOL takesBlend = [style.supportedParameterKeys containsObject:NFKParameterStrength];  // YES
```

The language backend declares the core sampling keys, `NFKParameterJSONSchema`,
`NFKParameterOutputFormat`, `NFKParameterChoices`, and every `NFKMLXGenerationParameterKey`. A
backend that reads one input and no parameters declares exactly that, which is a different answer
from declaring nothing.

Variant models expose an `@objc` enum: `NFKMLXRealESRGANVariant` (x4 / anime / x2), `NFKMLXDepthVariant`
(small / base / large), `NFKMLXU2NetVariant` (full / light). Single-config models
(`NFKMLXNAFNet`, `NFKMLXSAM`, `NFKMLXLaMa`, `NFKMLXStableDiffusionInpaint`, `NFKMLXMarigold`,
`NFKMLXSDUpscaler`, `NFKMLXRIFE`, `NFKMLXRAFT`, `NFKMLXWhisper`, `NFKMLXDemucs`) use
`backendWithWeightsURL:error:` and `backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:`. The
`register()` / `registerAll()` / `NFKMLXHub` registry path below still works and is unchanged.

### The registry — for custom / bring-your-own models

An MLX model's forward pass is Swift over `MLXArray`, so the bring-your-own-closure backends
(`NFKMLXModuleBackend` / `NFKMLXMattingBackend` / `NFKMLXTensorBackend` / `NFKMLXSpeechBackend`) are
constructed from Swift. To let an Objective-C consumer build and run one of *those* without writing
Swift, register it once by name with `NFKMLXModelRegistry` from Swift, then construct it by that name
from Objective-C.

A model author registers a factory (Swift) — the forward and any weight loading live here:

```swift
NFKMLXModelRegistry.register(name: "corridor-key") { weightsURL in
    let model = GreenFormer(weightsURL: weightsURL)         // load the learned keyer's weights
    var configuration = NFKMattingConfiguration()
    configuration.emitsMatte = true
    configuration.outputsTexture = true                     // hand a Metal host textures
    return NFKMLXMattingBackend(identifier: "corridor-key", configuration: configuration) { plate, hint in
        model.matte(plate: plate, hint: hint)               // [H, W, 4]: straight foreground + matte
    }
}
```

The consumer builds and drives it (Objective-C) — no `MLXArray` in sight:

```objc
@import InferKitMLX;

NSError *error = nil;
id<NFKInferenceBackend> keyer = [NFKMLXModelRegistry backendNamed:@"corridor-key" weightsURL:weightsURL error:&error];
NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputImage: plate, NFKInputMask: hint }];
NFKInferenceResult *result = [keyer runInferenceForRequest:request error:&error];
id<MTLTexture> composited = [result outputForKey:NFKOutputImage];   // MTLTexture, per the configuration
id<MTLTexture> matte = [result outputForKey:NFKOutputMask];
```

InferKitMLX ships a reference model — a simple green-screen keyer registered with
`[NFKMLXReferenceModels registerGreenScreenKeyer]` under `"green-screen-keyer"`; CorridorKey's
GreenFormer registers the same way. Everything else — the `CGImage` or `MTLTexture` bridge, the alpha
matte, tiling — is already handled by `NFKMLXMattingBackend`, so implementing a keyer is writing its
forward and registering it.

#### Registered model names

`[NFKMLXReferenceModels registerAll]` registers every shipped model at once; then build any by name.
Two categories share the registry:

| Category | Names | What they are |
| --- | --- | --- |
| Real models | `real-esrgan-x4` / `-x4-anime` / `-x2`, `depth-anything-v2-small` / `-base` / `-large`, `u2net`, `u2netp`, `nafnet`, `sam`, `rife`, `raft`, `whisper-tiny`, `demucs`, `htdemucs`, `lama-inpaint`, `sd-inpaint`, `marigold-depth`, `sd-x4-upscaler`, `minimax-music3` | Real architectures that load a real safetensors checkpoint (upscale, depth, background removal, restoration, segmentation, interpolation, optical flow, transcription, stem separation, inpaint, music generation — `minimax-music3` takes the release DIRECTORY as its weights URL). |
| Reference stand-ins | `green-screen-keyer`, `tone-speech`, `diffusion-upscaler`, `diffusion-depth`, `diffusion-inpaint` | Working pipelines with a synthetic (non-trained) forward. They prove the I/O shape and the loop; the `diffusion-*` ones are oracle-driven stand-ins for the diffusion backend, not trained models. |

So `diffusion-depth` (a diffusion-loop stand-in) and `depth-anything-v2-small` (a real depth network)
are distinct entries; use the real name for a real result.

#### Latent diffusion with real weights

Three latent-diffusion models build on `NFKMLXDiffusionBackend` over one shared pair of networks,
`NFKMLXSDUNet` and `NFKMLXSDAutoencoder`: `sd-inpaint` (`NFKMLXStableDiffusionInpaint`),
`marigold-depth` (`NFKMLXMarigold`, image → depth), and `sd-x4-upscaler` (`NFKMLXSDUpscaler`,
image → ×4 image). They differ in a configuration, not in structure.

The released weights come as two diffusers checkpoints, one per network, plus the text embedding the
trained UNet cross-attends to. The text tower is not part of the model here — the caller supplies the
embedding, and a model that takes no prompt still expects the embedding of an empty one.

<!-- objc-check: given id mask = nil; -->
```objc
id<NFKInferenceBackend> inpainter = [NFKMLXStableDiffusionInpaint backendWithUNetWeightsURL:unetURL
                                                                             vaeWeightsURL:vaeURL
                                                                            textContextURL:promptURL
                                                                                     error:&error];
NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{NFKInputImage: plate,
                                                                        NFKInputMask: mask}];
CGImageRef filled = (__bridge CGImageRef)[[inpainter runInferenceForRequest:request error:&error]
                                          outputForKey:NFKOutputImage];
```

`backendWithWeightsURL:` takes the single-file form instead — both networks under `unet.`/`vae.`,
which is what `NFKMLXWeights` writes, so a fine-tuned pipeline reloads through one path.

### Download the weights from Hugging Face and build in one call

`NFKMLXHub` combines the core's `NFKHFHub` download layer with the registry: it downloads a model's
weights and builds the registered backend around them. The registered factory receives the local
weights URL.

```objc
NSError *error = nil;
id<NFKInferenceBackend> keyer =
    [NFKMLXHub backendNamed:@"corridor-key"
                       repo:@"org/corridor-key"
                weightsPath:@"model.safetensors"
                   revision:nil
          cacheDirectoryURL:nil            // nil = default cache
                      error:&error];       // downloads (blocking) then builds; run off the render thread
```

The blocking form has an asynchronous peer on `NFKMLXHub` and on every direct model factory — the
`…completionHandler:` selector runs the download on a background queue and delivers the backend (or an
error) to the handler, so the caller does not hand-thread it off the render thread:

```objc
[NFKMLXHub backendNamed:@"corridor-key"
                   repo:@"org/corridor-key"
            weightsPath:@"model.safetensors"
               revision:nil
      cacheDirectoryURL:nil
      completionHandler:^(id<NFKInferenceBackend> backend, NSError *error) {
          // runs on the download's background queue; hop to the main thread if the UI needs it
      }];
```

`NFKHFHub` itself has no MLX and does not run models — it fetches files. The bundled Stable Diffusion
backend downloads through it as well, into `NFKMLXBackend.cacheDirectoryURL`. A gated repository needs
an access token: set `accessToken` on a hub you make, or `NFKHFHub.defaultAccessToken` for the hubs the
download-and-build factories make on their own. With neither, `HF_TOKEN` in the environment supplies
one, which a command-line tool has and an app does not.

### Downloading a whole release

Every model that builds from a release directory has a download peer: the directory selector with
`DirectoryURL:` replaced by `Repo:revision:cacheDirectoryURL:`, plus its `…completionHandler:` form. It
fetches the files the model reads, and nothing else, into the `NFKHFHub` cache, then builds from the
snapshot folder. That covers the language models (`NFKMLXLanguage`, the Gemma family, Granite 4.0-H,
Nemotron-H, Mamba), the vision-language models (SmolVLM2, Qwen3-VL, Pixtral, Florence-2, Sa2VA, TrOCR,
Table Transformer, V-JEPA 2), the embedders and rerankers, the speech recognizers (Parakeet and Canary
from their `.nemo` archives, Granite Speech, Voxtral), Kokoro (one voice), Chatterbox, VoiceRestore,
Resemble Enhance, MossFormer2 super-resolution, FLUX, FLUX.2, and MiniMax Music 3.

```objc
NFKHFHub.defaultAccessToken = token;     // only for a gated repository; from the app's own secure storage
id<NFKInferenceBackend> chat = [NFKMLXLanguage backendWithRepo:@"Qwen/Qwen3-0.6B" revision:nil
                                              cacheDirectoryURL:nil error:&error];
[NFKMLXParakeet backendWithRepo:@"nvidia/parakeet-tdt-0.6b-v2" revision:nil cacheDirectoryURL:nil
              completionHandler:^(id<NFKInferenceBackend> backend, NSError *error) { /* background queue */ }];
```

```swift
let speculative = try NFKMLXLanguage.backend(repo: "Qwen/Qwen3-4B", revision: nil,
                                             draftRepo: "Qwen/Qwen3-0.6B", draftRevision: nil,
                                             cacheDirectoryURL: nil)
let embedder = try NFKMLXQwen3Embedding.backend(repo: "Qwen/Qwen3-Embedding-0.6B", revision: nil,
                                                cacheDirectoryURL: nil)
```

Pass a commit as `revision` to pin a release; nil follows `main`, and a cached `main` is not checked
again. The weights of the larger releases run from several gigabytes (Qwen3-VL-2B) to tens of gigabytes
(FLUX, MiniMax Music 3), and the download blocks until every shard is in the cache.

### MLX runtime knobs from Objective-C

MLX's global seed, GPU memory management, and device selection ship as Swift-only free functions,
enums, and a struct; `NFKMLXRandom` / `NFKMLXGPU` / `NFKMLXDevice` wrap them for Objective-C. Seed for
reproducible weight init and sampling; cap the GPU cache to bound memory in a plugin or app; select the
CPU where a graphics device is contended.

```objc
[NFKMLXRandom seed:42];                       // reproducible init/sampling

// Where MLX loads its Metal library from, found the way its loader looks. nil means the first
// evaluation fails, so check it at launch rather than at the first inference.
NSURL *metalLibrary = NFKMLXGPU.metalLibraryURL;

[NFKMLXGPU setCacheLimit:48 * 1024 * 1024];   // bound the buffer cache (bytes)
NSInteger active = NFKMLXGPU.activeMemory;    // live bytes; also cacheMemory / peakMemory
[NFKMLXGPU clearCache];                        // return the cache to the system

// What the machine has, for sizing a model before loading it. The recommended working set is
// Metal's own budget and is well below the physical total, which is the number to size against.
NSInteger budget = NFKMLXGPU.recommendedWorkingSetSize;
NSInteger reclaimable = NFKMLXGPU.reclaimableMemory;   // the cache; clearCache returns exactly this
double pressure = NFKMLXGPU.memoryPressure;            // active memory as a share of the budget

// One call at startup instead of remembering clearCache at every model boundary: a standing cache
// cap plus a soft memory limit derived from that budget.
[NFKMLXGPU applyStandingLimits];

__block NFKInferenceResult *result = nil;
__block NSError *deviceError = nil;
[NFKMLXDevice performOnDeviceType:NFKMLXDeviceTypeCPU block:^{
    result = [backend runInferenceForRequest:request error:&deviceError];
}];
```

The device selection covers work the block does **on the calling thread**. It does not reach another
thread, so it wraps a synchronous `runInferenceForRequest:` and not `submitInferenceJobForRequest:`,
whose queue takes the global device. Run the synchronous call inside the block from your own background
thread. Selecting the CPU does not avoid shipping the Metal library: MLX builds the Metal device when it
initializes, whichever device the work names.

## Text → video

`NFKMLXLTXVideoGenerator` (LTX-Video 0.9.0) and `NFKMLXWanVideoGenerator` (Wan 2.1 T2V and Wan 2.2
TI2V-5B) assemble a text-to-video model from its diffusers release directory, and hold it as an
`NFKMLXResidency` says. The T5 text encoder runs once per clip and the transformer and autoencoder
after it, so `.automatic` stages a release that does not fit whole: LTX-Video's float32 T5-XXL is 19 GB
beside a 7.7 GB transformer. Each generator's glue (the prompt padded and masked, the schedule, guidance,
the latent statistics, the decode) matches diffusers' own `LTXPipeline` and `WanPipeline`.

```swift
let ltx = try NFKMLXLTXVideoGenerator.generator(directoryURL: ltxRelease, residency: .automatic)
let clip = try ltx.video(forPrompt: "a red fox walking through fresh snow", frames: 121,
                         width: 704, height: 480, seed: 0)           // [121, 480, 704, 3] in 0…1

let wan = try NFKMLXWanVideoGenerator.generator(repo: nil, revision: nil, cacheDirectoryURL: nil,
                                                residency: .automatic)  // Wan-AI/Wan2.1-T2V-1.3B-Diffusers
wan.steps = 30
let frames = try wan.video(forPrompt: "a red fox walking through fresh snow", frames: 33)
```

<!-- objc-check: given NSURL *wanRelease; -->
```objc
NFKMLXWanVideoGenerator *wan = [NFKMLXWanVideoGenerator generatorWithDirectoryURL:wanRelease
                                                                        residency:NFKMLXResidencyAutomatic
                                                                            error:&error];
NSArray *frames = [wan framesForPrompt:@"a red fox walking through fresh snow" negativePrompt:nil
                                frames:33 width:832 height:480 seed:0 error:&error];
CGImageRef first = (__bridge CGImageRef)frames[0];
```

A frame count rounds down to one more than a multiple of the autoencoder's temporal compression (8 for
LTX-Video, 4 for Wan), and a side to a multiple of its spatial compression. Both negative prompts
default to empty, and guidance above 1 guides against it. The Wan 2.1 14B transformer is 28 GB on its
own, beyond a 32 GB machine in any placement.

## Video (clip → clip)

A video backend reads an `NFKVideoAsset`, hands **every frame** to its transform as `[H, W, 3]` in
`0...1`, and writes the result back out as a new asset. The transform takes the whole sequence rather
than one frame, because that is what the models need: interpolation reads pairs and returns more
frames than it took, and BasicVSR propagates state forward and backward through time.

```swift
// Frame interpolation: n frames become 2n - 1, at twice the source rate. The clip plays smoother,
// not slower — the duration is what stays fixed.
let interpolator = try NFKMLXRIFE.clipBackend(weightsURL: weights)
let request = NFKInferenceRequest(inputs: [NFKInputVideo: NFKVideoAsset(fileURL: sourceURL)])
let smoothed = try interpolator.runInference(for: request).output(forKey: NFKOutputVideo) as? NFKVideoAsset

// Video super-resolution: ×4 at the same rate.
let upscaler = try NFKMLXVideoSR.clipBackend(weightsURL: weights)

// Bring your own model: any [MLXArray] -> [MLXArray] over the frames.
var configuration = NFKMLXVideoConfiguration()
configuration.frameRateMultiplier = 2            // the transform returns twice as many frames
let custom = NFKMLXVideoBackend(identifier: "my-video-model", configuration: configuration) { frames in
    frames.flatMap { [$0, $0] }
}
```

Run a clip off the render thread: it is one forward pass per frame. `NFKMLXVideoFile` is the
decode/encode layer underneath, and is usable on its own.

### V-JEPA 2 video features (`NFKMLXVJEPA2`)

V-JEPA 2 is a self-supervised video encoder: a clip (or a single image) becomes a mean-pooled feature
embedding for retrieval or as a video encoder for a vision-language model. A classification release
(Something-Something v2 or Diving48) also ranks its classes. The directory factory reads the release's
`config.json` for the geometry (ViT-L, ViT-H, or ViT-g), and the backend accepts a video or an image.

```objc
NSError *error = nil;
id<NFKInferenceBackend> vjepa2 = [NFKMLXVJEPA2 backendWithDirectoryURL:releaseDirectory error:&error];

NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputVideo: asset }];
NFKInferenceResult *result = [vjepa2 runInferenceForRequest:request error:&error];
NSArray<NSNumber *> *embedding = [result outputForKey:NFKOutputEmbedding];   // the pooled feature vector
NSArray<NFKClassification *> *classes = [result outputForKey:NFKOutputClassifications];   // a classifier's
```

Each release is at reference parity against transformers' own `VJEPA2Model` or
`VJEPA2ForVideoClassification`; [model-parity.md](model-parity.md) lists the measured cosines. A probe on
your own classes trains on the device ("Training a video classifier on your own clips").

### Cosmos image and video tokens (`NFKMLXCosmosTokenizer`)

The Cosmos Tokenizer compresses an image or a clip into a continuous latent or a grid of discrete tokens
and reconstructs it. Each of the ten releases (`nvidia/Cosmos-0.1-Tokenizer-*`) is one
`NFKMLXCosmosTokenizerVariant`, and each loads from the release's own `autoencoder.jit`. The backend
reconstructs an image (and, for a video variant, a clip under `NFKInputVideo`); the tokenizer object
returns the tokens themselves.

<!-- objc-check: given NSURL *autoencoderJIT = nil; -->
```objc
NSError *error = nil;
id<NFKInferenceBackend> cosmos = [NFKMLXCosmosTokenizer backendWithVariant:NFKMLXCosmosTokenizerVariantContinuousVideo8x8x8
                                                                 weightsURL:autoencoderJIT
                                                                      error:&error];
NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputVideo: asset }];
NFKInferenceResult *result = [cosmos runInferenceForRequest:request error:&error];
NFKVideoAsset *reconstruction = [result outputForKey:NFKOutputVideo];

// Discrete tokens for an image: a 64,000-entry vocabulary, one token per 16×16 block.
NFKMLXCosmosTokenizer *tokenizer = [NFKMLXCosmosTokenizer tokenizerWithVariant:NFKMLXCosmosTokenizerVariantDiscreteImage16x16
                                                                    weightsURL:autoencoderJIT
                                                                         error:&error];
NFKMLXCosmosTokenizerCode *tokens = [tokenizer codeForImage:photo error:&error];   // int32 [h, w]
NSArray *decoded = [tokenizer framesForCode:tokens error:&error];                   // one CGImage
```

The image's sides must be multiples of 16 for `codeForImage:error:` (the backend pads and crops as the
reference does), and a clip for `codeForFrames:error:` holds one more frame than a multiple of the
temporal compression. Every variant is at reference parity against NVIDIA's own tokenizer modules on
its released weights.

## Faces in a photograph

`NFKMLXCodeFormer` restores an **aligned** 512×512 face. `photoBackend` does the finding for you —
detect, align, restore, composite back — using Vision for detection and alignment, so that step needs
no weights and no download. It is not the reference pipeline's RetinaFace, so a crop differs slightly
from facexlib's; what the model does to a crop is unchanged.

```objc
id<NFKInferenceBackend> restorer =
    [NFKMLXCodeFormer photoBackendWithFidelity:0.5
                                    weightsURL:weights
                            detectorWeightsURL:detectorWeights
                                         error:&error];
NFKInferenceRequest *request =
    [NFKInferenceRequest requestWithInputs:@{ NFKInputImage: (__bridge id)photograph }];
CGImageRef restored = (__bridge CGImageRef)[[restorer runInferenceForRequest:request error:&error]
                                            outputForKey:NFKOutputImage];
```

An image with no detectable face passes through unchanged. `NFKMLXFaceAlignment` is usable directly
when you want the crop rather than the restoration.

**Which detector.** The call above uses RetinaFace, the detector the reference pipeline runs, at
measured parity and in a 1.7 MB checkpoint. It is the default because it makes the crop the
reference's own. Vision is the alternative when a download-free path matters more:

```swift
let restorer = try NFKMLXCodeFormer.photoBackend(fidelity: 0.5, weightsURL: weights,
                                                 detector: NFKMLXVisionFaceDetector())
```

The two disagree enough to matter: on a 960×1200 portrait their boxes overlap at IoU 0.65 and their
landmarks differ by up to 15.7 px, which moves the aligned crop and therefore the restoration.

## Remote providers

Point at a hosted or local service by name. Every preset carries the endpoint and protocol; you supply
the key and the model.

```objc
// Any OpenAI-compatible provider — OpenAI, Grok, Gemini, Groq, Mistral, DeepSeek, Together,
// OpenRouter, or a local Ollama / LM Studio / llama.cpp / vLLM server.
NFKRemoteProvider *provider = [NFKRemoteProvider providerWithIdentifier:@"ollama"];
id<NFKInferenceBackend> backend = [NFKRemoteProvider backendForProvider:provider
                                                                apiKey:nil            // local: no key
                                                             modelName:@"llama3.2"];

NFKInferenceRequest *request =
    [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Summarize this in one line." }
                                parameters:@{ NFKParameterMaxTokens: @128 }
                            outputModality:NFKModalityText];
NSString *reply = [[backend runInferenceForRequest:request error:&error]
                   outputForKey:NFKRemoteBackendTextKey];
```

Anthropic speaks a different protocol, and the factory returns the right backend for it — the calling
code above is unchanged:

```objc
id<NFKInferenceBackend> claude =
    [NFKRemoteProvider backendForProvider:NFKRemoteProvider.anthropic
                                   apiKey:key
                                modelName:@"claude-sonnet-4-5"];
```

A system turn is written the same way for both; Anthropic's backend lifts it into the top-level field
the Messages API expects:

```objc
NFKInferenceRequest *chat = [NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: @[
    @{ @"role": @"system", @"content": @"Answer in one sentence." },
    @{ @"role": @"user",   @"content": @"What is InferKit?" },
] }];
```

**No preset carries a default model name** — identifiers change faster than releases do. Ask the
provider for its current list instead, and set `modelName` from one of those. Every preset answers the
same way, so one call serves a hosted API and a local runner alike:

```objc
// Blocks; run it off the render thread. A runner that is not running fails with
// kNFKError_RemoteUnreachable, which is a different answer from an empty list.
NSError *error = nil;
NSArray<NFKRemoteModel *> *models = [NFKRemoteProvider.ollama modelsWithAPIKey:nil error:&error];
if (error.code == kNFKError_RemoteUnreachable) {
    // Ollama is not running.
}
for (NFKRemoteModel *model in models) {
    NSLog(@"%@ (%@)", model.identifier, model.displayName);   // "llama3.2:latest"
}

// The same, off the calling thread.
[NFKRemoteProvider.openAI modelsWithAPIKey:key completionHandler:^(NSArray<NFKRemoteModel *> *models,
                                                                    NSError *error) {
    // populate a picker
}];
```

```swift
let models = try NFKRemoteProvider.ollama.models(withAPIKey: nil)
let names = models.map(\.identifier)
```

The list is returned as the provider orders it and is not filtered: the envelope says nothing about
what a model does, so a hosted list includes embedding and speech models the chat endpoint rejects.
`NFKRemoteModel.raw` keeps each entry for a field the type does not normalize (`ownedBy`, `createdAt`,
and `contextLength` are read where the provider publishes them). `NFKRemoteModelCatalog` is the object
under the convenience, for a timeout, a session, or `isReachableWithError:`.

A preset re-points at another address with one field changed — a runner on another port, or on another
machine on the network — keeping its identity and protocol:

```objc
NFKRemoteProvider *lanOllama = [NFKRemoteProvider.ollama providerWithBaseURL:
                                [NSURL URLWithString:@"http://192.168.1.20:11434/v1"]];
NSURL *transcribe = [NFKRemoteProvider.openAI URLForPath:@"audio/transcriptions"];
```

Embeddings are the same shape everywhere (`POST /embeddings`), and the vector comes back under the
core key the on-device embedders use, so search code does not change with the engine:

```objc
NFKRemoteEmbeddingBackend *embedder =
    [NFKRemoteEmbeddingBackend backendForProvider:NFKRemoteProvider.ollama
                                           apiKey:nil
                                        modelName:@"nomic-embed-text"];   // nil for Anthropic: no endpoint
NFKInferenceResult *one = [embedder runInferenceForRequest:
    [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"a red barn" }] error:&error];
NSArray<NSNumber *> *vector = one.embedding;

// A corpus, one request; the ith vector belongs to the ith text.
NSArray<NSArray<NSNumber *> *> *vectors = [embedder embeddingsForTexts:documents error:&error];
```

**Which local runner is up is a question the code should not have to answer.** Discovery probes the
local presets and hands back the ones that reply, so an app serves a user running Ollama, LM Studio,
llama.cpp, or vLLM without being told which:

```objc
// The four local presets, in the order discovery probes them.
NSArray<NFKRemoteProvider *> *local = NFKRemoteProvider.localProviders;   // ollama, lmstudio, llamacpp, vllm

// One call instead of a choice the app cannot make. nil means none of them is running. Blocks, so
// run it off the render thread.
NFKRemoteProvider *running = NFKRemoteProvider.firstAvailableLocalProvider;
id<NFKInferenceBackend> backend =
    [NFKRemoteProvider backendForFirstAvailableLocalProviderWithModelName:@"llama3.2"];

// The whole list, for a picker of the runners this machine has up right now.
for (NFKRemoteProvider *provider in NFKRemoteProvider.availableLocalProviders) {
    NSLog(@"%@ is running", provider.displayName);
}

// One address on its own: any HTTP reply counts, so a rejected key is still a server that is there.
NFKRemoteProvider *stopped = [NFKRemoteProvider.ollama providerWithBaseURL:
                              [NSURL URLWithString:@"http://127.0.0.1:9/v1"]];
NSError *error = nil;
BOOL up = [stopped isReachableWithAPIKey:nil timeout:2.0 error:&error];   // NO, kNFKError_RemoteUnreachable
```

```swift
let running = NFKRemoteProvider.firstAvailableLocalProvider()
let backend = NFKRemoteProvider.backendForFirstAvailableLocalProvider(withModelName: "llama3.2")

// Another port or another machine: pass the list to probe.
let lan = [NFKRemoteProvider.ollama.withBaseURL(URL(string: "http://192.168.1.20:11434/v1")!)]
let reachable = NFKRemoteProvider.availableProviders(among: lan, timeout: 2)

// The completion-handler forms import as async calls under a probe name.
let discovered = await NFKRemoteProvider.probeAvailableLocalProviders()
```

`availableProvidersAmong:timeout:` probes concurrently, so the call costs one timeout rather than one
per address; `firstAvailableProviderAmong:timeout:` probes in order and stops at the first reply. Two
providers are equal when their identifier and base match, so a discovered provider compares to the
preset it came from. `availableLocalProvidersWithCompletionHandler:` and
`firstAvailableLocalProviderWithCompletionHandler:` are the forms that do not block the caller.

**Local runners have a second surface.** The OpenAI-compatible endpoints say nothing about what is
installed, what is loaded, how large a model is, or how to get one. The preset hands back an adapter
over the runner's native API — Ollama and LM Studio have one; llama.cpp and vLLM have nothing beyond
the OpenAI surface, so they answer nil:

```objc
id<NFKLocalModelRunner> runner = NFKRemoteProvider.ollama.localRunner;   // http://localhost:11434
if (!runner.isRunning) { /* show "start Ollama" */ }

for (NFKRemoteModel *model in [runner installedModelsWithError:&error]) {
    // "gpt-oss:20b" · 13.8 GB · MXFP4 · 131072 tokens · completion, tools, thinking
    NSLog(@"%@ %@ %@ %@ %@", model.identifier, model.sizeBytes, model.quantization,
          model.contextLength, model.capabilities);
}
NSArray<NFKRemoteModel *> *loaded = [runner loadedModelsWithError:&error];   // in memory now

// The actions that change the machine are offered only where the runner has them.
if ([runner respondsToSelector:@selector(pullModel:)]) {
    NFKInferenceJob *pull = [runner pullModel:@"llama3.2"];   // Ollama: streams the download
    pull.progressHandler = ^(NFKInferenceJob *job) {
        // job.progress is the fraction of the layer being fetched;
        // job.partialResult carries the runner's status line under NFKOutputText
    };
    pull.completionHandler = ^(NFKInferenceJob *job) { /* job.result or job.error */ };
    // [pull cancel] stops the download
}
```

```swift
let runner = NFKRemoteProvider.ollama.localRunner
let installed = try runner?.installedModels()
let embedder = NFKRemoteEmbeddingBackend.backend(for: .ollama, apiKey: nil, modelName: "nomic-embed-text")
```

A pull of a name the registry does not have fails with the runner's own message — measured against
Ollama 0.33, which answers HTTP 200 and puts the failure in an error line inside the stream, so the
job reads every line rather than trusting the status.

**The remaining modalities have remote backends too**, so every direction an on-device engine serves
has a hosted counterpart answering with the same key. Text to speech (`POST /audio/speech`; served by
openai, groq, together, xai, mistral, and openrouter, verified at release) returns an `NFKAudioAsset`
under `NFKOutputAudio`, a WAV by default, which is what `NFKMLXSpeechBackend` writes; a voice is required
and has no default, for the reason a model name has none:

```objc
NFKRemoteSpeechBackend *speaker =
    [NFKRemoteSpeechBackend backendForProvider:NFKRemoteProvider.openAI
                                        apiKey:key modelName:@"gpt-4o-mini-tts" voice:@"alloy"];
NFKAudioAsset *clip = [[speaker runInferenceForRequest:
    [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Rendering complete." }] error:&error]
    outputForKey:NFKOutputAudio];                       // clip.fileURL is a .wav
```

Image generation chooses its operation from the request the way `NFKMLXBackend` does: a prompt alone is
text-to-image (`POST /images/generations`; openai, together, xai, openrouter), an image under
`NFKInputImage` is an edit of it (`POST /images/edits`, multipart; openai, xai), and an image with a
mask under `NFKInputMask` is an inpaint of the mask's region. `NFKParameterWidth`/`Height` become the
service's size; the result is a 32BGRA `CVPixelBuffer` under `NFKOutputImage`:

```objc
NFKRemoteImageBackend *painter =
    [NFKRemoteImageBackend backendForProvider:NFKRemoteProvider.openAI apiKey:key modelName:@"gpt-image-1"];
NFKInferenceRequest *generate =
    [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"a lighthouse at dusk" }
                                parameters:@{ NFKParameterWidth: @1024, NFKParameterHeight: @1024 }
                            outputModality:NFKModalityImage];
CVPixelBufferRef image = (__bridge CVPixelBufferRef)[[painter runInferenceForRequest:generate error:&error]
                                                      outputForKey:NFKOutputImage];

NFKInferenceRequest *edit =                             // image + prompt → image
    [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"make the sky stormy",
                                              NFKInputImage: (__bridge id)image }];
NFKInferenceRequest *inpaint =                          // + mask → the masked region only
    [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"add a sailboat",
                                              NFKInputImage: (__bridge id)image,
                                              NFKInputMask: (__bridge id)mask }];
```

And an image beside a prompt is a vision question through the ordinary chat backend — no new class.
`NFKRemoteBackend` puts it on the last user turn as an `image_url` content part (inline, base64 PNG),
which every OpenAI-compatible vision model reads, local runners included; `NFKAnthropicBackend` puts
it there as an `image` block. Measured against a live Ollama with `qwen3.5:27b`: a flat blue square
and "what color is this?" come back "blue".

<!-- objc-check: given CGImageRef frame = NULL; -->
```objc
id<NFKInferenceBackend> eyes = [NFKRemoteProvider backendForProvider:NFKRemoteProvider.ollama
                                                              apiKey:nil modelName:@"qwen3.5:27b"];
NFKInferenceRequest *look = [NFKInferenceRequest requestWithInputs:@{
    NFKInputPrompt: @"What is in this frame?",
    NFKInputImage: (__bridge id)frame }];               // CGImage, CVPixelBuffer, or a BGRA/RGBA MTLTexture
NSString *answer = [[eyes runInferenceForRequest:look error:&error] outputForKey:NFKRemoteBackendTextKey];
```

```swift
let speaker = NFKRemoteSpeechBackend(for: .openAI, apiKey: key, modelName: "gpt-4o-mini-tts", voice: "alloy")
let painter = NFKRemoteImageBackend(for: .openAI, apiKey: key, modelName: "gpt-image-1")
let png = NFKImageCoding.pngData(for: image)            // the codec under both, public
```

The three image representations the contract carries (`CGImage`, `CVPixelBuffer`, `MTLTexture`) are
accepted wherever an image is taken, through `NFKImageCoding`, which is public: PNG bytes or a data
URL out of any of them, and a 32BGRA pixel buffer back from any format ImageIO reads. Several images
go under `NFKInputImages` (an array), attached in order after `NFKInputImage`.

**The chat backends stream.** `submitInferenceJobForRequest:` on `NFKRemoteBackend` and
`NFKAnthropicBackend` sends the request with streaming on and reads the reply as server-sent events:
each token appends to the job's `partialResult`, the job finishes with the same result the blocking
form returns, and **cancelling the job cancels the request**, so an abandoned completion stops costing
at that moment. `NFKInferenceSubmit` reaches this form too, so the snippet in the contract section
above streams from a remote provider exactly as it does from an on-device one. Measured against a live
Ollama: the reply arrives in more than one piece and the last partial is the final text.

```objc
NFKInferenceJob *job = [backend submitInferenceJobForRequest:request];   // any remote chat backend
job.progressHandler = ^(NFKInferenceJob *j) {
    NSString *soFar = j.partialResult.text;                                 // the text so far, not the delta
};
job.completionHandler = ^(NFKInferenceJob *j) {
    NSString *text = j.result.text;                                         // or j.error
};
[job cancel];                                                               // closes the connection
```

**Tools and structured output** use two contract keys, each translated into the provider's own shape:

```objc
NSDictionary *weather = @{ @"name": @"get_weather",
                           @"description": @"Current weather in a city.",
                           @"parameters": @{ @"type": @"object",
                                             @"properties": @{ @"city": @{ @"type": @"string" } },
                                             @"required": @[ @"city" ] } };
NFKInferenceRequest *ask =
    [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"What's the weather in Paris?" }
                                parameters:@{ NFKParameterTools: @[ weather ] }
                            outputModality:NFKModalityText];
NFKInferenceResult *turn = [backend runInferenceForRequest:ask error:&error];
for (NSDictionary *call in turn.toolCalls) {          // {id, name, arguments (parsed), argumentsJSON}
    // run the tool, then send its result back as a tool message
}

NSDictionary *schema = @{ @"type": @"object", @"properties": @{ @"answer": @{ @"type": @"integer" } } };
NFKInferenceRequest *structured =
    [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"What is 6 × 7?" }
                                parameters:@{ NFKParameterJSONSchema: schema }
                            outputModality:NFKModalityText];
NSDictionary *answer = [backend runInferenceForRequest:structured error:&error].structured;   // @{ answer: 42 }
```

`NFKRemoteBackend` sends the tools as OpenAI `function` tools and the schema as a `json_schema`
response format; `NFKAnthropicBackend` sends the tools in the Messages shape and, since that API has
no response format, asks for the schema through a forced tool whose input is the reply. Both assemble
a streamed tool call across its argument deltas. A tool already written in a provider's wire shape
passes through unwrapped. Measured against a live Ollama: asked for the weather with the tool declared,
`qwen3.5:27b` calls `get_weather` with `{"city": "Paris"}`.

**A rate limit is retried.** Every blocking remote call retries a 429, 502, 503, or 504 after the
provider's `Retry-After` or an exponential delay from half a second — `NFKRemoteTransport.retryAttempts`
(default 2) more times, never after a delay above `NFKRemoteTransport.maximumRetryDelay` (default 8 s),
where waiting would cost more than failing. A refused connection is not retried: a server that is not
there is an answer in itself.

**More directions, in and out.** The chat backends take three more inputs beside the prompt, each
riding on the last user turn in the provider's own shape, and one more output:

```objc
// Audio in (OpenAI-compatible only; the Messages API refuses it rather than dropping it):
NFKInferenceRequest *heard = [NFKInferenceRequest requestWithInputs:@{
    NFKInputPrompt: @"What did I ask for?",
    NFKInputAudio: [NFKAudioAsset audioAssetWithFileURL:recording] }];    // wav or mp3, by extension

// A document (PDF), on both — a `file` part for OpenAI, a `document` block for Anthropic:
NFKInferenceRequest *summarized = [NFKInferenceRequest requestWithInputs:@{
    NFKInputPrompt: @"Summarize the brief.",
    NFKInputDocument: briefPDFURL,                                      // NSURL or NSData
    NFKInputDocuments: @[ appendixPDFData ] }];

// A clip, sampled into frames for a vision model (video → text), on both:
NFKInferenceRequest *watched = [NFKInferenceRequest requestWithInputs:@{
    NFKInputPrompt: @"What happens in this clip?",
    NFKInputVideo: [NFKVideoAsset videoAssetWithFileURL:clipURL] }
    parameters:@{ NFKParameterVideoFrameCount: @8 }                    // evenly spaced; 8 by default
    outputModality:NFKModalityText];

// A spoken reply beside the text (OpenAI-compatible only), streamed or not:
NFKInferenceRequest *spoken = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Read me the summary." }
    parameters:@{ NFKParameterAudioOutput: @{ @"voice": @"alloy" } }      // format wav by default
    outputModality:NFKModalityAudio];
NFKInferenceResult *turn = [backend runInferenceForRequest:spoken error:&error];
NFKAudioAsset *voice = [turn outputForKey:NFKOutputAudio];                // turn.text is the transcript
```

`NFKVideoSampling` is the public piece under the clip input: `framesOfVideoAtURL:count:error:` returns
evenly spaced `CGImage`s. Measured against a live Ollama: four frames of a red–green–blue–white clip
and `qwen3.5:27b` names the colours.

The transcription backend gains what the on-device Whisper backend has: `emitsTimestamps` asks for the
verbose reply and adds the segments under `NFKOutputSegments` as `NFKAudioSegment`s, and `translates`
sends the audio to the sibling `/audio/translations` endpoint for an English transcript.

```objc
NFKRemoteTranscriptionBackend *ears = [NFKRemoteTranscriptionBackend backendForProvider:NFKRemoteProvider.groq
                                                                                 apiKey:key modelName:@"whisper-large-v3"];
ears.emitsTimestamps = YES;
NSArray<NFKAudioSegment *> *segments = [ears runInferenceForRequest:request error:&error].segments;
```

Three more services round out the remote surface:

```objc
// Text/image → video, the job-style shape: submit, poll, download. Gemini's Veo through its
// Sora-compatible path; naming xAI, Together, or OpenRouter instead reaches their video APIs with
// the same request, and apiStyle:NFKRemoteVideoAPIStyleGeminiVeo reaches Veo's native API.
NFKRemoteVideoBackend *director = [NFKRemoteVideoBackend backendForProvider:NFKRemoteProvider.googleGemini
                                                                     apiKey:key modelName:@"veo-3.1-generate-preview"];
NFKInferenceJob *shoot = [director submitInferenceJobForRequest:
    [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"a lighthouse at dusk",
                                              NFKInputImage: (__bridge id)firstFrame,        // optional
                                              NFKInputLastFrame: (__bridge id)lastFrame }    // optional
                                parameters:@{ NFKParameterDurationSeconds: @8,
                                              NFKParameterAspectRatio: @"16:9",
                                              NFKParameterResolution: @"1080p",
                                              @"person_generation": @"allow_adult" }         // a Veo option, by name
                            outputModality:NFKModalityVideo]];
shoot.completionHandler = ^(NFKInferenceJob *job) {
    NFKVideoAsset *clip = [job.result outputForKey:NFKOutputVideo];       // an .mp4 on disk
};

// The Responses API: a wire-shaped tool asks for a service-run one; the reply's id continues the
// conversation on the next request under NFKParameterPreviousResponseIdentifier.
NFKRemoteResponsesBackend *responses = [NFKRemoteResponsesBackend backendForProvider:NFKRemoteProvider.openAI
                                                                              apiKey:key modelName:@"gpt-5.6-sol"];
NFKInferenceResult *answer = [responses runInferenceForRequest:
    [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"What changed in Swift 7?" }
                                parameters:@{ NFKParameterTools: @[ @{ @"type": @"web_search" } ] }] error:&error];
NSArray *sources = [answer outputForKey:NFKOutputCitations];            // url_citation annotations
NSString *next = [answer outputForKey:NFKOutputResponseIdentifier];

// Gemini's Interactions API: the output modality picks speech, an image, music, or video.
NFKGeminiInteractionsBackend *gemini = [NFKGeminiInteractionsBackend backendWithAPIKey:key
                                                                             modelName:@"gemini-3.1-flash-tts-preview"];
gemini.voice = @"Kore";
NFKAudioAsset *spoken = [[gemini runInferenceForRequest:
    [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Say cheerfully: good morning!" }
                                parameters:@{} outputModality:NFKModalityAudio] error:&error] outputForKey:NFKOutputAudio];

// Fill-in-the-middle, OCR, and a token count before sending.
NFKRemoteCompletionBackend *infill = [NFKRemoteCompletionBackend backendForProvider:NFKRemoteProvider.mistral
                                                                             apiKey:key modelName:@"codestral-latest"];
NSString *middle = [infill runInferenceForRequest:
    [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"def add(a, b):\n", NFKInputSuffix: @"\nprint(add(1, 2))" }]
                                            error:&error].text;
NFKRemoteOCRBackend *reader = [NFKRemoteOCRBackend backendForProvider:NFKRemoteProvider.mistral apiKey:key modelName:@"mistral-ocr-latest"];
NSString *markdown = [reader runInferenceForRequest:
    [NFKInferenceRequest requestWithInputs:@{ NFKInputDocument: pdfURL }] error:&error].text;
NFKRemoteTokenCounter *counter = [NFKRemoteTokenCounter counterForProvider:NFKRemoteProvider.anthropic apiKey:key modelName:@"claude-opus-5-5"];
NSNumber *tokens = [counter tokenCountForRequest:request error:&error];

// A live spoken conversation over a WebSocket; the same calls reach xAI, Gemini Live, and the
// streaming transcription and speech sockets through apiStyle.
NFKRealtimeSession *live = [NFKRealtimeSession sessionForProvider:NFKRemoteProvider.openAI
                                                         apiStyle:NFKRealtimeAPIStyleOpenAIConversation
                                                           apiKey:key modelName:@"gpt-realtime-2.1"];
live.voice = @"marin";
live.audioHandler = ^(NSData *pcm) { /* 16-bit PCM at live.outputSampleRate, on the socket's queue */ };
live.textHandler = ^(NSString *text, NFKRealtimeTextKind kind) { NSLog(@"%@", text); };
[live connect];
[live appendAudio:microphonePCM];          // 16-bit mono at live.inputSampleRate
[live commitAudio];
[live requestResponse];

// Files: upload once, then name the file in any request that takes a document or image.
NFKRemoteFileStore *files = [NFKRemoteFileStore fileStoreForProvider:NFKRemoteProvider.anthropic apiKey:key];
NFKRemoteFile *contract = [files uploadFileAtURL:pdfURL purpose:nil error:&error];
NFKInferenceResult *summary = [claude runInferenceForRequest:
    [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Summarize the termination clause.",
                                              NFKInputDocument: contract }] error:&error];

// A hosted retrieval store: create it, add the uploaded file, search it.
NFKRemoteRetrievalStore *library = [NFKRemoteRetrievalStore retrievalStoreForProvider:NFKRemoteProvider.openAI apiKey:key];
NFKRetrievalStoreRecord *store = [library createStoreNamed:@"Contracts" options:nil error:&error];
[library uploadData:pdfData filename:@"contract.pdf" toStore:store.identifier error:&error];
NSArray<NFKRetrievalMatch *> *matches = [library searchStores:@[ store.identifier ] query:@"notice period"
                                                        limit:5 filter:nil error:&error];

// Usage and spend. The admin key comes from the app's user at run time, never from the binary.
NFKRemoteUsageReporter *usage = [NFKRemoteUsageReporter reporterForProvider:NFKRemoteProvider.anthropic apiKey:userAdminKey];
NSArray<NFKCostEntry *> *spend = [usage costsFromDate:monthStart toDate:nil groupBy:@[ @"description" ] error:&error];
NSDecimalNumber *total = [spend valueForKeyPath:@"@sum.amount"];   // dollars

// Rerank, the same shape as the on-device NFKMLXModernBERTReranker (Together and OpenRouter serve it):
NFKRemoteReranker *ranker = [NFKRemoteReranker rerankerForProvider:NFKRemoteProvider.together
                                                            apiKey:key modelName:@"Salesforce/Llama-Rank-V1"];
NSArray<NSNumber *> *order = [ranker rankedIndicesForQuery:query documents:shortlist error:&error];

// Moderation (OpenAI and Mistral): per-category scores, most confident first, and the verdict.
NFKRemoteModerationBackend *gate = [NFKRemoteModerationBackend backendForProvider:NFKRemoteProvider.openAI
                                                                           apiKey:key modelName:@"omni-moderation-latest"];
NFKInferenceResult *verdict = [gate runInferenceForRequest:
    [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: userText }] error:&error];
BOOL flagged = [verdict.structured[@"flagged"] boolValue];
NFKClassification *top = verdict.classifications.firstObject;           // e.g. "harassment" 0.91
```

### Typed decisions (`NFKTypeSafeBackend`, Jev)

Jev, TypeSafe AI's System One model, answers typed questions about a state instead of generating text:
a choice among named options, a score on an ordered scale, or a noul, which is the probability that a
statement holds. The wire shape shares nothing with the chat protocols, so the `typesafe` preset has
its own backend; the factory hands it back. The model is required, and `jev-latest` is the alias of the
current release.

```objc
NFKTypeSafeBackend *jev = (NFKTypeSafeBackend *)[NFKRemoteProvider backendForProvider:NFKRemoteProvider.typeSafe
                                                                               apiKey:key modelName:@"jev-latest"];

NSDictionary<NSString *, NFKDecisionQuestion *> *questions = @{
    @"department": [NFKDecisionQuestion choiceQuestionWithInstructions:@"Which team should handle this?"
                                                               options:@[ @"billing", @"technical", @"sales" ]
                                                          descriptions:@{ @"billing": @"Payments, invoicing, refunds",
                                                                          @"technical": @"Bugs, outages, integrations" }],
    @"severity": [NFKDecisionQuestion scoreQuestionWithInstructions:@"How severe is the problem?"
                                                             levels:@[ @"low", @"medium", @"high" ]],
    @"urgent":   [NFKDecisionQuestion noulQuestionWithInstructions:@"The customer needs an answer today."],
};

// The state is a string, or a JSON-serializable record or conversation. The answers come back keyed
// as the questions were, each carrying the fields its type has and the probabilities behind it.
NSDictionary<NSString *, NFKDecisionAnswer *> *answers =
    [jev answersForState:@"Help! My payouts have been failing for 3 days." questions:questions error:&error];
answers[@"department"].choice;                          // "technical"
answers[@"department"].probabilities[@"technical"];     // 0.85
answers[@"severity"].score;                             // 1.6, between "medium" and "high"
answers[@"urgent"].probability;                         // 0.91

// The same through the contract: NFKInputState + NFKInputQuestions in, NFKOutputAnswers out, the
// whole reply under NFKOutputStructured and the token count under NFKOutputUsage.
NFKInferenceRequest *ask = [NFKInferenceRequest requestWithInputs:@{ NFKInputState: record,
                                                                    NFKInputQuestions: questions }];
NFKInferenceResult *decided = [jev runInferenceForRequest:ask error:&error];
decided.answers[@"urgent"].probability;
```

```swift
let jev = NFKRemoteProvider.backend(for: .typeSafe, apiKey: key, modelName: "jev-latest") as! NFKTypeSafeBackend
let questions = [
    "department": NFKDecisionQuestion.choiceQuestion(withInstructions: "Which team should handle this?",
                                                     options: ["billing", "technical", "sales"]),
    "urgent": NFKDecisionQuestion.noulQuestion(withInstructions: "The customer needs an answer today."),
]
let answers = try jev.answers(forState: "Help! My payouts have been failing for 3 days.", questions: questions)
answers["department"]?.choice        // "technical"
```

A question already in the service's own shape (`{type, instructions, criteria}`) goes under
`NFKInputQuestions` as a dictionary and passes through. A rate limit or an overload (429, 529) is
retried through `NFKRemoteTransport` like every other blocking remote call.

Midjourney has no official public API, so there is no preset for it; `opencode.ai` is a coding agent
rather than an inference service; and Codex is OpenAI's agent using the OpenAI API, so it is the
`openai` preset.

## Model gallery

Every shipped MLX model is built the same way: a direct `@objc` factory (`+backendWith…weightsURL:` for
local weights, `nil` → random weights that run; `+backendWith…repo:weightsPath:…` to download and build).
The compiled `MLXModelGalleryExamples` builds and runs each of these. Grouped by task:

```swift
// Upscaling & restoration (image → image)
let upscaler   = try NFKMLXRealESRGAN.backend(variant: .x4, weightsURL: nil)   // "real-esrgan-x4"; .generalX4V3/.animeVideoV3 run the compact generator
let swinIR     = try NFKMLXSwinIR.backend(weightsURL: nil)                      // "swinir-x4"; every release: .classicalX2…X8, .lightweightSRX2…X4, .realWorldX4Medium/Large
let realWorld  = try NFKMLXSwinIR.backend(variant: .realWorldX4Large, weightsURL: nil)
let hat        = try NFKMLXHAT.backend(variant: .large, weightsURL: nil)        // "hat-l-x4"; .base / .realWorld
let denoiser   = try NFKMLXNAFNet.backend(weightsURL: nil)                      // "nafnet"; .sidd/.goPro/.reds/.siddWidth64/.goProWidth64
let lowLight   = try NFKMLXZeroDCE.backend(weightsURL: nil)                     // "zero-dce"
let lowLight2  = try NFKMLXZeroDCEPlus.backend(weightsURL: nil)                 // "zero-dce-plus"
let stylizer   = try NFKMLXStyleTransfer.backend(weightsURL: nil)              // "fast-style-transfer"
let adain      = try NFKMLXAdaIN.backend(encoderURL: nil, decoderURL: nil)      // "adain"; style image under NFKInputControl
let colorizer  = try NFKMLXColorizer.backend(weightsURL: nil)                  // "colorizer-eccv16"
let ddcolor    = try NFKMLXDDColor.backend(variant: .modelscope, weightsURL: nil)  // "ddcolor"; .paper / .artistic
let faceRestore = try NFKMLXCodeFormer.backend(weightsURL: nil)                // "codeformer"

// Depth (image → grayscale depth)
let depth = try NFKMLXDepthAnything.backend(variant: .small, weightsURL: nil)  // "depth-anything-v2-small"

// Matting (plate → foreground image + alpha under NFKOutputMask)
let cutout   = try NFKMLXU2Net.backend(variant: .full, weightsURL: nil)        // "u2net"
let dichotomous = try NFKMLXISNet.backend(weightsURL: nil)                     // "isnet"
let highRes  = try NFKMLXBiRefNet.backend(weightsURL: nil)                     // "birefnet"; resizes to 1024
let videoKey = try NFKMLXRVM.backend(weightsURL: nil)                          // "robust-video-matting" (MobileNetV3); .resNet50 is the heavier release
let portrait = try NFKMLXMODNet.backend(weightsURL: nil)                       // "modnet"
let sam2     = try NFKMLXSAM2.backend(variant: .tiny, release: .sam21, weightsURL: nil)  // "sam2"; a click under NFKSAMPointKey

// Semantic segmentation (image → grayscale label map; index = round(gray·(classCount−1)))
let segformer = try NFKMLXSegFormer.backend(weightsURL: nil)                   // "segformer-b0"
let deeplab   = try NFKMLXDeepLab.backend(weightsURL: nil)                     // "deeplabv3"
let bisenet   = try NFKMLXBiSeNet.backend(weightsURL: nil)                     // "bisenet"

// Detection & pose (new core value types)
let yolo = try NFKMLXYOLO.backend(weightsURL: nil, labels: cocoLabels)         // result.detections : [NFKDetection]
let yolo26 = try NFKMLXYOLOGenerations.backend(release: .v26Nano, weightsURL: nil, labels: cocoLabels)   // "yolo26n"; every v9/v10/11/12/26 size is a release
let pose = try NFKMLXPose.backend(weightsURL: nil, jointNames: cocoJoints)     // result.pose : [NFKKeypoint]
let vitPose = try NFKMLXVitPose.backend(variant: .base, weightsURL: nil, jointNames: cocoJoints)  // "vitpose-base"; .baseSimple is the simple decoder; DARK-refined [NFKKeypoint]

// Embeddings, video, promptable segmentation
let clip    = try NFKMLXCLIP.backend(weightsURL: nil)                          // result.embedding : [NSNumber]; .vitB16 / .vitL14 / .vitL14At336 too
let siglip2 = try NFKMLXSigLIP2.backend(weightsURL: nil)                       // SigLIP 2: result.embedding (NFKMLXSigLIP2.textEmbedding for text); every release under NFKMLXSigLIP2Variant
let taesd   = try NFKMLXTAESD.backend(weightsURL: nil)                         // tiny AE: image → latent → image (NFKMLXTAESD.encode/decode for previews)
let videoSR = try NFKMLXVideoSR.backend(weightsURL: nil)                       // "video-super-resolution"
let cosmos  = try NFKMLXCosmosTokenizer.backend(variant: .discreteImage8x8, weightsURL: nil)  // image/clip → latent or tokens → reconstruction; "cosmos-tokenizer-di8x8" (one name per variant)
let sam     = try NFKMLXSAM.backend(weightsURL: nil)                           // plate + point under NFKSAMPointKey; .vitB / .vitL / .vitH

// Audio
// Translation (text → text; every translator loads a release directory)
let opusMT     = try NFKMLXMarian.backend(sourceLanguage: "en", targetLanguage: "de", cacheDirectoryURL: nil) // "opus-mt"; downloads Helsinki-NLP/opus-mt-en-de
let m2m100     = try NFKMLXM2M100.backend(variant: .m418M, directoryURL: m2mDir)   // "m2m100"; .m1_2B, .small100 → "small100"
let madlad     = try NFKMLXMADLAD.backend(directoryURL: madladDir, half: true)     // "madlad400-3b-mt"; 400+ languages, bfloat16
let tgemma     = try NFKMLXTranslateGemma.backend(directoryURL: tgDir, precision: .checkpoint) // "translategemma"; Gemma 3 + the translation template
let transcriber = try NFKMLXWhisper.backend(weightsURL: nil)                   // audio → NFKOutputText; .tiny … .largeV3Turbo
let stems       = try NFKMLXDemucs.backend(weightsURL: nil)                    // audio → "drums"/"bass"/"other"/"vocals"
let speakers    = try NFKMLXConvTasNet.backend(weightsURL: nil)               // audio → "speaker-1"/"speaker-2"
let clean       = try NFKMLXDenoiser.backend(weightsURL: nil)                  // audio → NFKOutputAudio
let enhanced    = try NFKMLXMPSENetFactory.backend(weightsURL: nil)            // MP-SENet: magnitude+phase transformer → NFKOutputAudio
let realtime    = try NFKMLXGTCRNFactory.backend(weightsURL: nil)             // GTCRN: ~48K-param real-time enhancer → NFKOutputAudio
let dereverbed  = try NFKMLXSGMSE.backend(weightsURL: nil)                     // SGMSE+: score-based generative dereverb (reverse-SDE sampler) → NFKOutputAudio
let regenerated = try NFKMLXStoRM.backend(weightsURL: nil)                     // StoRM: few-step stochastic regeneration (predictor + conditioned score) → NFKOutputAudio
let fullband    = try NFKMLXMossFormer2Factory.backend(weightsURL: nil)       // MossFormer2 SE: full-band 48 kHz enhancement (Kaldi-fbank mask) → NFKOutputAudio
let bandwidth   = try NFKMLXMossFormer2SRFactory.backend(directoryURL: nil)   // MossFormer2 SR: mel→mel backbone + Snake HiFi-GAN + bandwidth substitution → 48 kHz NFKOutputAudio
let deepfilter  = try NFKMLXDeepFilterNetFactory.backend(weightsURL: nil)     // DeepFilterNet3: ~2.3M-param real-time 48 kHz denoiser (ERB mask + deep filter) → NFKOutputAudio
let voicerestore = try NFKMLXVoiceRestoreFactory.backend(weightsURL: transformerURL, vocoderURL: bigvganURL, steps: 32, cfgStrength: 0.5)  // VoiceRestore: ~301M flow-matching universal restorer (E2-TTS transformer + BigVGAN) → NFKOutputAudio
let resemble    = try NFKMLXResembleEnhanceFactory.backend(directoryURL: enhancerStage2Dir)  // Resemble Enhance: 5-network general restorer (STFT-mask denoiser + IRMAE/CFM + UnivNet LVC vocoder) → NFKOutputAudio
let metricgan   = try NFKMLXMetricGANPlus.backend(weightsURL: nil)             // MetricGAN+: 2-layer BLSTM magnitude mask over log1p(|X|) → NFKOutputAudio
let cmgan       = try NFKMLXCMGAN.backend(weightsURL: nil)                     // CMGAN: conformer metric GAN (mask + complex residual) → NFKOutputAudio
let frcrn       = try NFKMLXFRCRN.backend(weightsURL: nil)                     // FRCRN: two complex UNets with frequency-recurrent FSMNs → NFKOutputAudio
let nuwave      = try NFKMLXNUWave2.backend(weightsURL: nil)                   // NU-Wave 2: diffusion bandwidth extension (8-step DDIM) → 48 kHz NFKOutputAudio
let apollo      = try NFKMLXApollo.backend(weightsURL: nil)                    // Apollo: music codec-artifact restoration (80-band Roformer) → 44.1 kHz NFKOutputAudio
let vad         = try NFKMLXVAD.backend(weightsURL: nil)                       // result.segments : [NFKAudioSegment]
let sileroVAD   = try NFKMLXSileroVAD.backend(weightsURL: nil)                  // Silero v6: result.segments : [NFKAudioSegment]
let dac         = try NFKMLXDAC.backend(weightsURL: nil)                        // neural codec: audio → codes → audio (NFKMLXDAC.encode for the tokens)
let snac        = try NFKMLXSNAC.backend(weightsURL: nil)                       // multi-scale codec (NFKMLXSNAC.encode → per-codebook streams at different rates); .music32kHz / .music44kHz
let bigvgan     = try NFKMLXBigVGANFactory.backend(weightsURL: nil)             // BigVGAN v2 vocoder: audio → mel → waveform copy-synthesis (net(mel) is the generator)
let mimi        = try NFKMLXMimi.backend(weightsURL: nil)                       // Mimi: transformer-in-codec (NFKMLXMimi.encode → per-codebook streams, semantic + acoustic); audio → codes → audio
let tagger      = try NFKMLXAudioTagger.backend(weightsURL: nil, labels: nil)  // result.classifications : [NFKClassification]
let basicPitch  = try NFKMLXBasicPitch.backend(weightsURL: nil)                 // music transcription: result.midi : NFKMIDISequence (notes, bends, standardMIDIFileData())
let hft         = try NFKMLXHFTTransformer.backend(weightsURL: nil)             // piano transcription: onset/offset/multi-pitch/velocity → result.midi
let muscriptor  = try NFKMLXMuScriptor.backend(variant: .medium, weightsURL: nil)  // multi-instrument transcription: result.midi, one program per instrument
let allinone    = try NFKMLXAllInOne.backend(weightsURL: nil)                   // music structure: result.segments (sections), result.beats, NFKOutputTempo. Four stems in, or a mixture with demucsWeightsURL:

// Music generation (MiniMax Music 3): a description under NFKInputPrompt and lyrics under
// NFKInputLyrics become a stereo 44.1 kHz NFKAudioAsset. The factory takes the downloaded release
// DIRECTORY (the MiniMaxAI/MiniMax-Music3 tree, ~27 GB) — there is no random-weights form, and
// isReady reports whether the weights are present. The weights carry the MiniMax-Music3 Community
// License (UI attribution in commercial products); see Docs/companions.md.
let music = try NFKMLXMusic3.backend(directoryURL: releaseDirectory)           // "minimax-music3"
// NFKParameterDurationSeconds bounds the clip (the model may stop earlier, cap six minutes);
// NFKParameterSeed makes a take repeatable; NFKParameterSteps and NFKParameterGuidanceScale drive
// the flow-matching stage.
// A one-time quantize (4-bit language model, 8-bit DiT — the measured split) shrinks the release
// to ~9 GiB; the same factory takes the result, and a stack that small stays loaded between runs:
try NFKMLXMusic3.quantizeRelease(at: releaseDirectory, to: quantizedDirectory)
let residentMusic = try NFKMLXMusic3.backend(directoryURL: quantizedDirectory)
// The plain factory decides at each run whether the stages stay loaded (.automatic). .staged loads
// each stage for its turn and releases it; .resident holds them and fails a run they do not fit.
// Either choice writes the same audio.
let stagedMusic = try NFKMLXMusic3.backend(directoryURL: quantizedDirectory, residency: .staged)
```

Video models expose a clip-level Swift API, while the module/matting backend does one frame at a time.
The two work differently: `NFKMLXRVMNet.forward(_:state:)` threads a recurrent state forward through
the frames, so a caller passes each frame in turn and carries the state along.
`NFKMLXVideoSRNet.upscaleSequence(_:)` takes the whole clip at once instead, because BasicVSR
propagates in both directions — a frame's result draws on the frames after it, which no
frame-at-a-time call can supply.

## Customizing a model on a consumer's own data

`NFKMLXTrainer` fine-tunes a shipped MLX model in the app. The result of training is an ordinary
safetensors checkpoint, so the model's existing factory loads it with no separate route:

```swift
// 1. Build the network itself rather than a backend. nil weights trains from scratch; the released
//    checkpoint fine-tunes from it.
let net = try NFKMLXZeroDCE.network(weightsURL: releasedWeights)

// 2. Train. Zero-DCE is zero-reference — no brightened target, only the consumer's own dark photos.
//    `wellExposedLevel` is the preferred brightness, which is what personalizing this model means.
var objective = NFKMLXZeroDCEObjective()
objective.wellExposedLevel = 0.65

let history = try NFKMLXZeroDCE.fineTune(net, photos: { step in myPhotos[step % myPhotos.count] },
                                         objective: objective, steps: 400,
                                         checkpoint: NFKMLXTrainingCheckpoint(url: tuned, everySteps: 50)) { step in
    progress(Double(step.index) / Double(step.count))
    return !cancelled                        // return false to end the run early
}

// 3. Save, then load through the same factory a converted checkpoint uses.
try NFKMLXWeights.save(net, to: tuned)
let backend = try NFKMLXZeroDCE.backend(weightsURL: tuned)
```

### Retargeting a segmentation model to your own classes

The other shipped recipe. A consumer rarely wants ADE20K's 150 classes and usually wants their own few,
which is a decode-head problem — freezing the encoder is what makes the run fit on a device:

```swift
// A different class count leaves the classifier freshly initialized and loads everything else. This
// is not optional: MLX adopts a checkpoint's shapes rather than validating them, so keeping the old
// classifier would silently restore the old class set.
let net = try NFKMLXSegFormer.network(weightsURL: releasedWeights, classCount: 3)

let sampler = NFKMLXBatchSampler(count: myFrames.count, seed: 7)
try NFKMLXSegFormer.fineTune(net, examples: { step in
    let index = sampler.indices(forStep: step)[0]
    return (image: try! NFKMLXTrainingData.tensor(myFrames[index]),
            labels: try! NFKMLXTrainingData.labels(myMasks[index], classCount: 3))
}, trainable: .decodeHead, steps: 300)

try NFKMLXWeights.save(net, to: tuned)
```

`NFKMLXTrainingData` converts an app's `CGImage`s into what the trainer takes: `tensor` for an image,
`matte` for an alpha target, and `labels` for a class-index map — which inverts the encoding the
segmentation backends emit, so a mask painted in the app and a mask the model outputs are the same
thing. `NFKMLXBatchSampler` draws reshuffled passes from a seed, because cycling a handful of examples
in a fixed order lets the optimizer chase the sequence rather than the data.

The optimizer and schedule default to NVlabs' configuration: the head at 6e-4, the `poly` decay, and a
1,500-step linear warm-up, which a 300-step run never leaves. Pass `learningRateSchedule: .constant` to
hold the rate instead.

### Retargeting a promptable segmenter to your own subject

SAM 2 already segments what a click points at; what a consumer usually wants changed is what it reads
as the subject in THEIR images. That is the prompt encoder and mask decoder, with the Hiera trunk and
the whole memory path frozen — about 5M parameters of the tiny release's 39M.

```swift
let net = try NFKMLXSAM2.network(weightsURL: releasedWeights, variant: .tiny, release: .sam21)

try NFKMLXSAM2.fineTune(net, examples: { step in
    let index = sampler.indices(forStep: step)[0]
    // The click is in pixels of the model's own 1024-square input.
    return (image: myFrames[index], points: [(myClicks[index].x, myClicks[index].y, 1)],
            target: myMasks[index])
}, trainable: .maskDecoder, steps: 300)

try NFKMLXWeights.save(net, to: tuned)
let backend = try NFKMLXSAM2.backend(variant: .tiny, release: .sam21, weightsURL: tuned)
```

The objective is the reference's own, and its shape matters: an annotated frame with nothing in it
still trains, because the mask terms are multiplied by whether the target holds an object and the
object-score term is not. A consumer's negative examples are worth collecting.

### Retargeting a text-prompted detector to your own instances

SAM 3 names what a consumer asks for in words. What changes between datasets is what counts as an
instance, which is a detector problem: the ViT and the text tower stay frozen, and because they are
separate modules their output is computed once per image and reused at every step.

```swift
let detector = try NFKMLXSAM3.network(weightsURL: releasedWeights,
                                      configuration: try NFKMLXSAM3.detectorConfiguration(fromHuggingFace: configURL))

// The frozen half runs once per image, not once per step.
let encoded = myImages.map { NFKMLXSAM3.encode(image: $0.plate, tokens: $0.ids, valid: $0.mask,
                                               using: frozenModel) }

try NFKMLXSAM3.fineTune(detector, examples: { step in
    let index = sampler.indices(forStep: step)[0]
    let cached = encoded[index]
    // Boxes are (cx, cy, w, h) in 0...1. An empty set is an image the prompt names nothing in,
    // which trains the presence head.
    return (cached.levels, cached.positions, cached.prompt, myImages[index].mask, myBoxes[index])
}, trainable: .detector, steps: 300)

try NFKMLXWeights.save(detector, to: tuned)
```

### Adapting retrieval to your own corpus

Qwen3-VL-Embedding and Qwen3-VL-Reranker are general retrievers, and a consumer's corpus has its own
vocabulary and its own notion of relevance. What changes is small: a linear adapter over the
embeddings, or the pair scorer. The 2B backbone stays frozen and produces its embeddings once, so
nothing about it enters the training graph.

```swift
let embedder = try NFKMLXQwen3VLEmbedder.embedder(directoryURL: releaseDirectory)

// The frozen half runs once per example, not once per step.
let queries = MLXArray(myPairs.flatMap { embedder.embedding(forText: $0.query).map(\.floatValue) })
    .reshaped([myPairs.count, embedder.embeddingDimensions])
let documents = MLXArray(myPairs.flatMap { embedder.embedding(forText: $0.document).map(\.floatValue) })
    .reshaped([myPairs.count, embedder.embeddingDimensions])

// The adapter starts as the identity, so training moves away from the released space rather than
// from a random one. Every other document in the batch is a negative.
let adapter = try embedder.makeAdapter()
try embedder.fineTune(adapter: adapter, queries: queries, documents: [documents], steps: 200)

try NFKMLXWeights.save(adapter, to: tuned)
try embedder.loadAdapter(from: tuned)        // every later embedding is the adapted one
```

The reranker retargets the same way, over the last-position hidden states of labeled pairs and a
binary objective:

```swift
let head = try reranker.makeHead()           // starts at the release's own scoring direction
try reranker.fineTune(head: head, hidden: myPairStates, labels: myLabels, steps: 200)
try NFKMLXWeights.save(head, to: tunedHead)
try reranker.loadHead(from: tunedHead)
```

Both objectives are the ones sentence-transformers trains these releases with:
`MultipleNegativesRankingLoss` at its defaults for the embedder, `BinaryCrossEntropyLoss` over the
raw pair logit for the reranker. A full fine-tune of the backbone is an offline run: it needs the
optimizer state of 2 billion parameters and a batch of negatives large enough for the contrastive
objective to mean anything.

The text embedders carry the same adapter. `NFKMLXQwen3Embedding` and `NFKMLXEmbeddingGemma` return an
`NFKMLXTextEmbeddingBackend`, which encodes a corpus once, trains an adapter over the cached vectors, and
installs it so every later embedding is the adapted one:

```swift
let embedder = try NFKMLXEmbeddingGemma.backend(directoryURL: releaseDirectory) as! NFKMLXTextEmbeddingBackend

let queries = try embedder.embeddings(for: myPairs.map(\.query))          // run once
let documents = try embedder.embeddings(for: myPairs.map(\.document))

let adapter = try embedder.makeAdapter()
try embedder.fineTune(adapter: adapter, queries: queries, documents: [documents], steps: 200)

try NFKMLXWeights.save(adapter, to: tuned)
try embedder.loadAdapter(from: tuned)        // Objective-C: loadAdapterFromURL:error:
```

### LoRA, for models with no small head to train

A transformer stack has nowhere cheap to fine-tune: adapting CLIP or Whisper to a domain means reaching
into the attention blocks, and doing that fully needs optimizer state proportional to the whole model.
LoRA adds a trainable rank-r detour to each targeted `Linear` and freezes everything else:

```swift
// Target the attention projections rather than every Linear — far cheaper, and usually enough.
try NFKMLXLoRA.apply(to: net, rank: 8, alpha: 16) { path, _ in
    path.hasSuffix("q") || path.hasSuffix("v")
}

try NFKMLXTrainer.train(net, optimizer: AdamW(learningRate: 1e-4), steps: 500,
                        batch: { step in (inputs[step], targets[step]) },
                        loss: myLoss)

// Fold the detours back into the base weights, then save one ordinary checkpoint.
try NFKMLXLoRA.merge(into: net)
try NFKMLXWeights.save(net, to: tuned)
```

`apply` returns how many layers it adapted, so a predicate that matched nothing is visible rather than
silent. An adapted model starts out producing exactly what it produced before, because the adapter's
second factor begins at zero. After `merge` there are no adapter keys in the file and no adapter format
to carry around: the model's own factory loads it.

### A custom image classifier from a handful of photos

The cheapest useful customization here. CLIP's embedding already separates most visual concepts; what a
consumer lacks is the mapping to *their* categories. Both towers stay frozen, so the embeddings are
computed **once** and the training loop runs over cached vectors — seconds, not minutes:

```swift
let clip = try NFKMLXCLIP.network(weightsURL: releasedWeights)
let cached = try NFKMLXCLIP.embeddings(for: myPhotos, using: clip)      // run once

let probe = NFKMLXCLIPProbe(embedDimensions: 512, classCount: 3)
try NFKMLXCLIP.trainProbe(probe, embeddings: cached, labels: myLabels, steps: 300)

let classifier = NFKMLXCLIP.probeBackend(net: clip, probe: probe, labels: ["cat", "dog", "neither"])
let result = try classifier.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: photo]))
result.classifications      // [NFKClassification], most confident first, confidences summing to 1
```

A probe is a separate small model, so `NFKMLXWeights.save(probe, to:)` writes a companion file rather
than modified CLIP weights. Contrast a contrastive fine-tune of CLIP itself, which needs large batches
for negatives and is not a device workload.

SigLIP 2 trains the same probe, `NFKMLXEmbeddingProbe` (`NFKMLXCLIPProbe` is its CLIP name), over its
attention-pooled image embedding. The model encodes the photos, and a saved probe reloads through the
model, which is also how an Objective-C app installs one:

```swift
let siglip = try NFKMLXSigLIP2.model(variant: .basePatch16At224, weightsURL: releasedWeights)
let cached = try siglip.imageEmbeddings(for: myPhotos)                  // run once

let probe = NFKMLXEmbeddingProbe(embedDimensions: siglip.embeddingDimensions, classCount: 3)
try NFKMLXEmbeddingProbe.train(probe, embeddings: cached, labels: myLabels, steps: 300)
try NFKMLXWeights.save(probe, to: savedProbe)

let classifier = try siglip.probeBackend(probeURL: savedProbe, labels: ["cat", "dog", "neither"])
```

### Adapting speech recognition to your own domain

Whisper has no small head to retrain, so this is the recipe LoRA exists for. Only the decoder's query and
value projections are adapted; the encoder's audio features transfer across domains and stay frozen:

```swift
let net = try NFKMLXWhisper.network(weightsURL: releasedWeights)
try NFKMLXWhisper.fineTune(net, examples: { step in
    (mel: NFKMLXWhisper.spectrogram(for: myClips[step].samples, sampleRate: 16000),
     tokens: myClips[step].tokenIds)          // the full target sequence, decode prompt included
}, rank: 8, steps: 500)

try NFKMLXLoRA.merge(into: net)
try NFKMLXWeights.save(net, to: tuned)
```

`spectrogram` pads or trims to the 30-second window Whisper is trained on. That is not a detail: it was
the single biggest accuracy factor when this model was brought to reference parity.

### Fine-tuning a speech denoiser on your own recordings

GTCRN is 48.2K parameters, so every weight trains on a device. Pair each noisy recording with the same
speech recorded clean, at 16 kHz. The objective is the reference's own `HybridLoss`: compressed spectral
errors plus the scale-invariant SNR of the resynthesized waveform.

```swift
let net = try NFKMLXGTCRNFactory.network(weightsURL: releasedWeights)
try NFKMLXGTCRNFactory.fineTune(net, examples: { step in
    (noisy: myPairs[step].noisy, clean: myPairs[step].clean)
}, steps: 500)

try NFKMLXWeights.save(net, to: tuned)
let denoiser = try NFKMLXGTCRNFactory.backend(weightsURL: tuned)   // Objective-C: backendWithWeightsURL:error:
```

### Teaching bandwidth extension your own audio

NU-Wave 2 restores the high band of a narrow-band recording. Fine-tuning it on wide-band audio of the
kind it will restore trains every weight on the reference's own objective: the clip is noised along
the diffusion schedule and the network learns to predict the noise, given the narrow-band copy.

```swift
let net = try NFKMLXNUWave2.network(weightsURL: releasedWeights)
try NFKMLXNUWave2.fineTune(net, examples: { step in
    // 48 kHz wide-band audio; the narrow-band copy is band-limited to the rate you will restore from.
    NFKMLXNUWave2.trainingPair(wideband: myClips[step], narrowbandRate: 16000)
}, steps: 500)

try NFKMLXWeights.save(net, to: tuned)
let extender = try NFKMLXNUWave2.backend(weightsURL: tuned)       // Objective-C: backendWithWeightsURL:error:
```

### Teaching music structure analysis your own annotations

All-In-One reads a track's four stems and marks its beats, downbeats, section boundaries, and section
functions. Fine-tuning trains every weight on annotated tracks with the authors' own objective and
optimizer; the annotation becomes frame targets the way their dataset builds them.

```swift
let net = try NFKMLXAllInOne.network(weightsURL: releasedWeights)
let examples = try myTracks.map { track in
    let spectrograms = net.spectrograms(stems: track.stems)            // bass, drums, other, vocals
    let targets = try NFKMLXAllInOneTargets(beatTimes: track.beats, downbeatTimes: track.downbeats,
                                            sectionBoundaries: track.boundaries,
                                            sectionLabels: track.labels,      // e.g. start, intro, verse, …, end
                                            frameCount: spectrograms.dim(2))
    return (spectrograms: spectrograms, targets: targets)
}
try NFKMLXAllInOne.fineTune(net, examples: { examples[$0 % examples.count] }, steps: 500)

try NFKMLXWeights.save(net, to: tuned)
let analyzer = try NFKMLXAllInOne.backend(weightsURL: tuned)      // Objective-C: backendWithWeightsURL:error:
```

### Teaching voice activity detection your own audio

The MarbleNet VAD trains every weight with the release's own recipe: masked per-frame cross-entropy,
SGD with momentum, NeMo's warm-up, hold, and polynomial decay, and SpecAugment, dither, and dropout
while it trains. Label each 20 ms frame from the spans that hold speech.

```swift
let net = try NFKMLXVAD.network(weightsURL: releasedWeights)
let examples = myClips.map { clip in                     // 16 kHz samples and the seconds that are speech
    (samples: clip.samples,
     labels: NFKMLXVAD.frameLabels(speech: clip.speechSpans,
                                   frameCount: NFKMLXVAD.frameCount(samples: clip.samples.count)))
}
try NFKMLXVAD.fineTune(net, examples: { examples[$0 % examples.count] }, steps: 1000)

try NFKMLXWeights.save(net, to: tuned)
let detector = try NFKMLXVAD.backend(weightsURL: tuned)            // Objective-C: backendWithWeightsURL:error:
```

### Teaching speech separation your own speakers

Conv-TasNet trains every weight on mixtures paired with each speaker's own signal, with asteroid's
recipe for the release: the permutation-invariant negative SI-SDR, Adam at 1e-3, and gradient clipping
at 5. The speakers can come back in either order; the objective scores the better assignment.

```swift
let net = try NFKMLXConvTasNet.network(weightsURL: releasedWeights)     // the geometry comes from the file
try NFKMLXConvTasNet.fineTune(net, examples: { step in
    (mixture: myMixtures[step].samples, sources: myMixtures[step].speakers)   // 16 kHz, one length
}, steps: 1000)

try NFKMLXWeights.save(net, to: tuned)
let separator = try NFKMLXConvTasNet.backend(weightsURL: tuned)   // Objective-C: backendWithWeightsURL:error:
```

### Retargeting YOLO to your own classes

ultralytics' recipe, whole: a release transfers everything shaped alike into a network built for your
class count (the class branches start fresh at the reference's bias priors), every weight trains under
`v8DetectionLoss` with AdamW sized to the class count, the warm-up and per-epoch linear schedule, and
clipping at 10, and a moving average of the weights is what the run leaves behind. Boxes are corners in
each image's own pixels; augmentation is yours.

```swift
let net = try NFKMLXYOLO.network(variant: .nano, classCount: 3, weightsURL: releasedWeights)
try NFKMLXYOLO.fineTune(net, examples: { step in
    (images: myBatches[step].images,        // [batch, 640, 640, 3] in 0…1
     targets: myBatches[step].boxes)        // [[NFKMLXYOLOBox]], one list per image
}, steps: 3000, stepsPerEpoch: myBatches.count)

try NFKMLXWeights.save(net, to: tuned)
let detector = try NFKMLXYOLO.backend(variant: .nano, weightsURL: tuned, labels: ["cat", "dog", "fox"])
```

The later generations take the same shape: `NFKMLXYOLOGenerations.network(release:classCount:weightsURL:)`
and `fineTune`, and the end-to-end releases (v10, YOLO26) train both branches under `E2ELoss`.

### Adapting a language model to your own text

Granite 4.0-H is a decoder with no small head to retrain, so LoRA is the recipe. Only the attention
query and value projections are adapted; the Mamba layers, the embeddings, and the feed-forward stay
frozen. The objective is causal language-model teacher forcing: each position predicts the next token.

```swift
let config = try NFKMLXGraniteHybrid.configuration(fromDirectory: releaseDirectory)
let net = try NFKMLXGraniteHybrid.network(weightsURL: releaseWeights, configuration: config)
try NFKMLXGraniteHybrid.fineTune(net, examples: { step in myTokenizedText[step % myTokenizedText.count] },
                                 rank: 8, steps: 500)

try NFKMLXLoRA.merge(into: net)
try NFKMLXWeights.save(net, to: tuned)                       // reloads through network(weightsURL:)
```

Each example is one token sequence (the consumer's own text as ids). After `merge` the file carries no
adapter keys, so `NFKMLXGraniteHybrid.network(weightsURL:configuration:)` reads it back.

Nemotron Nano 2 uses the identical recipe through `NFKMLXNemotronH` — `configuration(fromDirectory:)`,
`network(weightsURL:configuration:)`, `fineTune`, and `NFKMLXNemotronObjective` — with the same LoRA
target (the attention query and value projections, its Mamba layers and feed-forwards frozen). Swap the
type name and the same four calls apply.

### Adapting a decision model to your own decisions

Laya's own README says the base checkpoints are near chance on a new decision task and that the
capability comes from fine-tuning on that task's examples. An example is a state, the question, and the
right answer; the objective is the reference's strictly proper scoring rule (`rl_common.proper_reward`:
log plus half the spherical score, minus the ranked probability score for an ordinal question), so
honest probabilities are the only way to lower it. The head alone trains by default, which a device
holds comfortably; `.all` trains the encoder too.

```swift
let laya = try NFKMLXLaya.laya(directoryURL: releaseDirectory)
let department = NFKDecisionQuestion.choiceQuestion(withInstructions: "Which team?", options: ["billing", "technical"])
let examples = [
    NFKMLXLayaExample(state: "My invoice is wrong.", question: department, label: 0),
    NFKMLXLayaExample(state: "The app crashes on launch.", question: department, label: 1),
    NFKMLXLayaExample(state: "I need this today.", question: .noulQuestion(withInstructions: "Urgent?"), holds: true),
]
let history = try laya.fineTune(examples: examples, steps: 200, learningRate: 1e-4, trainable: .head)
try NFKMLXWeights.save(laya.net, to: tunedURL)

// The fine-tuned file reloads through the factory, from Swift or Objective-C.
let tuned = try NFKMLXLaya.laya(directoryURL: releaseDirectory, weightsURL: tunedURL)
```

```objc
NFKMLXLaya *tuned = [NFKMLXLaya layaWithDirectoryURL:releaseDirectory weightsURL:tunedURL error:&error];
```

### Adapting a decision model to your own conversations

The reference trains a conversation as its prefixes, so the model learns to judge the outcome early:
each prefix is the context plus the turns so far under `conversation`, cut from the left so the newest
turns survive, and a longer conversation is sampled evenly down to the release's `max_prefixes` (6).
Every prefix asks the same noul, and its target is a TD(λ) blend of the outcome and the model's own
prediction on the next prefix; λ = 1, the release's setting, trains every prefix toward the outcome.

```swift
let resolved = NFKDecisionQuestion.noulQuestion(withInstructions: "The problem will be resolved by the end of the conversation.")
let episodes = [
    NFKMLXLayaEpisode(context: ["account": 88213, "channel": "chat"],
                      turns: [["role": "customer", "text": "My payouts keep failing."],
                              ["role": "agent", "text": "Escalating to payments now."]],
                      question: resolved, holds: true),
]
let history = try laya.fineTune(episodes: episodes, steps: 200, learningRate: 1e-4, lambda: 1)
laya.prefixes(of: episodes[0]).map(\.length)      // [1, 2]: the prompts each prefix builds
```

The prefix builder and the TD targets are measured against the release's own `rl_common`: an
eight-turn episode samples to prefixes `[1, 2, 4, 5, 7, 8]`, each token for token, and the targets at
λ = 1 and λ = 0.5 match.

### Adapting the community Jev reproductions

open-jev-deberta trains its head, or the encoder too, on labeled states with the release's objective,
cross-entropy plus the Brier score. One state carries any number of questions, and each label indexes
the right option:

```swift
let model = try NFKMLXOpenJevDeBERTa.openJev(directoryURL: releaseDirectory)
let examples = [
    NFKMLXOpenJevDeBERTaExample(state: "Charged twice, refund me.", questions: [team, refund], labels: [0, 1]),
    NFKMLXOpenJevDeBERTaExample(state: "The app crashes.", question: team, label: 1),
]
try model.fineTune(examples: examples, steps: 200, trainable: .head, batchSize: 8)   // .all: the encoder too, at 3e-5
try NFKMLXWeights.save(model.net, to: tunedURL)
let tuned = try NFKMLXOpenJevDeBERTa.openJev(directoryURL: releaseDirectory, weightsURL: tunedURL)
```

Open-Jev trains its LoRA adapter and head at the release's rates, the base frozen, and saves in the
release's own `checkpoint` layout, so the same factory reloads it over the same base:

```swift
let release = try NFKMLXOpenJev.download(variant: .twoB, revision: nil, cacheDirectoryURL: nil)
let model = try NFKMLXOpenJev.openJev(checkpointDirectoryURL: release.checkpointDirectoryURL,
                                      baseDirectoryURL: release.baseDirectoryURL, precision: .float32)
try model.fineTune(examples: [NFKMLXOpenJevExample(state: record, question: intent, label: 0),
                              NFKMLXOpenJevExample(state: record, question: refund, holds: true)],
                   steps: 100, batchSize: 4)
try model.save(to: tunedDirectory)          // adapter/, head.safetensors, model.json, temperature.json
let tuned = try NFKMLXOpenJev.openJev(checkpointDirectoryURL: tunedDirectory, baseDirectoryURL: release.baseDirectoryURL)
```

Both objectives are measured against the reference training code: open-jev-deberta's `decision_loss`
on a padded batch at Brier weights 1 and 0.5, and Open-Jev's per-record loss on each of the 2B record's
questions. Both recipes use PyTorch's AdamW with the releases' two learning rates. open-jev-deberta
follows its release's schedule, a warm-up over the first 6% of the run then a linear decay to zero,
unless `learningRateSchedule:` names another; Open-Jev's release trains at a constant rate. The
temperature is not refitted by a fine-tune.

### Adapting a translator to your own sentence pairs

The three translators share one recipe: LoRA on the decoder's query and value projections, the encoder
frozen, teacher forcing as the objective. The release's tokenizers produce each pair's ids:

```swift
let net = try NFKMLXMarian.network(directoryURL: releaseDir)
let release = try NFKMLXMarian.translator(net: net, directoryURL: releaseDir)
try NFKMLXMarian.fineTune(net, examples: { step in
    let pair = myPairs[step % myPairs.count]
    return (source: MLXArray(release.sourceIds(for: pair.source, target: nil).map { Int32($0) }),
            target: MLXArray((release.targetTokenizer.encode(pair.target, dummyPrefix: nil) + [0]).map { Int32($0) }))
}, rank: 8, steps: 500)

try NFKMLXLoRA.merge(into: net)
try NFKMLXWeights.save(net, to: tunedDir.appendingPathComponent("model.safetensors"))
let tuned = try NFKMLXMarian.translator(net: try NFKMLXMarian.network(directoryURL: tunedDir), directoryURL: releaseDir)
```

`NFKMLXM2M100.fineTune` and `NFKMLXMADLAD.fineTune` take the same shape (M2M-100's target ids lead with
the `__xx__` marker; MADLAD adapts a float32 load). `NFKMLXTranslateGemma.fineTune` adapts the Gemma 3
decoder on (prompt ids, model-turn ids) pairs from `promptTokens(text:sourceCode:targetCode:)`, the prompt
positions masked in the loss.

### Adapting an image or video tokenizer to your own footage

NVIDIA post-trains the Cosmos Tokenizers to new domains, and the recipe fits a device: the network is
about 80M parameters and trains on crops. The objective is the reference's post-training one, a mean
absolute pixel error plus 0.1 times a VGG-16 perceptual term, so it needs the VGG-16 ImageNet weights
(`timm/vgg16.tv_in1k`, `model.safetensors`):

```swift
let net = try NFKMLXCosmosTokenizer.network(variant: .continuousVideo8x8x8, weightsURL: autoencoderJIT)
let objective = try NFKMLXCosmosTokenizerObjective(vggWeightsURL: vggWeights)

try NFKMLXCosmosTokenizer.fineTune(net, examples: { step in
    myClips[step % myClips.count]          // [1, 17, 256, 256, 3] in [-1, 1]
}, trainable: .decoder, objective: objective, steps: 1_000)

try NFKMLXWeights.save(net, to: tuned)
let backend = try NFKMLXCosmosTokenizer.backend(variant: .continuousVideo8x8x8, weightsURL: tuned)
```

`.decoder` keeps the encoder, and so every latent and token, exactly as released, which is what a world
model trained on the released latents needs; `.everything` trains the whole network as the reference's
post-training does. The optimizer defaults to the reference's AdamW (1e-4, betas 0.5 and 0.999, weight
decay 0.01) and its 5,000-step linear warm-up; pass `learningRateSchedule: .constant` to hold the rate.

### Teaching Sa2VA your own referring segmentation

Sa2VA's authors fine-tune it with LoRA on the language model while training the `[SEG]` bridge and SAM's
mask decoder, scoring the answer text and each mask together. An example is an image, a turn in the
release's template whose answer carries one `[SEG]` per object, the labels with the prompt masked, and
one mask per `[SEG]`:

```swift
let net = try NFKMLXSa2VA.network(directoryURL: sa2vaRelease)                  // float32
let tokenizer = NFKMLXSa2VA.tokenizer(inDirectory: sa2vaRelease)!
let examples = try myPhotos.map { photo, request, mask in                        // CGImage, String, [1, H, W]
    let tiles = try NFKMLXSa2VAProcessor.dynamicTiles(photo, side: net.configuration.imageSize)
    let prompt = NFKMLXSa2VAProcessor.promptText("<image>" + request, imageTokens: tiles.count * net.configuration.tokensPerTile,
                                                 template: net.configuration.template)
    let promptIds = tokenizer.encode(prompt).map(\.int32Value)
    let answerIds = tokenizer.encode("Sure, [SEG].<|im_end|>").map(\.int32Value)
    return NFKMLXSa2VAExample(pixelValues: NFKMLXSa2VAProcessor.tilePixels(tiles, side: net.configuration.imageSize),
                              inputIds: MLXArray(promptIds + answerIds),
                              labels: MLXArray([Int32](repeating: -100, count: promptIds.count) + answerIds),
                              groundingImage: try NFKMLXSa2VAProcessor.groundingPixels(photo).transposed(0, 2, 3, 1),
                              masks: mask)
}
try NFKMLXSa2VA.fineTune(net, examples: { examples[$0 % examples.count] }, steps: 2_000)

try NFKMLXLoRA.merge(into: net)
try NFKMLXSa2VA.save(net, toDirectoryURL: tuned, release: sa2vaRelease)
let backend = try NFKMLXSa2VA.backend(directoryURL: tuned)
```

The defaults are the authors' (LoRA rank 128, AdamW at 4e-5 with weight decay 0.05, a 5% warm-up then a
cosine to zero); a smaller `rank` fits a smaller device. A Qwen-VL release builds its example through
`NFKMLXSa2VAQwenNet.imageProcessor` and passes the patch grid as `grid`; the LLaVA release uses
`NFKMLXSa2VALLaVA.pixelValues`. Both fine-tune through their own `fineTune` overloads.

### Adapting Florence-2 to your own task

Florence-2 answers a task prompt about an image. LoRA on its text decoder teaches it a new answer
format or domain from a few hundred examples; the objective is the release's own training loss:

```swift
let net = try NFKMLXFlorence2.network(directoryURL: florenceRelease)
let tokenizer = NFKMLXFlorence2Processor.tokenizer(inDirectory: florenceRelease)!
let prompt = NFKMLXFlorence2Processor.encodePrompt(NFKMLXFlorence2Processor.expandPrompt("<CAPTION>"),
                                                   tokenizer: tokenizer, eosTokenId: 2).reshaped([-1])
let examples = try myPhotos.map { image, caption in
    (try NFKMLXFlorence2Processor.pixelValues(image), prompt,
     NFKMLXFlorence2Processor.encodePrompt(caption, tokenizer: tokenizer, eosTokenId: 2).reshaped([-1]))
}
try NFKMLXFlorence2.fineTune(net, examples: { examples[$0 % examples.count] }, steps: 1_000)

try NFKMLXLoRA.merge(into: net)
try NFKMLXFlorence2.save(net, toDirectoryURL: tuned, release: florenceRelease)
let backend = try NFKMLXFlorence2.backend(directoryURL: tuned)
```

The adapters sit on the decoder's query and value projections (rank 8); `rank: nil` trains the whole
language model with the image tower frozen. Microsoft publishes no fine-tuning script, so the optimizer is
the translators' default (AdamW at 1e-4, clip 1).

### Retargeting a table detector to your own document classes

Table Transformer is a DETR detector; its authors train it with DETR's set objective, which assigns each
labeled box its query before scoring the class and the box. A new class set replaces the class head and
keeps everything else:

```swift
let net = try NFKMLXTableTransformer.network(directoryURL: tatrRelease, labels: ["table", "figure", "chart"])

// Each page: its pixels and an [N, 5] array of [class, cx, cy, w, h] rows, the box normalized to 0...1.
let pages = try myPages.map { image, boxes in
    (try NFKMLXTableTransformerProcessor.pixelValues(image, sizing: .release(at: tatrRelease)), boxes)
}
try NFKMLXTableTransformer.fineTune(net, examples: { pages[$0 % pages.count] }, steps: 2_000,
                                    stepsPerEpoch: pages.count)

try NFKMLXTableTransformer.save(net, toDirectoryURL: tuned, release: tatrRelease)
let backend = try NFKMLXTableTransformer.backend(directoryURL: tuned)     // detects "table", "figure", "chart"
```

The default trains what the reference trains (the transformer, the heads, and the backbone's last three
stages) with its AdamW, its gradient clip at 0.1, and its 0.9-per-epoch decay; `trainable: .heads`
trains the class and box heads alone.

### Adapting a handwriting reader to your own writing

TrOCR's authors fine-tuned every release the same way: all the weights, teacher-forced on transcribed
lines. The recipe runs on the device, from any release directory:

```swift
let net = try NFKMLXTrOCR.network(directoryURL: trocrRelease)
let tokenizer = NFKMLXTrOCRProcessor.tokenizer(inDirectory: trocrRelease)!
let lines = try myLines.map { image, text in                      // CGImage, its transcription
    (try NFKMLXTrOCRProcessor.pixelValues(image), NFKMLXTrOCRProcessor.targetIds(for: text, tokenizer: tokenizer, endToken: 2))
}
try NFKMLXTrOCR.fineTune(net, examples: { lines[$0 % lines.count] }, steps: 3_000)

try NFKMLXTrOCR.save(net, toDirectoryURL: tuned, release: trocrRelease)
let backend = try NFKMLXTrOCR.backend(directoryURL: tuned)
```

The small releases resize with a bicubic filter: pass `bicubic: true` to `pixelValues` for them, as the
backend does when the release's `preprocessor_config.json` names it. The optimizer is the reference's
(Adam with decoupled weight decay 1e-4 at 2e-5, a 500-update warm-up, then an inverse-square-root
decay); `trainable: .decoder` freezes the image encoder for a lighter run.

### Training a video classifier on your own clips

V-JEPA 2's authors classify video by training an attentive probe on the frozen encoder: a few attention
layers and a linear classifier over the encoder's tokens. The recipe is the same on a device. Start from
an encoder release, or from a classification release to reuse its trained pooler:

```swift
let net = try NFKMLXVJEPA2.network(directoryURL: vjepa2Release, labels: ["pour", "stir", "whisk"])

let clips = try myClips.map { frames, label in                   // [CGImage], class index
    (try NFKMLXVJEPA2Processor.clip(frames: frames, configuration: net.configuration), label)
}
try NFKMLXVJEPA2.fineTune(net, examples: { clips[$0 % clips.count] }, steps: 2_000)

try NFKMLXVJEPA2.save(net, toDirectoryURL: tuned)
let backend = try NFKMLXVJEPA2.backend(directoryURL: tuned)     // ranks "pour", "stir", "whisk"
```

`.probe` trains the pooler and the classifier and `.classifier` the classifier alone; the encoder stays
frozen in both. The optimizer is the reference's AdamW at 5e-3 with weight decay 0.01 on a cosine to
zero. The reference sweeps twenty rate and decay pairs and keeps the best on validation; pass
`learningRate:` and `weightDecay:` to run another. The saved directory loads through the Objective-C
`+[NFKMLXVJEPA2 backendWithDirectoryURL:error:]` as well.

### Any model

The loop is model-agnostic. Supply the loss for a supervised model, and freeze what should not move:

```swift
net.freeze()                                 // frozen parameters cost no gradient and no optimizer
net.head.unfreeze()                          // state, which is what makes a large model trainable here

try NFKMLXTrainer.train(net, optimizer: AdamW(learningRate: 1e-4), steps: 200,
                        batch: { step in (inputs[step], targets[step]) },
                        loss: { model, input, target in (model(input) - target).abs().mean() },
                        clipGradientNorm: 1.0)
```

Notes:

- A run is multi-second. Call it off the render thread.
- `train` throws `NFKMLXError.trainingDiverged` if a step's loss stops being finite, before that step
  can overwrite a checkpoint with ruined weights.
- `train` throws `NFKMLXError.nothingToTrain` when every parameter is frozen, so a predicate that
  matched no layer reports itself rather than running a loss curve over an update that changes nothing.
- A frozen group stays in evaluation mode for the run, so a frozen `BatchNorm` backbone normalizes with
  the statistics it was released with and does not fold the training batches into them.
- Checkpoints record the model's parameters, not the optimizer's state: an `SGD` run resumes exactly,
  an `Adam` run rebuilds its moment estimates and shows a brief rise in loss.

## Dynamic backend discovery (optional engines)

`NFKDynamicBackend` (core) activates a heavier engine only when its classes are linked into the build,
with no build dependency on it. Link a companion and its capability lights up:

```swift
// Linking InferKitMLX ships NFKStableDiffusionProvider and NFKMLXWhisperProvider.
if NFKDynamicBackend.isCapabilityAvailable(NFKCapabilityStableDiffusion) {
    let sd = try NFKDynamicBackend.stableDiffusionBackend()          // "mlx-stable-diffusion"
}
let stt = try NFKDynamicBackend.backend(forCapability: NFKCapabilityTranscription)

// Linking InferKitFoundationModels ships NFKFoundationModelsProvider for text generation.
let llm = try NFKDynamicBackend.backend(forCapability: NFKCapabilityTextGeneration)
```

A consumer brings any engine by adding a class conforming to `NFKDynamicBackendProvider`
(`+makeInferenceBackend`) and either naming it the built-in default (e.g. `NFKControlNetProvider` for
`NFKCapabilityControlNet`) or registering it: `NFKDynamicBackend.register(providerClassName:forCapability:)`.

## Structured output and tools

Apple's model (through `NFKFoundationModelsBackend`) returns a JSON object or calls app-provided
tools through the same core keys a remote backend and the MLX language backend read, with no
compile-time `@Generable` type.

### Structured output

```swift
let request = NFKInferenceRequest(
    inputs: [NFKInputPrompt: "Invent a fictional character."],
    parameters: [NFKParameterJSONSchema: [
        "type": "object",
        "properties": ["name": ["type": "string", "description": "the character's full name"],
                       "age": ["type": "integer", "description": "the character's age in years"]],
        "required": ["name", "age"],
    ]])
let result = try backend.runInference(for: request)
let fields = result.structured    // ["name": "Elara Windrider", "age": 28]
let json = result.text            // the same as JSON
// NFKParameterChoices: ["yes", "no"] constrains the reply to exactly one of the strings instead.
```

### Tool calling

```swift
backend.tools = [
    NFKFoundationTool(
        name: "get_temperature",
        description: "Get the current temperature for a city.",
        parameters: ["type": "object",
                     "properties": ["city": ["type": "string", "description": "the city"]],
                     "required": ["city"]],
        handler: { arguments in
            let city = arguments["city"] as? String ?? ""
            return "It is 21°C in \(city)."                 // the model reads this and continues its reply
        })
]
// "How warm is it in Paris?" → the model calls get_temperature(city: "Paris").
// A request carrying NFKParameterTools offers its own declarations and takes handlers from
// backend.tools by name; a declared tool with no handler ends the turn with the call under
// NFKOutputToolCalls, and the next request answers it with a `tool` message.
```

### Sampling

```swift
let request = NFKInferenceRequest(
    inputs: [NFKInputPrompt: "Name one color."],
    parameters: [NFKParameterTopK: 40, NFKParameterSeed: 7, NFKParameterMaxTokens: 16])
// NFKParameterTopK / NFKParameterTopP / NFKParameterSeed choose Apple's sampling mode;
// NFKParameterTemperature: 0 is greedy decoding.
```

## Audio → text (transcription)

`NFKRemoteTranscriptionBackend` transcribes audio through an OpenAI-compatible audio-transcriptions
endpoint (a Whisper API, or a local server that speaks the same protocol). Audio goes under
`NFKInputAudio` as an `NFKAudioAsset` (its file is read) or `NSData` holding an encoded file; the
transcript returns under `NFKOutputText` and the parsed response under `NFKOutputStructured`.

```objc
NFKRemoteTranscriptionBackend *backend =
    [NFKRemoteTranscriptionBackend backendWithEndpointURL:[NSURL URLWithString:@"https://api.example.com/v1/audio/transcriptions"]];
backend.modelName = @"whisper-1";
backend.apiKey = apiKey;

NFKAudioAsset *clip = [NFKAudioAsset audioAssetWithFileURL:recordingURL];
NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: clip }
                                                          parameters:@{ @"language": @"en" }
                                                      outputModality:NFKModalityText];
NSError *error = nil;
NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];   // off the render thread
NSString *transcript = result.text;
```

Request parameters fold in as multipart form fields, so a caller sets `language`, `prompt`,
`response_format`, and `temperature` by name.

For **on-device** transcription, `NFKMLXWhisper` runs the Whisper encoder-decoder transformer in MLX
(audio → log-mel → encoder → greedy decoder). Register it, then read audio under `NFKInputAudio`:

```swift
NFKMLXWhisper.register()
let whisper = try NFKMLXModelRegistry.backend(named: NFKMLXWhisper.modelName, weightsURL: checkpointURL)
let text = try whisper.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: NFKAudioAsset.audioAsset(withFileURL: recordingURL)]))
    .text
```

`Tools/whisper-to-safetensors/convert.py` converts an OpenAI Whisper `.pt` (names already match the
module). The backend reads 16-bit PCM WAV; without a supplied `NFKTokenizer` it returns token ids, and
the mel filterbank / 16 kHz assumption are sweep items for exact parity.

### Segment times

`emitsTimestamps` asks the decoder for the spans as well as the words. The result then carries
`NSArray<NFKAudioSegment *>` under `NFKOutputSegments`, each span labelled with the text inside it,
beside the whole transcript under `NFKOutputText`.

```swift
let backend = try NFKMLXWhisper.backend(weightsURL: checkpointURL, tokenizer: tokenizer,
                                        timestamps: true)

let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: clip]))
for span in result.segments ?? [] {
    print("\(span.startSeconds)-\(span.endSeconds): \(span.label ?? "")")
}
```

This is a different decode rather than a different reading of one — the prompt drops
`<|notimestamps|>` and the timestamp range stays open — so the model may choose different words than
the plain path does. That is why it is off by default.

## Text → audio (speech)

`NFKMLXVoice` is the complete trained voice: the espnet FastSpeech2 conformer acoustic model, its
paired HiFi-GAN vocoder, and the release's own 78-symbol ARPAbet phoneme vocabulary. Both checkpoints
are at measured reference parity, and the end-to-end test has the package's own Whisper transcribe
the synthesized audio back.

```swift
let voice = try NFKMLXVoice.voice(acousticURL: acousticWeights,      // espnet/fastspeech2_conformer
                                  vocoderURL: vocoderWeights,        // its PAIRED HiFi-GAN
                                  vocabularyURL: vocabularyJSON)     // the release's vocab.json
let samples = voice.speak(phonemes: ["HH", "AH0", "L", "OW1"])       // 22050 Hz, -1...1

let backend = voice.makeSpeechBackend { text in myPhonemizer(text) } // text → NFKOutputAudio
```

The vocoder must be the paired release (`espnet/fastspeech2_conformer_with_hifigan`): the acoustic
model emits mels normalized by its training statistics, and a raw-log-mel vocoder — the universal
jik876 generator has the identical geometry — turns them into loud noise.

`NFKMLXSpeechBackend` runs a bring-your-own MLX text-to-speech model: supply a
`@Sendable (String, Int) -> MLXArray` closure returning a mono waveform in `-1...1`, generated at the
given sample rate. The backend reads the prompt (`NFKInputPrompt`, or the user content of
`NFKInputMessages`), runs the closure, writes a 16-bit PCM WAV file, and returns an `NFKAudioAsset`
under `NFKOutputAudio`. `NFKParameterSampleRate` on the request overrides the configured rate and
reaches the closure, so the pitch stays correct.

```swift
import InferKit
import InferKitMLX
import MLX

let speech = NFKMLXSpeechBackend(configuration: NFKMLXSpeechConfiguration(sampleRate: 24000)) { text, sampleRate in
    myTTSModel(text, sampleRate: sampleRate)               // -> [N] samples in -1...1
}
let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "Hello there."],
                                  parameters: [NFKParameterSampleRate: 16000])
let result = try speech.runInference(for: request)
let audio = result.output(forKey: NFKOutputAudio) as? NFKAudioAsset   // asset.fileURL is a playable WAV
```

InferKitMLX ships a reference synth registered under `"tone-speech"`
(`NFKMLXReferenceModels.registerToneSpeech()`) that turns each character into a tone, so an Objective-C
consumer builds and runs the text-to-audio path by name (the WAV writer is Foundation-only, unit-tested).

A full TTS voice needs a text→phoneme front-end. Two paths ship (both conform to `NFKMLXPhonemizer`):

```swift
// In-toolkit neural G2P — no external dependency, works on iOS and macOS:
let g2p = NFKMLXNeuralG2P(phonemeSymbols: symbols)      // load a checkpoint for real output
let phonemes = g2p.phonemes(for: "hello world")

// Or a system espeak-ng if installed (macOS; run Tools/espeak/install.sh first):
if let espeak = NFKMLXEspeakPhonemizer() {              // nil when not installed
    let phonemes = espeak.phonemes(for: "hello world")
}
```

espeak-ng is GPLv3, so InferKit does not bundle it — `Tools/espeak/install.sh` installs it onto your
system and the phonemizer uses it only when present.

`NFKMLXTTS` completes the voice: it chains a phonemizer + an acoustic model (`NFKMLXAcousticNet`,
FastSpeech2-style, phonemes → mel) + the HiFi-GAN vocoder (`NFKMLXHiFiGAN`, mel → waveform), and hands
back a speech backend that renders text to a WAV.

```swift
let g2p = NFKMLXNeuralG2P(phonemeSymbols: symbols)
let tts = NFKMLXTTS(phonemizer: g2p, symbols: symbols)
try tts.loadWeights(acousticURL: acousticURL, vocoderURL: vocoderURL)   // trained checkpoints
let speech = tts.makeSpeechBackend(sampleRate: 22050)
let audio = try speech.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: "Hello there."]))
    .output(forKey: NFKOutputAudio) as? NFKAudioAsset                   // a playable WAV
```

## Audio → notes and structure (music)

A music-transcription backend returns its notes as an `NFKMIDISequence` under `NFKOutputMIDI`, and a
music-structure backend returns labeled spans under `NFKOutputSegments`, beats under
`NFKOutputBeats`, and the tempo under `NFKOutputTempo`. All four are core value types, so a consumer
reads them without linking InferKitMLX.

```objc
NFKMIDINote *root = [NFKMIDINote noteWithPitch:60 startSeconds:0.0 endSeconds:0.5 velocity:100];
NFKMIDINote *third = [[NFKMIDINote alloc] initWithPitch:64
										   startSeconds:0.5
											 endSeconds:1.0
											   velocity:90
												program:4
											 percussion:NO
											  pitchBend:@[ @0.0, @0.25, @0.5 ]];
NFKMIDISequence *sequence = [NFKMIDISequence sequenceWithNotes:@[ third, root ] tempoBPM:96.0];

// The notes are ordered by time whatever order they arrive in.
sequence.notes.firstObject.pitch;            // 60
sequence.durationSeconds;                    // 1.0

// A Standard MIDI File a DAW opens: a conductor track with the tempo, one track per program,
// percussion on channel 10, and each note's pitch bend written across it.
NSData *midi = [sequence standardMIDIFileData];
[sequence writeToURL:url error:&error];
```

The structure result reads the same way. `positionInBar` counts from 1, so position 1 is a downbeat
and the highest position a track reaches is its bar length.

```objc
NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];   // a music-structure backend
for (NFKAudioSegment *section in result.segments) {
	NSLog(@"%@ %.2f–%.2fs", section.label, section.startSeconds, section.endSeconds);
}
result.beats.firstObject.isDownbeat;         // YES when the beat starts a bar
[[result outputForKey:NFKOutputTempo] doubleValue];
```

In Swift the importer renames the file writer to a method:

```swift
let sequence = NFKMIDISequence(notes: notes, tempoBPM: 96)
let data = sequence.standardMIDIFileData()
result.midi?.notes.count
result.segments?.first?.label                // "intro"
result.beats?.first?.isDownbeat              // true
```

The models behind these results are in InferKitMLX: `NFKMLXBasicPitch` transcribes any instrument to
notes, `NFKMLXHFTTransformer` transcribes piano more accurately, `NFKMLXMuScriptor` transcribes a
mixture to one MIDI track per instrument, and `NFKMLXAllInOne` divides a track into sections and
tracks its beats.

```swift
let transcriber = try NFKMLXBasicPitch.backend(weightsURL: basicPitchURL)
let midi = try transcriber.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: asset])).midi

// All-In-One reads the four HT Demucs stems, so it takes either an array of them or a mixture with a
// separator attached.
let analyzer = try NFKMLXAllInOne.backend(weightsURL: allInOneURL, demucsWeightsURL: demucsURL)
let structure = try analyzer.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: mixture]))
```

## Audio → stems (Demucs)

`NFKMLXDemucsBackend` separates a music mix into stems (drums, bass, other, vocals) — a time-domain
convolutional U-Net. Audio goes under `NFKInputAudio`; each stem returns as an `NFKAudioAsset` (a WAV)
under its name.

```swift
NFKMLXDemucs.register()
let demucs = try NFKMLXModelRegistry.backend(named: NFKMLXDemucs.modelName, weightsURL: checkpointURL)
let result = try demucs.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: NFKAudioAsset.audioAsset(withFileURL: songURL)]))
let vocals = result.output(forKey: "vocals") as? NFKAudioAsset      // also "drums", "bass", "other"
```

### Audio → stems (Demucs v4)

`NFKMLXHTDemucs` is the hybrid transformer release: a spectrogram branch and a waveform branch run in
parallel, a cross-transformer at the bottleneck lets each read the other, and the two reconstructions
are added. Same request and result shape as `NFKMLXDemucs`.

```objc
id<NFKInferenceBackend> htdemucs = [NFKMLXHTDemucs backendWithWeightsURL:checkpointURL error:&error];
NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{NFKInputAudio: song}];
NFKAudioAsset *vocals = [[htdemucs runInferenceForRequest:request error:&error] outputForKey:@"vocals"];
```

A clip shorter than the release's 7.8-second training segment is zero-padded up to it and the result
trimmed back, which is what the reference does at inference.

## Loading a PyTorch checkpoint directly

A consumer's own `.pth` (or `.pt`, `.ckpt`, `.th`, HF `.bin`) loads with no Python toolchain.
`NFKMLXWeights` sniffs a checkpoint's leading bytes, so every `weightsURL:` factory and registry
build accepts a raw PyTorch file wherever it accepts a converted safetensors — both the modern ZIP
container and the pre-1.6 stream, memory-mapped, with no pickle code ever executing.
`NFKMLXTorchCheckpoint` is the inspection and conversion API over the same reader:

```swift
let checkpoint = try NFKMLXTorchCheckpoint.checkpoint(contentsOf: pthURL)
print(checkpoint.tensorNames)                                // the flattened state dict
let info = checkpoint.info(forTensor: "conv_first.weight")   // shape + stored element type
try checkpoint.writeSafetensors(to: convertedURL)            // what the Tools converter produces
let arrays = try checkpoint.arrays()                         // Swift-only: [String: MLXArray]
```

```objc
NSError *error = nil;
NFKMLXTorchCheckpoint *checkpoint = [NFKMLXTorchCheckpoint checkpointWithContentsOfURL:pthURL error:&error];
NFKMLXTorchTensorInfo *info = [checkpoint infoForTensor:@"conv_first.weight"];
[checkpoint writeSafetensorsToURL:convertedURL error:&error];
```

Training wrappers unwrap as the offline converters do (`state_dict`, `params_ema`, `model`, …),
non-tensor sidecars drop, tensors stored as strided views (Whisper's transposed Linear weights)
gather to row-major, and float64 narrows to float32 on conversion. Every model's loader carries its
converter's renames and transforms itself (U²-Net's index rename, HiFi-GAN's weight-norm fusion,
the colorizer's Sequential table), so a raw release loads end to end wherever a converted one does.
Every checkpoint shape the shipped models use loads: a plain state dict, a pickled live
`nn.Module` tree (YOLO's ultralytics DetectionModel), a TorchScript archive (CLIP, walked through
its attribute-keyed scripted-module state), and a `.nemo` tar (unwrapped to the checkpoint inside).
No class is constructed and no serialized code is interpreted.

### Reading a GGUF model (`NFKMLXGGUF`)

GGUF is the format most quantized language models are distributed in. `NFKMLXGGUF` reads the container
and dequantizes the block-quant formats a real model uses — the k-quants `Q4_K`/`Q6_K` a `Q4_K_M` model
is built from, plus `Q8_0`, `Q5_0`, `Q4_0`, `F16`, `F32` — with no Python and no llama.cpp.

```objc
NFKMLXGGUF *gguf = [NFKMLXGGUF GGUFWithContentsOfURL:ggufURL error:&error];
NSString *architecture = [gguf metadataStringForKey:@"general.architecture"];
NFKMLXGGUFTensorInfo *info = [gguf infoForTensor:@"token_embd.weight"];   // .shape, .typeName (@"Q8_0")
```

```swift
let weight = try gguf.array(forTensor: "blk.0.ffn_down.weight")    // Swift-only: a dequantized MLXArray
let all = try gguf.arrays()                                        // every tensor, keyed by name
```

The dequantization is bit-exact against the `gguf` package on a real Q4_K_M model. A type the reader does
not implement is refused per-tensor (the tensor is still listed, reading it throws), not per-file. GGUF
stores the fastest-varying dimension first, so a tensor's row-major shape is the reverse of its stored
dimensions.

**Running a GGUF release end to end.** One `.gguf` file is the whole model — geometry, weights, and
tokenizer — so a single call builds a text-generation backend from it. The reader maps the metadata to a
configuration, remaps the llama.cpp tensor names onto the decoder's keys (un-permuting the rotary
query/key projections, which llama.cpp stores interleaved), dequantizes the weights, and rebuilds the
embedded byte-level BPE tokenizer:

```objc
// Objective-C — the dominant path for a quantized release.
id<NFKInferenceBackend> llm = [NFKMLXLanguage backendWithGGUFURL:ggufURL error:&error];
NFKInferenceRequest *request = [[NFKInferenceRequest alloc]
    initWithInputs:@{ NFKInputPrompt: @"The capital of France is" } parameters:@{}];
NFKInferenceResult *result = [llm runInferenceForRequest:request error:&error];   // off the render thread
```

```swift
let llm = try NFKMLXLanguage.backend(ggufURL: ggufURL)   // Swift
```

Only the dense `llama`/`qwen2`/`qwen3` families are read; another architecture throws. Loaded against
transformers reading the same GGUF, the decoder's logits match to cosine 0.9999999999.

**Running a Gemma 4 decoder.** The Gemma decoders (E2B/E4B, the 26B-A4B mixture, the 12B unified) have
their own backend, dispatched from the release's config model type. Gemma runs prefill-only (no
key-value cache), and its byte-fallback BPE tokenizer is read from the release:

```objc
// Objective-C.
id<NFKInferenceBackend> gemma = [NFKMLXGemmaLanguage gemmaBackendWithDirectoryURL:releaseDirectory error:&error];
```

```swift
let gemma = try NFKMLXGemmaLanguage.backend(directoryURL: releaseDirectory)   // Swift
// "The capital of France is" → " Paris." on the released E2B.
```

The 26B-A4B mixture takes a residency, and pages its routed experts as the language backend does:

```objc
id<NFKInferenceBackend> mixture = [NFKMLXGemmaLanguage gemmaBackendWithDirectoryURL:releaseDirectory
                                                                          residency:NFKMLXResidencyPaged
                                                                              error:&error];
```

## Choosing a backend at runtime

Because every engine adopts `NFKInferenceBackend`, a caller selects one at runtime and builds the
request the same way — the hub pattern.

```swift
func makeTextBackend() -> NFKInferenceBackend {
    let apple = NFKFoundationModelsBackend()
    if apple.isReady { return apple }                       // Apple's model when available
    if let local = try? loadedLocalBackend() { return local }   // else a converted local model
    let remote = NFKRemoteBackend()                         // else a remote endpoint
    remote.endpointURL = serverURL
    return remote
}
```

## Subsystems

### Jobs, progress, streaming, cancellation

`NFKInferenceJob` is the async handle: thread-safe, terminal states are final, and a `completionHandler`
set after the job finishes fires immediately. `NFKInferenceSubmit(backend, request, queue)` wraps a
synchronous backend into a job (default background queue when `queue` is `NULL`), so a caller gets a
job either way. A streaming backend reports partial text through `partialResult` (see above).

### Tokenizers (`NFKTokenizer`)

The tokenizer a converted model ships, built from its manifest. A class cluster: byte-level BPE
(`bpe-bytelevel`), its CLIP variant (`clip`, which CLIP and the Stable Diffusion text encoders take),
SentencePiece unigram (`unigram`), or WordPiece (`wordpiece`), per `tokenizer.type`.

```objc
NSData *data = [NSData dataWithContentsOfURL:[dir URLByAppendingPathComponent:@"manifest.json"]];
NSDictionary *manifest = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
NSError *error = nil;
NFKTokenizer *tokenizer = [NFKTokenizer tokenizerForManifest:manifest directory:dir error:&error];
NSArray<NSNumber *> *ids = [tokenizer encode:@"hello world"];
NSString *text = [tokenizer decode:ids];
```

`encode:` returns the ids for the text alone. A model input's start and end markers and its padding
belong to the model's geometry, so they are added where the context length is known — for the
diffusion path, by `NFKMLXSDPromptTokenizer`.

### Tensor conversion (`NFKTensorConversion`, `NFKMLMultiArray`)

RGBA-interleaved float images ↔ planar CHW / HWC tensors with per-channel normalization, and the
`MLMultiArray` bridge most Core ML vision models expect.

```objc
// Normalize an RGBA image (tightly packed floats, 0...1) into a planar [1, 3, H, W] tensor.
NFKTensorSpec spec = NFKTensorSpecMake(width, height, 3);       // defaults: CHW, RGBA order, mean 0 / scale 1
float *tensor = malloc(sizeof(float) * NFKTensorElementCount(spec));
NFKInterleavedToTensor(interleavedRGBA, tensor, spec);

MLMultiArray *array = NFKMultiArrayFromInterleaved(interleavedRGBA, spec, &error);  // [1, 3, H, W] float32
// … run a Core ML model, then read a model output back:
NFKInterleavedFromMultiArray(outputArray, interleavedRGBA, spec);
```

### Hugging Face hub (`NFKHFHub`)

Resolve, download, checksum, and cache public model files. The raw hub takes an explicit cache folder;
`+defaultCacheDirectoryURL` is a ready location under Application Support (`InferKit/models`). A
sandboxed host passes its own security-scoped URL instead. The download blocks, so run it off the main
thread — or use the async form.

```objc
NFKHFHub *hub = [NFKHFHub hubWithCacheDirectoryURL:NFKHFHub.defaultCacheDirectoryURL];
NSError *error = nil;
NSURL *localURL = [hub downloadRepo:@"Qwen/Qwen2.5-0.5B-Instruct"
                          revision:nil                          // defaults to main
                              path:@"tokenizer.json"
                            sha256:nil
                             error:&error];                     // blocking; call off the main thread

// Or asynchronously (background queue; Swift imports it as `try await hub.downloadRepo(...)`):
[hub downloadRepo:@"Qwen/Qwen2.5-0.5B-Instruct" revision:nil path:@"tokenizer.json" sha256:nil
completionHandler:^(NSURL *url, NSError *asyncError) { /* ready */ }];
```

The cache has a size limit and a backup setting. Each has a process-wide class default that every new
hub starts from, the hubs the companion factories create included, and a per-hub override. The limit
defaults to `NFKHFHubUnlimitedCacheSize` (-1). Over the limit, a download evicts whole
`<repo>/<revision>` snapshots, least recently used first, and keeps the one it just fetched. Only a
snapshot the hub owns is evicted: one it downloaded into, or one adopted from an older cache. A
pinned snapshot is never evicted. Backup
exclusion defaults to `YES`, because the cache can always be downloaded again: the first download
marks the folder with `NSURLIsExcludedFromBackupKey`, which is Time Machine's sticky exclusion on
macOS and the iCloud backup exclusion on iOS.

```objc
NFKHFHub.defaultCacheSizeLimit = 20LL * 1024 * 1024 * 1024;   // every new hub, companions' included
hub.cacheSizeLimit = NFKHFHubUnlimitedCacheSize;              // this hub only

[NFKHFHub setExcludedFromBackup:YES forURL:folder error:&error];   // any folder, managed or not
[hub pinCachedRepo:@"Qwen/Qwen2.5-0.5B-Instruct" revision:nil error:&error];    // never evicted
[hub adoptCachedRepo:@"org/older-model" revision:nil error:&error];  // cached before 0.4.0
[hub trimCacheToSizeLimitWithError:&error];                   // after lowering a limit
[hub removeCachedRepo:@"Qwen/Qwen2.5-0.5B-Instruct" revision:nil error:&error];
long long bytes = [hub cacheSize];
```

### Converting a model (`Tools/inferkit-convert`)

Convert a Hugging Face causal-LM checkpoint to a Core ML model directory for
`NFKCoreMLLanguageBackend`. See [the tool's README](../Tools/inferkit-convert/README.md).

```bash
cd Tools/inferkit-convert
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python convert.py --model Qwen/Qwen2.5-0.5B-Instruct --output ./out --quantize int8
```

## Testing without weights

`NFKPassthroughBackend` returns its inputs as outputs (optionally remapping keys), so an effect
renders its source unchanged when no model is present and tests stay green.

```objc
NFKPassthroughBackend *backend = [NFKPassthroughBackend backend];
backend.outputMap = @{ NFKOutputImage: NFKInputImage };        // each output key maps to an input key
NFKInferenceResult *result = [backend runInferenceForRequest:request error:NULL];
```

## Running notes

- Run inference off the main/render thread; cache results by frame.
- MLX evaluation needs its Metal library, which the Xcode build system bundles but a plain CLI
  `swift build`/`swift test` does not. Build and test the MLX companion with
  `xcodebuild test -scheme InferKitMLX -destination 'platform=macOS' -skipPackagePluginValidation`.
- `NFKCoreMLLanguageBackend` needs macOS 15 / iOS 18 (Core ML state); `NFKFoundationModelsBackend`
  needs macOS 26 / iOS 26 with Apple Intelligence.
