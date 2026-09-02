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
| text | structured | `NFKFoundationModelsBackend` | [Structured output](#structured-output-and-tools) |
| text | image | `NFKMLXBackend` (text-to-image) | [Text → image](#text--image-and-image--image) |
| text + image | image | `NFKMLXBackend` (image-to-image) | [Image → image](#text--image-and-image--image) |
| image | image | `NFKMLXModuleBackend`, `NFKMLXRealESRGAN` (upscale), `NFKMLXDepthAnything` (depth), `NFKMLXNAFNet` (restore), `NFKCoreMLBackend` | [Image → image](#image--image) |
| image (+ hint) | image + mask | `NFKMLXMattingBackend`, `NFKMLXU2Net` (bg removal), `NFKMLXSAM` (segment) | [Image → image + mask](#image--image--mask) |
| image + mask | image | `NFKMLXLaMa` (inpaint) | [LaMa inpainting](#lama-inpainting-mtimlxlama-a-shipped-mlx-model) |
| image(s) | image(s) | `NFKMLXTensorBackend`, `NFKMLXRIFE` (frame interpolation), `NFKMLXRAFT` (optical flow) | [Many tensors](#many-tensors-in-and-out) |
| image (+ mask) | image | `NFKMLXDiffusionBackend` (upscale, depth, inpaint) | [Diffusion](#diffusion-upscale-depth-inpaint) |
| audio | text | `NFKRemoteTranscriptionBackend` (remote), `NFKMLXWhisper` (local) | [Audio → text](#audio--text-transcription) |
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

### Local, on device through MLX (`NFKMLXLanguageBackend`, Swift)

The dense decoder — Qwen3 and Llama are the same structure at different settings — reading a released
Hugging Face directory. The module's keys are the checkpoint's, so nothing is remapped.

```swift
let backend = try NFKMLXLanguage.backend(directoryURL: releaseDirectory)
let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "Explain diffraction in one sentence."])
let text = try backend.runInference(for: request).text
```

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
switch on. `linear` and `yarn` are implemented and measured against `transformers`' own
initializers. A kind that is not implemented throws rather than loading:

```swift
// Throws NFKMLXError.unsupportedConfiguration for `dynamic`, `llama3`, or `longrope`.
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

**A mixture of experts.** Qwen3-MoE and Mixtral are this decoder with a routed feed-forward, and the
same factory reads them: the config names the family, the loader stacks the released per-expert
tensors into one `[experts, out, in]` tensor per projection, and a step runs one gathered matrix
multiplication over the experts each token chose. Both families match `transformers`' arithmetic
layer by layer at a tiny configuration, and every one of Qwen3-30B-A3B's 18,867 released tensors is
accounted for by shape. Quantizing packs the stacked experts too. The released sizes need quantized
weights to fit a 32 GB machine.

```swift
let configuration = try NFKMLXLanguage.configuration(fromHuggingFace: configURL)   // Qwen3MoeForCausalLM
configuration.isMixtureOfExperts        // true; expertCount 128, activeExpertCount 8
let backend = try NFKMLXLanguage.backend(directoryURL: releaseDirectory)
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

// Or build one against the release's own token bytes:
let vocabulary = NFKMLXVocabulary(tokenizer: tokenizer, size: configuration.vocabularySize)
options.constraint = NFKMLXJSONConstraint(vocabulary: vocabulary, root: .object)
```

Two things the grammar cannot do for the model. JSON admits unbounded whitespace, and a model whose
preferred next token is forbidden takes the whitespace it is offered indefinitely, so the grammar
caps a whitespace run (eight bytes by default). And a thinking model opens with a `<think>` block the
grammar forbids; the prompt closes that block itself (`<think>\n\n</think>\n\n` after the assistant
marker, as Qwen3's own template does for its no-think mode) so the answer starts at the document.

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

### Streaming and cancellation

```swift
let job = backend.submitInferenceJob(for: request)
job.progressHandler = { j in
    if let text = j.partialResult?.output(forKey: NFKOutputText) as? String { render(text) }
}
job.completionHandler = { j in /* j.result or j.error */ }
// job.cancel()
```

## Text → image and image → image

`NFKMLXBackend` runs a bundled Stable Diffusion release (Swift; Apple Silicon, macOS and iOS). No
image input runs text-to-image; a `CGImage` under `NFKInputImage` runs image-to-image. Three releases
are bundled: `.stableDiffusion15`, `.stableDiffusion21Base`, and `.sdxlTurbo`. The release's files
download from Hugging Face on first use; Stable Diffusion 2.1 base is a gated repository, so it needs
an access token (`NFKHFHub.accessToken`, or `HF_TOKEN` in the environment).

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
covers the remaining block/neck key remap. SAM 2's Hiera encoder / video memory are future variants.

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

```objc
id<NFKInferenceBackend> inpainter = [NFKMLXStableDiffusionInpaint backendWithUNetWeightsURL:unetURL
                                                                             vaeWeightsURL:vaeURL
                                                                            textContextURL:promptURL
                                                                                     error:&error];
NFKInferenceRequest *request = [[NFKInferenceRequest alloc] initWithInputs:@{NFKInputImage: plate,
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
an access token: set `NFKHFHub.accessToken`, or leave it nil and let `HF_TOKEN` in the environment
supply one.

### MLX runtime knobs from Objective-C

MLX's global seed, GPU memory management, and device selection ship as Swift-only free functions,
enums, and a struct; `NFKMLXRandom` / `NFKMLXGPU` / `NFKMLXDevice` wrap them for Objective-C. Seed for
reproducible weight init and sampling; cap the GPU cache to bound memory in a plugin or app; select the
CPU where a graphics device is contended.

```objc
[NFKMLXRandom seed:42];                       // reproducible init/sampling

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

[NFKMLXDevice performOnDeviceType:NFKMLXDeviceTypeCPU block:^{
    result = [backend runInferenceForRequest:request error:&error];
}];
```

The device selection covers work the block does **on the calling thread**. It does not reach another
thread, so it wraps a synchronous `runInferenceForRequest:` and not `submitInferenceJobForRequest:`,
whose queue takes the global device. Run the synchronous call inside the block from your own background
thread. Selecting the CPU does not avoid shipping the Metal library: MLX builds the Metal device when it
initializes, whichever device the work names.

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

**No preset carries a default model name** — identifiers change faster than releases do. Each
provider's `modelsURL` points at its list. Midjourney has no official public API, so there is no preset
for it; `opencode.ai` is a coding agent rather than an inference service; and Codex is OpenAI's agent
using the OpenAI API, so it is the `openai` preset.

## Model gallery

Every shipped MLX model is built the same way: a direct `@objc` factory (`+backendWith…weightsURL:` for
local weights, `nil` → random weights that run; `+backendWith…repo:weightsPath:…` to download and build).
The compiled `MLXModelGalleryExamples` builds and runs each of these. Grouped by task:

```swift
// Upscaling & restoration (image → image)
let upscaler   = try NFKMLXRealESRGAN.backend(variant: .x4, weightsURL: nil)   // "real-esrgan-x4"
let swinIR     = try NFKMLXSwinIR.backend(weightsURL: nil)                      // "swinir-x4"
let denoiser   = try NFKMLXNAFNet.backend(weightsURL: nil)                      // "nafnet"
let lowLight   = try NFKMLXZeroDCE.backend(weightsURL: nil)                     // "zero-dce"
let stylizer   = try NFKMLXStyleTransfer.backend(weightsURL: nil)              // "fast-style-transfer"
let colorizer  = try NFKMLXColorizer.backend(weightsURL: nil)                  // "colorizer-eccv16"
let faceRestore = try NFKMLXCodeFormer.backend(weightsURL: nil)                // "codeformer"

// Depth (image → grayscale depth)
let depth = try NFKMLXDepthAnything.backend(variant: .small, weightsURL: nil)  // "depth-anything-v2-small"

// Matting (plate → foreground image + alpha under NFKOutputMask)
let cutout   = try NFKMLXU2Net.backend(variant: .full, weightsURL: nil)        // "u2net"
let videoKey = try NFKMLXRVM.backend(weightsURL: nil)                          // "robust-video-matting"
let portrait = try NFKMLXMODNet.backend(weightsURL: nil)                       // "modnet"

// Semantic segmentation (image → grayscale label map; index = round(gray·(classCount−1)))
let segformer = try NFKMLXSegFormer.backend(weightsURL: nil)                   // "segformer-b0"
let deeplab   = try NFKMLXDeepLab.backend(weightsURL: nil)                     // "deeplabv3"
let bisenet   = try NFKMLXBiSeNet.backend(weightsURL: nil)                     // "bisenet"

// Detection & pose (new core value types)
let yolo = try NFKMLXYOLO.backend(weightsURL: nil, labels: cocoLabels)         // result.detections : [NFKDetection]
let pose = try NFKMLXPose.backend(weightsURL: nil, jointNames: cocoJoints)     // result.pose : [NFKKeypoint]

// Embeddings, video, promptable segmentation
let clip    = try NFKMLXCLIP.backend(weightsURL: nil)                          // result.embedding : [NSNumber]
let videoSR = try NFKMLXVideoSR.backend(weightsURL: nil)                       // "video-super-resolution"
let sam     = try NFKMLXSAM.backend(weightsURL: nil)                           // plate + point under NFKSAMPointKey

// Audio
let transcriber = try NFKMLXWhisper.backend(weightsURL: nil)                   // audio → NFKOutputText
let stems       = try NFKMLXDemucs.backend(weightsURL: nil)                    // audio → "drums"/"bass"/"other"/"vocals"
let speakers    = try NFKMLXConvTasNet.backend(weightsURL: nil)               // audio → "speaker-1"/"speaker-2"
let clean       = try NFKMLXDenoiser.backend(weightsURL: nil)                  // audio → NFKOutputAudio
let vad         = try NFKMLXVAD.backend(weightsURL: nil)                       // result.segments : [NFKAudioSegment]
let tagger      = try NFKMLXAudioTagger.backend(weightsURL: nil, labels: nil)  // result.classifications : [NFKClassification]

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

### LoRA, for models with no small head to train

A transformer stack has nowhere cheap to fine-tune: adapting CLIP or Whisper to a domain means reaching
into the attention blocks, and doing that fully needs optimizer state proportional to the whole model.
LoRA adds a trainable rank-r detour to each targeted `Linear` and freezes everything else:

```swift
// Target the attention projections rather than every Linear — far cheaper, and usually enough.
NFKMLXLoRA.apply(to: net, rank: 8, alpha: 16) { path, _ in
    path.hasSuffix("q") || path.hasSuffix("v")
}

try NFKMLXTrainer.train(net, optimizer: AdamW(learningRate: 1e-4), steps: 500,
                        batch: { step in (inputs[step], targets[step]) },
                        loss: myLoss)

// Fold the detours back into the base weights, then save one ordinary checkpoint.
NFKMLXLoRA.merge(into: net)
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

Apple's model (through `NFKFoundationModelsBackend`) can return typed fields or call app-provided
tools, both defined at runtime with no compile-time `@Generable` type.

### Structured output

```swift
backend.responseSchema = [
    NFKFoundationToolParameter(name: "name", description: "the character's full name", type: .string, required: true),
    NFKFoundationToolParameter(name: "age", description: "the character's age in years", type: .integer, required: true),
]
let result = try backend.runInference(for: request)
let fields = result.structured    // ["name": "Aria Thompson", "age": 27]
let json = result.text            // the same as JSON
```

### Tool calling

```swift
backend.tools = [
    NFKFoundationTool(
        name: "get_temperature",
        description: "Get the current temperature for a city.",
        parameters: [NFKFoundationToolParameter(name: "city", description: "the city", type: .string, required: true)],
        handler: { arguments in
            let city = arguments["city"] as? String ?? ""
            return "It is 21°C in \(city)."                 // the model reads this and continues its reply
        })
]
// "How warm is it in Paris?" → the model calls get_temperature(city: "Paris").
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
NFKInferenceRequest *request = [[NFKInferenceRequest alloc] initWithInputs:@{NFKInputAudio: song}];
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
