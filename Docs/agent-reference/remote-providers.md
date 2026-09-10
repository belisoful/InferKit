<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# Remote providers

`NFKRemoteProvider` carries the endpoint and protocol for the services a consumer is likely to call, so
pointing at one is a name rather than a hand-typed URL. Every endpoint was verified to exist when the
preset was added (a 401 or 405 without credentials is what confirms the path).

- **OpenAI-compatible** — one wire format, so `NFKRemoteBackend` serves them all: `openai`, `xai`
  (Grok), `gemini` (Google's OpenAI-compatible layer), `groq`, `mistral`, `deepseek`, `together`,
  `openrouter`, and the local servers `ollama`, `lmstudio`, `llamacpp`, `vllm`.
- **`anthropic`** is the exception and has its own backend, `NFKAnthropicBackend`. Four differences a
  URL swap cannot cover: the key is an `x-api-key` header rather than a Bearer token, an
  `anthropic-version` header is required, `max_tokens` is required rather than optional, and a system
  prompt is a top-level field rather than a message with a role. The reply is a list of typed blocks,
  so the text blocks are joined. A caller writes the same `NFKInputMessages` either way — a leading
  system turn is lifted into the top-level field.

No preset carries a default model name. Model identifiers change faster than a release does, and a
stale default fails at the first call with a message about the model rather than about the default.
The provider's own list is the source instead: `modelsWithAPIKey:error:` (and a completion-handler
form at user-initiated QoS) returns `NFKRemoteModel`s — identifier, display name, and where the
provider publishes them `ownedBy` / `createdAt` / `contextLength`, with the entry kept under `raw`.
Every preset answers the same `data[].id` envelope, hosted and local alike, so one parser serves all
thirteen; only the credential headers differ, and Anthropic paginates (`has_more` / `last_id` →
`after_id`, page size raised to 1000 from its default 20), which the catalog follows to the end. The
list is deliberately not filtered to chat models: the envelope carries no capability field, so any
filter would be a name heuristic that breaks on the next release. `NFKRemoteModelCatalog` is the
object under the convenience (timeout, session, `isReachableWithError:`, and the overridable
`sendRequest:` seam the tests stub). Readiness is not the same question:
`NFKAnthropicBackend` reports not-ready without a model because the API requires one, while an
OpenAI-compatible backend is ready with an endpoint alone, which **llama.cpp depends on** — its server
answers for whatever model it has loaded.

A preset carries one address, `baseURL`, and derives the rest (`endpointURL` = base +
`/chat/completions`, or `/messages` for Anthropic; `modelsURL` = base + `/models`; `URLForPath:` for
anything else, joining with exactly one slash). The bases differ per provider in ways a caller should
not have to know (Gemini's is `/v1beta/openai`, Groq's `/openai/v1`, OpenRouter's `/api/v1`), and the
derivation reproduces the literals the presets carried before, asserted by
`testEveryOperationURLIsDerivedFromTheBase`. `providerWithBaseURL:` re-points a preset — Ollama on
another port, or on a LAN machine — keeping its identity, protocol, and key requirement.

`NFKRemoteTransport` is the shared plumbing the chat, Anthropic, transcription, and catalog classes
used to each carry a copy of: the semaphore-blocked send, `authorizeRequest:apiKey:style:` (Bearer,
or `x-api-key` + `anthropic-version`), and `errorForResponse:data:` (a non-2xx status becomes
`kNFKError_InferenceBackendFailure` carrying the body, which is where a provider explains a rejected key
or an unknown model). Each class keeps its own `sendRequest:response:error:` override seam delegating
there, so a test stubs one class without touching the others. A failure that produced no response
at all is `kNFKError_RemoteUnreachable` with the URL-loading error under `NSUnderlyingErrorKey` — a
runner that is not running is a different answer from one that answered with an error or an empty
list, and an app shows "start Ollama" on that code. It is measured against a refused connection on the
discard port (`testTheTransportReportsARefusedConnectionAsUnreachable`), not stubbed.

`NFKRemoteEmbeddingBackend` is the embeddings counterpart (`POST /embeddings`, every preset but
Anthropic): `NFKInputPrompt` or joined `NFKInputMessages` → `NFKOutputEmbedding`, the key the MLX
embedders answer with, so a consumer's search code is engine-agnostic. `embeddingsForTexts:error:`
batches, ordered by the provider's `index` rather than by arrival (measured: a stub returning
index 1 before index 0 comes back in text order). `backendForProvider:apiKey:modelName:` answers nil
for Anthropic — it imports to Swift as the failable initializer `NFKRemoteEmbeddingBackend(for:apiKey:modelName:)`,
which the Swift example pins.

The local runners' native APIs are a second surface, and they are built (`NFKLocalModelRunner`,
`-[NFKRemoteProvider localRunner]` — a property, so Swift reads `.localRunner`; as a method it imported
as a function value and `runner?.x` failed to compile). The OpenAI endpoints say nothing about what is
installed, loaded, how large, or how to get one; the protocol's required set reads (`isRunning`,
`installedModelsWithError:`, `loadedModelsWithError:`, `detailsForModel:error:`) and its `@optional`
set changes the machine (`versionWithError:`, `pullModel:` → `NFKInferenceJob`, `deleteModel:error:`),
so an adapter adopts only what its runner has and a caller checks `respondsToSelector:` before
offering a button. `NFKOllamaRunner` adopts everything; `NFKLMStudioRunner` the reading set over
`/api/v0/models` (entries carry `state: loaded`, `type`, `quantization`, `max_context_length`; LM
Studio was not running here, so its shapes are its documented v0 rest API, stub-tested only);
llama.cpp and vLLM answer nil (nothing beyond the OpenAI surface to adapt — `/health` is what
`isReachableWithError:` already answers via `/v1/models`). The native base is the provider's base
minus `/v1` (`NFKLocalRunnerNativeBase`, private `NFKLocalRunnerSupport.h`), so a re-based preset
keeps its runner.
Measured against a live Ollama 0.33.2, read-only plus a pull/delete of a nonexistent name
(`testALiveOllamaAnswersItsNativeAPI`, gated on `INFERKIT_LIVE_LOCAL_MODEL`): `/api/tags` Already
carries `details.context_length`, `details.quantization_level`, and `capabilities` per model, so a
picker fills in one call with no `/api/show` per model; `/api/show` carries no id and keys the context
length by architecture inside `model_info` (`gptoss.context_length`), which the adapter lifts to where
`NFKRemoteModel` reads it; and a failing `/api/pull` answers HTTP 200 with `{"error":…}` as a line
Inside the NDJSON stream, so the pull job reads every line and treats an error line as failure and
`status: success` as completion — trusting the status code would report a failed pull as success. The
stream seam is `streamRequest:lineHandler:completionHandler:cancellation:` (overridable; the stub feeds
staged lines), distinct from `sendRequest:` because a streamed body cannot go through a blocking send.
`NFKRemoteModel` gained `sizeBytes` / `quantization` / `capabilities` and takes its id from `model` or
`name` where a list carries no `id` (Ollama's). A colon is legal in a URL path segment and is how
Ollama spells a tag (`llama3.2:latest`), but Foundation's `URLPathAllowedCharacterSet` encodes it to
`%3A`; `modelWithIdentifier:` and the LM Studio detail add `:` to the allowed set, pinned by a test.
**Two hazards from this round:** an `@[ a, b ]` literal inside an `XCTAssert…` macro argument splits
the macro on its comma unless the whole expression is parenthesized; and adjacent string-literal
concatenation as a direct element of an `@[ ]`/`@{ }` literal raises `-Wobjc-string-concatenation`
(parenthesize the element) — four of those had shipped in the catalog tests a round earlier because a
`tail` on the test log hid them. Grep the log for `warning:` without a tail.

The remaining modalities have remote backends, so every on-device direction has a hosted
counterpart on the same key (`NFKRemoteSpeechBackend` → `NFKOutputAudio` as an `NFKAudioAsset`, WAV
by default to match `NFKMLXSpeechBackend`; `NFKRemoteImageBackend` → 32BGRA `CVPixelBuffer` under
`NFKOutputImage`, choosing generations / edits / inpaint from `NFKInputImage` + `NFKInputMask` exactly
as `NFKMLXBackend` chooses; vision rides through the existing chat backends — `NFKRemoteBackend` attaches
`NFKInputImage` to the last user turn as an inline `image_url` content part, `NFKAnthropicBackend` as a
base64 `image` block before the text). Which presets serve which path was measured by probe, and the
probe needs a control: a `401` on a host that walls every path (DeepSeek answers 401 to
`/v1/nonesuch` too) proves nothing, so each host was also sent a nonsense path — served means the real
path answers 401/422/validation while the nonsense path 404s. Speech: openai, groq, together, xai,
mistral, openrouter. Generations: openai, together, xai, openrouter. Edits: openai, xai. Gemini's
OpenAI layer and all four local runners serve none; DeepSeek is undeterminable. Recorded on each
factory's `@discussion`. Vision is measured live (`testALocalVisionModelSeesTheImage`, gated on
`INFERKIT_LIVE_VISION_MODEL`): Ollama `qwen3.5:27b` given a flat blue square answers "blue" through the
content-parts shape, 23 s with the model load. `NFKImageCoding` is the public codec (ImageIO, now linked
by the core in `Package.swift` And the podspec): CGImage / 32BGRA-32RGBA CVPixelBuffer / BGRA8-RGBA8
MTLTexture → PNG or data URL; ImageIO-readable bytes → 32BGRA pixel buffer. Its private `CGImage…`
helpers must carry `CF_RETURNS_RETAINED` in a class extension — the public method promises +1 and the
analyzer reported five RetainCount issues when the helpers it delegates to did not. Importer names the
Swift examples pin: the class factories are failable initializers
(`NFKRemoteSpeechBackend(for:apiKey:modelName:voice:)`, `NFKRemoteImageBackend(for:apiKey:modelName:)`),
`NFKImageCoding.pngData(forImage:)` keeps `forImage:` (the parameter is not the type name), and a
`CF_RETURNS_RETAINED` CF return imports managed (no `takeRetainedValue()`). Two more ObjC traps from the
round: `inline` is a C keyword and not a variable name; and the `@{ a, b }`-inside-an-XCTAssert-macro
comma split struck twice more — hoist the request into a local before the macro, every time.

The remote chat path streams, cancels, calls tools, and returns structured output — the last
asymmetries with the on-device engines are closed. `submitInferenceJobForRequest:` on
`NFKRemoteBackend` / `NFKAnthropicBackend` sends `stream: true` and parses SSE through
`NFKRemoteTransport.streamRequest:session:lineHandler:completionHandler:` (the line-delimited primitive
the Ollama pull now shares; a failing status's body is collected whole and handed to the completion,
since a provider explains a rejected request in JSON, not a stream) and `SSEDataForLine:`. Each backend
keeps an overridable `streamRequest:lineHandler:completionHandler:` seam returning the cancel block,
which becomes `job.cancellationHandler` — the generic `NFKInferenceSubmit` wrapper never wired
cancellation, so a cancelled remote job used to run to the end on the server; now it reaches the
streamed form (`respondsToSelector:`) and the task is cancelled. OpenAI deltas: `choices[0].delta.content`
appends; `delta.tool_calls[]` assemble by index (id/name in the first delta, `function.arguments`
fragments after); `data: [DONE]` finishes; a stream closing without `[DONE]` still delivers what it
delivered. Anthropic events: `content_block_start` opens a block by index, `text_delta` /
`input_json_delta` append, `message_stop` finishes, an `error` event fails. **Tools:**
`NFKParameterTools` (`{name, description, parameters}`) → OpenAI `{type: function, function}` /
Anthropic `{name, description, input_schema}`; replies → `NFKOutputToolCalls` = `{id, name, arguments
(parsed), argumentsJSON}` (`result.toolCalls`). The key is spelled `"tools"`, the wire field's own
name, so an entry already in wire shape (has `type`, or `input_schema`) passes through unwrapped —
otherwise a caller who folded OpenAI tools by name gets double-wrapped. **Schema:**
`NFKParameterJSONSchema` → `response_format: {type: json_schema, json_schema: {name: response, schema}}`
(no `strict`, which demands `additionalProperties: false` throughout) / Anthropic has no response
format, so it is a forced tool `structured_output` (`tool_choice: {type: tool, name}`) whose `input` is
`NFKOutputStructured` and is not listed as a tool call. JSON is promoted to `structured` only when JSON
was asked for (schema, or a folded `response_format` of type `json_object`/`json_schema`) — JSON-looking
text is not guessed at. `NFKInputImages` attaches further images after `NFKInputImage`. **Retry:** the
transport's blocking send retries 429/502/503/504 after `Retry-After` (seconds; an HTTP-date falls to
the schedule) or `0.5·2^attempt`, `retryAttempts` (2) more times, never past `maximumRetryDelay` (8 s)
— a longer Retry-After ends the retries rather than waiting; a refused connection is not retried
(unreachable is an answer). Tested through an `NSURLProtocol` registered on a session configuration,
which drives a real `NSURLSession` with no network — the retry schedule, the ragged-chunk line splitter
(a CR is dropped, an unterminated last line arrives), and the whole-body collection on a 401.
Measured live on Ollama `qwen3.5:27b`: streaming delivers the reply in more than one partial with
the last partial equal to the final text (`testALocalRunnerStreamsTokenByToken`,
`INFERKIT_LIVE_LOCAL_MODEL`), and a declared `get_weather` tool is called with `{"city": "Paris"}`
(`testALocalModelCallsTheTool`, `INFERKIT_LIVE_TOOL_MODEL`). Swift importer: `submitInferenceJob(for:)`,
`result.toolCalls`, `NFKRemoteTransport.retryAttempts` as a class property. The two stubs' stream seams
deliver lines synchronously, so a test that wants to read a partial holds the stream open
(`holdOpen`) and asserts `.running`; a `[DONE]` inside the staged lines finishes the job before the
call returns.

The chat backends take audio, documents, and video in, and speak out; three more services close
the surface. `NFKRemoteAttachments` (private, `NFKRemoteMediaSupport.h`) gathers a request's media
Once — images + `NFKInputImages` + the frames sampled from `NFKInputVideo` as PNG, `NFKInputAudio` as
bytes + a format from the file extension, `NFKInputDocument(s)` as `{data, filename}` — and each backend
writes its own wire shape: OpenAI `image_url` / `input_audio` / `file` parts, Anthropic `image` /
`document` blocks (the Messages API takes no audio, so `NFKAnthropicBackend` Refuses `NFKInputAudio`
and `NFKParameterAudioOutput` with `kNFKError_InferenceUnsupported` rather than dropping them).
`NFKParameterAudioOutput` → `modalities: [text, audio]` + `audio: {voice, format}` (format defaults to
wav, the container `NFKMLXSpeechBackend` writes); the reply's `message.audio.data` (base64) is written
through `NFKRemoteWriteMediaFile` → `NFKOutputAudio`, and `message.audio.transcript` stands in for the
null `content`; streamed `delta.audio.data` chunks are concatenated AS BASE64 then decoded once.
`NFKVideoSampling` (public; core now links AVFoundation — Package.swift and the podspec) samples
`count` frames at `(i + 0.5)/count` of the duration **plus one millisecond**, because a clip whose
frames divide the count evenly lands every midpoint on a frame edge, with a half-frame seek tolerance
from the track's `nominalFrameRate` (loaded through `loadValuesAsynchronouslyForKeys:@[@"tracks"]` —
`loadTracksWithMediaType:` is macOS 12 / iOS 15, above the core's floor, and trips
`-Wunguarded-availability-new`). **Measured on this machine:** sampling a clip
right after Ollama's 27B model had occupied the GPU fails with `AVFoundationErrorDomain -11821`
"Cannot Decode", underlying OSStatus **-12911 `kVTVideoDecoderMalfunctionErr`**, after a ~4-minute
timeout — a broken hardware decode session — and the next session works; the sampler recreates its
generator once on `AVErrorDecodeFailed`, which turns that run from a failure into a slow pass
(`testFewerSamplesAreSpacedEvenlyThroughTheClip` at 245 s in the live ordering, 0.1 s otherwise). Two
boundary theories preceded that finding and were wrong; the sampler's own error, once the test
printed it, was what settled it. The test clip writer (`NFKTestClip`, AVAssetWriter, 64×64 H.264 at
2 fps) must hold every pixel buffer until `finishWriting` — the pool hands a released buffer straight
back and the encoder may still be reading it — and H.264 chroma subsampling bleeds ~0.27 into a pure
primary's other channels, so colour assertions carry a 0.35 tolerance. Video → text is measured
live: `qwen3.5:27b` names red and blue from four frames of a red–green–blue–white clip
(`testALocalVisionModelDescribesASampledClip`, `INFERKIT_LIVE_VISION_MODEL`).
`NFKRemoteTranscriptionBackend` gained `emitsTimestamps` (`response_format=verbose_json` unless the
caller set one; `segments[]` → `NFKAudioSegment` with `exp(avg_logprob)` as confidence, matching the
on-device Whisper backend's `NFKOutputSegments`) and `translates` (the path's last component swapped
to `translations`). `NFKRemoteVideoBackend` is the first shipped `NFKAsyncGenerationBackend`
subclass (OpenAI `/v1/videos`, verified by probe; no other preset serves one): JSON submit, or
multipart with `input_reference` when `NFKInputImage` is present — the base gained the
`submitRequestForRequest:` hook for that and `failureReasonFromStatusResponse:` so the service's
`error.message` reaches the job — percentage `progress` → fraction, poll every 5 s, then get
`/videos/{id}/content` → `.mp4` `NFKVideoAsset`. `NFKRemoteReranker` (`/rerank`, together +
openrouter verified; results arrive in relevance order and are put back in the documents' order) and
`NFKRemoteModerationBackend` (`/moderations`, openai + mistral; `category_scores` →
`NFKClassification`s most confident first, the verdict under `NFKOutputStructured`). Unverified live:
audio in/out, PDFs, video generation, rerank, moderation — all need paid keys; their envelopes are
stub-tested and their paths probe-verified.

**Deliberately absent.** Midjourney has no official public API (its API host does not resolve), so
shipping a preset would imply one exists. `opencode.ai` answers `Not Found` on its API path — it is a
coding agent that calls other providers rather than an inference service. Codex is OpenAI's coding
agent, not a separate endpoint; it is the `openai` preset.

`NFKRemoteProviderTests` stubs the transport to assert Anthropic's request shape without a network, and
carries one live test gated on `INFERKIT_LIVE_LOCAL_MODEL` that runs against a local server when one is
listening.
