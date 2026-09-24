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
- **`typesafe`** (TypeSafe AI's System One API, which serves Jev) has its own backend,
  `NFKTypeSafeBackend`, because it does not generate text: see "Typed decisions" below.
- **`anthropic`** is the other exception and has its own backend, `NFKAnthropicBackend`. Four differences a
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
fourteen; only the credential headers differ, and Anthropic paginates (`has_more` / `last_id` →
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
or `x-api-key` + `anthropic-version`), and `errorForResponse:data:`, which turns a non-2xx status into
the code an app acts on: 429, 402, and 529 are `kNFKError_InferenceRateLimited` (back off; the
provider's `Retry-After`, seconds or an HTTP-date, rides as a date under `NFKRemoteErrorRetryAfterKey`),
400, 413, and 422 are `kNFKError_InferenceRefused` (the request is the problem, which is also where an
OpenAI-compatible content-policy rejection arrives), and every other status, a rejected key, an
unknown model, a missing path, a server fault, is `kNFKError_InferenceBackendFailure`; the status and
the body ride under `NFKRemoteErrorStatusCodeKey` / `NFKRemoteErrorBodyKey` beside the description,
which is where a provider explains itself. The two chat backends also read the model's own refusal
from a successful reply: `finish_reason: content_filter` or a `message.refusal` string on the OpenAI
shape (streamed too, from the chunk's `finish_reason` and `delta.refusal`), and `stop_reason: refusal`
on the Messages API (streamed as `message_delta`'s `stop_reason`), each `kNFKError_InferenceRefused`;
an Anthropic stream `error` event maps its `type` (`overloaded_error` / `rate_limit_error` → rate
limited, `invalid_request_error` → refused). `NFKAsyncGenerationBackend`'s own transport goes through
the same mapping, so a 429 on submit fails the job under its code instead of being parsed for a job
id. Each class keeps its own `sendRequest:response:error:` override seam delegating there, so a test
stubs one class without touching the others. A failure that produced no response
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

**Discovery of the local runners** removes the question a consumer cannot answer, which app the user
has running. `localProviders` is the four local presets in probe order (ollama, lmstudio, llamacpp,
vllm); `availableProvidersAmong:timeout:` probes a list concurrently through a `dispatch_group` and
answers in the list's order, so the call costs one timeout rather than four and the same machine gives
the same answer twice; `firstAvailableProviderAmong:timeout:` probes in order and stops at the first
reply, which is one probe when the first address is up; `availableLocalProviders` /
`firstAvailableLocalProvider` are those two over `localProviders` at
`NFKRemoteProviderProbeTimeout` (2 s), with completion-handler forms and
`backendForFirstAvailableLocalProviderWithModelName:` as the one-call path. Every probe goes through
the instance seam `isReachableWithAPIKey:timeout:error:`, which is `NFKRemoteModelCatalog`'s
`isReachableWithError:` with the timeout set, so any HTTP reply counts and a subclass decides what
available means. No key is sent, so a hosted preset answers 401 and reads as reachable; the call is
aimed at the local ports, where nothing listening is the answer that matters.
`NFKRemoteProvider` gained value equality for it: every preset getter builds a new instance, so
`firstAvailableLocalProvider` used to compare unequal to `NFKRemoteProvider.ollama` under `==` and
`containsObject:`, which the Swift example caught.

Two things this round pinned. **The tests bind a real loopback socket** (`NFKLoopbackServer` in
`NFKRemoteProviderTests`, port 0 so the kernel picks a free one, one 200 with `{}` per connection):
`NSURLProtocol` cannot serve the probe, because the catalog uses `NSURLSession.sharedSession` and a
session only consults the protocol classes in its own configuration, so a stubbed protocol is
invisible to it. Dead addresses in those tests are the discard port (9), which loopback refuses at
once. **The Swift async import shadows a blocking method of the same name:**
`+availableLocalProvidersWithCompletionHandler:` imports as `availableLocalProviders() async`, which
took the blocking `+availableLocalProviders`'s Swift name and made it unreachable ("'async' call in a
function that does not support concurrency" at the *sync* call site). The completion forms therefore
carry `NS_SWIFT_ASYNC_NAME(probeAvailableLocalProviders())` /
`NS_SWIFT_ASYNC_NAME(probeFirstAvailableLocalProvider())`. The pair that already shipped,
`modelsWithAPIKey:error:` and `modelsWithAPIKey:completionHandler:`, escapes this because the blocking
one throws and `try` disambiguates. The importer also drops a trailing noun that repeats the return
type, so `+firstAvailableLocalProvider` arrived as `firstAvailableLocal()` and now carries
`NS_SWIFT_NAME(firstAvailableLocalProvider())`; `availableProvidersAmong:timeout:` and
`firstAvailableProviderAmong:timeout:` carry their `among:timeout:` names for the same reason.
Measured on this machine with Ollama 0.33 on 11434 and nothing on 1234 / 8080 / 8000:
`firstAvailableLocalProvider` answers ollama, `availableLocalProviders` answers ollama alone.

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
`input_json_delta` append, `message_stop` finishes, an `error` event fails. **Sampling keys (2026-09-19):** the contract's text parameters are camelCase and the services read
underscored names, so `NFKRemoteWireNames()` in `NFKRemoteBackend.m` renames them: `maxTokens` →
`max_tokens`, `topP` → `top_p`, `topK` → `top_k`, `stopSequences` → `stop`, and `repetitionPenalty`
→ **both** `repetition_penalty` (vLLM, TGI) and `repeat_penalty` (llama.cpp, Ollama), which name the
same multiplicative penalty with no server reading both. `temperature` and `seed` already carry the
wire spelling. The renaming runs before the fold, so a caller who writes `max_tokens` themselves
keeps their value. Before that, those four keys folded in as `maxTokens` and friends and every
OpenAI-compatible service ignored them, while `Docs/inference-guide.md` promised the contract's keys
work on any text engine. `NFKAnthropicBackend` reads the same five explicitly (`stop_sequences` is
its spelling for the stops) and has no repetition penalty. **Reasoning and usage (2026-09-19):**
`NFKParameterReasoningEffort` renames its value as well as its key, so it has its own table,
`NFKRemoteReasoningEfforts()`: light → `low`, moderate → `medium`, deep → `high` under
`reasoning_effort`, any other string written as it stands. The Messages API takes a budget rather
than a level only on models before Claude Opus 4.6, so there `NFKAnthropicThinkingBudgets()` maps the
three to 2048 / 8192 / 16384 under `thinking: {type: enabled, budget_tokens}`, a numeric string is an
exact budget, and anything else is refused in `urlRequestForRequest:`. **Model generations
(2026-09-22):** the Anthropic backend picks the shape from `modelName` by family substring (so
`anthropic.claude-…` and `claude-…@date` match). `NFKAnthropicBudgetThinkingFamilies()` (claude-3, the
4.0 / 4.1 / 4.5 models) keeps the budget. Every other name takes `thinking: {type: adaptive}` plus
`output_config.effort` (light → `low`, moderate → `medium`, deep → `high`, other strings as written,
a numeric string refused) because Opus 4.6 onward returns 400 on `budget_tokens`. An unknown name is
treated as current, because every release since 4.6 refuses the budget. Sampling goes out only on the
budget families and `NFKAnthropicSummarizingFamilies()` (Opus / Sonnet 4.6), and only when the model is
not thinking: Opus 4.7 onward, Sonnet 5, Fable, and Opus 5 / 5.5 return 400 on `temperature` / `top_p` /
`top_k`. Outside the 4.6 families the request adds `display: summarized`, since the default display
from 4.7 on is `omitted` (an empty `thinking` string, now skipped when joining the reasoning).
Thinking counts toward `max_tokens`, so an effort raises the limit by the same budget table (deep for a
pass-through level). `NFKAnthropicUnforcedToolFamilies()` (Opus 5.5, Fable 5.1, Mythos 5.1) returns 400 on a forced
`tool_choice`, so there the schema goes out as `output_config.format` and the reply's JSON text is
parsed into `NFKOutputStructured`. Every model also refuses a forced tool beside
`thinking: {type: enabled}`, so a budget-family request that carries both a schema and an effort takes
`output_config.format` too. Opus 5.5 also always thinks: it refuses `thinking: {type: disabled}`,
which the backend never sends. Its effort default is `medium`, one level below Opus 5. **OpenAI
reasoning families (2026-09-22):** `NFKRemoteBackend` spells the limit `max_completion_tokens` when
`modelName` has the prefix `gpt-5`, `gpt-6`, `o1`, `o3`, or `o4`, because those refuse `max_tokens` on
Chat Completions. The match is a prefix, so a router's `openai/gpt-…` keeps `max_tokens`. GPT-5.6 Sol
(`gpt-5.6-sol`) and Luna (`gpt-5.6-luna`, the low-cost tier; Terra sits between them) serve Chat
Completions with `reasoning_effort` `none` / `low` / `medium` (default) / `high` / `xhigh` / `max`, which
pass through. OpenAI's latest-model guide says to remove `temperature`, `top_p`, and `top_logprobs`
(and `logprobs` on Chat Completions) whenever the effort is not `none`, and the families default to
`medium`, so `removeFieldsTheModelRefusesFromBody:` drops all four for the same prefixes unless the body's
`reasoning_effort` is `none`.

**Model audit (2026-09-22).** The providers' official model pages were checked against the request
shapes:
- **Anthropic:** there is no Claude Opus 5.1. The Opus line runs `claude-opus-5` → `claude-opus-5-5`
  (released 2026-09-22); "5.1" is Fable 5.1 and Mythos 5.1. The deprecations table and the thinking
  troubleshooting page are the sources for the family tables above. `claude-mythos-preview` is
  deprecated but served, accepts a budget, and does not refuse a forced tool.
- **OpenAI:** `gpt-6-astra` / `-sol` / `-luna`, `gpt-5.6-sol` / `-terra` / `-luna`, `gpt-5.5`, and
  `gpt-5.4` (+ mini / nano) serve Chat Completions; the `-pro` models and `gpt-5.3-codex` are
  Responses-only. `gpt-6-astra` has no `none` effort. `max` exists only on the GPT-6 and GPT-5.6 families.
  None of the model pages says whether `max_tokens` is refused; `max_completion_tokens` is the
  documented field.
- **xAI:** "presencePenalty, frequencyPenalty, and stop cannot be used with reasoning models"
  (grok-4.5 / 4.6 / 4.7, the 4.20 reasoning models). `NFKRemoteModelRefusesStopSequences` covers
  `grok-4…` and `grok-3-mini…` except names containing `non-reasoning`; the backend drops `stop` and cuts
  the finished text at the earliest stop (the streamed partials run past it). Chat Completions is
  xAI's legacy endpoint.
- **Mistral:** a reasoning reply (`reasoning_effort` set on `mistral-medium-3-5` / `mistral-small-4`)
  returns `message.content` as chunks, `{type: thinking, thinking: [{type: text, text}]}` then
  `{type: text, text}`, blocking and streamed; `appendContentChunks:toText:reasoning:` reads both into
  the text and `NFKOutputReasoning`. Its effort values are `none|minimal|low|medium|high|xhigh`.
- **Gemini (OpenAI layer):** current `gemini-3.5`–`3.8-flash`, `3.1-pro-preview`, `3.1-flash-lite`; the
  2.0 models shut down 2026-06-01. 3.x cannot turn reasoning off. `temperature` / `top_p` / `top_k` were
  deprecated 2026-07-21 with no stated rejection, so the backend still sends them.
- **DeepSeek:** `deepseek-flash` (V4.1-Flash) and `deepseek-v4-pro`; effort `none|low|high|max`
  (`medium` is not listed), and thinking mode ignores `temperature`. Only `json_object` is documented,
  so a `json_schema` response format may be refused (unverified).
- **Groq:** `max_tokens` is deprecated for `max_completion_tokens`, not refused; `openai/gpt-oss-*`
  take `low|medium|high`, `qwen/qwen3.8-27b` `none|default|low|medium|high`.
- **Together / OpenRouter:** both accept the request shape. OpenRouter lists `anthropic/claude-opus-5.5`,
  `openai/gpt-5.6-sol`, and `openai/gpt-6-*`, accepts `max_tokens` for OpenAI models, and drops
  parameters a model does not take rather than refusing them. The prefix match leaves those
  namespaced names alone on purpose.
Unknown-field rejection (`top_k`, `repetition_penalty`, `repeat_penalty`) is undocumented for Gemini,
xAI, Mistral, DeepSeek, and Together; those go out only when a caller sets them.

**Media and retrieval backends audit (2026-09-22).** Every `backendForProvider:` factory below accepts
any OpenAI-style preset, so a preset that does not serve the path answers 404 at request time.
- **Video (`NFKRemoteVideoBackend`):** OpenAI removes the Videos API and `sora-2` / `sora-2-pro` on
  **2026-09-24**, with no replacement named (announced 2026-03-24). The backend now speaks every
  hosted video API through `apiStyle` (2026-09-22); the per-style contract is below.
- **Images (`NFKRemoteImageBackend`):** OpenAI `gpt-image-2` (plus `gpt-image-2.5-sunburst` /
  `-flare`) takes any `WxH` with sides a multiple of 16 and returns `b64_json` only; `seed` and `steps`
  are undocumented there. `gpt-image-1.5` / `-1-mini` shut down 2026-12-01; DALL·E shut down
  2026-05-12. xAI takes `aspect_ratio` / `resolution` instead of `size` and edits by JSON only (the
  multipart edit is refused). Together takes `width` / `height`, `steps`, and `seed` instead of `size`.
  Gemini ignores unknown fields and documents no edits. OpenRouter's path is `/api/v1/images`, so
  `images/generations` does not reach it. Groq, Mistral, and DeepSeek serve no images.
- **Transcription:** `whisper-1` and the `gpt-4o-*-transcribe` models shut down 2027-02-26 for
  `gpt-transcribe` (same path, no documented `verbose_json` / `segments[]`) or `gpt-live-transcribe`
  (Realtime only). `whisper-1` is also OpenAI's only `/audio/translations` model. Groq `whisper-large-v3`
  (+ turbo, no translations) returns the verbose shape with `avg_logprob`; Together's segments lack
  `avg_logprob` (confidence reads 1); Mistral `voxtral-mini-latest` has no `response_format` field and
  returns `segments[]` with `speaker`. xAI transcribes on `/v1/stt` and Gemini's layer has no audio path.
- **Speech:** OpenAI `gpt-4o-mini-tts` (13 voices, `instructions`), `tts-1` / `tts-1-hd` (9 voices);
  `wav` works everywhere it is served. Groq Orpheus (`canopylabs/orpheus-v1-english`) is wav-only and
  caps input at 200 characters. xAI speaks on `/v1/tts`.
- **Embeddings:** served by openai (`text-embedding-3-*`), mistral (`mistral-embed-23-12`,
  `codestral-embed-25-05`), gemini (`gemini-embedding-2`, `-001`; `embedding-2-preview` shut down
  2026-08-10), and openrouter; not xai, groq, deepseek, or serverless Together.
- **Moderation:** openai (`omni-moderation-latest`) and mistral (Moderation 2, written both
  `mistral-moderation-2603` and `-26-03`; `-2411` retired 2026-06-30). Mistral returns `categories` and
  `category_scores` without `flagged` and takes strings only, so `verdictWithFlag:` derives the flag
  from the categories.
- **Rerank:** together (dedicated endpoints only: `mixedbread-ai/mxbai-rerank-large-v2`,
  `Salesforce/Llama-Rank-V1`) and openrouter (`cohere/rerank-v3.5` and others). OpenRouter's schema has
  no `return_documents`; whether it ignores the field is unconfirmed.
- **TypeSafe:** `jev-latest` and `jev-preview` still resolve to `jev-1.13.0`.

**Audio and image styles (2026-09-22).** The transcription, speech, and image backends choose a wire
shape the way the video backend does, and their factories return nil for a preset without the path.
- **Transcription** (`NFKRemoteTranscriptionAPIStyle`): OpenAI style for openai / groq / together /
  openrouter / vllm (`verbose_json` plus `timestamp_granularities[]`, `diarized_json` +
  `chunking_strategy: auto` when the model name contains `diarize`, a plain `diarize` field otherwise,
  `keywords[]` for the vocabulary); Mistral (no `response_format` field at all, repeated
  `timestamp_granularities`, `diarize`, `context_bias`, segments with `speaker_id` and `score`); xAI
  `/v1/stt` (`diarize`, `keyterm`, a reply of `words[]` only, grouped into turns at a speaker change or a
  sentence end). The file part always goes last, because xAI ignores fields after it. A hosted clip
  travels as `url` (Groq, xAI), `file` (Together), or `file_url` (Mistral); elsewhere it is fetched and
  uploaded. Streaming (`streams`): OpenAI `transcript.text.delta` / `.done`, Mistral
  `transcription.text.delta` / `.segment` / `.done`; `whisper-1` refuses `stream`, so it is opt-in.
- **Speech** (`NFKRemoteSpeechAPIStyle`): OpenAI style for openai / groq / together / openrouter
  (`stream_format: sse` + `stream` when streamed; OpenRouter's voice clone as `input_references`);
  Mistral (`voice_id` or `ref_audio`, reply `{audio_data}` base64, stream `speech.audio.delta
  {audio_data}`); xAI `/v1/tts` (`text`, `voice_id` optional, `language` defaulting to `auto`,
  `output_format {codec, sample_rate}`). Stream deltas are read from `audio`, `b64`, or `audio_data`.
  Groq's 200-character cap is `maximumInputLength`; pieces are cut with
  `NSStringEnumerationBySentences`, which needs a capitalized next sentence to see a break. Voice
  listing: xAI `/v1/tts/voices`, Mistral `/v1/audio/voices`, Together `/v1/voices?model=`.
- **Images** (`NFKRemoteImageAPIStyle`): OpenAI (openai; gemini with no edits URL), xAI (JSON edits,
  `image` for one source and `images[]` for several, `{url: data URI}`, ratio from the size,
  `response_format: b64_json`), Together (edits on the generations path, `width` / `height`,
  `image_url` + `reference_images` as data URIs, `response_format: base64`), OpenRouter (`/images`,
  `input_references`). A mask is OpenAI-only. Streaming reads `*.partial_image` into the partial
  result and `*.completed` into the final one.

**Chat dialects and reply extras (2026-09-22).** `NFKRemoteBackend.chatDialect` is set by
`NFKRemoteProvider backendForProvider:` (mistral, openrouter, vllm, llamacpp; every other preset is
standard). `NFKRemoteAttachments attachmentsForRequest:keepsVideo:error:` keeps a clip whole under
`videoData` for the OpenRouter / vLLM / llama.cpp dialects unless `NFKParameterVideoFrameCount` is set.
Whole-clip parts: `video_url {url: data:video/<ext>;base64}` (OpenRouter, vLLM), `input_video {data,
format}` (llama.cpp; the shape is inferred from its `input_audio` and not yet verified live). Mistral:
`document_url` + `document_name`, and `input_audio` as a bare base64 string. Documents carry a
`mediaType`: `text/plain` for an `NSString` or a `.txt` / `.md` / `.markdown` / `.csv` / `.json` /
`.html` / `.xml` file, which rides as a text part on chat and as a `text` source on Anthropic. Reply
extras read by `extraOutputsInMessage:` from the blocking reply only: `message.images[]`
(OpenRouter, data URLs), `message.annotations[].url_citation`, and Groq's `message.executed_tools`.
Anthropic reads `citations[]` on text blocks (and `citations_delta` when streamed) and pairs
`server_tool_use` blocks with `*_tool_result` blocks by `tool_use_id`. Gemini embeddings use the native
`models/{m}:embedContent` with `inline_data` parts and `x-goog-api-key`, `:batchEmbedContents` for
`embeddingsForTexts:`, `dimensions` → `outputDimensionality`, `task_type` → `taskType`.

**Responses, Interactions, and the single-purpose backends (2026-09-22).**
- `NFKRemoteResponsesBackend` (`/responses`; openai, xai, groq, deepseek, openrouter, lmstudio, ollama,
  vllm, llamacpp): a leading system turn → `instructions`; turns → `{role, content: [input_text |
  output_text]}` with `input_image` (data URL) and `input_file {filename, file_data}` on the last user
  turn (a text document as `input_text`); audio is refused. Tools flatten to `{type: function, name,
  description, parameters}`; a wire-shaped tool passes through. `text.format {type: json_schema, name,
  schema}`, `reasoning {effort, summary: auto}`, `max_output_tokens`, `previous_response_id`. OpenAI's
  reasoning families lose `temperature` / `top_p` unless the effort is `none`, as on chat. Output items:
  `message` (text, `url_citation` annotations, `refusal` → kNFKError_InferenceRefused), `reasoning`
  (`summary[].text`), `function_call` (`call_id`, JSON-string `arguments`), `image_generation_call`
  (`result` base64), any other `*_call` → server tool results. Background: `background: true`, then
  `GET /responses/{id}` until `completed` / `failed` / `cancelled` / `incomplete`; cancelling the job
  posts `/responses/{id}/cancel`. Stream: `response.output_text.delta`, `response.reasoning*.delta`,
  `response.completed` / `.incomplete` carry the whole response, `response.failed` / `error`.
- `NFKGeminiInteractionsBackend` (`POST /v1beta/interactions`, `x-goog-api-key`, which Google's curl
  uses although the reference page shows a Bearer header): a prompt alone goes as the string `input`;
  otherwise steps `{type: user_input | model_output, content: [blocks]}` with `system_instruction`, and
  media blocks `{type: image | audio | video | document, mime_type, data}`. `outputModality` picks
  `response_format`: image (`aspect_ratio`, `image_size`, `mime_type`), audio (`speech_config` is an
  array of `{voice}` or `{speaker, voice}`), video (`aspect_ratio`, `resolution`), or JSON text
  (`mime_type: application/json`, `schema`). Transcription: `transcription_config {language_codes,
  custom_vocabulary, mode: {type: verbatim, diarization_mode: speaker, timestamp_granularities:
  [word]}}`, read back from `word_info` annotations (`start_offset` / `end_offset` as `"1.2s"` strings
  or numbers). `thinking_level` + `thinking_summaries: auto`; everything else goes into
  `generation_config`. Reply steps: `thought` → reasoning, `function_call` (`id`, `name`, object
  `arguments`), `model_output` blocks → text / image / audio (PCM wrapped in a 24 kHz mono WAV unless
  the MIME type names mp3 or wav) / video (`uri` fetched with the key on Google's host); other step types
  → server tool results. Usage fields are `total_input_tokens`, `total_cached_tokens`,
  `total_output_tokens`, `total_thought_tokens`. Background polls `GET /interactions/{id}` until a
  terminal status (`requires_action` included). The stream is `?alt=sse` with `event_type` events;
  `step.delta {delta: {type: text, text}}` grows the text and the finished interaction is read back by
  the id from `interaction.created`.
- `NFKRemoteCompletionBackend`: OpenAI style `{prompt, suffix}` → `choices[].text`; DeepSeek's path is
  `…/beta/completions` off the version root; Mistral `/fim/completions` answers in the chat shape and
  names the seed `random_seed`; llama.cpp posts `/infill {input_prefix, input_suffix}` or `/completion
  {prompt}` at the server root with `n_predict`, answered with `content`.
- `NFKRemoteOCRBackend` (Mistral `/ocr`): `document {type: document_url, document_url}` (a data URI for
  local bytes) or `{type: image_url, image_url}`; `document_annotation_format {type: json_schema,
  json_schema: {name, schema}}` from `NFKParameterJSONSchema`, whose reply `document_annotation` is a JSON
  string; `pages[].markdown` joined by blank lines; `pages[].images[].image_base64` (a data URI) decoded.
- `NFKRemoteClassifierBackend`: Mistral `results[0]` is `{target: {scores: {label: score}}}` (labels
  prefixed `target/` when there are several targets), chat input `{messages}` on `/chat/classifications`;
  vLLM `/classify` at the server root answers `data[0] {label, probs[]}`, only the top class named.
- `NFKRemoteTokenCounter`: Anthropic `count_tokens` (Messages body, system lifted) → `input_tokens`;
  Gemini `models/{m}:countTokens` → `totalTokens`; xAI `tokenize-text` → `token_ids[{token_id}]`;
  llama.cpp `/tokenize {content}` → `tokens`.

**Realtime sessions (`NFKRealtimeSession`, 2026-09-22).** One class, twelve protocols; the socket is
`NFKRealtimeSocket` (`NFKRealtimeWebSocket` on `NSURLSessionWebSocketTask`, one receive per message),
and `socketForRequest:` is the test seam. `sessionForProvider:apiStyle:` builds the ws(s) URL from the
provider's base (`http` → `ws`, else `wss`); Gemini's two sockets are fixed `…/ws/google.ai.
generativelanguage.v1beta.GenerativeService.BidiGenerateContent` and `….v1alpha.…BidiGenerateMusic`
with the key as `?key=`; every other style sends `Authorization: Bearer`. Handshake queries: `model`
(OpenAI conversation and translation, xAI conversation, Mistral), xAI STT `sample_rate`, `encoding=pcm`,
`interim_results=true`, `language`; xAI TTS `voice`, `language` (default `auto`), `codec=pcm`,
`sample_rate`; Together STT `model`, `input_audio_format=pcm_s16le_<rate>`; Together TTS `model`,
`voice`. Configuration on connect: OpenAI / xAI `session.update {session: {type: realtime (OpenAI only),
instructions, audio: {input: {format: {type: audio/pcm, rate}}, output: {format, voice}}, tools}}`;
OpenAI transcription `session.update {session: {type: transcription, audio: {input: {format,
transcription: {model, language}}}}}` (the transcription-session URL shape is inferred, not seen in a
raw example); OpenAI translation `session.update {session: {audio: {output: {language}}}}`; Mistral
`session.update {session: {audio_format: {encoding: pcm_s16le, sample_rate}}}` (from the mistralai SDK
source, which is the only raw spelling Mistral publishes); vLLM `session.update {model}`; Gemini
`{setup: {model: models/<m>, generationConfig: {responseModalities: [AUDIO], speechConfig},
systemInstruction, inputAudioTranscription: {}, outputAudioTranscription: {}, tools:
[{functionDeclarations}]}}`; Lyria `{setup: {model}}`. Sending: audio is `input_audio_buffer.append
{audio}` (OpenAI, xAI conversation, Together, vLLM), `session.input_audio_buffer.append` (translation),
`input_audio.append` (Mistral), `realtimeInput.audio {data, mimeType: audio/pcm;rate=N}` (Gemini), or
binary frames (xAI STT). Commit / finish per style: `input_audio_buffer.commit` (+ `final: true` for
vLLM), `input_audio.flush` / `input_audio.end` (Mistral), `finalize` / `audio.done` (xAI STT),
`text.done` (xAI TTS), `input_text_buffer.commit` (Together TTS), `session.close` (translation),
`realtimeInput.audioStreamEnd` (Gemini). Lyria: `clientContent.weightedPrompts`,
`musicGenerationConfig`, `playbackControl: PLAY | PAUSE | STOP`. Receiving: any typed event ending
`audio.delta` (not a transcript) or `conversation.item.audio_output.delta` is audio; text deltas are
sorted into response text, output transcript, and input transcript (partial or final, including xAI's
`transcript.partial.is_final`); `response.function_call_arguments.done` and Gemini `toolCall.functionCalls`
become tool calls with parsed arguments; `response.done`, `audio.done`, `transcription.done`, and Gemini
`turnComplete` end a turn. Gemini frames may arrive binary; both are parsed as JSON.

**Files, retrieval stores, and usage reports (2026-09-22).** Three storage and reporting objects, none
of them a backend. Each has a `sendRequest:response:error:` seam and a `…ForProvider:apiKey:` factory
that returns nil for a preset without the API.

- `NFKRemoteFileStore` styles: OpenAI (openai and deepseek purpose `user_data`, xai `assistants`,
  mistral `ocr`, groq `batch`, together `fine-tune`; Together's multipart names the part `upload`
  beside `file_name`; DeepSeek's root is `https://api.deepseek.com/files`, off the `/v1` base),
  Anthropic (`/v1/files`), and Gemini (resumable two-step upload: `X-Goog-Upload-Protocol: resumable`
  start, then `upload, finalize`; files are named `files/…`, referenced by `uri`, and wait in
  `PROCESSING` until `ACTIVE`, which `fileWhenReadyWithIdentifier:timeout:` polls). Listing reads every
  page. Mistral's signed URL is `GET /files/{id}/url?expiry=`.
- File references by request shape: Chat Completions `{type: file, file: {file_id}}`; DeepSeek the
  flat `{type: file, file_id}`; Mistral chat a `document_url` holding the signed URL, fetched through
  the chat backend's own `sendRequest` so a test stub sees it; Anthropic a `document` or `image` block
  with source `{type: file, file_id}`; Responses `input_file` / `input_image` with `file_id`; Gemini
  Interactions `{uri, mime_type}`; Mistral OCR `{type: file, file_id}`; Gemini embeddings
  `file_data {mime_type, file_uri}`.
- `NFKRemoteRetrievalStore` styles: OpenAI `/vector_stores` with `OpenAI-Beta: assistants=v2`, files
  added by id, `POST /vector_stores/{id}/search` (`query`, `max_num_results`, `filters`); xAI
  collections on `https://management-api.x.ai/v1` with `managementAPIKey`, search on
  `api.x.ai/v1/documents/search` with `apiKey` and an AIP-160 filter string; Gemini
  `v1beta/fileSearchStores`, add by `:importFile` (answers an operation), delete with `force=true`, no
  direct search; Mistral `/libraries`, documents only by multipart upload, no direct search.
- `NFKRemoteUsageReporter` contracts. Money is normalized to dollars, time to `NSDate`, and every page
  (`has_more` / `next_page` → `page`) is read.
  - Anthropic: `x-api-key` + `anthropic-version`; `organizations/usage_report/messages` (`starting_at`
    / `ending_at` RFC 3339, `bucket_width` 1d / 1h / 1m, repeated `group_by[]`); `cost_report` (1d
    only, `amount` a decimal string in cents, divided by 100); `usage_report/claude_code` takes
    `starting_at` as a bare day and answers one record per actor and day.
  - OpenAI: Bearer admin key; `organization/usage/{report}` (completions, embeddings, images,
    audio_speeches, audio_transcriptions, moderations, vector_stores, code_interpreter_sessions,
    web_search_calls, file_search_calls) with Unix-second `start_time` / `end_time`; `organization/costs`
    answers `amount {value (dollars), currency (lower case)}` and `line_item`. The repeated
    `group_by` spelling is not confirmed against a live call.
  - xAI: `POST billing/teams/{team}/usage` with an `analyticsRequest` (`timeRange` in
    `yyyy-MM-dd HH:mm:ss`, `Etc/GMT`, `TIME_UNIT_DAY`, `usd` summed, `groupBy` default `description`);
    each `timeSeries[].dataPoints[]` becomes a row. `prepaid/balance` answers `total.val`, read as
    negative cents (credit); the unit is inferred, not documented.
  - OpenRouter: `credits` (`total_credits − total_usage`, management key), `activity` (last 30
    completed days, `usage` in dollars), filtered to the asked span and bucketed by day.
  - DeepSeek: `user/balance` → one balance per `balance_infos[]` currency. No usage or cost API.
  - Mistral: `v1/admin/usage?month=&year=` with the admin key as `x-api-key`; the reply shape is
    undocumented, so it returns whole as one bucket.
  - No reporting API: Groq, Together, Gemini.

**Video styles (`NFKRemoteVideoAPIStyle`, 2026-09-22).** `backendForProvider:` picks gemini →
Sora-compatible, openai → OpenAI, xai / together / openrouter → their own, and nil for every other
preset; `backendForProvider:apiStyle:` reaches Gemini's native Veo. Moving the default between
services is a change of provider, not of code. Shared behavior: a contract key the service has no
field for is dropped, every other parameter goes out by name (into Veo's `parameters`), a request the
style cannot express fails with `kNFKError_InferenceUnsupported` before the submit (`refusalForRequest:`),
the key goes with a download only to the submit host, and a download whose bytes open with the EBML
signature is written `.webm`. Width and height become a reduced ratio and a short-side tier
(`720p`, `1080p`, `4k`) for services that take those.
- **OpenAI** (`/v1/videos`): JSON `model`, `prompt`, `seconds` (string), `size`; a first frame makes it
  multipart with an `input_reference` file; edit → `/videos/edits`, extend → `/videos/extensions`, both
  `{prompt, video: {id}}` from `NFKParameterSourceVideoIdentifier`; clip at `/videos/{id}/content`.
- **Gemini Sora-compatible** (`v1beta/openai/videos`, the default for `gemini`): always multipart, as
  Google's curl sample is. `model` and `prompt` are the documented top-level fields; the Veo options
  (`duration_seconds`, `aspect_ratio`, `resolution`, `frame_rate`, `negative_prompt`, `seed`,
  `image` base64, `last_frame` base64, `reference_images[]`, `extend_video_id`, `person_generation`,
  `style`) go out as further form fields, which is where the OpenAI SDKs put `extra_body` on a multipart
  request. Google shows no wire example of those fields, and the `last_frame` object shape is
  undocumented; both are unverified live. Status `processing` → `completed` / `failed`; clip at `url`.
  Edit is refused (Veo extends only).
- **Gemini Veo native** (`v1beta/models/{model}:predictLongRunning`, `x-goog-api-key`): `instances[0]`
  carries `prompt`, `image` / `lastFrame` as `{inlineData: {mimeType, data}}`, `referenceImages`
  (`referenceType: asset`), and `video: {uri}` to extend; `parameters` carries `durationSeconds`,
  `aspectRatio`, `resolution`, `numberOfVideos`, `negativePrompt`, `seed`, `personGeneration`. The job
  is the operation `name`, polled at `v1beta/{name}` until `done`; every
  `generateVideoResponse.generatedSamples[].video.uri` is downloaded. Veo facts: 4/6/8 s (8 for 1080p,
  4k, references, extension), 16:9 or 9:16, audio always on, clips deleted after 2 days.
- **xAI** (`/v1/videos/generations`, `/edits`, `/extensions`): JSON `duration`, `aspect_ratio`,
  `resolution` (480p/720p/1080p), `image` / `last_frame` / `reference_images[]` as `{url: data URI}`,
  `generate_audio`; edit and extend send `video: {url}` (a local clip inline as a data URI) or
  `{file_id}`. No seed. Job `request_id`; status `pending` → `done` / `failed` / `expired`, progress
  0–100; clip at `video.url` on `vidgen.x.ai` (fetched without the key).
- **Together** (`/v2/videos` on the preset's host): JSON `width`, `height`, `seconds` (string), `fps`,
  `steps`, `seed`, `guidance_scale`, `output_format` (`MP4` / `WEBM`), `negative_prompt`,
  `generate_audio`, `resolution` (`720P`), `ratio`, and `media.frame_images` (`{input_image: raw base64,
  frame: first | last}`), `media.reference_images`, `media.source_video` (edit), `media.frame_videos`
  (extend). A source clip must be a hosted URL. Status `queued` / `in_progress` → `completed` /
  `failed` / `cancelled`; clip at `outputs.video_url` (expires).
- **OpenRouter** (`/api/v1/videos`): JSON `duration`, `size`, `aspect_ratio`, `resolution`, `seed`,
  `generate_audio`, `frame_images` (`{type: image_url, image_url: {url}, frame_type: first_frame |
  last_frame}`), `input_references`, `previous_job_id` for edit and extend, `provider` and the rest by
  name. Status `pending` / `in_progress` → `completed` / `failed` / `cancelled` / `expired`; clips at
  `unsigned_urls`, relative ones resolved on its host. Coming back: `message.reasoning_content` or `message.reasoning` (both spellings are in use)
and Anthropic's `thinking` blocks → `NFKOutputReasoning`; `usage` → `NFKOutputUsage` through the
shared `NFKRemoteUsage(...)`, which leaves out a count the provider did not report. Streaming:
`reasoning_content` deltas and `thinking_delta` grow the chain on the partial result beside the
text; OpenAI puts the counts on a choice-less final chunk, which it sends only when the request set
`stream_options.include_usage` (a caller adds that itself, since it folds in by name), while
Anthropic splits them across `message_start` (input) and `message_delta` (output). **Tools:**
`NFKParameterTools` (`{name, description, parameters}`) → OpenAI `{type: function, function}` /
Anthropic `{name, description, input_schema}`; replies → `NFKOutputToolCalls` = `{id, name, arguments
(parsed), argumentsJSON}` (`result.toolCalls`). The key is spelled `"tools"`, the wire field's own
name, so an entry already in wire shape (has `type`, or `input_schema`) passes through unwrapped —
otherwise a caller who folded OpenAI tools by name gets double-wrapped. **Schema:**
`NFKParameterJSONSchema` → `response_format: {type: json_schema, json_schema: {name: response, schema}}`
(no `strict`, which demands `additionalProperties: false` throughout) / Anthropic has no response
format on most models, so it is a forced tool `structured_output` (`tool_choice: {type: tool, name}`) whose `input` is
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

**Typed decisions (`NFKTypeSafeBackend`, the `typesafe` preset).** TypeSafe AI released Jev on
2026-09-15 as a "System One" model: `POST https://api.typesafe.ai/v1/systemone` with a Bearer token
takes `{model, state, questions}` and answers `{model, answers, usage}`, no text anywhere. `state` is a
string, an object, or an array; each question is `{type, instructions, criteria}` where `type` is
`choice` (criteria a map of up to 255 option names to a description or null), `score` (criteria an
ordered array of 2 to 10 level descriptions), or `noul` (criteria optional, the meanings of `true`
and `false`); each answer is `{type: choice, choice, probabilities, confidence}`,
`{type: score, score, legend, probabilities, confidence}`, or `{type: noul, noul}`, where `noul` is the
probability the statement holds. The model is required; `jev-latest` and `jev-preview` alias
`jev-1.13.0` at release, and `GET /v1/models` lists them, so the catalog serves the preset through the
Bearer path (the style is `NFKRemoteAPIStyleSystemOne`, and `authorizeRequest:` treats every style but
Anthropic's as Bearer). Limits at release: 64k tokens per request, 32k for the state plus the longest
question; pricing $0.042 per million input tokens, output free. Errors: 401, 422 (invalid body), 429,
and 529 (overloaded), the last two to be retried with backoff, which is why 529 joined the transport's
retried statuses (Anthropic uses the same code). The endpoint answered **403** to an unauthenticated
POST at release, not the 401/405 the other presets answer, so the path exists but the house probe
rule is not met the same way; the answer shape is asserted only by `testALiveEndpointAnswersATypedDecision`,
gated on `INFERKIT_TYPESAFE_API_KEY`, which no machine here has had (access is waitlisted). One
community write-up names `thejevai.com` as the API host; it is not TypeSafe's and is not used.

The core's vocabulary for the shape is engine-neutral, because the open reproduction (Laya,
`convaiinnovations/laya`, Apache 2.0, ModernBERT-large plus a decision head that scores every option
at its own mask token) speaks the same request; it ships as `NFKMLXLaya` in InferKitMLX
(`mlx-models-embeddings-retrieval.md`), so a feature moves between the hosted model and the device by
swapping the object:
`NFKInputState` / `NFKInputQuestions` in, `NFKOutputAnswers` out, `NFKDecisionQuestion` (three
factories; `dictionaryRepresentation` is the wire shape, and a dictionary already in that shape passes
through) and `NFKDecisionAnswer` (`answerWithDictionary:`, secure coding like the rest of the result
family, `raw` for what the type does not read). The backend accepts `NFKInputPrompt` and then
`NFKInputMessages` as the state when `NFKInputState` is absent, folds any other parameter into the
body under its own name without overriding `model` / `state` / `questions`, declares no supported
parameter key (nothing is sampled), and puts the whole reply under `NFKOutputStructured` beside
`NFKOutputUsage`. The Swift importer turns `+answerWithDictionary:` into `NFKDecisionAnswer(dictionary:)`
and leaves the question factories as static methods (`choiceQuestion(withInstructions:options:)`),
which the Swift example pins.

**Deliberately absent.** Midjourney has no official public API (its API host does not resolve), so
shipping a preset would imply one exists. `opencode.ai` answers `Not Found` on its API path — it is a
coding agent that calls other providers rather than an inference service. Codex is OpenAI's coding
agent, not a separate endpoint; it is the `openai` preset.

`NFKRemoteProviderTests` stubs the transport to assert Anthropic's request shape without a network, and
carries one live test gated on `INFERKIT_LIVE_LOCAL_MODEL` that runs against a local server when one is
listening.
