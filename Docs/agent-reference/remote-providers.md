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
- **`inferkit`** is an `NFKInferenceServer` (see "Serving" below). It speaks the OpenAI shape, the
  Responses and Messages APIs, and a native route, and `backendForProvider:` builds the native client,
  `NFKRemoteInferKitBackend`.
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
fifteen; only the credential headers differ, and Anthropic paginates (`has_more` / `last_id` →
`after_id`, page size raised to 1000 from its default 20), which the catalog follows to the end. The
list is deliberately not filtered to chat models: the envelope carries no capability field, so any
filter would be a name heuristic that breaks on the next release. `NFKRemoteModelCatalog` is the
object under the convenience (timeout, session, `isReachableWithError:`, and the overridable
`sendRequest:` seam the tests stub). Readiness is not the same question:
`NFKAnthropicBackend` reports not-ready without a model because the API requires one, while an
OpenAI-compatible backend is ready with an endpoint alone, which **llama.cpp depends on** — its server
answers for whatever model it has loaded.

A preset carries one address, `baseURL`, and derives the rest (`endpointURL` = base +
`/chat/completions`, `/messages` for Anthropic, `/systemone` for TypeSafe, or `/inferkit/run` for an
InferKit server; `modelsURL` = base + `/models`; `URLForPath:` for
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

`NFKRemoteEmbeddingBackend` is the embeddings counterpart (`POST /embeddings`; openai, gemini,
mistral, together, openrouter, the four local runners, and an InferKit server): `NFKInputPrompt` or joined `NFKInputMessages` → `NFKOutputEmbedding`, the key the MLX
embedders answer with, so a consumer's search code is engine-agnostic. `embeddingsForTexts:error:`
batches, ordered by the provider's `index` rather than by arrival (measured: a stub returning
index 1 before index 0 comes back in text order). `backendForProvider:apiKey:modelName:` answers nil
for every other preset — it imports to Swift as the failable initializer `NFKRemoteEmbeddingBackend(for:apiKey:modelName:)`,
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
`%3A`; `modelWithIdentifier:error:` and the LM Studio detail add `:` to the allowed set, pinned by a test.
**Two hazards from this round:** an `@[ a, b ]` literal inside an `XCTAssert…` macro argument splits
the macro on its comma unless the whole expression is parenthesized; and adjacent string-literal
concatenation as a direct element of an `@[ ]`/`@{ }` literal raises `-Wobjc-string-concatenation`
(parenthesize the element) — four of those had shipped in the catalog tests a round earlier because a
`tail` on the test log hid them. Grep the log for `warning:` without a tail.

**Discovery of the local runners** removes the question a consumer cannot answer, which app the user
has running. `localProviders` is the five local presets in probe order (ollama, lmstudio, llamacpp,
vllm, inferkit); `availableProvidersAmong:timeout:` probes a list concurrently through a `dispatch_group` and
answers in the list's order, so the call costs one timeout rather than one per address and the same machine gives
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
OpenAI layer serves generations and no speech or edits. The four local runners serve none; DeepSeek is
undeterminable. Recorded on each
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
exact budget, and anything else is refused in `urlRequestForRequest:streaming:error:`. **Model generations
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
pass-through level). `NFKAnthropicUnforcedToolFamilies()` (Opus 5.5, Sonnet 5.5, Fable 5.1, Mythos 5.1) returns 400 on a forced
`tool_choice`, so there the schema goes out as `output_config.format` and the reply's JSON text is
parsed into `NFKOutputStructured`. Every model also refuses a forced tool beside
`thinking: {type: enabled}`, so a budget-family request that carries both a schema and an effort takes
`output_config.format` too. Opus 5.5 also always thinks: it refuses `thinking: {type: disabled}`,
which the backend never sends. Its effort default is `medium`, one level below Opus 5. Sonnet 5.5
(2026-09-28) joined the unforced list from its release notes ("Forced tool use returns 400"). Haiku
5.5 (2026-10-07) did not: its notes list sampling, `budget_tokens`, prefill, and `computer_20250124`
as 400s and say nothing about `tool_choice`. Neither 5.5 model takes sampling, which the
current-model default already covers. **OpenAI
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
- **Gemini (OpenAI layer):** current `gemini-4-argon` (GA 2026-10-13), `gemini-3.8-flash`,
  `gemini-3.6-flash`, `3.5-flash-lite`, `3.1-pro-preview`, `3.1-flash-lite`; `gemini-3.7-flash` and
  `gemini-3.5-flash` were deprecated 2026-10-08 and their names route to 3.8 and 3.6. The 2.0 models
  shut down 2026-06-01. 3.x cannot turn reasoning off. `temperature` / `top_p` / `top_k` were
  deprecated 2026-07-21 with no stated rejection, so the backend still sends them; Argon ignores them.
  The OpenAI page maps Argon's `reasoning_effort` minimal→low and documents no long decode on this
  layer.
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

**Media and retrieval backends audit (2026-09-22).** Every `backendForProvider:` factory below returns
nil for a preset that does not serve the path.
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
`NFKRemoteProvider backendForProvider:` (mistral, openrouter, vllm, llamacpp, deepseek; every other preset is
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
`embeddingsForTexts:error:`, `dimensions` → `outputDimensionality`, `task_type` → `taskType`.

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
- `NFKGeminiGenerateContentBackend` (`POST {models}/{model}:generateContent`, `x-goog-api-key`,
  2026-10-13; read from the API reference at `ai.google.dev/api/generate-content`). Google serves it
  at `/v1beta/models` beside the OpenAI layer the `gemini` preset points at, and OpenCode Zen at
  `/inference/v1/models`.
  - Request: `contents[] {role: user | model, parts[]}`, `systemInstruction {parts}`, media as
    `inlineData {mimeType, data}` on the last user turn, an `NFKRemoteFile` as `fileData {mimeType,
    fileUri}`, a tool answer as `functionResponse {id?, name, response: {output | error}, parts?}`
    with consecutive answers in one user turn. Tools are `[{functionDeclarations: [{name,
    description, parametersJsonSchema}]}]`; an entry without a name (`{googleSearch: {}}`) passes as
    its own tool.
  - `generationConfig`: `maxOutputTokens`, `temperature`, `topP`, `topK`, `seed`, `stopSequences`,
    `thinkingConfig {thinkingLevel: LOW | MEDIUM | HIGH | thinkingBudget, includeThoughts: true}`, and
    `responseFormat {text: {mimeType: APPLICATION_JSON, schema}}`. The reference marks
    `responseSchema` and `_responseJsonSchema` deprecated in favor of `responseFormat`.
    `safetySettings`, `toolConfig`, `cachedContent`, `labels`, `serviceTier`, and `store` go to the
    top level; the contract keys without a field (repetition penalty, conversation key, MCP servers)
    are dropped.
  - Reply: `candidates[0].content.parts[]`. A `thought: true` part is a summary, a `functionCall`
    `{id?, name, args}` a call (one without an id gets `gemini-call-N`, and its answer goes back
    without an id), an image `inlineData` an image output. `usageMetadata` `promptTokenCount`,
    `cachedContentTokenCount`, `candidatesTokenCount + thoughtsTokenCount` (output), and
    `thoughtsTokenCount` (reasoning). `groundingSupports` pair a segment with `groundingChunks[].web`;
    `citationMetadata.citationSources` spans are byte offsets into the reply.
  - Thought signatures: the thinking guide says generateContent has no thought blocks, so a
    `thoughtSignature` can sit on any part (a `functionCall`, or the last part). The backend records
    the parts verbatim under the `gemini-generatecontent` turn format and replays them to the same
    endpoint. The stream merges text into the previous part of the same kind until that part is
    signed, so a signature that arrives on an empty trailing text part lands on the text it closes.
  - Refusals: `promptFeedback.blockReason`, and a finish reason of SAFETY, RECITATION, BLOCKLIST,
    PROHIBITED_CONTENT, SPII, or an image-safety reason with no text, call, or image.
  - Long decode (`gemini-4-argon`, 2026-10-13, `ai.google.dev/gemini-api/docs/generate-content/
    long-decode`): a hop ends with `finishReason: CONTINUATION` and a top-level `continuationToken`;
    the next hop resends the same `contents` and `generationConfig` with `continuationToken` at the top
    level. Non-streaming REST hops cap at 32K tokens, streaming hops at 64K. `MAX_TOKENS` also carries a
    token, which the backend does not follow: the caller's `maxOutputTokens` stands. Both paths join
    the hops through the stream state's part merge. A `CONTINUATION` without a token, a token seen
    on an earlier hop, or a 65th hop fails with `kNFKError_InferenceBackendFailure`, so a cut reply
    never reads as a whole one and a service that never stops cannot hold the call. A streamed hop
    checks the job for a cancel that arrived during the handover and cancels its own stream.
  - Not verified live: no Gemini or OpenCode key is set on this machine.
    `testALiveGenerateContentCallAnswers` is gated on `INFERKIT_GEMINI_API_KEY`.
- `NFKGeminiInteractionsBackend` (`POST /v1beta/interactions`, `x-goog-api-key`, which Google's curl
  uses although the reference page shows a Bearer header): a prompt alone goes as the string `input`;
  otherwise steps `{type: user_input | model_output, content: [blocks]}` with `system_instruction`, and
  media blocks `{type: image | audio | video | document, mime_type, data}`. `outputModality` picks
  `response_format`: image (`aspect_ratio`, `image_size`, `mime_type`), audio (`speech_config` is a
  one-entry array `[{voice}]` for one voice and the object `{speakers: [{speaker, voice}]}` for two),
  video (`aspect_ratio`, `resolution`), or JSON text
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
  - 2026-10-13 audit (the API reference and the agent, long-decode, and video guides; the guides send
    `Api-Revision: 2026-05-20`, which the backend does not):
    - Top-level fields by parameter name: `service_tier` (`flex | standard | priority | deferred`),
      `safety_settings [{type, threshold, method}]` (the harm category goes in `type`),
      `webhook_config {uris, user_metadata}` (background only), `labels`, `store`, `environment`,
      `agent_config`.
    - Agents: `agent` replaces `model`, `agent_config` replaces `generation_config` (a union on `type`:
      `antigravity | deep-research | code-mender | dynamic`). Agents need `background: true` and
      `store: true`. Deep Research config: `thinking_summaries`, `visualization`,
      `collaborative_planning` (a plan comes back as a completed interaction; the next turn approves
      it with `false`). Antigravity config: `model` (`gemini-3.8-flash` default, `gemini-4-argon`,
      `gemini-3.6-flash`, `gemini-3.5-flash-lite`), `max_total_tokens` (reaching it ends `incomplete`;
      `input: "continue"` with `previous_interaction_id` and `environment` resumes). Antigravity
      rejects `temperature`, `top_p`, `top_k`, `stop_sequences`, `max_output_tokens` with a 400.
    - The reference's agent enum still says `antigravity-preview-05-2026` (shut down 2026-10-05); the
      guides and the deprecations page say `-09-2026`. `deep-research-pro-preview-12-2025` shuts down
      2026-10-23.
    - An agent's built-in tools (`write_to_file`, `replace_file_content`, `view_file`, `list_dir`,
      `find_by_name`, `grep_search`, PascalCase arguments) come back as `function_call` steps with a
      matching `function_result` (`call_id`). The backend reports an answered call as a server tool
      and only an unanswered call as `NFKOutputToolCalls`; a caller's answer goes back as a
      `function_result` step with `previous_interaction_id` and `environment`.
    - `thought` steps carry `{signature, summary: [blocks]}` in the reference and `content` in the Omni
      guide; both are read.
    - Long decode (`gemini-4-argon`): `max_output_tokens` above 65536 needs `background: true` (400
      otherwise) and `store: true`; each 64K hop adds a `model_output` step; `queued` means waiting for
      off-peak capacity. The backend switches to background above 65536 on its own.
    - Background polling: the blocking call and the job share one loop. `backgroundTimeout` (3600 s
      by default, 0 for none) cancels the run on the service and fails. A failed poll, which the
      transport has already retried for 429 and 5xx, is polled again; the third in a row fails with
      an error naming the interaction, which `store: true` keeps on the service.
    - MCP: `NFKParameterMCPServers` → `{type: mcp_server, name, url,
      headers: {Authorization: Bearer …}, allowed_tools: [{tools}]}` (streamable HTTP only; SSE is not
      supported). The name must match
      `^[a-z0-9_-]+$` (an uppercase letter is a bare 400), so the backend lowercases it and turns other
      characters into `-`. The reference types `allowed_tools` as `[{mode, tools}]`; the Antigravity
      guide calls it a list of names. Calls come back as `mcp_server_tool_call` /
      `mcp_server_tool_result` steps with `server_name`. This is how the hosted Antigravity agent
      reaches an `NFKInferenceServer`'s MCP endpoint behind a public tunnel.
    - Agentic video: `processing: "agentic"` on a video block; the model's `processing_call` /
      `processing_result` steps need no answer and come back as server tools.
    - `ResponseModality` `document` appears only in the usage breakdowns (`*_tokens_by_modality`). No
      model produces document or 3D output; `response_format` types are text, image, audio, video.
    - Cancel is `POST /interactions/{id}/cancel` in the reference (the Antigravity guide writes
      `:cancel`); delete is `DELETE /interactions/{id}`.
- `NFKGeminiEnvironmentStore` (`/v1beta/environments`, 2026-10-13, `ai.google.dev/gemini-api/docs/
  agent-environment`): the Antigravity sandbox. `GET /environments` → `{environments: [{id, status,
  sources, network}], next_page_token}` (the page parameter is taken to be `page_token`, as on the
  voices and credentials collections; the environments page shows only `pageSize`). `GET
  /environments/{id}/files[/{path}][?recursive=true]` → `{files: [{name, path, type: FILE | DIRECTORY,
  size_bytes (a string), mime_type, created, modified}]}`; `?alt=media` returns a file's bytes or a
  directory as an uncompressed POSIX tar. Writes are `PUT /upload/v1beta/environments/{id}/files/{path}`
  with `overwrite=true` and `extract=true` (a tar into a directory); a write without the `/upload`
  prefix is a 400 and an existing file without `overwrite` a 409. `DELETE /environments/{id}`. An
  environment idles after 15 minutes and is deleted after 7 days without use. Files never come back
  inside the interaction. The store puts `upload/` before the endpoint's version segment, so a proxy
  base `…/g/v1beta/environments` uploads to `…/g/upload/v1beta/environments`. An id or path component
  that is empty, `.`, `..`, or carries a slash is refused before sending, since the URL would resolve
  it to another resource under the same key; every other character is percent-encoded per component.
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
and `socketForRequest:` is the test seam. `sessionForProvider:apiStyle:apiKey:modelName:` builds the ws(s) URL from the
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
`musicGenerationConfig`, `playbackControl: PLAY | PAUSE | STOP | RESET_CONTEXT`. Receiving: any typed
event ending `audio.delta` (not a transcript) or `conversation.item.audio_output.delta` is audio; text
deltas are sorted into response text, output transcript, and input transcript (partial or final,
including xAI's `transcript.partial.is_final`); `response.function_call_arguments.done` and Gemini
`toolCall.functionCalls` become tool calls with parsed arguments; `response.done`, `audio.done`,
`transcription.done`, and Gemini `turnComplete` end a turn. Gemini frames may arrive binary; both are
parsed as JSON.

Gemini Live, 2026-10-13 audit (`ai.google.dev/api/live` updated 2026-10-09, the thinking,
live-translate, live-transcribe, and ephemeral-token guides, google-genai v2.30.1):
- `gemini-3.8-live`: audio out only, no `thinkingConfig`, proactive audio always on (`proactive_audio:
  false` is an error), `NON_BLOCKING` functions by default, `scheduling` on a function response
  (`INTERRUPT | WHEN_IDLE | SILENT`; the model page's `INTERRUPTED` is a typo against the SDK).
- `gemini-3.8-live-extended-thinking`: `thinkingConfig {thinkingLevel: low | medium | high}`
  (`minimal` rejected), non-blocking functions only. Background reasoning adds
  `serverContent.interactionStatus: IN_PROGRESS | IDLE` beside `turnComplete`; a filler's
  `turnComplete` arrives with `IN_PROGRESS` and the task ends at `IDLE`. The thinking guide's sample
  also puts `interactionStatus` at the top level beside `toolCall`; the session reads both places and
  ends a turn only when the status is not `IN_PROGRESS`: a `turnComplete` with `IN_PROGRESS` waits,
  and the `IDLE` that follows, alone or beside a `toolCall`, reports it. Thoughts arrive as
  `thought: true` text parts (`NFKRealtimeTextKindReasoning`) when the caller asks for them. The
  guide shows no `includeThoughts` sample; the caller puts it in `extraConfiguration` as
  `generationConfig.thinkingConfig.includeThoughts`, the field generateContent's `ThinkingConfig`
  documents, and the session sends it as given.
- An extra `generationConfig` merges into the session's own (responseModalities, speechConfig), so
  `thinkingConfig` reaches the setup without dropping the rest.
- `gemini-3.5-transcribe-live`: `responseModalities: [TEXT]`, `inputAudioTranscription {languageCodes,
  customVocabulary, mode: VERBATIM | SMART}`; `serverContent.interimInputTranscription` is speculative
  and `inputTranscription` final. Ten-minute sessions, no diarization. (Vertex names it
  `…-live-preview`.)
- `gemini-3.5-live-translate-preview`: `generationConfig.translationConfig {targetLanguageCode,
  echoTargetLanguage}`; the source is detected; audio in only, no instructions or tools.
- Live Avatar (`setup.avatarConfig {avatarName | customizedAvatar {imageData, imageMimeType}}`,
  `responseModalities: [VIDEO]`) is documented for Vertex AI only, and no frame shape is published. The
  session hands any non-audio `inlineData` to `mediaHandler` with its MIME type.
- Ephemeral tokens: `POST /v1beta/auth_tokens` (server side, with the real key) returns a `name` of the
  form `auth_tokens/…`. A session whose `apiKey` has that prefix connects to
  `…v1beta.GenerativeService.BidiGenerateContentConstrained?access_token=…`. The ephemeral-token
  guide (updated 2026-09-15) says tokens work only with `v1beta`, for minting and for the socket, and
  the SDK's v1alpha-only warning is older than that page. Minting is not wrapped: it belongs on the
  app's server, beside the key.
- New server fields read through `eventHandler` only: `voiceActivity`, `waitingForInput`,
  `turnCompleteReason`, `setupComplete.sessionId`, `goAway`, `sessionResumptionUpdate`,
  `toolCallCancellation`.
- Lyria RealTime: still `lyria-realtime-exp` only; `RESET_CONTEXT` applies a bpm or scale change.

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
preset; `backendForProvider:apiStyle:apiKey:modelName:` reaches Gemini's native Veo. Moving the default between
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
otherwise a caller who folded OpenAI tools by name gets double-wrapped. **Tool conversations
(2026-10-10):** a conversation in the chat shape (assistant `tool_calls`, `role: tool` answers; see
`core-runtime-notes.md`, "Tools and the tool loop") converts per backend. OpenAI chat: sent as is,
with an answer's error flag as an `Error: ` text prefix and a tool's images in a user turn after the
answers (a `tool` message carries text only). Anthropic: calls → `tool_use` blocks after the turn's
text; a run of answers → `tool_result` blocks (`is_error`, image blocks in `content`) opening the next
user turn, which absorbs a following user message (conversion runs after the media attach, so request
images stay on the real user turn). Responses: calls → `function_call` items, answers →
`function_call_output` (`Error: ` prefix; `input_text` + `input_image` parts for images). Interactions:
calls → `function_call` steps, answers → `function_result` (`call_id`, `name` looked up from the call
when the answer lacks one, `is_error`, image content). Every chat backend returns
`NFKOutputAssistantMessage` with its record: OpenAI chat keeps `content`, `reasoning_content`,
`reasoning`, `reasoning_details`, and `audio.id` (DeepSeek and OpenRouter read reasoning back during
tool use); Anthropic its content blocks minus the forced `structured_output` call (nothing answers
it); Responses its whole `output` (reasoning items by id, which `store` keeps); Interactions every
step but `user_input` (thought `signature`s). **Schema:**
`NFKParameterJSONSchema` → `response_format: {type: json_schema, json_schema: {name: response, schema}}`
(no `strict`, which demands `additionalProperties: false` throughout) / Anthropic has no response
format on most models, so it is a forced tool `structured_output` (`tool_choice: {type: tool, name}`) whose `input` is
`NFKOutputStructured` and is not listed as a tool call. JSON is promoted to `structured` only when JSON
was asked for (schema, or a folded `response_format` of type `json_object`/`json_schema`) — JSON-looking
text is not guessed at. `NFKInputImages` attaches further images after `NFKInputImage`. **Retry:** the
transport's blocking send retries 429/502/503/504/529 after `Retry-After` (seconds; an HTTP-date falls to
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
subclass (OpenAI `/v1/videos` first; the styles for Gemini, xAI, Together, and OpenRouter are under
"Video styles"): JSON submit, or
multipart with `input_reference` when `NFKInputImage` is present — the base gained the
`submitRequestForRequest:` hook for that and `failureReasonFromStatusResponse:` so the service's
`error.message` reaches the job — percentage `progress` → fraction, poll every 5 s, then get
`/videos/{id}/content` → `.mp4` `NFKVideoAsset`. `NFKRemoteReranker` (`/rerank`; the factory accepts
together, openrouter, llamacpp, and vllm, with together and openrouter verified; results arrive in relevance order and are put back in the documents' order) and
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
Bearer path (the style is `NFKRemoteAPIStyleSystemOne`, and `authorizeRequest:apiKey:style:` treats every style but
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
`NFKInputState` / `NFKInputQuestions` in, `NFKOutputAnswers` out, `NFKDecisionQuestion` (five
factories over the three question types; `dictionaryRepresentation` is the wire shape, and a dictionary already in that shape passes
through) and `NFKDecisionAnswer` (`answerWithDictionary:`, secure coding like the rest of the result
family, `raw` for what the type does not read). The backend accepts `NFKInputPrompt` and then
`NFKInputMessages` as the state when `NFKInputState` is absent, folds any other parameter into the
body under its own name without overriding `model` / `state` / `questions`, declares no supported
parameter key (nothing is sampled), and puts the whole reply under `NFKOutputStructured` beside
`NFKOutputUsage`. The Swift importer turns `+answerWithDictionary:` into `NFKDecisionAnswer(dictionary:)`
and leaves the question factories as static methods (`choiceQuestion(withInstructions:options:)`),
which the Swift example pins.

**OpenAI's Decisions API (`NFKOpenAIDecisionsBackend`).** Announced at DevDay on 2026-09-29 as a
limited preview, then opened to every developer as a public beta on 2026-10-06. `POST
https://api.openai.com/v1/decisions` with a Bearer token takes `{model, input, questions,
safety_identifier?}` and answers `{model, answers, usage}`. The schema here comes from the
openai-python SDK's generated types (`src/openai/types/decision*.py`, v3.28.0), because the guide at
`developers.openai.com/api/docs/guides/decisions` shows only excerpts and no API reference page is
published. The differences from System One:

- `input` is a string or an array of user messages (`{role: user, content}`, where content is a
  string or `input_text` / `input_image` parts). Images are base64 data URLs only: no hosted URL, no
  `file_id`, at most 128 image parts a request. Non-user roles, audio, and files are rejected.
- `questions` is an array. Each question has an optional `name` that must be unique; `predicate`
  takes only instructions, `choice` takes `choices: [{value, description?}]` (a value may be a JSON
  boolean, distinct from the string), and `score` takes `levels: [{label, description?}]`, lowest first.
- Each answer echoes `name`. `predicate` → `probability`; `choice` → `choice`, `probabilities: [{value,
  probability}]`, `confidence`; `score` → `score` (the probability-weighted level index, from 0),
  `probabilities: [{value, label, probability}]`, `confidence`; `refusal` → `name` only (the host
  declines one question without disclosing its score).
- `usage` is the Responses shape (`input_tokens`, `input_tokens_details.cached_tokens`,
  `output_tokens`, `output_tokens_details.reasoning_tokens`, `total_tokens`).
- Only `gpt-6-luna` is accepted at release. Pricing is $0.10 per million input tokens, with output and
  cache reads and writes free. The guide documents no error codes, rate limits, or size limits.

The backend maps this onto the engine-neutral vocabulary rather than adding a type:

- The question's `NFKInputQuestions` identifier is its `name`, and questions go in sorted identifier
  order, so an unnamed answer is matched by position.
- A noul is sent as a predicate. Its true/false meanings have no field, so they follow the
  instructions as `True means:` / `False means:` lines.
- A score's level descriptions go as labels. A dictionary state is sent as its sorted-key JSON text,
  and `NFKInputMessages` as a `role: content` transcript.
- `NFKDecisionAnswer.answerWithDictionary:` reads `predicate` and the array forms. A refusal names no
  type, so the backend builds it with `refusalForType:raw:` from its question's type.
- No `NFKDecisionType` case was added for a refusal. The InferKitMLX decision models switch over the
  enum, and a new case warns there. `refused` is a flag on the answer instead.
- The Decisions API is a route on the `openai` preset, like `/moderations`, so it has no
  `NFKRemoteAPIStyle` value. `+backendForProvider:apiKey:modelName:` answers for `openai` only, at
  its own base or a re-pointed one.

A standard key got **403** `"Decision API is not enabled for this user."` (`invalid_request_error`)
during the preview (eesel AI, 2026-10-01/02). That body reaches the caller through
`NFKRemoteTransport`'s error. The answer shape is asserted live only by
`-[NFKOpenAIDecisionsBackendTests testALiveEndpointAnswersATypedDecision]`, gated on
`INFERKIT_OPENAI_API_KEY`.

**Gemini's voice library (`NFKGeminiVoiceLibrary`).** The Voices API shipped on 2026-09-22 beside the
Gemini 3.8 Flash TTS models. The field names come from the google-genai Python SDK's generated types
(`google/genai/_gaos/types/voices/`, v2.29.0), because the guides
(`ai.google.dev/gemini-api/docs/voice-design`, `voice-replication`, `speech-generation#voice-library`)
show requests but no response bodies. Shape, under `https://generativelanguage.googleapis.com/v1beta`,
with `x-goog-api-key`:

- `GET /voices` reads the filters `language_code`, `region_code`, `accent`, `gender`, `pitch`,
  `persona`, `context`, `type` (prebuilt, prompted, replicated), `search`, and `page_size` (default 50,
  at most 1000), plus `page_token`. A repeated filter matches as alternatives, and different filters
  must all match. It answers `{voices, next_page_token}`, custom voices first.
- `GET /voices/{id}` and `DELETE /voices/{id}`. The delete answers `{}`.
- `POST /voices` takes `{store, voice}`. `voice` carries `type`, `model` (optional; the latest
  voice-design model otherwise), `display_name`, the discovery fields, and either `prompted: {input}`
  or `replicated: {source_audio, consent_audio}`, each `{mime_type, data}` with data in base64.
- A reply voice carries `id` (stored) or `key` (`voicekey_…`, replicated with `store: false`), plus
  `sample_audio` (a WAV preview on create and get, not in a listing), `expire_time`, and `usage`.

Rules from the SDK docstrings: `store` defaults to false, and a prompted voice requires it true, so
`designVoiceWithDescription:` always stores. A stored voice expires a year after its last use, a key
after seven days. A project holds 200 stored voices, and going over answers `RESOURCE_EXHAUSTED` (429,
retried by the transport). The consent clip is the same adult reading Google's per-locale consent
statement, and the source clip is 10 to 30 seconds. Both are recommended as 24 kHz mono 16-bit WAV.

The library reads a clip's type from a file's extension or from the bytes' signature (`RIFF`, `fLaC`,
`ID3` or an MPEG frame sync). Anything else is refused before sending, because the service needs the
type named. The voice identifier is the id, or the key when there is no id. That string goes into
the Interactions backend's `speech_config[].voice` unchanged; a custom voice works in a
single-speaker reply only.

`speech_config` is typed in the SDK (`_gaos/types/interactions/generationconfig.py`, v2.29.0) as a list
of `{voice, speaker?, language?}` or the object `{speakers: [...]}` (`speakerconfig.py`). The speech
guide's multi-speaker example uses the object, with an optional `mode: conversational` that the SDK
type omits. So a `speakers` array parameter goes out as `{speakers}` without `mode`, and a
`speakers` dictionary passes through, which is how a caller sends `mode`. Neither form has been
checked against a live call. The live test (`testALiveListingNamesThePrebuiltVoices`) is
gated on `INFERKIT_GEMINI_API_KEY`.

**OpenCode gateways (`opencode`, `opencode-go`, 2026-10-13).** OpenCode runs two inference gateways
behind one console key: Zen (pay-as-you-go, `https://opencode.ai/inference/v1`) and Go (subscription,
`https://opencode.ai/inference/go/v1`). The earlier "deliberately absent" note probed
`opencode.ai/api/…`, which is the site and answers 404; the gateways were never under `/api`. The
older `opencode.ai/zen/v1` and `opencode.ai/zen/go/v1` bases still answer the same model lists, and
the presets use the documented `/inference/` bases.

- Each model is served on its maker's protocol. OpenCode's docs (`opencode.ai/docs/zen`,
  `opencode.ai/docs/go`) publish a table of model → endpoint → AI SDK package. The presets carry the
  routes that are not chat completions in `modelAPIStyles`; every other model goes to
  `/chat/completions`.
- The tables differ between the gateways for the same name: `qwen3.8-max`, `minimax-m3`, and
  `minimax-m2.7` are chat on Zen and Messages on Go. Model-family prefixes do not predict a route.
- `GET /models` returns ids only (`id`, `object`, `created`, `owned_by`), so the route cannot be read
  from the service. models.dev (`https://models.dev/api.json`, providers `opencode` and `opencode-go`)
  carries the same routes as a per-model `provider.npm` and agreed with the docs on every model both
  listed on 2026-10-13. models.dev also lists about 37 retired Zen models, so the docs table is the
  source and models.dev the cross-check. `muse-spark-1.2-contributor-free` is live and absent from
  the docs table; models.dev routes it to Responses, and the preset follows.
  `claude-opus-5-5-backup` is live and in neither source, so it has no route.
- Gemini models (Zen only) use `@ai-sdk/google`, which posts to `{base}/models/{id}:generateContent`
  with an `x-goog-api-key` header. The factory builds an `NFKGeminiGenerateContentBackend` on
  `{base}/models` for them. Whether OpenCode takes the key in that header, and whether Zen's
  `/chat/completions` also answers a Gemini id, is unknown without a key.
- Authentication: the docs show a Bearer token for chat and System One. `NFKAnthropicBackend` sends
  `x-api-key` on the Messages route, which is what `@ai-sdk/anthropic` sends. No request has been
  made with a key; `testOpenCodeAnswersOnEachRoutedProtocol` (gated on `INFERKIT_OPENCODE_API_KEY`)
  lists both gateways and asks one free model per text protocol on Zen.
- Every per-capability factory except `NFKRemoteResponsesBackend` returns nil for both presets:
  OpenCode documents no embeddings, audio, image, file, or usage routes.
- Refreshing the table: extract the endpoint tables from both docs pages, diff them against the live
  `/models` lists and models.dev, and update `openCodeZen` / `openCodeGo` in `NFKRemoteProvider.m`
  with the date. A caller reaches a newer model before a release through
  `providerWithAPIStyle:forModelName:`.

**Deliberately absent.** Midjourney has no official public API (its API host does not resolve), so
shipping a preset would imply one exists. Codex is OpenAI's coding agent, not a separate endpoint; it
is the `openai` preset.

`NFKRemoteProviderTests` stubs the transport to assert Anthropic's request shape without a network, and
carries two live tests: one gated on `INFERKIT_LIVE_LOCAL_MODEL` that runs against a local server when
one is listening, and the OpenCode one gated on `INFERKIT_OPENCODE_API_KEY`.

**Serving (`NFKInferenceServer`, 2026-09-30).** The server side of the remote subsystem: another
machine's remote clients reach a model this process hosts. Files:

- `NFKHTTPServer.h/.m` (private): HTTP/1.1 over Network.framework (`nw_listener`, available from the
  core's floor). Keep-alive, `Expect: 100-continue`, Content-Length and chunked request bodies, chunked
  responses for server-sent events, multipart parsing, one request at a time per connection, a
  120-second idle timeout, and a 64 KB head limit (431). The client closing its side while a reply is
  pending is treated as a disconnect, which cancels the run; a client that half-closes and then reads
  is not served. `dispatch_data_t` is used directly as `NSData`.
- `NFKServedOpenAI.h/.m` (private): each OpenAI route inverts the core client that speaks it, so the
  hosted backend sees the keys the caller set. Chat content parts become `NFKInputImage` (first) plus
  `NFKInputImages` (the rest), `NFKInputAudio`, `NFKInputVideo`, and `NFKInputDocument(s)`, and a lone
  user turn also arrives as `NFKInputPrompt`. `modalities` containing `image` maps back to
  `NFKModalityImage`, which is what a default-modality request sends (`NFKInferenceRequest` defaults to
  image, and `NFKRemoteBackend` then asks for `modalities: [image, text]`). Remote media URLs are
  refused: the server fetches nothing for a client. Speech converts the hosted clip with `AVAudioFile`
  to wav, flac, aac (written as m4a), m4a, caf, or raw pcm; MP3 and Ogg Opus have no system encoder
  and are served only when the model produced them.
- `NFKServedAnthropic.h/.m` and `NFKServedResponses.h/.m` (private, 2026-10-15): Anthropic's Messages
  API and OpenAI's Responses API, each the inverse of the core client that speaks it
  (`NFKAnthropicBackend`, `NFKRemoteResponsesBackend`), with the routes' serving code and a stream
  writer class per family. See "Messages and Responses routes" below.
- `NFKServedRun.h/.m` (private): the run a model's queue holds, and the unmeasured token-count run.
  `NFKInferenceServerPrivate.h` (private) gives a route family what it reaches of the server: the body,
  `modelNamed:error:`, the error reply in a wire's shape, and `enqueue:on:request:response:`.
- `NFKInferenceWireCoding.h/.m` (private): the native route's tagged JSON (`"$nfk"`), shared by server
  and client. Pixel buffers cross as each plane's rows with padding, so any format crosses without a
  bytes-per-pixel table; the decoder first checks the planes against the format's own description
  (`CVPixelFormatDescriptionCreateWithPixelFormatType`) so the buffer it allocates is no larger than
  what was sent. Multi-arrays and PCM buffers are checked the same way before allocation.
- `NFKInferenceServer.m`: registry, per-model FIFO (`NFKServedModel`), routing, auth, streaming.
  A synchronous backend holds its slot until its call returns, cancelled or not, because it cannot be
  interrupted and the next run must not overlap it. A job backend releases its slot when the job ends.
- `NFKRemoteInferKitBackend.m`: the native client. `respondsToSelector:` answers NO for the two
  supported-key properties until `prepareWithError:` has read them, which is the protocol's
  "not declared" answer.
- `NFKServedHostStatus.h/.m` (private): the status route's host object. Owned by the server from
  `init`, so the first CPU-usage reading covers the time since the server was created.
- `NFKInferKitDiscovery.h/.m` (private): `DNSServiceBrowse` then `DNSServiceResolve` on one serial
  queue for the timeout. Resolving yields the advertising machine's host name (`Studio.local`), which
  `NSURLSession` resolves over mDNS, so no scoped IPv6 literal is needed. The TXT record carries `path`,
  `tls`, `auth`, `mcp`, and `version`.
- `NFKServedMCP.h/.m` (private, 2026-10-11): MCP over Streamable HTTP at `/mcp`, outside `/v1`
  because MCP clients take the endpoint as one URL. Choices:
  - Sessions: see "MCP sessions" below. A POST reply is plain JSON, never an event stream.
  - A batch answers in order. Notifications get 202. A parse error is 400 with -32700, an unknown tool
    is -32602 (the spec's protocol error), and an unknown method -32601.
  - A failing call is an `isError` result, never a protocol error, so the client's model reads it.
  - Versions: 2025-11-25, 2025-06-18, 2025-03-26, and 2024-11-05. `initialize` echoes a version the
    client asks for when it is one of these, and the newest otherwise. An `MCP-Protocol-Version`
    header outside them is 400.
  - `Origin`: absent is allowed, since a non-browser client sends none. localhost, 127.0.0.1, `::1`,
    or the request's own `Host` is allowed. Anything else, `null` included, is 403, against DNS
    rebinding.
  - A call runs on a user-initiated global queue through `invokeWithArguments:`, so the schema check
    runs server-side too.
  - Content: `NFKMCPCallResult` writes text, then each `imagePNGs` image, then `NFKOutputAudio`'s
    file as `audio` with a MIME type by extension, `NFKOutputVideo` as an embedded resource
    (`inferkit:///<file name>`), and a dictionary `NFKOutputStructured` as `structuredContent`.
  Only `addTool:` tools are listed, which is the user's decision of 2026-10-10: a hosted model is not
  exposed until the host writes it a tool with a description.
- `NFKMCPClient.m` (public): blocking and serialized by a lock, over either transport. It
  initializes on first use, sends `notifications/initialized`, and pages `tools/list` by
  `nextCursor`. A JSON-RPC error is nil plus an error whose description is `<method>: <message>`.
  `NFKMCPClientTool` re-runs the schema check locally before the remote call. `disconnect` sends
  DELETE for an HTTP session the server assigned (the served route answers 204 and ends it) or stops
  a stdio server, and forgets the session either way.
  - Streamable HTTP: each event's `data:` lines join with newlines into one message, and the reply
    whose `id` matches is taken, past any progress notification. The session id the server assigns
    goes back with each later request, beside `MCP-Protocol-Version`.
  - stdio (2026-10-15, the 2025-06-18 spec's stdio transport): one message per line, written with
    `NSJSONWritingWithoutEscapingSlashes`. `NSJSONSerialization` writes a newline inside a string as
    `\n`, so a message never holds a raw newline. `apiKey` and `additionalHeaders` do not apply.
  - Replies match by `id` through a pending table under its own lock. The reader runs on the
    subprocess's event queue and never takes the call lock, so it delivers while a call waits. A reply
    is stored under the pending lock, so a wait that times out at the same moment still reads it.
  - A line that is not JSON is skipped. A server notification is ignored.
  - A server request: `ping` → `{}`; any other method → `-32601` "InferKit's MCP client does not
    handle <method>", since the client declares no capabilities. Answers go out on a separate serial
    queue: written from the reader, a server blocked on a full stdout pipe and a client blocked on a
    full stdin pipe would deadlock.
  - Deadlines (2026-10-15): `tools/call` waits `timeout` (600 s), and every other request waits
    `protocolTimeout` (30 s), on both transports. A value that is not positive takes its default, so
    no wait is unbounded. A timeout removes the pending entry, sends `notifications/cancelled` with
    `requestId`, and returns `kNFKError_RemoteUnreachable`. The late reply finds no entry and is
    dropped. Measured cause: the Swift example's stand-in server had no `ping` branch, so the
    example's `ping()` waited the single 600 s timeout every request then used, and a test run
    sampled it 9.5 minutes in, still in `semaphore_timedwait_trap`. A ping or a handshake now
    gives up after 30 s.
  - An exit or a failed launch fails every pending request with `kNFKError_RemoteUnreachable`, naming
    the exit status or signal and the server's last nonblank standard-error line, and marks the
    session ended.
  - A server that exits stays down (decided 2026-10-15). A call launches the process
    only when `launchesOnNextCall` is set: at creation, after `connectWithError:`, and after
    `disconnect`. Any other call on a process that is not running returns the exit's error at once.
    The restart policy relaunches without the client; the next call finds it running and initializes
    again, since the exit marked the session ended.
  - Environment (decided 2026-10-15): the client sets
    `serverProcess.inheritedEnvironmentVariables` to HOME, LOGNAME, PATH, SHELL, TERM, USER, TMPDIR,
    LANG, and LC_ALL, which is the MCP TypeScript SDK's `getDefaultEnvironment()` list for macOS plus
    the last three. A third party's MCP server should not see an application's keys.
    `inheritsFullEnvironment` sets it to nil. `NFKSubprocess` alone inherits everything, since a
    plugin host is InferKit's own child.
  - The client reads the process through `ownerLineHandler` and `ownerStateHandler`
    (`NFKSubprocessPrivate.h`), which run before the public `lineHandler` and `stateHandler`, so an
    application observes the same traffic without displacing the client. The client's `dealloc`
    stops its server without waiting.
- `NFKSubprocess.h/.m` (public, macOS only, 2026-10-15): the launcher and supervisor under the stdio
  client, built to host a plugin process InferKit launches as well. Choices:
  - The launch uses `posix_spawn` for flags `NSTask` does not expose. `POSIX_SPAWN_CLOEXEC_DEFAULT`
    leaves only the three `dup2` targets open in the child. `POSIX_SPAWN_SETPGROUP` with 0 gives the
    child its own process group, so a stop reaches what the server started (`npx` → `node`).
    `SETSIGDEF` for every signal and an empty `SETSIGMASK` keep a host's ignored SIGPIPE out of the
    child.
  - `posix_spawn_file_actions_addchdir_np` sets the working directory. The macOS 26 SDK deprecates it
    for the name without `_np`, which earlier systems lack, so the call silences the deprecation. A
    directory that does not exist fails the spawn with ENOENT, which reads as `FailedToLaunch`.
  - A command without a slash is searched on the PATH of the child's merged environment (or
    `/usr/bin:/bin:/usr/sbin:/sbin` without one), then in `NFKAgentCLIToolDirectories()`
    (`NFKAgentCLIPrivate.h`), the list `NFKAgentCLI` searches after PATH. A command found only there
    has its directory appended to the child's PATH, since `npx` and the other Node launchers run
    `#!/usr/bin/env node`. A command found nowhere fails with ENOENT and a description naming every
    directory searched. An absolute path, or a path relative to the working directory, is used as
    given.
  - Parent ends: `FD_CLOEXEC`, `O_NONBLOCK`, and `F_SETNOSIGPIPE` on standard input, so a write to an
    exited child returns EPIPE and raises no signal.
  - Output: two read sources on one serial user-initiated queue. Framing scans only the new bytes,
    since the pending buffer never holds a newline. A carriage return before a newline is dropped,
    output left without a newline arrives as a final line, and a line past `maximumLineLength` is
    dropped through its newline.
  - Exit: a `DISPATCH_SOURCE_TYPE_PROC` source and `waitpid(WNOHANG)`, plus one `waitpid` queued right
    after the source starts, for a child that exited before it registered. `Exited` is entered after
    both pipes reach EOF, or 0.5 s after the exit when a grandchild holds them, so every line precedes
    the state.
  - Writes block under a lock and poll `POLLOUT` in 100 ms slices, up to `writeTimeout` (30 s; a value
    that is not positive takes 30). `closeStandardInput` sets a flag before it takes the lock, so a
    writer the child is not reading never holds it up for long. The blocking `stop` waits at most both
    stop timeouts plus 5.5 s and returns whether the process exited.
  - Stop: close standard input → `stopInputTimeout` (2 s) → SIGTERM to the group and the process →
    `stopTerminateTimeout` (3 s) → SIGKILL. Signals go out only before the run is reaped, so a reused
    process identifier is never signaled.
  - Restart (`OnCrash`): a nonzero status, a signal, or an unreadable status, unless a stop caused the
    exit. The delay is `restartDelay`·2ⁿ, capped at `maximumRestartDelay`, for at most
    `maximumRestarts` consecutive crashes. A run that lasts `restartResetInterval` resets the count,
    and so does an explicit start, which also replaces a pending relaunch. `restartPending` is set
    before `Exited` is reported.
  - Lifetime: the dispatch sources retain the subprocess, so a running one outlives its owner until its
    process exits.
  - Tests: `NFKSubprocessTests` (3), `NFKMCPStdioTests` (13), and `NFKMCPClientDisconnectTests` (1).
    The fake server is a POSIX shell script the test writes to a temporary directory and runs as
    `/bin/sh <script> <mode>`. `/bin/sh`, `/usr/bin/sed`, `/bin/sleep`, and `/bin/pwd` ship with
    macOS, while `python3` is a Command Line Tools stub and a helper executable has no SwiftPM
    test-target home. The script matches requests by substring and answers with `printf`. Modes
    `ignore-input-close` and `ignore-sigterm` (`trap '' TERM`, which the `sleep` it runs inherits)
    drive the stop escalation, and `silent-ping` leaves `ping` unanswered for the protocol timeout.
    A stand-in server in an example answers every method the example sends, `ping` included, and
    the example sets short timeouts.
- `NFKAgentCLI.h/.m` (public, macOS only, 2026-10-11): registers an MCP server with an agent CLI
  by running its `mcp` subcommands through `NSTask`. It edits no configuration file itself; the
  user left config-file agents (Cursor, VS Code's remove, Claude Desktop) out for now. Each CLI
  differs, measured against Claude Code 2.1.90 and read from the other CLIs' sources:
  | CLI | add | duplicate name on add | read back | absent |
  |---|---|---|---|---|
  | `claude` | `mcp add --transport http --scope S N URL --header H` | refused, exit 1 | `mcp get N` text: `Scope:`, `Status: ✓ Connected`, `URL:`, headers under `Headers:` | exit 1, "No MCP server found with name" |
  | `codex` | `mcp add N --url URL --bearer-token-env-var V` (no header flag, global only) | overwritten | `mcp get N --json`: `transport.{type: streamable_http, url, bearer_token_env_var, http_headers}` | non-zero, "No MCP server named" |
  | `gemini` | `mcp add --scope S --transport http N URL --header H` (writes `url` + `type`) | overwritten | `mcp list` text only, no headers, connects to each server | not listed |
  | `qwen` | same flags as gemini (writes `httpUrl`) | overwritten | `mcp list`, same line format | not listed |
  | `agy` | `mcp add --type http N URL` (flags before the name; writes `serverUrl`, `disabled: false`, `headers` into `~/.gemini/config/mcp_config.json`) | overwritten ("Add or update") | `mcp list` table `NAME TYPE STATUS COMMAND/URL`, rows `N http enabled\|disabled URL`; does not connect, shows no headers | `remove` exit 1, `Error: MCP server "N" not found`; empty list prints `No MCP servers configured.` |
  - Tokens: Claude Code stores `Authorization: Bearer ${V}` literally and expands it when it connects
    (local scope included, which the live test proves by a connect with the right key and a refused
    one with the wrong key). Gemini CLI and Qwen Code resolve `${V}` in every settings string at load.
    Codex takes the variable name. No path passes a literal key.
  - Antigravity (`agy`, added 2026-10-13, measured against agy 1.3.5 under a scratch `HOME`):
    Google's coding-agent CLI. Its `~/.gemini/config/mcp_config.json` is shared by
    the Antigravity desktop app, the IDE, and the CLI (Google's codelab), so one user-scope
    registration reaches all three; the app picks it up on Refresh or Restart under Installed MCP
    Servers. A probe listener received `Authorization: Bearer ${INFERKIT_PROBE_TOKEN}` verbatim when
    agy connected, so agy does not expand variables; the user decided (2026-10-13) that a keyed
    registration is refused, since agy would send the literal key. agy connects to configured MCP
    servers before its own Google sign-in; `mcp add`, `list`, and `remove` need no sign-in. agy has no
    scope flag; the workspace `.agents/mcp_config.json` is not written by it.
  - Repeatable header options (Commander's `-H <header...>`, yargs arrays) take the words after
    them, so the header goes last on the command line.
  - Link: Claude Code checks with `get`, returns when current, else `remove --scope` then `add`.
    Codex checks with `get --json` and adds (overwrite). Gemini, Qwen, and Antigravity add without
    a check: Gemini's and Qwen's lists show no headers and connect to every server, and `agy mcp add`
    adds or updates.
  - Claude's `get` finds a name in any scope; a registration whose `Scope:` names another scope
    reads as absent for the requested one. Remove "not found" counts as success.
  - A GUI app's PATH is `/usr/bin:/bin:/usr/sbin:/sbin`, so the search adds (through
    `NFKAgentCLIToolDirectories()`, which `NFKSubprocess` shares) `~/.local/bin`,
    `~/.claude/local`, `/opt/homebrew/bin`, `/usr/local/bin`, npm-global, Bun, Volta, Cargo, and nvm
    bins, and puts them on the child's PATH, since the Node CLIs run `#!/usr/bin/env node`.
  - A timed-out command gets SIGTERM, then SIGKILL after 2 s. A grandchild can keep the pipes open
    after the child dies, so the pipe readers stop 2 s after exit regardless.
  - `APP_SANDBOX_CONTAINER_ID` in the environment means the App Sandbox, which cannot run the CLI.
  - Tests: a stub executable per preset records argv and the working directory and answers each
    subcommand in that CLI's format. The live Claude Code test runs only with
    `INFERKIT_LIVE_AGENT_CLI=1`, since it writes a local-scope entry to `~/.claude.json` (removed at
    the end).
- **Provider-run MCP (`NFKParameterMCPServers`, 2026-10-11).** Each server is
  `{url, name, authorizationToken?, allowedTools?, description?}`.
  - Claude: `mcp_servers: [{type: url, url, name, authorization_token}]` plus one
    `{type: mcp_toolset, mcp_server_name}` per server in `tools`. An allowlist becomes
    `default_config: {enabled: false}` with `configs: {tool: {enabled: true}}`. The
    `anthropic-beta: mcp-client-2025-11-20` header goes only with a server. The docs list a newer
    `mcp-client-2026-09-15` beta for pinned tool lists, which is not used. Replies' `mcp_tool_use`
    blocks pair with their `mcp_tool_result` like server tools.
  - Responses: `{type: mcp, server_label, server_url, require_approval: never, authorization?,
    allowed_tools?, server_description?}`. `mcp_call` items read as server tools through the `*_call`
    rule; `mcp_list_tools` is skipped.
  Both providers fetch the URL themselves, so it must be public HTTPS. Shapes from the providers'
  current docs (platform.claude.com MCP connector; developers.openai.com tools-connectors-mcp), read
  2026-10-11.

Measured on this machine (`NFKInferenceServerTests`, 34 tests, and `NFKBalancedBackendTests`, 20, every one against a real listener):
`nw_parameters_set_required_interface_type(…, nw_interface_type_loopback)` binds IPv4 and IPv6 loopback
together, and a connect from the machine's LAN address is refused. Bonjour advertises and resolves
inside `swift test` with no permission prompt, and the resolved host name reaches the server.

Traps met while building it:

- `NSJSONSerialization` does not return every double identical: a NaturalLanguage sentence embedding
  came back up to 2.78e-17 off per component. The native route packs an all-floating array as
  little-endian float32 or float64 words, and a scalar that would not survive the round trip travels
  as its 17 significant digits (`strtod` reads them back exactly). Measured afterwards: the same
  embedding served natively equals the in-process vector bit for bit. The OpenAI `/embeddings` route
  keeps JSON numbers (or base64 float32 on request), which is what its clients expect.
- `NFKSpeechSynthesisBackend` delivers its buffers on the main run loop, so a probe that blocks the
  main thread on a served speech call waits out the backend's 120-second deadline. A server app's main
  run loop runs; a test calls from a background queue and waits on an expectation.
- `[@"text" hasPrefix:@""]` is NO. The streamed-delta helper answered "no extension" for the first
  delta, so every chat stream fell back to one piece at the end. An empty prefix is special-cased.
- The server removed the temporary files it wrote for inline media before encoding the reply, so a
  result that names one of them (an echo, a restoration that writes beside its input) failed to
  encode. The run removes them after its completion handler has built the reply.
- `NFKRemoteWriteMediaFile` joins the extension into the path unchecked, so a client-named extension
  of `../x` leaves the temporary directory. Every server write goes through `NFKServedFileExtension`
  (one to eight letters and digits, else `bin`). The client-side callers pass their own extensions.
- The analyzer flags an `NSError **` out-parameter written inside an `@autoreleasepool`
  (`osx.cocoa.AutoreleaseWrite`); write to a strong local outside the pool and copy it out after.
- `NFKRemoteTransport`'s blocking send retries 502, 503, and 504, so a served "not ready" (503) is
  retried like a gateway error. Tests that expect a failing status set `retryAttempts` to 0.

**Messages and Responses routes (2026-10-15).** Asked for so Claude Code and Codex run on a hosted
model. `POST /v1/messages`, `/v1/messages/count_tokens`, `/v1/responses`, and `/v1/responses/input_tokens`,
all `NFKServerKeyScopeInference`. `NFKInferenceServer.m` holds only the routing; each family's serving
code is in its own file and reaches the server through `NFKInferenceServerPrivate.h`.

- One conversation shape: both families convert to the OpenAI chat shape `NFKInputMessages` takes, built
  with `NFKToolMessages` (`assistantMessageWithText:toolCalls:…` for a turn's calls,
  `messageForToolCallIdentifier:name:result:` for each answer, so a tool_result's `is_error` and images
  ride under `NFKMessageMetadataKey`). A tool answer's name is found from the call earlier in the same
  request; an answer to a call the request does not carry is 400, as both providers refuse it.
- Messages: `system` (a string or text blocks joined by a blank line) becomes the first turn. A user
  turn's `tool_result` blocks become tool turns ahead of its text. `thinking` and `redacted_thinking`
  blocks are dropped: their signatures verify only at Anthropic. Server tools (a `type` and no
  `input_schema`) are not offered. `tool_choice` `auto`/`any`/`none` pass on as the chat route's
  `tool_choice` (`auto`/`required`/`none`), and `disable_parallel_tool_use` as `parallel_tool_calls: false`.
- A forced tool (`tool_choice: {type: tool}`) is structured output: its `input_schema` becomes
  `NFKParameterJSONSchema`, the other tools are dropped, and the reply is a `tool_use` of that tool whose
  input is `NFKOutputStructured` (or the text parsed as a JSON object). This is also what makes
  `NFKAnthropicBackend`'s schema path (a forced `structured_output` tool) work against the server; when
  streamed, the text deltas go out as the tool's `input_json_delta`.
- Thinking: `{type: enabled, budget_tokens: N}` → effort `"N"` (the contract passes other strings
  through, and `NFKAnthropicBackend` reads a numeric string as a budget); `output_config.effort`
  low/medium/high → light/moderate/deep, other levels unchanged; `{type: adaptive}` without an effort →
  deep, the API's default; `disabled` → none. `NFKAnthropicBackend` on a name outside its known
  families sends adaptive thinking with an effort, and on such a name it drops sampling parameters,
  which is why the interoperability test checks temperature over raw HTTP only.
- Usage: `input_tokens` is the input minus the cached tokens and `cache_read_input_tokens` the cached
  ones, as Anthropic counts. `NFKAnthropicBackend` sums `input_tokens`, `cache_read_input_tokens`, and
  `cache_creation_input_tokens` into `NFKUsageInputTokens`, which the contract defines as cached
  included, and keeps `cache_read_input_tokens` as `NFKUsageCachedTokens`, so a cached count round-trips
  through the route and the client. Every Message carries `usage`, zeros included, since Claude Code
  reads it.
- Stop reasons: `tool_use` with any call; `max_tokens` when the reported output tokens reach
  `max_tokens`; `end_turn` otherwise. The contract has no finish reason, so `stop_sequence` is never
  named. Responses: `incomplete` with `incomplete_details.reason: max_output_tokens` on the same test.
- Responses: `instructions` and every system or developer turn before the first other turn join into
  one system turn, because many chat templates take one system turn, first, and Codex sends its
  instructions and developer turns apart. Item `type` absent means `message`. `reasoning` items are
  dropped (their `encrypted_content` is OpenAI's). Consecutive `function_call` items after an assistant
  message join that turn. A `custom` tool (Codex's `apply_patch`) becomes a function taking
  `{input: string}`, and a call to it returns as `custom_tool_call` with that string; built-in tools
  (`web_search`, `local_shell`, `file_search`, `image_generation`, `mcp`) are not offered. Keys the core
  client sends under their own contract names (`stopSequences`, `topK`, `seed`) pass through, as on the
  chat route.
- Stored responses: none (decision 2026-10-15). Every reply says `store: false`; `previous_response_id`,
  `conversation`, and `item_reference` are 400; `GET`/`DELETE /v1/responses/{id}` are 404. Reasons:
  Codex sends the whole conversation each turn (`store: false`, reasoning replayed from
  `encrypted_content`), so storage buys it nothing; stored responses would need a retention policy, a
  memory bound, and per-key isolation; and a stateless route works the same behind `NFKBalancedBackend`
  and across residency changes. `background: true` is answered at once as completed, which the core
  client takes without polling.
- Streams: `NFKHTTPResponse.sendEventNamed:data:` writes `event:` and `data:` lines; both families name
  the event and repeat it as the data's `type`. Each stream opens at the first event, so a failure
  before it is an ordinary error reply with its status; after it, Messages sends an `error` event and
  Responses `response.failed`. Text and reasoning deltas come from cumulative partials through
  `NFKServedSuffix`, as on the chat route. Responses events carry `sequence_number` from 0, and
  `response.completed` lists the items the stream finished, with the same ids. A thinking block's
  `signature` is the empty string.
- Errors: Messages routes answer `{type: "error", error: {type, message}}`, the type by status (401
  authentication, 403 and 421 permission, 404 not_found, 413 request_too_large, 429 rate_limit, 503
  overloaded, other 5xx api_error, other 4xx invalid_request), with `inferkit_domain` and
  `inferkit_code` beside the message so `NFKRemoteTransport.servedCodeInData:` reads the exact code.
  The route refusal, 405, an unknown `/v1/messages/*` path, and a body that is not JSON take the same
  shape. Responses keeps the chat route's OpenAI error body.
- Keys: on the Messages routes only, `x-api-key` is a second place for the key (Bearer is tried first),
  since Claude Code sends `ANTHROPIC_API_KEY` there; `anthropic-version` and `anthropic-beta` are
  accepted and ignored.
- Model list: `GET /v1/models` and `/v1/models/{id}` with an `anthropic-version` header answer in
  Anthropic's shape (`{data: [{type: "model", id, display_name, created_at}], has_more, first_id,
  last_id}`, `display_name` the hosted name, `created_at` the server's start in RFC 3339) and take the
  key as `x-api-key`, with errors in Anthropic's shape. Without the header the list keeps OpenAI's shape.
- Token counting: `NFKInferenceBackend` gained the optional `tokenCountForRequest:error:` (the selector
  `NFKRemoteTokenCounter` already used). Both counting routes run it as an `NFKServedRun` whose `work`
  calls it, so it takes the model's slot (never overlapping a run, building a model that is not
  resident) and is `unmeasured`: it stays out of the averages, the token totals, and the
  completed/failed counts. A backend without it answers 400 `kNFKError_InferenceUnsupported`.
  Implemented by the MLX language backends (`mlx-models-language.md`, "Token counts"), each counting the
  prompt its run builds (chat template, tools, an image's or a clip's tokens) through the same code
  path, so a count equals the run's reported input tokens;
  `NFKFoundationModelsBackend` on macOS 26.4 / iOS 26.4 with the on-device model (the context check's
  count: prompt, transcript, tools, schema; Private Cloud Compute and older systems throw
  `kNFKError_InferenceUnsupported`); and `NFKAnthropicBackend`, which posts its run body, reduced to
  `model`, `messages`, `system`, `tools`, `tool_choice`, `thinking`, and `mcp_servers`, to
  `…/messages/count_tokens`. `NFKRemoteTokenCounter`'s Anthropic style builds the same body through
  `-[NFKAnthropicBackend countTokensBodyForRequest:error:]` (private, `NFKAnthropicBackendPrivate.h`),
  with an `NFKAnthropicBackend` pointed at the counter's endpoint less `count_tokens`, so the two count
  the same thing; the counter does not send the `anthropic-beta` MCP header, so a request carrying
  `NFKParameterMCPServers` counts through the backend.
- Model names: every route resolves `model` through `modelNamed:error:`, the one place presets will
  extend. Claude Code also sends its background model (`ANTHROPIC_DEFAULT_HAIKU_MODEL`), which must be
  hosted too.
- Tests: `NFKServedAnthropicTests` (24) and `NFKServedResponsesTests` (15), each over a real loopback
  listener with raw HTTP and with the core client pointed at the server (non-streaming, streaming,
  tool round trips, images, schemas, error codes).

**Status and load (2026-10-05).** `GET /v1/inferkit/status` and the `load` object in each
`/models` entry come from `NFKServedModel`. It keeps an array of its running runs, each run's enqueue
and start times on `systemUptime`, and the job's last reported progress.

- The wait estimate gives each slot its running run's remaining time (from progress when the job
  reports one, else the average run minus the elapsed time), then hands each queued run the soonest
  slot for one average run. It is nil while a needed average does not exist yet, and 0 with a free
  slot. A cancelled run still in the queue is not counted.
- Only runs that return a result feed the run average, so a fast refusal does not shorten it. The
  averages are exponential, newest run weighted 0.2, first run taken whole.
- `backendStatus` (2026-10-07) is read on every status request and on every `/models` entry, so an
  adopter answers from a snapshot and never takes a lock a run holds: the MLX language backend
  refreshes its cache figures under its generation lock after each run and reads them under a
  separate lock. `NFKServedJSONObject` keeps strings, finite numbers, booleans, and arrays and
  string-keyed dictionaries of them, eight levels deep; an empty result leaves `status` out. The
  balancer's report: `policy`, `servers` (`base_url`, `discovered`, `hosts_model`, `strained`,
  `outstanding`, `resting_seconds` while backing off, `status_age_seconds`), `conversation_affinity`,
  `conversations`, and `conversation_moves`, counted when a conversation's record changes server.
- Token totals (2026-10-07) are lifetime sums, not averages: `input_tokens` over every completed run
  whose `NFKOutputUsage` names `NFKUsageInputTokens`, and `cached_input_tokens` / `cached_input_share`
  over the runs that also name `NFKUsageCachedTokens`, so a backend that never reports a cached count
  does not dilute the share. A cached count above the input is clamped to it. Lifetime sums let a
  client take the share over any window from two readings; an exponential average could not be
  differenced.
- The `X-InferKit-*` headers come from `NFKHTTPResponse.headerProvider`, read when the head is
  written. A plain reply's head is written after its run released the slot, so `Running` excludes
  it. The provider is set in `enqueue:on:response:`, so a 401, a 404, or a malformed body carries none.
- Host readings, all public API except the GPU: `NSProcessInfo.thermalState`; memory pressure from
  `sysctlbyname("kern.memorystatus_vm_pressure_level")` (1, 2, 4), falling back to a
  `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` source where the sysctl is unreadable; CPU usage as the busy
  share of `host_processor_info` ticks since the previous reading; `getloadavg`;
  `task_info(TASK_VM_INFO).phys_footprint`; `NSURLVolumeAvailableCapacityForImportantUsageKey`, which
  tvOS marks unavailable, so tvOS reads `NSURLVolumeAvailableCapacityKey`.
- GPU utilization is `IOAccelerator`'s `PerformanceStatistics` → `"Device Utilization %"`,
  undocumented, readable without an entitlement on an M1 Max under `swift test` (2026-10-05). The
  busiest accelerator is reported, as a fraction. It covers every process, so it stays out of
  routing (user decision, 2026-10-05).
- Under App Sandbox (an ad-hoc-signed `.app` with only `app-sandbox` and `network.server`, macOS
  26.6.2, M1 Max, 2026-10-07), every host field reads as it does unsandboxed: the IORegistry GPU
  figure, the memory-pressure sysctl (no fallback taken), processor ticks, the footprint, and storage,
  which then measures the container's volume. A bare executable signed with the sandbox entitlement
  stops at launch with SIGTRAP; the sandbox needs a bundle.
- A real temperature lives in the SMC, which is private; the thermal state is the public reading.
- `modelInfo` (optional on the protocol) is reduced to string and finite-number values before it is
  served, so a backend that puts a date or a NaN there loses that entry and nothing else. Core ML's
  storage size is the compiled `.mlmodelc`'s allocated bytes (`NFKModelInfoSupport`); a model Core ML
  compiled at load is measured in its temporary location.
- `NFKServerStatus` (public) reads the route; `NFKRemoteInferKitBackend.fetchServerStatusWithError:`
  derives the base from `endpointURL` the way `prepareWithError:` does and goes through the same
  `JSONObjectForRequest:` seam, so a refusal arrives as the server's own error (401 →
  `NFKInferenceServerErrorUnauthorized`). A value of the wrong JSON type reads as nil or 0, and an
  unknown thermal or pressure name as the enum's Unknown. `advertisedProperties` is the discovered
  TXT record as resolved, and `providerWithBaseURL:` does not carry it to the re-pointed copy.
- `NFKBalancedBackend` (2026-10-05). Its per-request client is a private `NFKRemoteInferKitBackend`
  subclass whose `sendRequest:` seam calls `+[NFKRemoteTransport sendOnce:…]` (exposed through the
  private `NFKRemoteTransportPrivate.h`), because the transport's retry count is process-wide and a
  retried 503 would hold a request on a busy server for up to `maximumRetryDelay`. "Never started" is
  `NFKInferenceServerErrorBusy`, `…ModelNotFound`, `kNFKError_InferenceNotReady`, or
  `kNFKError_RemoteUnreachable` whose underlying `NSURLError` is one of cannot-connect, cannot-find-
  host, DNS failure, or not-connected. A lost connection or a timeout is not, since the run may be
  executing. The streamed form fails over only before any progress above 0 or partial result.
- A status reading that times out (`kNFKError_RemoteUnreachable` over `NSURLErrorTimedOut`) changes
  nothing: the server keeps its previous reading and that reading's date, so the next request reads
  it again. Recording the timeout as "does not host the model" made every request fail for
  `statusInterval` on a loaded machine, whose status route answered in over 2 s (2026-10-10). A server
  with no reading yet stays out until one succeeds.
- Choosing: a load reading is the status entry or the last reply's `NFKServerLoad`, whichever is
  newer, plus every request sent after it and not yet finished. Expected wait = the reading's
  `estimated_wait_seconds` + sent-since × average run ÷ limit, or, without an estimate,
  max(outstanding − limit + 1, 0) × average run ÷ limit. Any candidate without an average run turns
  the choice into fewest-outstanding for all, so the scores share a unit. Ties rotate, and so does
  `prepareWithError:`, which chooses a server to read the keys from.
- Conversation affinity (2026-10-07). A conversation's name is `key:` + `NFKParameterConversationKey`, or
  `messages:` + the SHA-256 of the messages through the first user message, serialized as JSON with
  sorted keys; messages that are not JSON (a part holding an image or data object) name no conversation.
  The opening, not the whole history, is hashed because each turn adds the previous reply, which a
  client may reformat. Two chats that open identically share a server, which costs nothing but
  balance. The pinned server is kept when it is among the candidates (healthy ones first, as for any
  choice) and no candidate's expected wait is more than `conversationWaitAllowance` below its own;
  a server with no wait estimate keeps its conversations. Keeping a conversation skips
  `pickAmong:`, so round robin's turn advances only on requests it places. The record is written
  when a server answers, so a failover re-pins. At most 4096 conversations are kept; past that the
  least recently used go. A name derived from the messages is added to the request it sends as
  `NFKParameterConversationKey` (the `messages:` digest), so a server's backend keeps a keyless
  chat's prompt too; a caller's own key goes out as written, and a request in no conversation gains
  none. MLX's language backend keeps a prompt cache per named conversation
  (`conversationCacheByteBudget`) and reuses it without
  `NFKMLXGenerationParameterKey.reusesPromptCache`.
- `lastServerBaseURL` is the server's base URL as added. Rebuilding it from the client's endpoint by
  deleting path components leaves a trailing slash, and the URLs then compare unequal.
- Runtime reports: the server resolves each registered class name with `NSClassFromString` and asks
  it only when it conforms to `NFKServingRuntimeStatus`, so the core names `NFKMLXGPU` without linking
  MLX. A `Class<Protocol>` variable takes message sends, not dot syntax, for class properties.

**HTTPS with a self-signed identity (2026-10-13).** `NFKServerIdentity` makes and keeps the identity
`TLSIdentity` takes, and clients pin its SHA-256 fingerprint (`pinnedCertificateSHA256` on
`NFKRemoteInferKitBackend`, `NFKRemoteProvider`, and `NFKBalancedBackend`; `NFKServerStatus`'s pinned fetch).
A pinned `NFKRemoteProvider` pins every client a factory builds from it: each backend class's
`backendForProvider:`, the realtime session, the stores, reporters, runners, and the model catalog.
The Bonjour TXT record carries the fingerprint under `cert_sha256`, which is advisory unless the client
turns automatic pinning on (`NFKRemoteProvider.discoveryPinnedServers`, an `NFKPinnedServers`); then
discovery and `NFKBalancedBackend` pin it on trust-on-first-use terms. The persistence measurements, the
trust model, and the automatic-pinning decisions are in `core-runtime-notes.md` ("Serving identity and
certificate pinning").

**Plain loopback listener (2026-10-14).** Asked for by an app's service: the agent CLIs it registers
`/mcp` with (Claude Code, OpenCode, Gemini CLI) trust no self-signed certificate, so HTTPS for other
Macs broke every local agent link, and a second `NFKInferenceServer` would build every model twice.

- `servesPlainHTTPOnLoopback` starts a second `NFKHTTPListener` on `plainLoopbackPort` (0 lets the
  system choose; `plainLoopbackListeningPort` reads the bound one), loopback-only, without TLS, and
  never advertised. Both listeners hand requests to the same `routeRequest:response:`, so models,
  residency, counts, keys, app routes, MCP sessions, and observation are one set. A request on it has
  `fromLoopback` set, so `loopbackExemptScopes` applies as it does on the main listener's loopback side.
- DNS rebinding (the Commit Manager's review, 2026-10-14): TLS kept a browser page out, since a page on
  `attacker.example` re-resolved to 127.0.0.1 fails the handshake on a certificate that does not name
  it; a plain listener with every scope loopback-exempt would serve that page with no key. Decision:
  a loopback request whose `Host` names anything but `localhost`, 127.0.0.0/8, or `::1` (port optional;
  no `Host` passes) gets no exemption on any listener. It is served only with a key that opens the
  route, and is otherwise 421 (`NFKInferenceServerErrorMisdirected`) before routing. A browser cannot
  forge `Host`; agent CLIs send `localhost`; a reverse proxy that forwards a public `Host` presents a
  key, as the proxy advice already says. `/mcp` keeps its `Origin` check.
- `localBaseURL` and `localMCPURL` name the plain listener while it runs (a stable local endpoint
  whether TLS is on or off), otherwise the main one. `networkBaseURL` and `networkMCPURL` name the main
  listener by `gethostname()`, with `.local` added to a bare name; nil while stopped or `loopbackOnly`.
- `startWithError:` starts both or neither: a plain listener that fails to bind stops the main one and
  reports `NFKInferenceServerErrorListenFailed`. `stop` stops both, and `NFKHTTPListener.stop` waits
  (up to 2 s) for the listener's cancelled state, since `nw_listener_cancel` frees the port
  asynchronously: in the full suite a start on the just-stopped port failed with EADDRINUSE
  (2026-10-14), which a fixed-port restart (stop, change, start) would hit too. On a listener's own
  queue, where the cancelled state is delivered, stop skips the wait (a queue-specific key marks it).
- Tests: `NFKServerPlainLoopbackTests` (8): one model serving both listeners (counts shared, the main
  listener still pinned), the local URLs without the option, the network URLs, a taken plain port
  starting neither and a stop releasing it, keys and a revocation applying on both, an MCP session
  issued on one listener starting on the other, and, over hand-written requests (a URL session
  replaces `Host`), a foreign `Host` refused with 421 unless its key opens the route and `/mcp`'s
  `Origin` check.

**Residency and unloading (2026-10-13).** Asked for by an app's service, which lists models without
keeping them in memory, and unloads idle ones.

- `addBackendProvider:forModelName:` hosts a model as a block that builds its backend.
  `NFKServedModel` holds `residency` (not loaded, loading, resident) on its lock. A run that arrives
  while the model is not resident joins `pending`, and the first one starts the build on a global
  user-initiated queue; later ones wait in the same queue. A finished build starts up to `limit`
  waiting runs. A failed one fails every waiting run with the provider's error unchanged (so its
  status mapping holds), records `load_error`, and returns to not loaded. Runs waiting for a build
  that will start them at once (up to `limit`) do not count against `maximumQueuedRunsPerModel`.
- Every unload goes through `unloadPassingTest:`, which releases the backend only when the model is
  provider-hosted, resident, and has nothing in `running` or `pending`, and takes the check and the
  release under one lock hold. The backend is released off the lock inside an `@autoreleasepool`, so it
  is gone before the runtime-cache release. `unloadModelNamed:` sets `unloadRequested`, which the end
  of the last run (or of a build) acts on, so an unload never cuts a run. `loadModelNamed:` clears it.
- An `addBackend:` model has no provider and is never unloaded; `unloadModelNamed:` answers NO. A model
  that is not resident reports `ready: true`, since `NFKRemoteInferKitBackend.prepareWithError:` and
  the balancer refuse a model that is not ready, and it describes the backend it last held (identifier,
  inputs, parameters, `modelInfo`), snapshotted when the build finished.
- The estimated wait of a model that is not resident adds the last build's duration (less the time
  this build has run), and is nil before any build has finished.
- Idle timeout = min(idle interval, pressure timeout). An idle interval of 0 (server or per model) is
  infinity. The pressure timeout is infinity below `memoryPressureOnset`; from the onset it runs
  linearly from the idle interval (or `pressureUnloadInterval` when idle unloading is off) at the onset
  to `shortestIdleUnloadInterval` at 1. The onset is inclusive, so the kernel's warning level (0.6)
  at the default onset (0.6) already applies `pressureUnloadInterval` to a model whose idle unloading
  is off.
- The default pressure is `NFKServedHostStatus.memoryPressureFraction`: the higher of the kernel's
  level (normal 0, warning 0.6, critical 1, read as the status route reads it) and the process
  footprint over `NFKHardwareProfile.recommendedWorkingSetSize`, clamped to 1.
- `unloadIdleModels` (the timer, every `residencyCheckInterval` on a utility serial queue, or the
  host) unloads every model past its idle interval, then of the models past only a pressure-shortened
  timeout, the least recently used one, at most once per `pressureUnloadHysteresis`. A falling pressure
  reading is held for the same interval, a rising one is taken at once. `unloadGracePeriod` counts from
  the build's end. `lastUsedAt` moves when a run takes or releases a slot and when a build finishes.
- After an unload the server asks each registered runtime class that implements the optional
  `NFKServingRuntimeStatus.releaseCachedMemory` to return its cache, from the residency queue and only
  when no hosted model has a run in flight or a build in progress, because an MLX cache trim frees
  buffers in-flight work still reads (`mlx-runtime-gotchas.md`). `NFKMLXGPU` implements it as
  `synchronize()` then `clearCache()`, and does nothing without the Metal library. Without it an
  unloaded MLX model's buffers stay in MLX's cache and the footprint does not fall.
- Status: each model's `residency` object (`state`, `unloadable`, `last_used` in seconds since 1970,
  `idle_timeout_seconds` when finite, `load_seconds`, `load_error`, `loads`, `unloads`, `resident_bytes`,
  `peak_bytes`) in both the status route and the `/models` entry; `NFKServerModelStatus` reads it. A
  server without it reads as `NFKServerModelResidencyUnknown`.
- Load figures (2026-10-15). `build` starts an `NFKServedLoadMeter` before it calls the provider: each
  registered runtime that implements the optional `allocatedByteCount`, `peakAllocatedByteCount`, and
  `resetPeakAllocatedByteCount` has its peak reset and its allocation read. After the build, a backend
  whose `residentByteCount` is above 0 gives both figures (the peak raised to at least the resident
  bytes). Otherwise the figures are the summed rise in allocation and in peak over the runtimes that
  read both times (a negative reading means unreadable and is skipped). A rise of 0 records nothing. The
  figures are stored on the model's lock in `buildEndedWithBackend:residentBytes:peakBytes:error:`, only
  when present, so they describe the last load that measured them and stay through an unload. A model
  hosted with `addBackend:` takes its backend's own figures. `NFKMLXGPU` implements the three members
  over `activeMemory`, `peakMemory`, and `resetPeakMemory`, reading -1 without the Metal library.
  Accuracy limits, one per cause:
  - The peak is one reading for the whole process, so overlapping loads and runs fall into each other's
    figures.
  - Arrays a loader leaves lazy for the first run are not counted.
  - MLX's cache release after an unload sees only the runs and loads of the MLX lifecycle backends, not
    MLX work a caller runs itself.
- Backend unload (2026-10-15). `unloadPassingTest:reason:` and `unloadIdleModels` call
  `NFKInferenceUnload` on each taken backend inside the `@autoreleasepool`, off the lock, before the
  release. The event order is unchanged: the Unloaded event is still reported on the model's lock in
  `takeBackendPassingTest:reason:`, before the backend's unload runs. A backend another object also
  holds (an app's reference, an MLX wrapper) frees its weights at that call.
  `retire` (a removed or replaced model) returns the backend it held, and `removeBackendForModelName:`
  and `host:` unload it off the lock. Runs in flight keep the backend, and since an unload never waits,
  the last of them frees the model. `host:` skips the unload when the new model holds the same backend.
- Tests: `NFKServedResidencyTests` replaces the server's clock (`uptimeProvider`, a private property
  the test declares in a category) and its pressure provider, and sets `residencyCheckInterval` to 0,
  so every timing is exact and no timer races the test. `NFKMLXModelInfoTests` checks that an unload
  empties MLX's cache. `NFKBackendLifecycleTests` (9) covers the protocol helpers, a backend's own
  figures, a registered test runtime's readings around a build, a model nothing measured, the unload
  call from an explicit, an idle, a removing, and a replacing unload, and the status reader; its runtime reads -1 outside its
  tests, since a registration is permanent. `NFKMLXLifecycleTests` checks an MLX model's figures through
  the server and that the server's unload reaches the backend.

**Observation (2026-10-13).** An app's service shows runs, loads, clients, and connected agents without
polling its own status route. `NFKServerObservation.h/.m` holds the delegate protocol and the event
classes; the classes' setters are in `NFKServerObservationPrivate.h`, a class extension the server
imports.

- Delivery: every event is built on the thread where it happened and dispatched to `delegateQueue`
  (a serial utility queue of the server's, or the host's), so a serial queue sees them in order. An
  event is built only when the delegate implements its method. The delegate is `atomic, weak`.
- Runs: `NFKServedModel.runObserver` reports a start from `start:` once the run passed its cancelled
  check (`run.began`), and an end from `run:endedWithResult:error:`, and an end without a start for a
  refused run (queue full), a run whose build failed, and a run of a removed model. A run cancelled
  while queued ends through `start:`'s cancelled branch, so it reports Cancelled with duration 0. Each
  run carries a UUID, the request path, the client address, and the key name (set in
  `enqueue:on:request:response:`). The wait and duration use the server's clock, so a test clock
  shapes them too.
- Models: `modelObserver` reports Loaded or LoadFailed from `buildEndedWithBackend:error:` before the
  waiting runs start (so a load precedes the runs it admits), with the build's reason recorded when it
  began (`beginBuildIfNeededFor:`: Request from admission, Explicit from `loadModelNamed:`), and
  Unloaded from `unloadPassingTest:reason:` (Explicit from a requested unload, Idle and Pressure from
  `unloadIdleModels`' two passes) and from `retire` when a resident model is removed or replaced.
  A build that ends after its model was removed reports nothing.
- Clients: `NFKHTTPConnection` counts its requests, and `NFKHTTPRequest.firstOnConnection` marks the
  first. The server reports a client when that request passes the key check, so a connection whose
  first request is refused reports nothing. A keep-alive client is one event.
- `currentStatus` wraps `statusJSONObject`, the dictionary the status route serializes, in
  `+[NFKServerStatus statusWithJSONObject:]`, so the in-process reading and the route cannot drift.
- The delegate protocol is `NS_SWIFT_SENDABLE`; measured with the isolation probe, a main-actor
  conformance runs on the delegate queue without a trap, and traps without the attribute.
- Tests: `NFKServerObservationTests` (8): a run's two events and a failed run, a refused run with no
  start, the seven model events of a load/unload/idle/failed-build/removal sequence in order, a
  pressure unload, one client event for two keep-alive requests with the key's name and scope, the
  delegate queue, the snapshot against the route, and a server that has not started (no start date,
  no uptime).

**MCP sessions (2026-10-13).** Per the Streamable HTTP transport (2025-06-18 and later), asked for so
an app can list the agents connected to its `/mcp` and see them leave.

- `initialize` (alone or in a batch) issues an id (a UUID, returned as `Mcp-Session-Id` on that reply)
  and holds a pending record in `NFKServedMCPSessionStore` (private, in `NFKServedMCP.m`) with
  `clientInfo.name`, `clientInfo.version`, the negotiated `protocolVersion`, the client address, and
  the key name. An `Mcp-Session-Id` sent with `initialize` is ignored: initialize always issues a new
  id.
- Decision: the session starts at the first POST or GET carrying the id under the same key (a
  conforming client sends `notifications/initialized` with it at once). Only then is it listed in
  `MCPSessions` and reported to the delegate, so a client that never echoes the id leaves no session
  and fires no event. A pending record lapses after 60 s, or the idle timeout when that is shorter and
  positive, even with the idle timeout off; at most 256 wait, and one more drops the oldest. A DELETE
  of a pending id is 204 with no event; a revocation and a stop discard pending records.
- The start and end events are queued under the store's lock, so a session's end is never reported
  before its start. `openStream` installs the stream's `disconnectHandler` before publishing the
  stream and clears it when the client already left, so a drop in between cannot leave a dead stream.
- A later request with an id: an unknown or ended id, or one started under another key name, is 404
  (the spec's signal to initialize again). Decision: a request without an id is served statelessly,
  since the spec lets a server not require sessions, and clients that drop the id keep working. A GET
  without an id is 405 (`Allow: GET, POST, DELETE`), which the TypeScript SDK reads as "no stream
  offered"; a DELETE without one is 400.
- GET with an id opens the event stream (200, `text/event-stream`, the id echoed) and writes the SSE
  comment `: open` at once, since `NSURLSession` delivers a streamed response to its delegate only
  when body bytes arrive (measured 2026-10-14: with pings off, the response never reached the client).
  A second GET replaces and closes the first. The store's timer, on its own serial utility queue, runs while an
  active session has a ping or idle interval or a pending record waits, starts or stops whenever a
  session or a setting changes, and ticks at a quarter of the shorter of `MCPPingInterval` and
  `MCPSessionIdleTimeout` (within 10 ms and 1 s). Each tick sends `{"jsonrpc":"2.0","id":
  "inferkit-ping-<n>","method":"ping"}` on every stream due one, and ends a session with
  `MCPMissedPingLimit` pings outstanding. A POSTed JSON-RPC response whose id starts with
  `inferkit-ping-` clears the count (it is acknowledged 202, like any response). Pings off (0) leave a
  session with an open stream to end only by DELETE, the host, a revocation, or a stop.
- A session with no stream ends after `MCPSessionIdleTimeout` without activity (a request starting or
  ending, a ping answer, the stream closing); a request still in flight keeps it. Defaults: idle 1800 s,
  ping 30 s, 3 missed pings.
- A session also ends on DELETE (204), `endMCPSessionWithIdentifier:`, `removeAPIKeyNamed:` of the key
  it started with, and `stop`. Ending one closes its stream with the chunked terminator; the stream's
  `disconnectHandler` only fires for a client that left first, which marks the stream closed and starts
  the idle clock.
- The HTTP connection's idle timer only closes a connection with no reply pending, so an open stream
  is never closed for idleness; pings are what detect a client that went away without closing.
- In process: `MCPSessions` (snapshots, oldest first) and the delegate's start and end events; the
  status route does not list sessions, since their key names stay in process (the scoped-keys decision).
- Tests: `NFKServedMCPSessionTests` (9): the issued id and client info, a client that never echoes the
  id, unknown ids on every method, DELETE, idle expiry, the host's end and a stop, answered pings
  keeping a session past its idle timeout and missed ones ending it and closing the stream, raising an
  interval starting the timer, and a session bound to its key and ended by the key's revocation. The
  expiry tests drive the store's injected clock (`uptimeProvider`) and call its sweep, so no
  wall-clock interval decides them. `NFKServedMCPTests`' bare GET expects `Allow: GET, POST, DELETE`.

**Hosting updates (2026-10-13).** Asked for by an app's service, which restarts the same server for a
port or exposure change and pins models its user marks.

- `maximumConcurrentRunsPerModel` and `maximumQueuedRunsPerModel` have setters that apply the pair to
  every hosted model (`applyLimit:queueLimit:`), and `host:` applies them again after inserting a model,
  so a value set while a model was being made reaches it. A raised limit starts waiting runs at once
  through `takeStartableRuns`, the one drain the build's end and a run's end also use. A lowered limit
  cancels nothing: the admission test (`running.count < limit`) keeps later runs waiting until enough
  running runs end. A lowered queue limit refuses only later arrivals; runs already queued beyond it
  stay queued.
- The wait estimate with more running runs than the limit drops the soonest-ending surplus, since the
  first slot opens only when the surplus and one more run have ended.
- `stop` closes the listener and keeps everything else (models, residency, counts, tools, the residency
  timer); `startWithError:` builds a new listener from the current settings. The status route's
  `started` is the latest start and is absent before the first (`NFKServerStatus.startDate` nil);
  `uptime_seconds` is 0 while the server is stopped. `unloadIdleModels` takes the backends out under
  `residencyLock` and releases them after leaving it, so an MLX teardown does not hold the status route
  or the pin and idle accessors. A model's load and unload events are enqueued on its own lock, where
  its residency changes, so a request that rebuilds a model during another's teardown cannot reach the
  delegate's serial queue ahead of the unload. The unload handler, which returns the runtimes' caches,
  runs after the release.
- Pinning is by name, like the per-model idle intervals, under `residencyLock`: a pinned model is
  skipped by `unloadIdleModels`, its idle timeout is INFINITY (so `idle_timeout_seconds` is absent), and
  `unloadModelNamed:` still unloads it. The residency object carries `pinned`, which
  `NFKServerModelStatus.isPinned` reads. Pinning builds nothing; the host loads a pinned model itself.
- Tests (`NFKServedResidencyTests`): raising and lowering the concurrency limit with gated runs, lowering
  the queue limit with runs queued past it, a stop and two starts of one server across a
  loopback/network/advertising change, and a pinned model under idle and full pressure.

**Scoped keys (2026-10-13).** The server holds a set of named keys, each opening a set of
`NFKServerKeyScope`s, for an app service's per-client keys (a balancer, an agent, its own app API).

- Scopes: Inference (every `/v1` route but status, unknown `/v1` paths included), Status
  (`/v1/inferkit/status`, so a balancer or dashboard key reads load without running anything), MCP,
  and AppRoutes (the app's prefixes and every other path no route claims, which then 404s).
  `NFKServerKeyScopeAll` is the OR of the four, not `NSUIntegerMax`, so `requiresAPIKeyOnLoopback`
  can compare against it.
- `apiKey` is a computed property over the key named `NFKInferenceServerDefaultAPIKeyName`
  (`default`): setting it adds that key with every scope, nil removes it, and reading returns that
  key's value whatever its scopes. `requiresAPIKeyOnLoopback` is computed over `loopbackExemptScopes`:
  YES empties it, NO fills it, and it reads YES when any scope needs a key on loopback. A loopback
  client exempt from a scope passes whatever it presents.
- Matching: the presented token's SHA-256 digest is compared in constant time with every key's
  digest, without stopping at a match, so neither the length of a guess nor which key matched shows in
  the timing. The first matching name in
  sorted order wins. The keys are read under their lock on each request, so a revocation applies to the
  next request; a streamed reply already authorized finishes.
- The routing order: trim the path, find its scope and the app prefix that claims it, check the
  key for that scope, then dispatch. The matched key's name is stored on `NFKHTTPRequest.keyName` and
  reaches app routes as `NFKServerRouteRequest.keyName`.
- Decision: neither key values nor key names are reported over HTTP, in the status route or in a
  refusal, since a client with only the Status scope would otherwise learn the server's key inventory.
  `apiKeyNames` and `scopesForAPIKeyNamed:` list them in process. The Bonjour `auth` entry still
  follows `requiresAPIKey`.
- Refusals: no key or an unknown one is 401 (`NFKInferenceServerErrorUnauthorized`, `WWW-Authenticate:
  Bearer`); a held key without the route's scope is 403 (`NFKInferenceServerErrorInsufficientScope`,
  `WWW-Authenticate: Bearer error="insufficient_scope"`), as RFC 6750 specifies.
- `startWithError:` refuses with `NFKInferenceServerErrorAPIKeyRequired` when another machine needs a
  key or any scope needs one on loopback, and no key is set.
- Tests: `NFKServerKeyTests` (5): each scope and a two-scope key against one route of each scope, a
  revocation and a replaced key, per-scope loopback exemption and the legacy property, the start check,
  and `apiKey` as the default key.

**App routes (2026-10-13).** An app's service serves its own HTTP API beside the models, on the same
port, TLS identity, Bonjour name, and keys. `addRouteForPathPrefix:handler:error:` registers a handler
under a prefix; `NFKServerRoute.h/.m` holds the public request and response classes and the prefix
rules.

- Prefixes are normalized to one leading and one trailing slash. Refused with
  `NFKInferenceServerErrorInvalidConfiguration`: no leading slash, the root, an empty, `.`, or `..`
  segment, and a first segment of `v1` or `mcp` in any case. `/v1beta/` is the app's: the server's own
  base test is `/v1` exactly or `/v1/…`, which the routing uses too.
- Routing order: the key check, `/mcp`, the `/v1` routes, then the app routes by the longest matching
  prefix (`/myapp/admin/` before `/myapp/`; a prefix matches whole segments, so `/myapp/administer`
  is `/myapp/`'s), then 404. A path under `/v1` that no server route answers stays a 404 of the
  server's.
- The handler runs on the request's global user-initiated queue, the same one the server's routes run
  on. `NFKServerRouteResponse` forwards to `NFKHTTPResponse`, whose writes may come from any thread and
  whose first reply wins; a handler that never replies holds the connection until the client leaves
  (`disconnectHandler` fires then). Both classes are `NS_SWIFT_SENDABLE`.
- `NFKHTTPResponse.disconnectHandler` is read and written on the reply's lock. One set after the client
  has gone runs at once, so a run queued (`enqueue:on:request:response:`) or a stream opened for a
  client that already left is cancelled or closed; one set after the reply ends is not kept, so a
  handler that captures its reply forms no cycle.
- `NFKHTTPRequest.clientAddress`: `inet_ntop` of the peer endpoint, an IPv4-mapped IPv6
  address printed as IPv4. A loopback client over `localhost` reads `::1`.
- The body limit is the listener's (`maximumRequestBodyBytes`): an oversized body is a 413 before
  routing, so the handler never sees it. The reason-phrase table gained 201, 202, 301, 302, 304, and
  409 for app replies; an unlisted status still writes "Status".
- Tests: `NFKServerRouteTests` (9), over NSURLSession on a loopback port, two of them a finished
  reply releasing a disconnect handler that captures it and a handler set after the client left
  running at once.

**Sendable blocks (2026-10-13).** An app's Swift 6 service (`-default-isolation MainActor`) trapped in
`_swift_task_checkIsolatedSwift` → `dispatch_assert_queue_fail` when the server called a provider
closure written inline: the block type was not Sendable, so Swift inferred the closure main-actor
isolated and checked the queue when the server called it on its own. Every block InferKit calls off the
caller's queue is `NS_SWIFT_SENDABLE` (written `(^ NS_SWIFT_SENDABLE)` in a parameter,
`(^ NS_SWIFT_SENDABLE name)` in a typedef or property), which imports as `@Sendable` and leaves the
closure nonisolated. The attribute changes nothing for Objective-C callers or overriders. The audit:

- Server: `NFKServedBackendProvider` → a global user-initiated queue; `loadModelNamed:`'s handler → a
  global user-initiated queue; `memoryPressureProvider` → the residency queue, a request's queue, or
  the reader of `memoryPressure`.
- `NFKInferenceJob`: `completionHandler` → the thread that finished the job; `progressHandler` → the
  reporting thread; `cancellationHandler` → the thread that calls `cancel`.
- Tools: `NFKToolHandler`, `NFKBackendToolRequestBuilder`, `NFKBackendToolResultDescriber` → the
  invoking thread (a loop's thread, the MCP route's global queue); `NFKToolLoop.approvalHandler` and
  `invocationHandler` → the loop's thread.
- Completion handlers on a background queue: `NFKRemoteModelCatalog`, `NFKRemoteProvider` (local
  providers, first local provider, models, Bonjour discovery), `NFKServerStatus` fetches, `NFKHFHub`
  download, `NFKComputePlan` (Core ML's queue), `NFKSpeechRecognitionBackend` authorization (an
  arbitrary thread).
- `NFKSubprocess`: `lineHandler`, `standardErrorLineHandler`, and `stateHandler` → the subprocess's
  event queue; `stopWithCompletionHandler:`'s handler → the same queue.
- Transport seams: `NFKRemoteTransport`'s stream (line handler and `NFKRemoteStreamCompletion` on the
  session's delegate queue, the returned cancel block on the cancelling thread) and every backend's
  `streamRequest:lineHandler:completionHandler:` override point; `NFKOllamaRunner`'s pull seam.
- `NFKRealtimeSession`: the seven handler properties and the `NFKRealtimeSocket` open handlers → the
  socket's queue.

Not annotated: blocks InferKit calls synchronously on the caller's thread, and notifications
(`NFKPinnedServers` posts on the thread that changed the pins). `NFKServedBackendBuilder` is the
throwing Swift form of the provider: a protocol (Swift cannot write a throwing `@convention(block)`
closure, and a pure Objective-C target ships no Swift overlay) whose method imports as
`backend(forModelName:) throws`. The protocol itself is `NS_SWIFT_SENDABLE`; measured with the probe
below, a conformance declared in main-actor code then runs nonisolated, and without the attribute the
same conformance traps. `Tools/swift-isolation-probe/run.sh` compiles `main.swift` as Swift 6 with
`-default-isolation MainActor` against the built core and calls each kind of block from InferKit's
queues; it exits 0, and the same file against headers stripped of the attribute exits with SIGTRAP
(133) at the first provider. `SwiftExamples` stays in Swift 5 mode (tools version 5.9), where the
isolation check is not emitted, so its `testServingModelsBuiltOnDemandFromMainActorCode` pins the
imported shapes and the probe pins the behavior.

Errors carry `inferkit_domain`, `inferkit_code`, and `inferkit_user_info` beside the OpenAI `message`,
`type`, and `code`. The transport takes an `NFKInferenceErrorDomain` code from that body over its
status reading, and the native client rebuilds the whole `NSError`. Status mapping: not ready 503,
missing input, unsupported, and refused 400, rate limited 429 (with `Retry-After` from
`NFKRemoteErrorRetryAfterKey`), upstream unreachable 502, other failures 500, unknown model 404,
missing or unknown key 401 with `WWW-Authenticate: Bearer`, a held key without the route's scope 403
(`NFKInferenceServerErrorInsufficientScope`) with `WWW-Authenticate: Bearer error="insufficient_scope"`
as RFC 6750 specifies, a loopback request naming another host without a key that opens the route 421
(`NFKInferenceServerErrorMisdirected`), a full queue (`NFKInferenceServerErrorBusy`) 503 with `Retry-After`. The transport retries that 503 only when the
`Retry-After` is within `maximumRetryDelay` (8 s by default).

