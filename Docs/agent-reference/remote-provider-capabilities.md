<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes for its subject. An agent that learns something durable about this subject adds it to this
file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# Remote provider capabilities

Every inference mode each preset serves, as input → output, set against the backend that reaches it.
Surveyed 2026-09-22 from each provider's official documentation (five research passes; field-level
detail per mode lives in the provider's reference, linked at the end of each section). Account
management, fine-tuning, and agent-management APIs are out of scope; a hosted agent run as a
request with a result (Gemini's Deep Research and Antigravity) is in scope. Files, retrieval stores,
and usage reports ship (phase 5), and batches (2026-10-23). The request shapes
the backends already speak are in `remote-providers.md`; this file is the coverage map and the build
order for what is missing.

Status words: **covered** (a backend speaks it), **partial** (a backend reaches it but drops or
misshapes part of it), **gap** (nothing reaches it).

## Coverage by provider

### OpenAI (`openai`)

| Mode | Endpoint | Status |
|---|---|---|
| text + image + audio + PDF → text / audio | `/chat/completions` | covered (`NFKRemoteBackend`) |
| text + image + file → text, built-in tools, reasoning summaries, background jobs | `/responses` | covered (`NFKRemoteResponsesBackend`) |
| prompt-cache controls: key, retention, options (mode, TTL, prewarm), per-part breakpoints | `/chat/completions`, `/responses` | covered (2026-10-23); prewarm and the 12-hour TTL are Responses-only; the cache-write usage counts are not read |
| text + image → image inside Responses | `image_generation` tool | covered |
| text → image, image(s) + mask → image | `/images/generations`, `/images/edits` | covered (multi-image `image[]`, streamed partial images) |
| text → speech | `/audio/speech` | covered (`stream_format: sse`); a custom voice object goes by name |
| audio → text | `/audio/transcriptions` | covered (`gpt-transcribe` text and stream, diarized_json, words); `whisper-1` family shuts down 2027-02-26 |
| audio → English text | `/audio/translations` | covered until `whisper-1` shuts down |
| text + image → moderation | `/moderations` | covered |
| text + image → typed decisions (predicate, choice, score) | `/decisions` | covered (`NFKOpenAIDecisionsBackend`); public beta from 2026-10-06 |
| text → embedding | `/embeddings` | covered |
| text + image → video | `/videos` | covered; shuts down 2026-09-24 |
| speech ↔ speech, realtime transcription, realtime translation | `wss …/v1/realtime`, `…/realtime/translations` | covered (`NFKRealtimeSession`) |
| full-duplex voice | `/v1/live/sessions` (WebRTC SDP) | gap; WebRTC is outside the core's dependency rule |
| batches of chat, Responses, embedding, completion, moderation, and image (generate; edit as JSON) requests | `/v1/files` (`purpose: batch`) + `/v1/batches` | covered (`NFKRemoteBatch`, OpenAI style, 2026-10-23; images 2026-10-24); no video batch, since `/v1/videos` answers 404 (the Videos API was removed 2026-09-24) |

### Anthropic (`anthropic`)

| Mode | Endpoint | Status |
|---|---|---|
| text + image + PDF → text, tools, thinking | `/messages` | covered (`NFKAnthropicBackend`) |
| plain-text documents; custom-content documents and Files API `file_id` sources | `/messages` | covered: plain text, and `file_id` sources through `NFKRemoteFileStore` |
| citations on the reply | `/messages` | covered (`NFKParameterCitations`, `NFKOutputCitations`) |
| server tools: web search, web fetch, code execution (file outputs) | `/messages` | covered: the caller passes the tool by wire shape (any entry with a `type`); calls and results come back under `NFKOutputServerToolResults` |
| token count | `/messages/count_tokens` | covered (`NFKRemoteTokenCounter`) |
| skills: create, version, list, delete; load in a request | `/v1/skills`, Messages `container.skills` + code execution | covered (`NFKRemoteSkillStore`, `NFKParameterSkills`, 2026-10-23) |
| audio, image output, embeddings | — | not offered |
| batches of Messages requests | `/v1/messages/batches` (inline) | covered (`NFKRemoteBatch`, Anthropic style, 2026-10-23) |

### Google Gemini (`gemini`)

The Interactions API (`POST /v1beta/interactions`, GA 2026-06) is Google's recommended native surface;
`generateContent` is legacy and supported. The OpenAI layer serves chat, embeddings,
`images/generations`, and `videos` only.

| Mode | Endpoint | Status |
|---|---|---|
| text + image + audio + video + PDF → text, tools, structured output | Interactions / OpenAI-layer chat | covered (`NFKGeminiInteractionsBackend`) |
| text + image + audio + video + PDF → text, tools, thinking, structured output, grounding | `/v1beta/models/{id}:generateContent` | covered (`NFKGeminiGenerateContentBackend`) |
| long decode to 1M output tokens (`gemini-4-argon`) | Interactions background hops; generateContent `continuationToken` | covered (both backends, 2026-10-13) |
| text → cited report (Deep Research agents) | Interactions `agent` + `agent_config {type: deep-research}` | covered (`agentName`, 2026-10-13) |
| text + image → code and files in a hosted sandbox (Antigravity agent; e.g. a CAD script and its STL) | Interactions `agent` + `environment`; `/v1beta/environments` files | covered (`agentName`, `NFKGeminiEnvironmentStore`, 2026-10-13) |
| agentic video understanding | Interactions `processing: agentic` on a video block | covered |
| remote MCP servers the service calls (models, Deep Research, Antigravity) | Interactions `mcp_server` tool | covered (`NFKParameterMCPServers`, 2026-10-13) |
| service tier, safety settings, webhooks, labels | Interactions top-level fields | covered (by parameter name) |
| text + images → image (generate and edit, Nano Banana) | Interactions `response_format: image` | covered |
| text → speech (30 voices, two speakers) | Interactions `response_format: audio` | covered (Interactions) |
| voice library: list, get, design, replicate, delete | `/v1beta/voices` (2026-09-22) | covered (`NFKGeminiVoiceLibrary`) |
| audio → text with diarization and word times | Interactions `transcription_config` (`gemini-3.5-transcribe`) | covered (Interactions) |
| text (+ images) → music | Interactions (`lyria-3.5`, `lyria-3-*`) | covered |
| text + image + video + audio → video, video editing | Interactions (`gemini-omni-1.1-flash`) | covered (Interactions, background); Google names it the default video model over Veo |
| text + images → video | Veo `predictLongRunning` / OpenAI-layer `videos` | covered (`NFKRemoteVideoBackend`) |
| text + image + audio + video + PDF → embedding | `:embedContent` (`gemini-embedding-2`) | covered (native `embedContent`, the preset's default) |
| token count | `:countTokens` | covered |
| live speech ↔ speech, live music | Live `BidiGenerateContent`, `BidiGenerateMusic` WebSockets | covered (`NFKRealtimeSession`; live video input by `sendEvent:`) |
| live speech → speech with background reasoning (`gemini-3.8-live-extended-thinking`) | Live, `interactionStatus` | covered (2026-10-13) |
| live speech → translated speech, live speech → text | Live (`gemini-3.5-live-translate-preview`, `gemini-3.5-transcribe-live`) | covered (2026-10-13) |
| ephemeral tokens | `BidiGenerateContentConstrained?access_token` | covered for connecting; minting (`POST /auth_tokens`) stays with the app's server |
| live avatar video | Live `avatarConfig` | Vertex AI only; frames reach `mediaHandler` |
| 3D or document output | — | not offered (2026-10-13: no model; `document` is a usage-breakdown modality) |
| Imagen | — | shut down |
| batches of generateContent requests | `/v1beta/models/{id}:batchGenerateContent` (inline, under 20 MB) | covered (`NFKRemoteBatch`, Gemini style, 2026-10-23); file input past 19 MB and embedding batches (`:asyncBatchEmbedContent`) 2026-10-24 |

### xAI (`xai`)

Chat Completions is marked legacy; new features ship on `/v1/responses` first.

| Mode | Endpoint | Status |
|---|---|---|
| text + image → text | `/chat/completions`, `/responses` | covered through chat and `NFKRemoteResponsesBackend` (built-in tools by wire shape) |
| text → image, image(s) → image | `/images/generations`, `/images/edits` (JSON only, `images[]` up to 5) | covered (JSON edits, `images[]`, derived ratio) |
| text + images + video → video, edit, extend | `/videos/*` | covered |
| audio → text | `POST /v1/stt` (multipart, `diarize`, word times), `wss /v1/stt` | covered, HTTP and WebSocket |
| text → speech | `POST /v1/tts` (JSON `text`, `voice_id`, `output_format`), `wss /v1/tts` | covered, HTTP and WebSocket |
| voice list | `GET /v1/tts/voices` | covered (`availableVoicesWithError:`) |
| speech ↔ speech | `wss /v1/realtime` | covered (`NFKRealtimeSession`) |
| tokenize | `POST /v1/tokenize-text` | covered |
| embeddings, moderation, rerank | — | not offered (embeddings unconfirmed) |
| batches of chat, Responses, image (generate, edit), and video (generate, edit, extend) requests | `/v1/batches` (named; requests added; paged results) | covered (`NFKRemoteBatch`, xAI style, 2026-10-23; media 2026-10-24) |

### Mistral (`mistral`)

| Mode | Endpoint | Status |
|---|---|---|
| text + image + audio + document → text, reasoning chunks | `/chat/completions` | covered |
| document / image → markdown, tables, annotations (OCR) | `POST /v1/ocr` | covered (`NFKRemoteOCRBackend`) |
| code infill | `POST /v1/fim/completions` | covered (`NFKRemoteCompletionBackend`) |
| text / chat → moderation | `/moderations`, `/chat/moderations` | covered |
| text / chat → classification | `/classifications`, `/chat/classifications` | covered (`NFKRemoteClassifierBackend`) |
| audio → text (diarize, context bias, word times, SSE) | `/audio/transcriptions` | covered (Voxtral 3 checked 2026-10-23); `speaker_references` is not sent |
| realtime audio → text | `wss /v1/audio/transcriptions/realtime` | covered (`NFKRealtimeSession`) |
| text → speech, voice cloning by reference clip | `/audio/speech` → JSON `{audio_data}` | covered, the realtime TTS model's streamed reply included |
| realtime text → speech (`voxtral-mini-tts-realtime-3-0`) | not published | gap (2026-10-23): Mistral documents no speech socket, and the probe cannot confirm one; reopens on a published reference or a keyed answer other than 403 (lead and detail in `remote-providers.md`, "Realtime sessions") |
| voices | `/audio/voices` | covered (listing) |
| text → embedding (`output_dimension`, `output_dtype`) | `/embeddings` | covered (fields by name) |
| text + image → text with built-in tools (web search, premium web search, code interpreter, image generation, document libraries, custom connectors), function tools, guardrails, server-kept conversations (continue, restart, history, delete) | `/v1/conversations` | covered (`NFKMistralConversationsBackend`, 2026-10-24); saved agents run by `agent_id`, while creating and managing them (`/v1/agents`) is out of scope |
| image generation | the Conversations API's `image_generation` tool | covered (`NFKMistralConversationsBackend`; the tool's file downloads as `NFKOutputImage`) |
| batches of chat, embedding, FIM, OCR, moderation, classification, transcription, and conversation requests | `/v1/batch/jobs` (inline to 10,000, else a file) | covered (`NFKRemoteBatch`, Mistral style, 2026-10-23; all but chat and embeddings 2026-10-24, transcription by `file_url`) |

### DeepSeek (`deepseek`)

| Mode | Endpoint | Status |
|---|---|---|
| text + image → text, thinking | `/chat/completions` | covered; `json_schema` is not offered (`json_object` only) |
| Responses | `/responses` (stateless) | covered |
| code infill / raw completion | `/beta/completions` (`suffix`) | covered |
| prefix completion | `/beta/chat/completions` (`prefix: true`) | by endpoint and message shape through `NFKRemoteBackend` |
| audio, embeddings, moderation | — | not offered |
| batches | — | not offered (no batch API; off-peak pricing instead) |

### Groq (`groq`)

| Mode | Endpoint | Status |
|---|---|---|
| text + image → text, reasoning, built-in `browser_search` / `code_interpreter` | `/chat/completions`, `/responses` | covered through chat (`executed_tools` under `NFKOutputServerToolResults`) and `NFKRemoteResponsesBackend` |
| audio → text / English | `/audio/transcriptions`, `/audio/translations` | covered |
| text → speech | `/audio/speech` (Orpheus, WAV, 200-character input) | covered (200-character pieces, WAV joined) |
| embeddings, moderation, rerank, images, video | — | not offered |
| batches of chat requests | `/openai/v1/files` + `/openai/v1/batches` (window 24h–7d, cursor paging) | covered (`NFKRemoteBatch`, OpenAI style, 2026-10-23); audio transcription and translation by URL 2026-10-24 |

### Together (`together`)

| Mode | Endpoint | Status |
|---|---|---|
| text + image + audio → text | `/chat/completions` | covered |
| raw text completion | `/completions` | covered |
| text → image, image → image (Kontext) | `/images/generations` (`width` / `height`, `image_url`, `reference_images`) | covered |
| text + media → video | `/v2/videos` | covered |
| text → speech (SSE chunks), voices | `/audio/speech`, `/voices` | covered (SSE, voice list) |
| audio → text with diarization | `/audio/transcriptions` | covered (`diarize`, speakers on segments) |
| realtime STT / TTS | `wss /v1/realtime`, `wss /v1/audio/speech/websocket` | covered (`NFKRealtimeSession`) |
| rerank | `/rerank` (dedicated endpoints) | covered |
| batches of chat requests | `/v1/files/upload` (`purpose: batch-api`) + `/v1/batches` | covered (`NFKRemoteBatch`, Together style, 2026-10-23); no request counts reported |

### OpenRouter (`openrouter`)

| Mode | Endpoint | Status |
|---|---|---|
| text + image + audio + PDF + video → text | `/chat/completions` | covered (`video_url`; `file` parts, with the `plugins` field by name) |
| chat → image, chat → audio | `/chat/completions` `modalities` | covered |
| text / images → image | `POST /images` | covered |
| text → speech, audio → text (multipart or JSON `input_audio`) | `/audio/speech`, `/audio/transcriptions` | covered; PCM default not handled |
| multimodal embeddings | `/embeddings` | covered (image and audio parts) |
| rerank, video | `/rerank`, `/videos` | covered |
| web search tool | `openrouter:web_search` | by wire shape; `url_citation` annotations under `NFKOutputCitations` |
| batches of chat, Responses, Messages, or embedding requests | `/api/v1/batches` (inline; results inline; no cancel) | covered (`NFKRemoteBatch`, OpenRouter style, 2026-10-23) |

### OpenCode Zen (`opencode`)

A gateway that serves each model on its maker's protocol; the route per model is in the preset's
`modelAPIStyles` (2026-10-13). Detail in `remote-providers.md` ("OpenCode gateways").

| Mode | Endpoint | Status |
|---|---|---|
| text + image → text, tools (GLM, Kimi, DeepSeek, MiniMax, Mistral, Qwen3.8 Max, the free models) | `/chat/completions` | covered (`NFKRemoteBackend`) |
| text + image → text, tools, reasoning (GPT, Grok, Muse Spark) | `/responses` | covered (`NFKRemoteResponsesBackend`) |
| text + image → text, tools, thinking (Claude, most Qwen) | `/messages` | covered (`NFKAnthropicBackend`) |
| state → typed decisions (Jev) | `/systemone` | covered (`NFKTypeSafeBackend`) |
| text + image → text, tools, thinking (Gemini) | `/models/{id}:generateContent` | covered (`NFKGeminiGenerateContentBackend`) |
| model list | `/models` | covered (ids only) |

### OpenCode Go (`opencode-go`)

The subscription gateway, same key as Zen. Its routes differ from Zen's for some names.

| Mode | Endpoint | Status |
|---|---|---|
| text + image → text, tools (GLM, Kimi, DeepSeek, MiMo, LongCat, Hy, the free models) | `/chat/completions` | covered (`NFKRemoteBackend`) |
| text + image → text, tools, reasoning (GPT Luna, Grok, Muse Spark Contributor) | `/responses` | covered (`NFKRemoteResponsesBackend`) |
| text + image → text, tools (Claude Haiku 5.5, MiniMax, Qwen3.7 Plus and Qwen3.8) | `/messages` | covered (`NFKAnthropicBackend`) |
| model list | `/models` | covered (ids only) |

### Alibaba Model Studio (`modelstudio`)

DashScope, Singapore base; other regions by `providerWithBaseURL:`. Added 2026-10-21 for the
Qwen-Audio TTS models, so only chat and speech are built. Detail in `remote-providers.md` ("Alibaba
Model Studio").

| Mode | Endpoint | Status |
|---|---|---|
| text + image → text, tools | `/compatible-mode/v1/chat/completions` | covered (`NFKRemoteBackend`) |
| text → speech (Qwen-Audio 3.0 / 3.1 Plus and Flash) | `/api/v1/services/audio/tts/SpeechSynthesizer` | covered (`NFKRemoteSpeechAPIStyleModelStudio`, unstreamed URL fetch and SSE) |
| text → speech (Qwen3-TTS) | `/api/v1/services/aigc/multimodal-generation/generation` | gap |
| model list | `/compatible-mode/v1/models` | covered if it answers OpenAI's `data[].id`; unconfirmed. Native `/api/v1/models` is a gap |
| text → embedding | `/compatible-mode/v1/embeddings` | gap (exists by probe) |
| audio → text, realtime speech, image, video | native routes | gap; not yet surveyed |
| batches of chat or embedding requests | `/compatible-mode/v1/files` + `/batches` (window 24h–336h) | covered (`NFKRemoteBatch`, OpenAI style, 2026-10-23) |

### Perplexity (`perplexity`)

Added 2026-10-23. Detail in `remote-providers.md` ("Perplexity"). No live call has been made (no key);
every shape is stub-tested.

| Mode | Endpoint | Status |
|---|---|---|
| text + image + file → text, tools (web, image, finance, people search, fetch, sandbox, function, MCP), structured output, background jobs | `POST /v1/agent` | covered (`NFKRemoteResponsesBackend`, `NFKRemoteResponsesDialectPerplexityAgent`) |
| sandbox output files | `GET /v1/agent/{id}/files`, `…/files/{file_id}/content` | covered (`outputFilesForResponseIdentifier:error:`, `contentsOfOutputFileWithIdentifier:responseIdentifier:error:`) |
| state + images → typed decisions (noul, choice, score) | `POST /v1/decisions` | covered (`NFKTypeSafeBackend`, by the `pplx-decider` prefix route) |
| text → embedding (int8 or binary) | `POST /v1/embeddings` | covered (`NFKRemoteEmbeddingBackend`, base64 decoded) |
| document chunks → contextualized embeddings | `POST /v1/contextualizedembeddings` | covered (`contextualizedEmbeddingsForDocuments:error:`) |
| query → ranked web pages | `POST /search` | covered (`NFKRemoteWebSearch`) |
| model list | `GET /v1/models` + published | covered: the Agent API's list, then the two decision models; no single-model route |
| Sonar chat completions | `/chat/completions`, `/v1/sonar` | support ended 2026-09-27; reformulated as Agent API requests |
| skills: create, replace, list, revisions, download, delete; load in a request | `/v1/skills`, Agent `skills` | covered (`NFKRemoteSkillStore`, `NFKParameterSkills`) |
| batches of Agent API requests, and the files they read and write | `/v1/files` (`purpose: batch`), `/v1/batches` (`/v1/responses` only) | covered (`NFKRemoteBatch`, OpenAI style; `NFKRemoteFileStore`, 2026-10-23) |
| auth tokens, Computer analytics | `/generate_auth_token`, `/revoke_auth_token`, analytics | out of scope: minting keys is key management, analytics account administration |
| OpenAI-compatible chat, responses, and messages | Router API, `https://api.perplexity.ai/router/v1` | private preview; no preset |

### Local servers

| Server | Beyond the OpenAI layer | Status |
|---|---|---|
| Ollama | `/api/generate` (`suffix` infill, `think`, `images`), `/api/embed`, `/v1/responses` | `NFKRemoteCompletionBackend` and `NFKRemoteResponsesBackend` accept `ollama`; `/api/generate` and `/api/embed` are not adapted |
| LM Studio | native `/api/v1/chat` (stateful, `reasoning`), `/v1/responses` | Responses covered (`NFKRemoteResponsesBackend`) |
| llama.cpp | `/completion`, `/infill`, `/reranking`, `/tokenize`, `input_video` / `input_audio` chat parts | completion covered (`NFKRemoteCompletionBackend`), tokenize covered (`NFKRemoteTokenCounter`), rerank covered (`NFKRemoteReranker`) |
| vLLM | `/v1/completions`, `/score`, `/rerank`, `/classify`, `/pooling`, `/v1/audio/*`, `wss /v1/realtime` STT, `video_url` parts | completion, classify, rerank, and `video_url` parts covered; `/score` and `/pooling` are gaps |

### InferKit server (`inferkit`)

`NFKInferenceServer` in another InferKit process. Every route reaches whatever backend the host
registered, so what a mode does depends on that backend.

| Mode | Endpoint | Status |
|---|---|---|
| any request → any result, streamed progress and partials | `POST /inferkit/run` | covered (`NFKRemoteInferKitBackend`) |
| text + image + audio + video + PDF → text, tools, schema, spoken reply | `/chat/completions` | covered (`NFKRemoteBackend`) |
| text → embedding | `/embeddings` | covered |
| audio → text / English | `/audio/transcriptions`, `/audio/translations` | covered |
| text → speech | `/audio/speech` | covered; MP3 and Opus only when the model produced them |
| text → image, image(s) + mask → image | `/images/generations`, `/images/edits` | covered |
| model list | `/models` | covered |
| Responses, realtime, files, moderation, rerank | — | not served; the native route carries these backends' requests |

## Progress

- **Phase 1 done (2026-09-22).** Transcription, speech, and image backends carry per-service styles;
  the media factories return nil for presets without the endpoint. The per-style contract is in
  `remote-providers.md` ("Audio and image styles").
- **Phase 2 done (2026-09-22).** Chat dialects (Mistral, OpenRouter, vLLM, llama.cpp), whole-video
  parts, image output through chat, citations and server-tool results (Anthropic, OpenAI-style
  annotations, Groq), Anthropic plain-text documents and citations, multimodal embeddings (OpenRouter
  parts, Gemini native), Mistral chat moderation. Detail in `remote-providers.md` ("Chat dialects and
  reply extras").
- **Phase 3 done (2026-09-22).** `NFKRemoteResponsesBackend`, `NFKGeminiInteractionsBackend`,
  `NFKRemoteCompletionBackend`, `NFKRemoteOCRBackend`, `NFKRemoteClassifierBackend`, and
  `NFKRemoteTokenCounter`. Detail in `remote-providers.md` ("Responses, Interactions, and the
  single-purpose backends"). Left out on purpose: Anthropic Files API `file_id` sources and OpenAI
  custom-voice creation (account and file management), xAI collections search and Together's code
  interpreter (hosted tools, not model inference), DeepSeek's prefix completion (a chat request with
  `prefix: true` on its `/beta` base, reachable through `NFKRemoteBackend` by endpoint and message
  shape).

- **Phase 4 done (2026-09-22).** `NFKRealtimeSession` speaks twelve realtime protocols over
  `NFKRealtimeSocket`. Detail in `remote-providers.md` ("Realtime sessions").
- **Phase 5 done (2026-09-22).** `NFKRemoteFileStore` with file references in every request shape,
  `NFKRemoteRetrievalStore`, and `NFKRemoteUsageReporter`. Detail in `remote-providers.md` ("Files,
  retrieval stores, and usage reports"). Anthropic `file_id` sources, left out in phase 3, ship here.

## Decisions

- **OpenAI GPT-Live over WebRTC (`/v1/live/sessions`): not implemented (decided 2026-09-22).** WebRTC
  needs a media stack the core does not take as a dependency. Revisit if a repository issue asks for
  it; until then the WebSocket realtime styles in `NFKRealtimeSession` are the live-voice path.

- **Together's code interpreter (`/tci/execute`): not implemented (decided 2026-09-22).** It runs code
  in a hosted sandbox rather than a model; the code execution that matters rides the service-run tools,
  whose results come back under `NFKOutputServerToolResults`.
- **Account management: usage and cost reports only (decided 2026-09-22).** Member, invite, workspace,
  and key management stay out. Usage and cost reporting ships because an app built on InferKit may
  offer its own users an admin key as a feature; the reporting class documents that an admin key is a
  credential to keep off shipped binaries unless the app's user supplies it.
- **Gemini's generateContent (decided 2026-10-13).** The OpenCode presets first shipped with the
  Gemini route recorded as a gap; `NFKGeminiGenerateContentBackend` closed it the same day and also
  serves the `gemini` preset. Open until a live key: OpenCode's auth header on that route, and
  whether its chat completions answers a Gemini id.
- **Gemini audit of 2026-10-13.** Checked against the changelog, the models page, the Interactions,
  Live, and generateContent references, and google-genai v2.30.1 after Gemini 4 Argon went GA. No
  Google model outputs STL or another 3D format; `gemini-4-argon` outputs text. Argon's launch demo
  (a 3D-printable sofa leg) is "Argon using Antigravity in the Gemini macOS app": the agent writes
  the geometry with code. Through the API that is the Antigravity agent with `agent_config.model:
  gemini-4-argon`, whose files come back through `NFKGeminiEnvironmentStore`. Deep Research and
  Antigravity are reached as models are (a request in, a result out), so they are in scope.
  Environment management beyond listing, files, and deletion (sources, network allowlists,
  credentials at `/v1beta/credentials`) goes through the `environment` parameter by wire shape.
- **Files and hosted retrieval (decided 2026-09-22):** file management (upload, list, fetch, delete,
  and file references in requests) and retrieval stores with management (create, add and remove files,
  search, delete) are in scope.

## Build order

Each phase ends at the Full Check. A phase adds contract keys only where no existing key carries the
meaning.

1. **Existing backends, per-provider shapes.** The transcription and speech backends gain an
   `apiStyle` the way the video backend has one (xAI `/v1/stt` and `/v1/tts`, Mistral's JSON speech
   reply and transcription fields, `gpt-transcribe`, Together diarization, streaming SSE for speech
   and transcription, Groq's 200-character chunking). The image backend gains styles (xAI JSON
   edits with `images[]`, Together width / height and reference images on one path, OpenRouter
   `/images`, streamed partial images, multi-image edits). Every media factory returns nil for a
   preset that serves no such endpoint.
2. **Chat surfaces.** Video parts (`video_url`, `input_video`), Mistral `document_url` and its
   `input_audio` spelling, OpenRouter `file` parts, image output (`message.images[]`), Anthropic
   plain-text documents, `file_id` sources, citations, and server-tool result blocks, Groq
   `executed_tools`, multimodal embedding input (OpenRouter, Gemini native).
3. **New request/response backends.** OCR (Mistral), text completion and infill (OpenAI-style
   `/completions`, Mistral FIM, DeepSeek beta, llama.cpp `/infill`, Ollama `suffix`), a Responses
   API backend (OpenAI, xAI, Groq, DeepSeek, OpenRouter, LM Studio, Ollama, vLLM), a Gemini
   Interactions backend (image, speech, transcription, music, and Omni video out), classification
   and scoring (Mistral classifiers, vLLM `/classify` and `/score`), token counting (Anthropic,
   Gemini, xAI, llama.cpp), and voice listing.
4. **Realtime sessions.** A WebSocket session type on `NSURLSessionWebSocketTask` for OpenAI
   Realtime (conversation, transcription, translation), Gemini Live and live music, xAI realtime and
   streaming STT / TTS, Mistral realtime STT, Together realtime STT / TTS, and vLLM realtime STT.
   OpenAI's WebRTC `live/sessions` stays out: WebRTC needs a third-party stack the core does not take.
