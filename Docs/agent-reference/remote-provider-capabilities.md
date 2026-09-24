<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes for its subject. An agent that learns something durable about this subject adds it to this
file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# Remote provider capabilities

Every inference mode each preset serves, as input → output, set against the backend that reaches it.
Surveyed 2026-09-22 from each provider's official documentation (five research passes; field-level
detail per mode lives in the provider's reference, linked at the end of each section). Account,
files, batch, fine-tuning, vector-store, and agent-hosting APIs are out of scope. The request shapes
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
| text + image → image inside Responses | `image_generation` tool | covered |
| text → image, image(s) + mask → image | `/images/generations`, `/images/edits` | covered (multi-image `image[]`, streamed partial images) |
| text → speech | `/audio/speech` | covered (`stream_format: sse`); a custom voice object goes by name |
| audio → text | `/audio/transcriptions` | covered (`gpt-transcribe` text and stream, diarized_json, words); `whisper-1` family shuts down 2027-02-26 |
| audio → English text | `/audio/translations` | covered until `whisper-1` shuts down |
| text + image → moderation | `/moderations` | covered |
| text → embedding | `/embeddings` | covered |
| text + image → video | `/videos` | covered; shuts down 2026-09-24 |
| speech ↔ speech, realtime transcription, realtime translation | `wss …/v1/realtime`, `…/realtime/translations` | covered (`NFKRealtimeSession`) |
| full-duplex voice | `/v1/live/sessions` (WebRTC SDP) | gap; WebRTC is outside the core's dependency rule |

### Anthropic (`anthropic`)

| Mode | Endpoint | Status |
|---|---|---|
| text + image + PDF → text, tools, thinking | `/messages` | covered (`NFKAnthropicBackend`) |
| plain-text documents; custom-content documents and Files API `file_id` sources | `/messages` | plain text covered; `file_id` sources need the Files API, which is out of scope |
| citations on the reply | `/messages` | covered (`NFKParameterCitations`, `NFKOutputCitations`) |
| server tools: web search, web fetch, code execution (file outputs) | `/messages` | covered: the caller passes the tool by wire shape; calls and results come back under `NFKOutputServerToolResults` |
| token count | `/messages/count_tokens` | covered (`NFKRemoteTokenCounter`) |
| audio, image output, embeddings | — | not offered |

### Google Gemini (`gemini`)

The Interactions API (`POST /v1beta/interactions`, GA 2026-06) is Google's recommended native surface;
`generateContent` is legacy and supported. The OpenAI layer serves chat, embeddings,
`images/generations`, and `videos` only.

| Mode | Endpoint | Status |
|---|---|---|
| text + image + audio + video + PDF → text, tools, structured output | Interactions / OpenAI-layer chat | covered (`NFKGeminiInteractionsBackend`) |
| text + images → image (generate and edit, Nano Banana) | Interactions `response_format: image` | covered |
| text → speech (30 voices, two speakers) | Interactions `response_format: audio` | covered (Interactions) |
| audio → text with diarization and word times | Interactions `transcription_config` (`gemini-3.5-transcribe`) | covered (Interactions) |
| text (+ images) → music | Interactions (`lyria-3.5`, `lyria-3-*`) | covered |
| text + image + video + audio → video, video editing | Interactions (`gemini-omni-1.1-flash`) | covered (Interactions, background); Google names it the default video model over Veo |
| text + images → video | Veo `predictLongRunning` / OpenAI-layer `videos` | covered (`NFKRemoteVideoBackend`) |
| text + image + audio + video + PDF → embedding | `:embedContent` (`gemini-embedding-2`) | covered (native `embedContent`, the preset's default) |
| token count | `:countTokens` | covered |
| live speech ↔ speech, live music | Live `BidiGenerateContent`, `BidiGenerateMusic` WebSockets | covered (`NFKRealtimeSession`; live video input by `sendEvent:`) |
| Imagen | — | shut down |

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

### Mistral (`mistral`)

| Mode | Endpoint | Status |
|---|---|---|
| text + image + audio + document → text, reasoning chunks | `/chat/completions` | covered |
| document / image → markdown, tables, annotations (OCR) | `POST /v1/ocr` | covered (`NFKRemoteOCRBackend`) |
| code infill | `POST /v1/fim/completions` | covered (`NFKRemoteCompletionBackend`) |
| text / chat → moderation | `/moderations`, `/chat/moderations` | covered |
| text / chat → classification | `/classifications`, `/chat/classifications` | covered (`NFKRemoteClassifierBackend`) |
| audio → text (diarize, context bias, word times, SSE) | `/audio/transcriptions` | covered |
| realtime audio → text | `wss /v1/audio/transcriptions/realtime` | covered (`NFKRealtimeSession`) |
| text → speech, voice cloning by reference clip | `/audio/speech` → JSON `{audio_data}` | covered |
| voices | `/audio/voices` | covered (listing) |
| text → embedding (`output_dimension`, `output_dtype`) | `/embeddings` | covered (fields by name) |
| image generation | agents tool only | not a standalone endpoint |

### DeepSeek (`deepseek`)

| Mode | Endpoint | Status |
|---|---|---|
| text + image → text, thinking | `/chat/completions` | covered; `json_schema` is not offered (`json_object` only) |
| Responses | `/responses` (stateless) | covered |
| code infill / raw completion | `/beta/completions` (`suffix`) | covered |
| prefix completion | `/beta/chat/completions` (`prefix: true`) | by endpoint and message shape through `NFKRemoteBackend` |
| audio, embeddings, moderation | — | not offered |

### Groq (`groq`)

| Mode | Endpoint | Status |
|---|---|---|
| text + image → text, reasoning, built-in `browser_search` / `code_interpreter` | `/chat/completions`, `/responses` | covered through chat (`executed_tools` under `NFKOutputServerToolResults`) and `NFKRemoteResponsesBackend` |
| audio → text / English | `/audio/transcriptions`, `/audio/translations` | covered |
| text → speech | `/audio/speech` (Orpheus, WAV, 200-character input) | covered (200-character pieces, WAV joined) |
| embeddings, moderation, rerank, images, video | — | not offered |

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

### Local servers

| Server | Beyond the OpenAI layer | Status |
|---|---|---|
| Ollama | `/api/generate` (`suffix` infill, `think`, `images`), `/api/embed`, `/v1/responses` | infill gap; image generation undocumented over HTTP |
| LM Studio | native `/api/v1/chat` (stateful, `reasoning`), `/v1/responses` | Responses gap |
| llama.cpp | `/completion`, `/infill`, `/reranking`, `/tokenize`, `input_video` / `input_audio` chat parts | infill, tokenize gaps; rerank reachable by URL |
| vLLM | `/v1/completions`, `/score`, `/rerank`, `/classify`, `/pooling`, `/v1/audio/*`, `wss /v1/realtime` STT, `video_url` parts | completion, score, classify gaps; video parts gap |

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
