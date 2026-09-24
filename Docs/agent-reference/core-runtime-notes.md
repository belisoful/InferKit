<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# Core runtime notes

Value-type accessors, tokenizers, grammar-constrained sampling, what a backend declares, and dynamic
backend discovery in the core.

## Value-type convenience accessors

`NFKInferenceResult` and `NFKInferenceRequest` expose typed convenience accessors over their
dictionaries for the keys with a single natural type: `result.text` / `result.structured` /
`result.embedding` (NSArray<NSNumber *> for `NFKOutputEmbedding`) / `result.detections`
(NSArray<NFKDetection *> for `NFKOutputDetections`) / `result.pose`
(NSArray<NFKKeypoint *> for `NFKOutputPose`) / `result.classifications`
(NSArray<NFKClassification *> for `NFKOutputClassifications`) / `result.segments`
(NSArray<NFKAudioSegment *> for `NFKOutputSegments`) and
`request.prompt` / `request.negativePrompt` / `request.messages`. Each is a read-only computed getter
that type-checks and returns nil on a mismatch (no crashing cast). Image / mask / video keys stay on
`outputForKey:` / `inputForKey:` because their representation is chosen by the backend or caller
(CVPixelBuffer, texture, CGImage) — do not add typed accessors for those.

## Tokenizers

`NFKTokenizer` is a class cluster: `tokenizerForManifest:directory:error:` reads the manifest's
`tokenizer.type` and returns the subclass named there. The concrete subclasses are private; the
factory is the public entry, so a new type needs no header change.

- `bpe-bytelevel` → `NFKByteLevelBPETokenizer`, the GPT-2 / Qwen / o200k scheme. The
  pre-tokenization pattern is selectable (`"pretokenizer": "gpt2"` (default), `"qwen2"`, or `"o200k"`
  — OpenAI's o200k_base / o200k_harmony, which gpt-oss ships: digits in runs of at most three and words
  split wherever their case pattern turns over — in the manifest spec), and
  the choice is load-bearing: a merge cannot cross a pretoken boundary, so a Qwen vocabulary encoded
  under the GPT-2 pattern produces different, valid-looking ids for the same text ("-pop" is one
  Qwen2 pretoken because a letter run may absorb one leading punctuation character; digits split
  singly; a punctuation run absorbs trailing newlines; whitespace runs ending in a newline hold
  together). Found by the MiniMax Music 3 prompt parity record, which mis-tokenized under the
  default; the Qwen3 text path (`NFKMLXLanguage.backend(directoryURL:)`) now names `qwen2` too.
  `"qwen35"` is Qwen3.5's: Qwen2's pattern with combining marks (`\p{M}`) inside letter runs and out of
  punctuation runs, and each segment NFC-normalized first, as the release's `tokenizer.json` declares.
  `NFKMLXLanguage.pretokenizationName` selects it from a `[\p{L}\p{M}]+` in the `Split` regex. An
  unknown pretokenization name is refused rather than silently defaulted.
- `clip` → `NFKCLIPTokenizer`, which the CLIP image-text model and every Stable Diffusion text encoder
  take. It **subclasses** the byte-level tokenizer through four hooks — `normalizedText:`,
  `pretokenizationPattern`, `symbolsForWord:`, `finalizedText:` — because CLIP shares byte-level BPE
  and differs in exactly those places: text lowercases and its whitespace collapses; the pattern takes
  a run of letters, **one** digit, or a run of other non-space characters, with no leading space (so
  "2024" is four tokens); a word's last character carries `</w>`, which the vocabulary distinguishes;
  and decoding turns `</w>` back into a space.
- `unigram` / `sentencepiece` → `NFKUnigramTokenizer`; `wordpiece` → `NFKWordPieceTokenizer`.

`bytesForTokenId:` (0.3.0) returns the bytes one id contributes to decoded text — a fragment of a
multi-byte character comes back as that fragment, a word-piece word start carries its space, a
special token its literal — which is what a byte-level grammar reasons over; `NFKMLXVocabulary`
reads the whole table once per tokenizer.

`encode:` returns the ids for the text alone. A model input's start and end markers and its padding
are the model's geometry, not the tokenizer's, so they are added where the context length is known —
`NFKMLXSDPromptTokenizer` for the diffusion path.

## Grammar-constrained sampling in the core

`NFKTokenConstraint` (`Sources/InferKit/NFKTokenConstraint.m`, 0.3.1) ports the MLX companion's
byte-level engine into the core so `NFKCoreMLLanguageBackend` can constrain its own sampler:
`NFKTokenVocabulary` holds every id's bytes (from `NFKTokenizer.bytesForTokenId:` at the model's logit
width, or explicit `NSData`s), `NFKJSONConstraint` is JSON syntax with an `NFKJSONRoot` and the same
8-byte whitespace cap the MLX grammar carries, `NFKChoiceConstraint` a fixed set, and a
`NFKTokenConstraintCursor` masks a `float *` logit buffer in place (`-inf` for the inadmissible; the end
token admitted only at a complete document, or when nothing else is admissible so the run stops rather
than emitting a refused token). The grammar state is a fixed-size C struct (`NFKConstraintState`,
64 nesting levels; deeper is refused) so a token's bytes are tried on a copy — no per-state mask cache,
which at the Core ML backend's vocabularies is a few milliseconds a step. The backend builds the
constraint lazily on the first sample (the logit width is only known then) from the core keys
`NFKParameterOutputFormat` (`"json"` / `"json-object"` / `"json-array"`) and `NFKParameterChoices`, masks
Before temperature and nucleus, feeds each emitted token back to the cursor, and returns the parsed
reply under `NFKOutputStructured` when JSON was asked for. The MLX backend honors the same two core keys
as aliases of its own, so a request is engine-agnostic. The schema grammar (`NFKMLXJSONSchemaConstraint`)
stays MLX-only. Two ObjC test traps struck while writing its tests: an `@[ ]` literal inside an
`XCTAssert` macro argument splits the macro (parenthesize the argument), and `NSSet` has
`isSubsetOfSet:` but no `isSupersetOfSet:`.

## What a backend declares

`NFKInferenceBackend` has two optional declarations (2026-09-18): `supportedParameterKeys` and
`supportedInputKeys`, the `NFKParameter*` and `NFKInput*` keys a backend acts on. They exist because
a caller sometimes has to know in advance what an engine honors: the Foundation Models provider
bridge derives Apple's `LanguageModelCapabilities` from them, and a router picks the engine that
honors the key a request needs. A backend that declares nothing is used the way it always was.

- A backend declares only what it acts on. The remote backend declares the keys it translates
  (`NFKParameterTools`, `NFKParameterJSONSchema`, `NFKParameterAudioOutput`,
  `NFKParameterVideoFrameCount`, and the text parameters renamed to the endpoint's spelling) plus
  temperature and seed, whose core spelling is already the endpoint's.
- **The core keys reach the endpoint by renaming (2026-09-19).** The core keys are camelCase
  (`maxTokens`, `topP`, `topK`, `stopSequences`) and a parameter the backend does not translate folds
  into the body under its own name, so those four silently missed every OpenAI-compatible service
  until `NFKRemoteWireNames()` in `NFKRemoteBackend.m` renamed them to `max_tokens`, `top_p`,
  `top_k`, and `stop`. `NFKParameterRepetitionPenalty` goes out under both `repetition_penalty`
  (vLLM, TGI) and `repeat_penalty` (llama.cpp, Ollama): the two name the same multiplicative
  penalty, and no server reads both. The renaming is written into the body before the fold, so a
  caller who sets the endpoint's own name keeps the value they wrote. `NFKAnthropicBackend` maps the
  same keys to the Messages API's own names (`stop_sequences` there), and has no repetition penalty
  to map. The image and video backends were already translating (`size`, `seconds`), which is the
  pattern the chat backends now follow.
- The Core ML language backend declares the sampling keys plus `NFKParameterOutputFormat` and
  `NFKParameterChoices`, and not `NFKParameterJSONSchema`: JSON comes from its token grammar.
- **Reasoning keys (2026-09-19).** `NFKParameterReasoningEffort` is a string, not a number, because
  no two providers agree on a scale: OpenAI names levels (`low` / `medium` / `high`), Anthropic takes
  a token budget, Apple names its own (`.light` / `.moderate` / `.deep`) and allows a custom string.
  The contract names three levels (`NFKReasoningEffortLight` / `Moderate` / `Deep`) and lets any
  other string through, so a caller reaches a level only one provider names. `NFKRemoteBackend` has
  `NFKRemoteReasoningEfforts()` beside `NFKRemoteWireNames()` because this key renames its **value**
  as well as its name. `NFKAnthropicBackend` has a budget table plus a numeric-string escape, and
  refuses anything else in `urlRequestForRequest:`, where there is an error out-parameter to report
  through (`bodyForRequest:attachments:` has only one nil meaning). Extended thinking there forbids
  temperature / `top_p` / `top_k` and needs `max_tokens` above the budget, so the backend drops the
  three and raises the limit.
- `NFKOutputUsage` is a dictionary rather than four output keys so a caller reads one key and finds
  what the provider reported. A count the provider leaves out is **absent**, never zero, which is
  what `NFKRemoteUsage(...)` in `NFKRemoteMediaSupport.m` enforces for both chat backends. The
  Messages API reports no reasoning count (thinking tokens are inside the output total), so that key
  is simply missing there.
- `NFKInferencePrepare(backend, &error)` is the prepare counterpart of `NFKInferenceSubmit`: it calls
  `prepareWithError:` where the backend implements it and returns `YES` where it does not. Swift
  calls it rather than `backend.prepare?()`, which crashes swift-frontend 6.4 in IRGen (the
  reabstraction thunk for an imported throwing function used as a value).

## Dynamic backend discovery

`NFKDynamicBackend` (core, Foundation-only) activates an optional engine only when its classes are
linked into the consumer's build, with no build dependency on that engine. InferKit ships only
zero-dependency backends; a heavier engine (Stable Diffusion, a Core ML/MLX model, a C/Rust runtime) is
brought by the consumer and discovered at runtime.

- A consumer adds a small class conforming to `NFKDynamicBackendProvider` (one method,
  `+makeInferenceBackend`) that builds a backend around their engine.
- InferKit resolves it by name through `NSClassFromString`, so it never references the engine's
  symbols. When the engine is not linked, the class is absent and resolution returns nil — the feature
  is simply unavailable, with no link error and no crash.
- Resolve by provider class name (`+backendForProviderClassName:error:`) or by capability: a consumer
  registers provider class names under a capability string (`+registerProviderClassName:forCapability:`),
  and `+backendForCapability:error:` activates the first present one (most-recently-registered first).
- 2026-09-22: a built-in capability names an ORDERED LIST of default providers, not one. The
  companion providers stay first and the core's own Apple-framework engines (`NFKAppleProviders.h`)
  come last, so a consumer who links nothing still resolves text recognition, segmentation, pose,
  face detection, image embedding, upscaling, optical flow, and transcription. Two rules that go
  with it: a provider returning nil from `makeInferenceBackend` is PASSED OVER rather than treated as
  a failure, which is how the VideoToolbox providers decline on hardware without the processor; and
  the registry is process-wide, so a test that registers into a built-in capability owns that
  capability for the whole run (`testARegisteredProviderStillWinsOverTheCoresOwn` owns face
  detection for exactly this reason).
- Each built-in capability has default provider class names (a `capability → class names` map in the
  core), tried last so a registered override wins:
  - `NFKCapabilityStableDiffusion` (`"stable-diffusion"`) → `NFKStableDiffusionProvider` — **InferKitMLX
    ships it** (wraps `NFKMLXBackend`), so linking InferKitMLX makes `stableDiffusionBackend()` work.
  - `NFKCapabilityTranscription` (`"transcription"`) → `NFKMLXWhisperProvider` — **InferKitMLX ships it**
    (wraps `NFKMLXWhisper`); a consumer's native engine (whisper.cpp) registers to override.
  - `NFKCapabilityTextGeneration` (`"text-generation"`) → `NFKFoundationModelsProvider` —
    **InferKitFoundationModels ships it** (wraps `NFKFoundationModelsBackend`), so linking that package
    activates on-device LLM.
  - `NFKCapabilityControlNet` (`"controlnet"`) → `NFKControlNetProvider` — no shipped default; a consumer
    brings a ControlNet/SD engine and adopts that name or registers their own.
  Providers build lazily (construction is cheap; weights/pipeline initialize on first use, off the
  render thread). This is how the **existing** SD / Whisper / Foundation Models implementations activate
  in the core only when the companion is linked, with no build dependency.

## Hugging Face hub cache policy

`NFKHFHub` manages its cache in snapshots, one `<cache>/<repo>/<revision>` folder each (added
2026-09-22, InferKit 0.4.0).

- Two marker files, each a separate flag:
  - `.inferkit-owned` → the hub owns the snapshot and may evict it. Its modification date is the
    snapshot's last use; a download and a cache hit both write it. The hub writes it into every
    snapshot it downloads into, so a new snapshot is owned from the start.
  - `.inferkit-keep` → pinned. Eviction skips it; `removeCachedRepo:` still removes it. Pinning
    creates the folder, so a model can be pinned before its first download.
- Ownership is positive on purpose. A missing marker means "not the hub's", so a user-chosen folder
  never loses converted packages, fine-tuned weights, or other files the hub cannot re-download. The
  layout alone cannot find snapshot boundaries (an org-less repo such as `gpt2/main` looks like
  `org/model`), and the marker is the only reliable last-use date (atime is lax on macOS).
- A cache written before 0.4.0 has no markers. A snapshot becomes owned on its next download or cache
  hit, or through `adoptCachedRepo:revision:error:`, which needs the repo name because the layout
  cannot be walked for it.
- Eviction runs after every successful fetch, never on a cache hit. It removes whole snapshots, least
  recently used first, and skips the snapshot just requested and any holding a `*.download` file. The
  transfer itself lands in the system temporary folder and is moved into the snapshot only at the
  end, so an in-flight download elsewhere is visible only briefly. A trim failure never fails the
  download.
- Size is `NSURLTotalFileAllocatedSizeKey` over every regular file, so a 5-byte file counts as one
  allocation block. Tests size their limits from `cacheSize` after one download, never from byte
  counts.
- The limit and the backup exclusion each have a process-wide class default that `init` copies.
  `NFKMLXDownload` and `NFKMLXSDRelease.download` create a hub per call, so the class default is the
  only setting that reaches them.
- The access token has a process-wide default too, `defaultAccessToken`, for the same reason: every
  InferKitMLX download factory (`NFKMLXReleaseDownload`, `NFKMLXDownload`, the SD release) makes its own
  hub, so before it existed a gated repository was reachable through them only from a process with
  `HF_TOKEN` set, which an app is not. Unlike the limit and the exclusion, it is read on each request
  rather than copied at `init`: a hub's own `accessToken`, then the default, then `HF_TOKEN`.
- Backup exclusion is `NSURLIsExcludedFromBackupKey` on the cache root: the xattr `tmutil
  addexclusion` (without `-p`) sets on macOS, the iCloud backup exclusion on iOS and tvOS. It defaults
  to on because App Review expects re-downloadable content kept out of iCloud backup, and the same
  reasoning holds for Time Machine. `NO` means the hub leaves the attribute alone; it never clears it.
  `isExcludedFromBackup:` drops the URL's cached resource value first, or a change made through
  another `NSURL` instance reads stale.
- `Tools/validation-assets/fetch.py` sets the same exclusion on its asset root and on the
  huggingface_hub cache the reference oracles fill (`--keep-in-backup` skips it).
