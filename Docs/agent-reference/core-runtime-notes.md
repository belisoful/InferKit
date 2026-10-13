<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# Core runtime notes

Value-type accessors, tokenizers, grammar-constrained sampling, what a backend declares, tools and the
tool loop, and dynamic backend discovery in the core.

## Value-type convenience accessors

`NFKInferenceResult` and `NFKInferenceRequest` expose typed convenience accessors over their
dictionaries for the keys with a single natural type: `result.text` / `result.structured` /
`result.embedding` (NSArray<NSNumber *> for `NFKOutputEmbedding`) / `result.detections`
(NSArray<NFKDetection *> for `NFKOutputDetections`) / `result.pose`
(NSArray<NFKKeypoint *> for `NFKOutputPose`) / `result.classifications`
(NSArray<NFKClassification *> for `NFKOutputClassifications`) / `result.segments`
(NSArray<NFKAudioSegment *> for `NFKOutputSegments`) / `result.toolCalls` / `result.midi` /
`result.beats` / `result.answers`, and
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
  refuses anything else in `urlRequestForRequest:streaming:error:`, where there is an error out-parameter to report
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

## Tools and the tool loop

`NFKInferenceTool` (name, description, JSON Schema, handler), `NFKBackendTool` (any backend as a
tool), `NFKToolLoop` (the call-and-answer loop, itself an `NFKInferenceBackend`), and `NFKToolMessages`
(the turns) are the core's tool layer (0.4.0). The decisions behind them:

- **The conversation is the OpenAI chat shape.** `NFKInputMessages` already took it, the server's
  OpenAI route already passes it through to the hosted backend, and Qwen-family chat templates render
  `tool_calls` and `role: tool` directly. Each chat backend converts it to its provider's form; the
  per-provider forms are in `remote-providers.md` ("Tool conversations").
- **What the chat shape cannot carry rides under `NFKMessageMetadataKey` (`"inferkit"`).** Sub-keys:
  `format` + `source` + `turn` (a backend's own record of an assistant turn), `toolName`, `isError`,
  and `images` (base64 PNG strings, so a conversation stays JSON and crosses the native route). Every
  backend strips the key before the wire: OpenAI rejects unknown message fields. A record replays only
  when `format` is the backend's and `source` is its `endpointURL.absoluteString`, so a conversation
  moved to another service falls back to the plain fields.
- **Call identifiers are fixed when the assistant turn is built.** A call without an id is given
  `call_<uuid>` by `NFKAssistantMessage` (private, `NFKToolMessageSupport.h`), and the loop reads its
  calls back from that turn (`toolCallsOfAssistantMessage:`), never from `result.toolCalls`, so the
  answer names the identifier the turn carries. The OpenAI backend's replay record takes its
  `tool_calls` from the same built turn for the same reason.
- **The loop stops for a tool it does not hold.** That is the client-run contract the Foundation
  Models backend already keeps for a declared tool without a handler, so a caller mixes its own tools
  with the loop's. The request's own `NFKParameterTools` entries are declared after the loop's, minus
  any name the loop holds.
- **A tool failure is an error result, never a loop failure.** `invokeWithArguments:` checks
  `required` and each top-level property's JSON `type` (an `NSNumber` that is a `CFBoolean` is a
  boolean, not a number; an integral double is an integer), and an `NSError` from the handler becomes
  `"<name> failed: <description>"`. The model reads both and corrects the call. Only a failing model
  turn ends the loop with an error.
- **The job path.** `submitInferenceJobForRequest:` runs on a `QOS_CLASS_USER_INITIATED` global queue,
  submits each model turn through `NFKInferenceSubmit` and waits on its completion, forwards the turn's
  partial results, and cancels the current turn's job from the outer job's cancellation handler. A
  cancel between steps is read from the outer job's status.
- **Wire forms.** `NFKInferenceWireCoding` tags `toolResult` and `toolInvocation`; a result's outputs
  encode recursively, so the audio file a tool made crosses with the invocation.
- **Default descriptions** follow the tool-use guidance providers publish: what the tool does, "Use it
  when …" (`purpose`), what it returns, and per-argument descriptions. The audio and video defaults
  tell the model it does not receive the media, so it does not describe sound it never heard.

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
    (wraps `NFKMLXWhisper`); a consumer's native engine (whisper.cpp) registers to override. Then
    `NFKSpeechAnalyzerProvider` (InferKitAppleSwift), then the core's `NFKSpeechRecognitionProvider`.
  - `NFKCapabilityTranslation` (`"translation"`) → `NFKMLXTranslationProvider` (InferKitMLX), then
    `NFKTranslationProvider` (InferKitAppleSwift). The core ships no translator.
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
  - `.inferkit-keep` → pinned. Eviction skips it; `removeCachedRepo:revision:error:` still removes it. Pinning
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
- The exclusion is eventually consistent when cleared (measured 2026-10-05, macOS 26, APFS, outside
  XCTest). Setting YES reads back at once: 1000 NO→YES folders showed no later change. Clearing an
  exclusion set moments earlier does not: across 3000 YES→NO round trips on fresh folders, the xattr
  was still present right after the clear in 88 and cleared later, and in 434 it cleared and then
  came back within 100 ms. The last change seen was 0.265 s after the clear, and every folder read
  correctly after 3 s. Something applies the earlier YES asynchronously against the PATH: a folder
  set excluded, deleted, and recreated at once at the same path came up excluded within 300 ms in
  86 of 400 trials. That is how
  `NFKHFHubTests` failed with a fixed `NSTemporaryDirectory()/NFKHFHubTests`: each test now gets a
  UUID folder, and the round-trip test waits for the cleared state to hold for 0.5 s.
- `Tools/validation-assets/fetch.py` sets the same exclusion on its asset root and on the
  huggingface_hub cache the reference oracles fill (`--keep-in-backup` skips it).

## Serving identity and certificate pinning

`NFKServerIdentity` makes the `SecIdentityRef` that `NFKInferenceServer.TLSIdentity` takes; the
remote classes trust it by pinning its certificate's SHA-256 fingerprint. The server needs no change.

- The certificate is built by hand in DER (`NFKServerIdentity.m`): X.509 v3, ECDSA P-256,
  ecdsa-with-SHA256 signed by `SecKeyCreateSignature`, subject = issuer = the common name, a random
  16-byte serial, notBefore an hour back, extensions basicConstraints (critical, not a CA), keyUsage
  digitalSignature (critical), extendedKeyUsage serverAuth, subjectAltName (IP entry when
  `inet_pton` parses the name, DNS entry otherwise), and subjectKeyIdentifier. Default validity is
  825 days, the most Apple accepts for a server certificate a client trusts through its trust
  settings. `openssl x509 -text`, `openssl verify`, and `SecTrust` with the certificate as its own
  anchor (SSL policy, localhost) all accept it (checked 2026-10-13, OpenSSL 4.0.3, LibreSSL 3.3.6).
- The key never touches a keychain. It is stored as PEM SEC 1 with the certificate in
  `identity.pem` (0600, written under a temporary name with that mode and renamed into place) and
  rebuilt with `SecKeyCreateWithData` + `SecIdentityCreate`. `SecIdentityCreate` is public in the
  Xcode 27 SDK headers for every platform and available at runtime since macOS 10.12 / iOS 11.2, so
  the macOS 11 floor holds. `kSecImportToMemoryOnly` (`SecPKCS12Import`) is the other in-memory
  route but is macOS 15 / iOS 18 only and needs a PKCS#12 builder.
- Why not the keychain (measured 2026-10-13, macOS 26.6.2, ad-hoc signed probes in `Local/`, user
  interaction disabled so a prompt reports an error instead of a dialog):
  - Login keychain, default ACL: the creating binary signs; a rebuilt binary (new cdhash) finds the
    item and fails to sign with `errSecAuthFailed` (-25293), which is the access prompt when
    interaction is allowed.
  - Login keychain with an any-application ACL passed through `kSecAttrAccess` to
    `SecKeyCreateRandomKey`: the same failure. `security dump-keychain -a` shows the attribute was
    ignored (the sign ACL names the creator's cdhash) and the item carries a `partition_id` of
    `cdhash:<creator>`, which no ACL setting lifts without the keychain password.
  - Data-protection keychain (`kSecUseDataProtectionKeychain`): `errSecMissingEntitlement` (-34018)
    for a tool without a keychain-access-groups entitlement.
  - File key + in-memory identity: no keychain call, so nothing prompts. Two differently built
    ad-hoc binaries loaded the same identity (same fingerprint), and a probe run as a LaunchAgent
    (`launchctl submit`) created, served, and was reached by pinned clients.
  A Developer ID tool has a stable designated requirement and `teamid:` partition, so the login
  keychain would stop prompting after its first approval; that was not measured.
- Trust model: the key is as safe as the user account. Any process running as the user, or an
  administrator, reads `identity.pem`. Loading refuses a file another user owns or a symlink, and
  narrows a looser mode to 0600. A lock file serializes load-or-create across processes.
- Pinning lives in `NFKRemoteTransport`: `sessionWithPinnedCertificateSHA256:` returns a session
  (one per fingerprint, cached) whose delegate is a private `NFKRemotePinnedTrust`. Every remote
  class already takes a `session`, so that is the whole seam. The transport recognizes a pinned
  session by its delegate's class and:
  - refuses an http URL or a malformed pin before sending (`kNFKError_InferenceBackendFailure`);
  - accepts the server trust challenge only when the leaf certificate's SHA-256 matches, skipping
    chain, host name, and date evaluation; otherwise cancels it and marks the task, and the marked
    task's failure is `kNFKError_RemoteUnreachable` with underlying
    `NSURLErrorServerCertificateUntrusted`;
  - refuses a redirect to http;
  - carries the pin into a stream: `streamRequest:` makes its own session with a line-stream
    delegate, which forwards the challenge to the pin. Any other session delegate a caller set is
    still not consulted by a stream.
  The challenge is the task-level delegate method only, which `NSURLSession` uses for server trust
  when the session-level one is absent. A refusal after a matching connection to the same server
  holds (TLS session reuse does not skip the challenge), measured in the probe and asserted in
  `NFKServerIdentityTests`.
- Entry points: `NFKRemoteInferKitBackend.pinnedCertificateSHA256` (sets `session`; reading reports
  the session's pin), `NFKBalancedBackend` `addServerWithBaseURL:apiKey:pinnedCertificateSHA256:`
  plus a fleet-wide `pinnedCertificateSHA256` used for servers without one and for discovered
  servers (applied to both status readings and runs without changing the overridable status seam),
  and `NFKRemoteProvider` `providerWithPinnedCertificateSHA256:`, which `backendForProvider:` and the
  model catalog apply. `providerWithBaseURL:` drops the pin because a pin names one server.
  `+[NFKServerStatus fetchFromBaseURL:apiKey:error:]` takes no pin; a pinned
  `NFKRemoteInferKitBackend`'s `fetchServerStatusWithError:` reads the same route.
