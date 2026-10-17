<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# Core runtime notes

Value-type accessors, tokenizers, grammar-constrained sampling, what a backend declares, tools and the
tool loop, dynamic backend discovery, and the model catalog in the core.

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
A third, `supportedOutputKeys` (2026-10-14), lists the `NFKOutput*` keys a backend writes; the model
catalog derives a descriptor's outputs from it ("Model catalog" below).

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

## Model catalog

`NFKModelCatalog` (`NFKModelCatalog.h`), `NFKModelDescriptor` and its value types
(`NFKModelDescriptor.h`), and the vocabularies (`NFKModelVocabulary.h`) were added 2026-10-14 for
InferKit 0.4.0. The consumer reference is `Docs/model-catalog.md`. The decisions behind them:

- **One schema for the core, the companions, and plugins.** A descriptor's dictionary is the entry a
  plugin manifest carries, so the shared catalog, a companion's provider, and a plugin's JSON file
  describe a model in one shape and an app reads all three the same way. The manifest envelope is
  `{"schemaVersion": 1, "kind": "inferkit.model-catalog", "models": [...]}`. A newer
  `schemaVersion` is refused (`NFKModelCatalogErrorUnsupportedSchemaVersion`); a missing or different
  `kind` is malformed.
- **Unknown fields are preserved.** Every object keeps the fields it does not read in `extensions`
  and writes them back beside its own, so a manifest from a newer writer passes through an older
  reader unchanged. A later field (an option's `availability`, a choice's memory figures) rides in
  `extensions` until it becomes a property. A known field passed in `extensions` to an initializer is
  dropped, so the property always wins.
- **Plain string vocabularies.** Value types, tasks, engines, residencies, option kinds and stages,
  file roles, and source kinds and layouts are `extern NSString * const` sets. A plugin extends a set
  by using a new string, and a reader passes an unknown one through. A task that an `NFKCapability*`
  already names keeps the capability's string (`NFKModelTaskPoseEstimation` is `pose`, and text
  recognition is `text-recognition`), so a task resolves through `NFKDynamicBackend` as it is.
- **Provider discovery by class name.** The shared catalog loads once from `NFKAppleModelDescriptors`,
  `NFKMLXModelDescriptors`, `NFKFoundationModelsDescriptors`, `NFKAppleSwiftModelDescriptors`, then
  each registered name, through `NSClassFromString`, as `NFKDynamicBackend` finds backend providers.
  A Swift provider needs an `@objc(Name)` so the runtime name is unmangled. A later provider's
  descriptor replaces an earlier one with the same identifier. Registering after the shared catalog
  loaded adds the provider at once, and registering a name again reloads it. The shared instance is
  published under the registry lock before the providers load, so a provider that registers another
  while it loads does not deadlock on the `dispatch_once`.
- **One key string, two types.** `NFKParameterCitations` (boolean) and `NFKOutputCitations` (JSON)
  share `"citations"`, the only core request key and result key that share a string with different
  types. So the table has two lookups: `NFKModelValueTypeForKey` resolves the request side first and
  `NFKModelValueTypeForOutputKey` the result side first. A descriptor types an `outputs` port without
  a `valueType` through the second. Every other shared string (`image`, `audio`, `messages`, …) has
  one type on both sides.
- **Minimum memory is the residency budget.** `minimumMemoryBytes` is the smallest recommended Metal
  working set whose budget admits the model at its default options beside that budget's reserve.
  `NFKModelMemoryBudget` (core) holds the fraction (0.85) and the reserve rule (4 GiB from a budget of
  8 GiB, half the budget below it; `hardware-and-model-sizing.md`); the MLX provider computes its
  figures through `NFKMLXResidencyBudget`, which reads the rule from the core and calls through to its
  planner (see "Load options, memory, and the
  build entry" below). A variant's `peakBytes` is the most it holds at once while loading or
  running at its default options, which can exceed `residentBytes`.
- **Writing is deterministic.** `manifestDataWithError:` writes pretty-printed, sorted-key JSON with
  the models sorted by identifier, so a re-export of an imported manifest is byte-identical
  (`testAManifestRoundTripsByteForByte`). A writer leaves an absent optional value out and never
  writes `null`; booleans, `requirements`, and the descriptor's arrays are always written, and an
  option's `choices` only when it has some.
- **Leniency where a field has an obvious default.** A missing `displayName` is the identifier, a
  missing option `kind` follows the key's value type, a missing license `name` is its `id`, a missing
  file `role` is `other`, a missing source `layout` is `none`, and missing `requirements` are none.
  `id`, `family`, and `engine` are required on a model; `key` on a port and an option; `kind` on a
  source; `path` on a file; `id` on a variant; `value` on a choice.
- **Refused where a value would mislead or run away.** A repeated identifier is malformed: two models
  of one manifest with one `id` (`MalformedManifest`, checked in `addDescriptorsFromManifestData:error:`),
  and within a model two inputs, outputs, or options with one `key`, two variants or presets with one
  `id`, or two stages of one variant's facts with one `id` (`MalformedDescriptor`, through
  `-[NFKModelFieldReader requireUniqueElements:ofField:keyPath:name:error:]`). A lookup by identifier
  returns the first match, so a repeat would hide an entry without a word. A count that drives a loop
  has a documented maximum: a stage's `layerCount` is at most 4096
  (`NFKModelStageFacts.maximumLayerCount`), because the planner builds one entry per layer and a
  manifest stating 1e12 would hang the estimate. It is the only such count in the schema; the option
  combinations already stop at 4096, and the other numbers enter arithmetic only. Data that does not
  parse keeps the parser's error as `NSUnderlyingErrorKey` and its debug description in the message.
- `descriptorWithIdentifier:derivedFromBackend:` reads the optional `supportedInputKeys`,
  `supportedParameterKeys`, and `supportedOutputKeys` (the last added to `NFKInferenceBackend` with
  the catalog), sorts each, and types each key from the table. The family is the backend's
  `NFKModelInfoArchitecture` when its `modelInfo` reports one, else its `backendIdentifier`.
- **Every core backend declares its outputs (2026-10-15).** Each set is read off the backend's own
  result paths, conditional keys included: `NFKRemoteBackendRawKey` is in the set of every remote
  backend that writes it, and the streamed partial results write a subset of the final keys. A set
  follows the backend's configuration where its result does: `NFKVisionFaceBackend` drops
  `NFKOutputPose` when `detectsLandmarks` is NO, `NFKVisionTrackingBackend` writes
  `NFKOutputStructured` for trajectories and `NFKOutputDetections` otherwise, `NFKCoreMLBackend`
  lists the loaded model's output features (none before load), and `NFKPassthroughBackend` lists its
  map's keys (none when it echoes). `NFKToolLoop` adds its four keys to its backend's set.
- **The server carries the outputs.** `NFKInferenceServer`'s model entry (`/v1/models/<name>`,
  under `inferkit`) carries `outputs` beside `inputs` and `parameters`, sorted, and only when the
  backend declares a set. The status route strips all three. `NFKRemoteInferKitBackend` reads
  `outputs` in `prepareWithError:` and answers `supportedOutputKeys` only when the entry had it;
  `NFKBalancedBackend` copies the chosen server's set the same way.
- **The Apple provider.** `NFKAppleModelDescriptors` (public header) is the core's provider: one
  descriptor per Apple-framework engine, engine `apple`, source `system` with layout `none`, license
  "Apple system framework", and the identifier as the registry name. `+backendNamed:` is the
  registry: it builds the backend a registry name names, and each descriptor's ports and options
  are read off that backend, so they cannot drift from it. VideoToolbox gets one descriptor per
  task (`videotoolbox-super-resolution`, `-frame-interpolation`, `-optical-flow`) because the
  tasks differ in inputs, floor, and capability; tracking gets `vision-tracking` and
  `vision-tracking-trajectory` because the kinds differ in outputs. `vision-coreml` has none: it
  runs a model the app supplies. The VideoToolbox descriptors set `requiresAppleSilicon`, as the
  frame processors need it; the rest do not.

## Load options, memory, and the build entry

Added 2026-10-15 for InferKit 0.4.0 (`NFKModelMemory.h`, `NFKModelOptions.h`, `NFKModelBuilding.h`,
`NFKModelPreset.h`). The consumer reference is `Docs/model-catalog.md` ("Memory", "Presets", "Load
options and the build"). The decisions behind them:

- **The settings field is `options`.** The descriptor field, the property, and the initializer label
  are `options`. The reader reads no other spelling, since schema 1 has no earlier writer to stay
  compatible with: a descriptor that names the field `"parameters"` keeps it in `extensions`.
- **The derived `variant` option is not written.** A descriptor with several variants and no declared
  `variant` option gets one from the variants, built in the initializer. It is in `loadOptions` and
  `optionWithKey:`, not in `options` or the manifest, so a round trip is unchanged.
- **`automatic` is a residency choice, the default, with no memory figure of its own.** The planner
  resolves it per machine; the estimate reports `requestedResidency` and the resolved `residency`. A
  constraint does not count `automatic` when it decides that exactly one residency fits, and
  `descriptorByDerivingMemoryFigures` leaves its choice without a figure.
- **A provider derives in its own catalog.** `+[NFKModelCatalog sharedCatalog]` calls each provider's
  `+modelDescriptors` inside its `dispatch_once`, so a provider that called
  `descriptorByDerivingMemoryFigures` there would reach the shared catalog again and trap.
  `descriptorByDerivingMemoryFiguresInCatalog:` takes the catalog that resolves model references and
  drafts; InferKitMLX passes one holding its own raw descriptors.
- **The planner is ported, not redesigned.** `NFKModelResidencyPlan +placementForStages:…refusal:`
  (private) is the Swift `plan(_:residency:budget:)` line for line: the same budget rules
  (`holds` = budget > 0 and bytes + reserve(budget) ≤ budget; `admits` = budget ≤ 0 or bytes +
  reserve(budget) ≤ budget, the reserve scaled to the budget since 2026-10-17), the same expert-cache
  headroom, the same streamed rooms, the same widening, and the same refusal sentences, whose GiB
  figures are `%.1f` of bytes / 2^30. A sentence ends "against a N GiB budget": the figure is the
  planner's `budget` argument, which a caller passes as `NFKModelMemoryBudget.byteCount` (0.85 of the
  working set), so the word names what it prints, and the reserve it names is that budget's. The
  Swift reference in `NFKMLXPlannerDifferentialTests` says the same. The public `planForStages:` returns
  the refusal as `NFKModelCatalogErrorDoesNotFit`; the estimate keeps the refused plan's figure and
  reports fits NO. `NFKMLXResidencyBudget` calls through to it. `NFKMLXPlannerDifferentialTests` keeps
  the Swift planner's last body as its reference and compares it with the core planner and the
  call-through over a grid of footprints, residencies, and budgets (`NFKModelMemoryTests` holds the
  sentences the two share).
- **The estimate puts the key-value cache, the draft, and the linear options on the cache stage**
  before planning, so a staged model's figure is its largest stage including them. With no budget,
  `automatic` plans staged, so a budget-free figure (`descriptorByDerivingMemoryFigures`) is the
  largest stage.
- **Quantization approximations.** A quantized stage's pageable and largest-layer bytes scale by the
  stage's own ratio, and a quantized stage does not widen. A precision a stage does not list converts
  from its checkpoint bytes by the element-size ratio, integer tensors included.
- **The linear maximum is analytic.** (budget − reserve − peak at 0) / bytes per unit, where the bytes
  per unit come from the facts' per-position cache at the combination's cache precision or bits, and
  is exact where the cache stage is the stage the plan holds largest. A combination fits when it fits at
  the linear options' given values, else their minimums.
- **Load-stage cache defaults ride in the request defaults.** The memory model's context length, cache
  bits, and cache group size are load options a request may override. The build puts
  their resolved values in `requestDefaults`, so the backend runs at the figure the estimate assumed.
- **Builders.** The descriptor's provider class (the catalog records which provider supplied each
  descriptor), then the engine builder, then `providerClass` through `NFKDynamicBackend` with only the
  property options. A builder applies the property options itself, so a rebuild reapplies them; the
  setter check runs before any value is set.
- **`NFKDefaultedBackend` and the unload work.** It implements `unload`, `isLoaded`,
  `residentByteCount`, and `peakByteCount` against a private protocol, forwarding through
  `respondsToSelector:`, so it compiles before and after `NFKInferenceBackend` declares them.
  `respondsToSelector:` answers the wrapped backend's for every optional member.
- **Requirements checks** read `NFKHardwareProfile.operatingSystemName` / `operatingSystemVersion` /
  `isAppleSilicon` / `hasNeuralAccelerators`. A snapshot profile answers from its dictionary, the
  neural accelerators included, so an app checks a remote machine. An unknown feature string is never
  met.
- **A draft counts its own cache.** The draft component is the draft stage's weights plus the draft's
  key-value cache (its default variant's facts) at the run's context length, cache precision, and cache
  bits. The context length's bytes per position add the draft's per-position cache, so the analytic
  maximum accounts for it.
- **The enumeration is capped at 4096 combinations.** Past the cap, each option's choices are evaluated
  one at a time with the others at the caller's selection, else the defaults; each combination is
  tagged with the option it varies, and an option's statuses read only its own combinations.
  `NFKModelMemoryConstraints.exhaustive` (JSON `exhaustive`, true when absent) says which happened.
- **A refusal carries the way out.** The build's DoesNotFit computes constraints with nothing fixed and
  the fitting combination nearest the requested load options (the private `preferred:` argument), puts
  it and its estimate in userInfo, and appends "Choosing <key> <value>[, …] fits: it holds <peak>." to
  the description, or the "No combination of options fits" sentence when nothing does.
- **`vocabularySize` is in the memory facts.** The draft rule (same family, same vocabulary size, fewer
  parameters) needed a field the spec did not list; the facts carry it.

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
  transfer streams into `<file>.download` beside its destination and is moved into place only whole,
  so every in-flight download protects its snapshot for its whole length. A trim failure never fails
  the download.
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

## Download plans, progress, status, and removal

`NFKDownload`, `NFKDownloadPlan`, and the `NFKHFHub (NFKDownloads)` category (`NFKHFHubDownloads.m`,
added 2026-10-15, InferKit 0.4.0). The transfer is `NFKHFHubTransfer` (private).

- One transfer path. `fetchURL:toFileURL:error:` stays the seam (the core tests' `FxHFHubStub` and
  InferKitMLX's `NFKOfflineHub` override it). Its default streams an `NSURLSessionDataTask` straight
  into the destination over a session made from `session.configuration` with the transfer as its
  delegate, so a test's `protocolClasses` carry over. Plan requests go through the second seam,
  `fetchMetadataFromURL:HTTPMethod:response:error:`. A response of another scheme (a `file://` URL)
  reads as status 200 with its length, which is how the compiled examples run offline.
- The current download is thread-local (`NSThread.threadDictionary`, a stack), the `NSProgress`
  `becomeCurrent` pattern. `fetchFileFromURL:…` reads it, reports the file, registers the snapshot,
  and passes an `NFKHFHubTransferReport` to the default transfer through the thread dictionary
  around the seam call. A subclass's transport therefore still lands the file; the download counts its
  bytes at the end from the file's size.
- A download finishes when its last activity ends (`beginActivity` / `endActivityWithError:`). The
  hub's plan-driven downloads hold one activity; `NFKMLXReleaseDownload.asyncValue` and
  `NFKMLXDownload.backend` each hold one around the factory's work; the core's legacy
  `downloadRepo:…completionHandler:` holds one too. A synchronous caller brackets its own.
- Progress accounting: a file the plan or the cache marks cached counts toward the file counts and
  adds no bytes. Received bytes per file are a running maximum, so a transfer that restarts from byte 0
  (the server ignored `Range`, or `If-Range` found a changed file) never lowers the total. Reports are
  throttled to ten a second and forced at every file start, file end, and state change, and are built
  under the lock and dispatched in order to one serial queue. The completion handler is queued after the
  final report, and `waitUntilFinishedWithTimeout:` returns only after both have run: reading
  `progressHandler`'s last snapshot right after the state turned final once found it a report short.
  Called on the report queue, the wait does not block and answers `isFinished`. A fetch that no plan
  declared and that fails with no bytes (an optional file the release lacks, a weights candidate it
  does not serve) leaves no record.
- Resumption: up to three resumes after `NSURLErrorNetworkConnectionLost`, `TimedOut`,
  `CannotConnectToHost`, or `ResourceUnavailable`. A resume sends `Range: bytes=N-` and `If-Range`
  with the first response's strong ETag; without a strong ETag the transfer restarts from zero. A 206
  whose `Content-Range` starts elsewhere, or a 416, restarts. A content-encoded response's
  `Content-Length` is the encoded length, so only an identity body's length checks the file size.
- Cancellation sets the flag first and then runs the registered handlers, and a transfer checks the
  flag under the same lock it publishes its task under, so a cancel that lands before the task exists
  still stops it.
- The access token goes only to `endpointURL`'s host (`requestForURL:`). The legacy transfer sent it
  to whatever URL it fetched; a plain URL source must not leak a Hugging Face token to another host.
- Errors: `NFKDownloadErrorDomain` (`NS_ERROR_ENUM`, so Swift catches `NFKDownloadError.cancelled`).
  HTTP 401 → `AuthenticationRequired`, worded by whether a token was sent; 403 →
  `AuthenticationRequired` (a gated repository whose terms the account has not accepted); 404 →
  `NotFound`. The Hugging Face API answers 401 for a repository that does not exist when no token is
  sent, so that message names all three causes. A checksum mismatch moved from
  `kNFKError_InferenceBackendFailure` to `NFKDownloadErrorChecksumMismatch` (pre-1.0). A missing cache
  folder stays `kNFKError_InferenceNotReady`. An unsafe name is `UnsafePath`; a file whose size differs
  from its listed size is `SizeMismatch`.
- Unsafe names: every repository, revision, path, listing entry, recorded path, URL host, and URL file
  name passes `NFKHFHubUnsafePathError` before it becomes a cache location. Refused: empty, a leading
  `/`, an empty, `.`, or `..` component, a backslash, a NUL; a host or final file name also may not
  hold `/`. The URL builders (`localURLForRepo:…`, `remoteURLForRepo:…`, `localURLForURL:fileName:`)
  return nil; every other entry point fails with `NFKDownloadErrorUnsafePath` before any request or
  file access, and a status reads absent. A listing with one unsafe file entry refuses the whole plan:
  the server is broken or hostile. A revision may hold `/` and is still one folder:
  `revisionFolderName:` stores `%` as `%25` and `/` as `%2F`, so `refs/pr/1` caches at
  `<repo>/refs%2Fpr%2F1` and a revision `main/x` sits beside `main`, never inside it (removing `main`
  once deleted `main/x` with it). A revision without either character keeps its plain name. The
  resolve URL keeps the revision as it was (`resolve/refs/pr/1/<path>`); the tree API percent-encodes
  it as one segment.
- Containment is the second guard: `requireURL:insideCacheURL:error:` runs before every write, trash
  move, and removal (`fetchFileFromURL:…`, `removeSnapshotAtURL:…`, `moveURLs:toTrashInCacheURL:…`,
  the completion record, pin, unpin, adopt). It applies `.` and `..` lexically and accepts a path
  strictly under any of three forms of the root: as given, `URLByResolvingSymlinksInPath` (which
  drops a leading `/private`), and `realpath` (which keeps it). `NSDirectoryEnumerator` over a
  `/var/folders/…` root hands back `/private/var/folders/…` URLs, so a trim needs the `realpath` form
  (probed 2026-10-16). The lexical check is followed by a real-path check unless
  `followsFolderLinksOutOfCache` is set: the folder the item lands in, resolved by `realpath` through
  its deepest existing ancestor (`NFKHFHubRealPath`), must lie under the real path of the cache root.
  The item itself is not resolved, because a removal moves a link and never its target, and a write
  replaces a link. So a file link inside a snapshot (the InferKitMLX release-store tests seed those) and
  a cache root that is itself a link stay supported, and a folder inside the cache that links to
  another volume needs the opt-in. The `.inferkit-owned` marker is removed first when it is a link,
  and the pin is written atomically, so neither writes through a link.
- `NSURL.lastPathComponent` percent-decodes and then splits, so `a%2F..%2Fx.bin` reads `x.bin`; a final
  `..` segment reads `..` and is refused.
- Plans: `<endpoint>/api/models/<repo>/tree/<revision>?recursive=1`, the revision percent-encoded as
  one segment (`refs/pr/1` → `refs%2Fpr%2F1`), following `Link: <…>; rel="next"` pages. A Large File
  Storage entry's `lfs.oid` is the content SHA-256 and `lfs.size` the size; a plain entry's `oid` is a
  Git blob ID and is not reported as a SHA-256. Globs are `fnmatch` without `FNM_PATHNAME`, as
  huggingface_hub's `allow_patterns` are. A plan marks a file cached when it exists at the listed size;
  it never hashes, because hashing gigabytes at plan time stalls a user interface.
- Listed size: `fetchFileFromURL:…` compares the finished `.download` file with the plan's byte count
  before it moves the file into place, after any transport, an overridden `fetchURL:toFileURL:error:`
  included. The transfer's own check compares with the response's `Content-Length`, so a 200 captive
  page with an honest length passes it and fails this one. A URL plan keeps a listed size over the
  HEAD answer; HEAD fills in only an unknown size. The LFS SHA-256 check follows the size check.
- Per-file places: an `NFKModelFile` with `repository`, `revision`, or `URL` comes from there
  (`repository` read at its own `revision` or "main"; `revision` alone is the source's repository at
  that revision; `URL` is a download URL named by `path`). `partsOfModelSource:` groups a source's
  files by place (repository + normalized revision, or one part per URL file), in first-file order. A
  source of one part that is its own repository takes the single-repository path unchanged. Otherwise
  `planParts:…` checks every part (`refusalOfPart:`: repository, revision, each path, the URL's host
  and file name) before the first request, lists each repository once, HEADs each URL, and joins the
  files; each `NFKDownloadPlanFile` carries its own `snapshotURL` (internal), which `runPlan:` registers
  in the in-use registry and fetches into. The plan's `localURL` and `snapshotURL` stay the source's.
  An optional URL file the server answers 404 is left out. No completion record is written: every
  part is a listed-file plan.
- Status of such a source reads each part in its own snapshot and joins present and missing paths:
  complete only when no required file of any part is missing. Removal makes one `NFKRemovalPlan` per
  part, joined under the internal `parts`; `removeFilesInPlan:` refuses when any part's snapshot is in
  use, plans each part again, and moves every file of every part into one trash folder, all or none.
  Claims run over every catalog variant's parts, so a base model's complete download keeps the files
  an adapter's removal names in the base repository, and a complete adapter keeps them through the
  base model's removal. A URL target claims by its cached file name; a URL source's removal names that
  file, so the source is the same download as itself.
- Admission: a plan whose total exceeds a non-negative `cacheSizeLimit` is refused before any
  transfer (the limit could never hold it), and so is one whose remaining bytes exceed the volume's
  `NSURLVolumeAvailableCapacityForImportantUsageKey` (`NSURLVolumeAvailableCapacityKey` on tvOS, which
  lacks the former).
- URL sources cache at `<cache>/@url/<host>/<digest>/<file name>`, digest the first 16 hexadecimal
  digits of the URL's SHA-256, a file URL's host `local`. `@` cannot begin a Hugging Face name. The
  folder is an ordinary snapshot to eviction, status, and removal.
- Completion record: `.inferkit-complete` in a snapshot, a JSON object with `paths` (the complete
  download's files) and `absent` (optional files the server refused with HTTP 404), each unioned on
  write under one process-wide lock; a path recorded present leaves `absent`. A bare JSON array, the
  first format, reads as `paths`. A whole-repository plan and every InferKitMLX release download write
  it (`recordCompleteDownloadOfRepo:revision:paths:absentPaths:error:`). `paths` is how a source that
  lists no files reads complete; without it such a snapshot reads partial whenever it holds a file.
  `absent` is how a cached release loads offline: the release layer skips a recorded refusal.
  `forgetRecordedAbsentPathsOfRepo:revision:error:` clears it and writes nothing when there is no
  record.
- Status counts a present file's allocated bytes at its link target. Optional files add bytes and do
  not decide completeness. A source of kind `system` or `none` is complete with no bytes.
- Removal plans read the cache alone. A file is kept when another catalog variant (`hub.modelCatalog`,
  default the shared catalog) at the same repository and normalized revision lists it and that
  variant's download is complete. A variant whose listed paths equal the target's (or that, like the
  target, lists none) is the same download and claims nothing. An unlisted variant claims the paths its
  snapshot's completion record names. Only complete variants claim: a partial one does not load, and
  counting it would make a file unremovable whenever the removal target's own files made the sibling
  look partial.
- The hub's layout is `<repo>/<revision>/<path>`, not huggingface_hub's `blobs/` + `snapshots/`, so
  sharing between snapshots exists only as links someone placed (the InferKitMLX release-store tests
  link a cache to the validation store; a user may link a Python cache). A symbolic link is removed and
  frees nothing; its target inside the cache goes with it unless another link in the cache resolves to
  it (kept, "linked from N other cached file(s)") or it is a file of another snapshot (kept, "a file of
  another cached snapshot"), and a target outside the cache is always kept. A regular file that a link
  outside the removal resolves to is kept the same way, so the other snapshot's completion record keeps
  matching its files. Another snapshot is recognized by a bookkeeping file (`.inferkit-owned`,
  `.inferkit-complete`, `.inferkit-keep`) in a folder between the target and the cache root; a file in
  an unmarked folder reads as a blob and goes with its last link. A hard link with other names is
  removed and frees nothing.
- All or none: the planned files and, when no model file is left, the `.inferkit-owned` and
  `.inferkit-complete` markers move into `<cache>/.inferkit-removing-<UUID>`; a failed move puts every
  moved file back. Only then is the folder deleted; one a crash left behind is deleted by the next
  removal. Folders left empty are removed up to the cache root. `.inferkit-keep` stays: a pin is the
  app's setting, not a downloaded file.
- `removeFilesInPlan:` plans again under the snapshot-registry lock and removes the intersection of the
  two plans, so a file that became shared meanwhile is kept. While the lock is held no download can
  register a snapshot, and a snapshot an unfinished `NFKDownload` registered refuses the removal with
  `NFKDownloadErrorInUse`. A download that only cache-hits registers nothing. `removeCachedRepo:…`
  refuses the same way.
- Eviction honors the registry too: `evictableSnapshotsInDirectoryAtURL:` leaves out a registered
  snapshot, and the trim checks and removes each snapshot under the registry lock, so a multi-file
  `NFKDownload` keeps its snapshot between files, when no `.download` file marks it. A fetch registers
  its snapshot before it creates the folder.
- Holds: `fetchFileFromURL:…` holds its snapshot in the registry for the whole call
  (`+[NFKDownload holdSnapshotURL:]` / `endHoldOfSnapshotKey:`), with or without a current
  `NFKDownload`, so a synchronous `downloadRepo:…error:` is safe from a concurrent trim and removal
  while it runs. Between calls, `holdSnapshotOfRepo:revision:error:` returns an `NFKHFHubSnapshotHold`
  that keeps the snapshot in use until `invalidate` or deallocation. Holds and downloads share one
  counted set, so a snapshot is free only when every hold and download has let go. InferKitMLX's two
  release helpers (`NFKMLXReleaseDownload.directory(…)`, the flat and the component form) hold the
  release for the fetch of all its files through `NFKHFHub.holdRelease(repo:revision:)`, and so do the
  loaders that fetch file by file outside them: `NFKMLXOpenJev` (release and base, each its own hold),
  `NFKMLXLayaRelease`, `NFKMLXOpenJevDeBERTa`, `NFKMLXVoiceRestoreBackend` (transformer and vocoder
  repositories), and `NFKMLXBackend`'s Stable Diffusion release download.
- InferKitMLX keeps each hold until the model build that reads the files returns. Its download
  factories open an `NFKMLXBuildHolds` scope around the fetch and the build, and `holdRelease` hands its
  hold to that scope, so no trim or removal deletes a release between its fetch and its read
  (`mlx-companion.md`, "Holds across a build").
- Streaming limit: each transfer stops with `NFKDownloadErrorSizeMismatch` once it would write past
  its limit: the listed size, else the length the response announces, lowered to a ceiling. The
  ceiling is `maximumFileByteCount` when set, else the cache volume's free space for a file of unknown
  size. The transfer report is now made on every fetch, a download or not, to carry the limit. A stub
  body that stalls after its first chunk proves the stop: without the limit the transfer waits for
  bytes that never come.
- Tests: `NFKDownloadTests` and `NFKDownloadRemovalTests` against `NFKStubHubProtocol` (an
  `NSURLProtocol` at `http://hub.test` with Range, a stall after the first chunk, and a drop part-way),
  never the network. A cancellation test polls `currentProgress`: the throttled handler may never
  report the single chunk that arrives before a stall.

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
  and `NFKRemoteProvider` `providerWithPinnedCertificateSHA256:`. Every factory that takes a provider
  applies it through the private `NFKRemotePinnedSessionForProvider` (`NFKRemoteTransportPrivate.h`):
  `backendForProvider:` on `NFKRemoteProvider` and on each remote backend class, the model catalog,
  `NFKRealtimeSession`'s `sessionForProvider:apiStyle:apiKey:modelName:` (its `URLSession`), the
  file, retrieval, usage, token-count, rerank, and voice-library factories, and the Ollama and LM
  Studio runners. A factory whose service has its own fixed host (usage reporters, Gemini's file and
  retrieval stores) sends there through the same pin, so it fails closed. A new factory that takes
  an `NFKRemoteProvider` applies the helper too. `NFKRemoteProvider`'s `isEqual:` compares pins
  normalized, so the colon-separated and plain spellings of one fingerprint are equal.
  `providerWithBaseURL:` drops the pin because a pin names one server.
  `+[NFKServerStatus fetchFromBaseURL:apiKey:pinnedCertificateSHA256:error:]` (and its
  completion-handler form) reads the status route through the pinned session; the unpinned
  `fetchFromBaseURL:apiKey:error:` is that call with a nil pin. A pinned `NFKRemoteInferKitBackend`'s
  `fetchServerStatusWithError:` reads the same route.
- A realtime socket on a pinned session opens only a `wss://` URL: the trust delegate checks the
  certificate, and `NFKRealtimeWebSocket` closes a `ws://` socket unopened with
  `kNFKError_InferenceBackendFailure`, since the transport's https preflight does not see socket tasks.
- The pinned-session cache is unbounded by design. Clients keep the shared session they were given,
  and a session cannot be reused once invalidated, so evicting one would break live clients, while
  dropping it uninvalidated leaks it anyway (a session holds its delegate until invalidated). It
  grows by one session per distinct fingerprint a process pins.
- The Bonjour TXT `cert_sha256` is advisory by default: discovery reports it in `advertisedProperties`
  and pins nothing. Automatic pinning (2026-10-13) is opt-in through
  `NFKRemoteProvider.discoveryPinnedServers`, an `NFKPinnedServers` (nil by default). With it set,
  `discoverInferKitServersWithTimeout:` (both forms), `startDiscoveryWithInterval:`, and the public
  per-advertisement factory `providerForInferKitServiceNamed:host:port:TXTRecord:pinnedServers:`
  (the seam for an app browsing with `NWBrowser`, and for the tests) decide per server, reported in
  `discoveredPinning`:
  - No remembered fingerprint for the service name: TLS with a well-formed fingerprint → remembered
    and pinned (`FirstUse`); TLS without one, or plain HTTP → unpinned (`None`), which refuses an
    untrusted self-signed certificate.
  - A remembered fingerprint: TLS advertising it or none → pinned to it (`Remembered`), so stripping
    the fingerprint from the record cannot unpin a server; a different fingerprint, or plain HTTP →
    unpinned (`Changed`). Discovery still lists the provider; the balancer skips it and removes it if
    it was already discovered. `NFKPinnedServersFingerprintChangedNotification` is posted once per
    advertised value until `acceptFingerprint:forServiceName:` or `forgetServiceName:`.
  - The key is the service name alone, compared case-insensitively as mDNS compares names. Host and
    port stay out: a server on port 0 moves at each launch, a machine can be renamed, and keying by
    host would let another machine advertising a remembered name be trusted as a new server. A new
    name is a new server and is trusted on first use, the inherent limit of TOFU.
  - The library persists nothing. `dictionaryRepresentation` (name → 64 lowercase hex digits) is a
    property list the app saves on `NFKPinnedServersDidChangeNotification` and restores with
    `initWithDictionaryRepresentation:`, which skips malformed values. Both notifications post
    synchronously on the deciding thread, outside the set's lock.
  - `NFKBalancedBackend mergeDiscoveredProviders:` (public, the browse's merge) applies each
    provider's pin to its discovered server on every browse, so an accepted fingerprint takes effect
    on the next browse; `pinnedCertificateSHA256ForServerWithBaseURL:` reads the effective pin
    (server's own, else the fleet's). Tests: `NFKPinnedServersTests`, weight-free and network-free.
