<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# Core runtime notes

Value-type accessors, tokenizers, grammar-constrained sampling, and dynamic backend discovery in the core.

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
  default; the Qwen3 text path (`NFKMLXLanguage.backend(directoryURL:)`) now names `qwen2` too. An
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

`NFKTokenConstraint` (`Sources/InferKit/NFKTokenConstraint.m`, 0.4.0) ports the MLX companion's
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
- Each built-in capability has a default provider class name (a `capability → class name` map in the
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
