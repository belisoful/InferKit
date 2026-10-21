<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. Add new notes on this subject to this file, not
to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# Model resolver

The resolver takes a Hugging Face repository and reports how InferKit runs it, how well that route is
verified, and what it costs, before any weights download. The consumer reference is
`Docs/model-catalog.md` ("Resolving a repository").

Status, 2026-10-20: phase 1 is implemented. The catalog route and the family route run, and InferKitMLX
judges the dense and mixture decoders `NFKMLXLanguage` loads. Tiers 2 to 4 are the plan below.

## Starting point

These pieces existed before the resolver and it composes them:

- The model catalog (`NFKModelCatalog`, `Docs/model-catalog.json`). Most variants name their
  `source.repository` and `measuredRevision`. A descriptor builds through
  `backendForBuildRequest:` (`NFKModelBuilding.h`).
- Config-reading loaders. `NFKMLXLanguage.backendWithDirectoryURL:error:` reads `config.json`,
  the tokenizer, and the shards of any release its configuration reader accepts. About 58 other
  `directoryURL` factories read their own family's `config.json`.
- The hub's recursive tree listing (`<endpoint>/api/models/<repo>/tree/<revision>?recursive=1`),
  which gives each file's path and size without a download.
- `NFKCoreMLLanguageBackend`, which runs a package written by `Tools/inferkit-convert`.
- The local-runner presets (`ollama`, `lmstudio`, `llamacpp`, `vllm`, `inferkit`),
  `+[NFKRemoteProvider availableProvidersAmong:timeout:]`, and `-[NFKRemoteProvider localRunner]`
  with `pullModel:`.

An architecture with no port has no in-process route, because MLX has no generic graph executor.

## Two steps

1. **Resolve.** It reads metadata only. It returns an `NFKModelResolution`.
2. **Build.** The app adds the resolution's descriptor to the catalog and builds it through the
   existing build path. This step downloads the weights and loads them.

The split lets an app show the route, the verification level, the download size, and the memory fit
before it commits gigabytes.

## Routes

The resolver tries the tiers in order and records each refusal.

| Tier | Route | Verification | Match | State |
| --- | --- | --- | --- | --- |
| 0 | `NFKModelRouteCatalog` | `measured` | A variant's `source.repository` is the repository, case-insensitively: any revision when none is named, else the variant's revision or `measuredRevision`. | shipped |
| 1 | `NFKModelRouteFamily` | `configCompatible` | A descriptor's `modelTypes` holds the configuration's `model_type`, and its provider class builds a descriptor for the repository. | shipped |
| 2 | Prebuilt Core ML package | `external` | The repository holds a `.mlpackage` and an `inferkit-convert` `manifest.json`. | planned |
| 3 | Local server | `external` | GGUF files → an Ollama or llama.cpp pull. Safetensors → a vLLM server that is already running. | planned |
| 4 | Conversion hint | `conversionRequired` | `model_type` is one of the nine `Tools/inferkit-convert/architectures.py` registers. | planned |
| none | `NFKModelRouteNone` | nil | No tier matched. The result carries every refusal. | shipped |

Tier 4 returns the `convert.py` command and runs nothing. `Tools/` ships in no distribution, and the
converter needs a Python environment.

Tier 3's pull identifiers (`hf.co/<owner>/<repo>:<quant>` for Ollama, `-hf` for llama.cpp) are taken
from those projects' documentation and are not probed here yet. Probe them against a live server before
the route ships, and record the probe in `remote-providers.md`.

## The resolution, step by step

`-[NFKModelResolver resolutionForRepository:revision:error:]` (`Sources/InferKit/NFKModelResolver.m`):

1. Tier 0 against the catalog with no request. A named revision matches the variant's revision or its
   `measuredRevision`.
2. `GET <endpoint>/api/models/<repo>/revision/<rev>` → the commit (`sha`), the model card (`cardData`),
   and the gating. A 401 or 403 → `NFKModelResolutionErrorGated`, described as private or missing.
   Another failure fails the call with the hub's error. Probed against huggingface.co on 2026-10-20
   with no token:
   - a repository that does not exist → 401 on this endpoint and on the listing, no `X-Error-Code`,
     "Invalid username or password.";
   - a gated repository (`meta-llama/Llama-3.2-1B`) → 200 on the listing, and 401 with
     `X-Error-Code: GatedRepo` on a file.
   A missing repository and a private one are therefore indistinguishable without a token, and the
   refusal names both.
3. Tier 0 again against the commit, so a branch that points at the measured commit resolves to the
   shipped variant.
4. The listing at the commit (`planDownloadOfRepo:revision:paths:error:`). An authentication failure →
   `NFKModelResolutionErrorGated` with the hub's error under `NSUnderlyingErrorKey`.
5. `config.json` through the hub cache, where a later download finds it again. Route-independent
   refusals: no `config.json`, `auto_map`, no `.safetensors` file, no `model_type`.
6. Each catalog descriptor whose `modelTypes` holds the `model_type`, in catalog order, once per
   provider class and family: the five Qwen3 sizes are one judgment, since the judgment is the family
   loader's. The class that supplied it (`providerClassNameForDescriptorIdentifier:`) answers
   `+descriptorForRepository:matchingDescriptor:error:`. The first descriptor returned wins. Each
   refusal is kept. Errors from the answer split three ways:
   - `NFKDownloadErrorAuthenticationRequired` → `Gated`, naming the route and the descriptor;
   - any other `NFKDownloadErrorDomain` or `NSURLErrorDomain` error → the call fails with it;
   - any other error → `UnsupportedConfiguration`, with the error under `NSUnderlyingErrorKey`.
7. Without `allowsUnverifiedRoutes`, the match becomes `UnverifiedRouteNotAllowed`, naming the route
   and the shipped descriptor. With it, the descriptor gains the `resolution` extension and the
   resolution carries its estimate at the default options against `memoryBudget` (the catalog's when
   nil).

A resolved descriptor's identifier is the repository name, so a catalog holds one revision of a
repository at a time.

## The InferKitMLX judgment

`NFKMLXModelDescriptors.descriptor(for:matching:)` (`NFKMLXRepositoryMatching.swift`) answers for the
entries `NFKMLXCatalogLanguage.readsReleases(of:)` names: the dense and mixture releases `NFKMLXLanguage`
loads. Their entries list one `model_type` each (`qwen3`, `qwen3_moe`, `qwen2_moe`, `mixtral`, `gpt_oss`,
`mistral`). Any other entry refuses with `UnsupportedArchitecture`.

The judgment refuses what the loader reads wrongly with no error:

- `NFKMLXLanguage.configuration(fromJSON:)` throws → `UnsupportedConfiguration`.
- A `quantization` key, or a `quantization_config` whose `quant_method` is not `mxfp4` →
  `UnsupportedQuantization`. So are `U32` tensors the configuration does not describe, and an InferKit
  save (`inferkit.quantization` metadata) split across files.
- The tokenizer → `UnsupportedTokenizer` unless every one of these holds:
  - `tokenizer.json` and `tokenizer_config.json` are listed;
  - `model.type` is `BPE`, without `byte_fallback`, `ignore_merges`, or a `Metaspace` step;
  - the pre-tokenizer has exactly one `Split` whose regex is byte-for-byte one of the Qwen2, Qwen3.5,
    and o200k patterns `NFKByteLevelBPETokenizer.m` implements.
  An NFC normalizer is accepted. The shipped Qwen3 releases declare one and the loader applies NFC only
  for the Qwen3.5 pattern, so the gap is the shipped model's own.
- Weights outside `model.safetensors` or the shards a `model.safetensors.index.json` names →
  `UnsupportedWeightFormat`.
- A header shape or offset that is negative, malformed, or past 2^53 alone or in total
  (`NFKMLXRepositoryJudgment.measurableLimit`), and a listed size past it → `UnsupportedWeightFormat`.
  The headers are remote input, and the bound keeps every sum the measurement forms from overflowing.

Of the six families, Qwen3, Qwen3-MoE, and Qwen1.5-MoE repositories resolve today. Mixtral, gpt-oss,
and Mistral repositories are refused at the tokenizer (a Metaspace BPE with byte fallback,
`ignore_merges`, and Llama 3's pattern) until the loader-gap fixes below land and the judgment widens.

The measurement ports `Tools/catalog/measure.py`'s hub path to Swift: download bytes over the template's
files plus the shards, safetensors tensor bytes by dtype from the headers, parameters, the repeated
layers (`NFKMLXTensorStructure`), routed-expert bytes, quantizable elements (rank-2 `.weight`, input width
divisible by 64, outside embeddings and norms), and the geometry (`NFKMLXMeasuredGeometry(configuration:)`).
The resolved entry is the template entry with the new identifier, license, and variant, run through
`NFKMLXCatalogEntry.measured(by:)`, the path the shipped entries take. The family's load plan and load
profile are keyed by family, so the resolved descriptor gets the same load options and memory model.
`testLiveMeasurementMatchesTheRecordedMeasurement` (gated on `INFERKIT_LIVE_HUB=1`) measures
`Qwen/Qwen3-0.6B` at its measured commit and compares every figure with the recorded row.

## Host requirements

A host that serves models from a long-running service process calls the resolver in that process and
reaches it from its own control protocol. The in-process API meets these requirements:

- **Threading.** `resolveRepository:revision:completionHandler:` is callable from any thread, including
  the main thread. It runs on a global queue at user-initiated quality of service, and the block is
  `NS_SWIFT_SENDABLE`.
- **Memory budget.** `memoryBudget` takes an `NFKModelMemoryBudget`, made from a byte count with
  `+budgetWithWorkingSetByteCount:`. Nil uses the catalog's budget, this machine's by default.
- **Persistence without a network.** A resolved descriptor's extensions hold JSON values only. The round
  trip carries it: `-[NFKModelDescriptor dictionaryRepresentation]` → JSON →
  `+[NFKModelDescriptor descriptorWithDictionary:error:]` → `-[NFKModelCatalog addDescriptor:]`. A host
  stores the repository, the revision, and that dictionary, and re-adds the descriptor at launch with no
  resolve call. `testAResolvedDescriptorRoundTripsThroughItsDictionaryIntoACatalog` and
  `testTheDescriptorRoundTripsThroughItsDictionary` pin the round trip.
- **Machine-readable refusals.** Each entry in `reasons` is an `NSError` in
  `NFKModelResolutionErrorDomain`, with `NFKModelResolutionRouteKey` and
  `NFKModelResolutionDescriptorKey` in its `userInfo`. `NFKModelResolution.dictionaryRepresentation`
  writes each as `{"domain", "code", "description", "route", "descriptor"}`, and
  `resolutionWithDictionary:error:` reads it back. Tiers 2 and 3 add codes for a model that does not fit
  and for no available server.
- **External routes.** A tier-2 or tier-3 result describes an action. The host shows it and performs it
  only after the user confirms.

No `NFKInferenceServer` route is planned for the first phase. A resolve route serves only an app with no
service process of its own, and is a later decision.

## Decisions

Developer decisions, 2026-10-20:

1. **Unverified routes are opt-in.** Tier 0 always resolves. Tiers 1 to 3 resolve only when the caller
   sets `allowsUnverifiedRoutes`. The `verification` value is always set on a resolved route. A tier-3
   pull uses disk and changes the server's state, so it is returned as an action for the app to confirm
   and is never performed by the resolver.
2. **The catalog states each family's signature.** A descriptor gains the `model_type` values its loader
   reads (`modelTypes`), and the manifest's `schemaVersion` is 2. A reader takes 1 and 2. An app explains
   a match from the catalog without loading the package.

## Loader gaps that limit tier 1

Each gap is a judgment refusal until the loader handles it. Otherwise the load succeeds and produces
wrong output with no error. Closing a gap removes its entry here and widens the judgment in the same
change.

The loader gaps this list held are closed. mlx_lm's quantized releases load, single-file and sharded,
dense and mixture (`NFKMLXQuantization.releaseQuantization(configJSON:)`). The release tokenizer reader
(`NFKMLXLanguage.tokenizerScheme(tokenizerJSON:tokenizerConfiguration:)`) reproduces each shipped
family's pre-tokenization (Qwen2, Qwen3.5, o200k, Llama 3, Nemotron, GPT-2, SmolLM), its NFC normalizer,
`ignore_merges`, and Mistral's `Metaspace` BPE with byte fallback, or refuses a scheme it does not
reproduce by name.

Open: the judgment (`NFKMLXRepositoryMatching.swift`) still refuses a `quantization` block,
`ignore_merges`, and the Llama 3, Nemotron, and SmolLM patterns. Widening it to what the loader now reads
is the resolver's next change, which removes this entry.

## Phasing

1. Metadata fetch, tier 0, the language-family judgment with the refusals above, the result type, and
   the catalog `modelTypes` field. Done.
2. Judgments for the other config-reading families (seq2seq and T5, wav2vec2 and HuBERT, Gemma,
   DeepSeek, and the remaining `directoryURL` factories), each listing its `modelTypes`.
3. Tiers 2 and 3, with the server probes recorded.
4. Tier 4 hints, and an optional post-load check: a short greedy generation tested for degenerate
   output. It is reported as a sanity check and never as parity.
