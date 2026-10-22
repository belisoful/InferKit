<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. Add new notes on this subject to this file, not
to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# Model resolver

The resolver takes a Hugging Face repository and reports how InferKit runs it, how well that route is
verified, and what it costs, before any weights download. The consumer reference is
`Docs/model-catalog.md` ("Resolving a repository").

Status, 2026-10-21: phases 1 to 4 are implemented. Every tier runs: the catalog route, the family route
(InferKitMLX judges every config-reading family in the judge table below), the Core ML package route,
the local-server route, and the conversion hint, with the post-load sanity check beside them.

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
| 2 | `NFKModelRouteCoreMLPackage` | `external` | The repository holds an `inferkit-convert` `manifest.json` for a causal language model, the package it names, and its tokenizer's files. | shipped |
| 3 | `NFKModelRouteLocalServer` | `external` | A server among `localServers` holds the repository (`use`), or the repository holds a single-file GGUF and a server whose runner pulls answers (`pull`). | shipped |
| 4 | `NFKModelRouteConversion` | `conversionRequired` | No route runs the repository, and its `model_type` is one of the nine `Tools/inferkit-convert/architectures.py` registers. | shipped |
| none | `NFKModelRouteNone` | nil | No tier matched. The result carries every refusal. | shipped |

Tier 4 returns the `convert.py` command and runs nothing. `Tools/` ships in no distribution, and the
converter needs a Python environment.

## Tier 4 and the sanity check

The conversion route (`conversionResolutionOfRepository:reasons:`) applies last, to a repository with a
readable `config.json` whose `model_type` is in `+[NFKModelResolver convertibleModelTypes]`, with no
`auto_map`, and with `.safetensors` weights. A pickled checkpoint (`pytorch_model*.bin`) gets no hint, since
loading one through `torch.load` can run code. The route gives no hint after a route refused only for the
caller's consent (`UnverifiedRouteNotAllowed`), since that route may run the repository once the caller
allows unverified routes. It adds no refusal where it does not apply: the routes before it say why. Its resolution
has route `NFKModelRouteConversion`, verification `conversionRequired`, no descriptor, and `isResolved`
NO, since the repository runs only once converted, through the Core ML package route. The action's kind is
`convert`, its provider `inferkit-convert`, its byte count the weights', and its command
`python convert.py --model '<repository>' --revision '<commit>' --output '<name>-coreml'`, run from
`Tools/inferkit-convert`, which converts the commit the resolution read. Each argument is single-quoted
and matches the hub's name form (`[A-Za-z0-9][A-Za-z0-9._-]*`, owner and name for the repository), so a
pasted command passes no character the shell reads; a repository, revision, or output name outside it
gets no hint.
`testTheConvertibleModelTypesAreTheConvertersRegistry` reads `architectures.py` and holds the core's list to
its `model_types`.

`NFKModelSanityCheck` (`NFKModelSanityCheck.h`) is the post-load check. It runs one greedy generation
(temperature 0, 48 tokens) from "The capital of France is" on a built backend, and reports:

- `Empty`: the reply is whitespace only;
- `Garbled`: the reply holds U+FFFD or a control character other than a tab or a line break;
- `Repetitive`: the reply ends in a stretch of up to 32 characters repeated (ten times for one or two
  characters, so a number such as 1000000 passes, six for three or four, three for a longer one), or a
  reply of 32 characters or more uses three distinct characters or fewer.

It is a sanity check, never parity: fluent but wrong text passes. `resultForOutput:` judges a text with no
backend.

Tier 3's Ollama pull identifier (`hf.co/<owner>/<repo>:<quant>`) is probed against Ollama 0.40.2; the
record is in `remote-providers.md` ("Pulling a Hugging Face GGUF"). llama.cpp's `-hf` is a launch argument
with no pull API and no runner adapter, so the route pulls through Ollama only and uses a running
llama.cpp, vLLM, LM Studio, or InferKit server that already lists the repository.

## Tiers 2 and 3

A family-route refusal passes to the next route. The resolver reads `config.json` for the family
route only: a repository without one, or without safetensors weights, is refused by that route with
`NFKModelRouteFamily` in the refusal, and the Core ML and local-server routes try next. A gated
`config.json` still ends the resolution, since no route reads the repository then.

The Core ML package route (`packageResolutionOfRepository:`, `NFKModelResolver.m`) applies to a listing
with a root `manifest.json`:

- a manifest that is not a JSON object, a `modelType` other than `causal-lm`, or a model it names (else
  `model.mlmodelc`, `model.mlpackage`, `model.mlmodel`) the listing does not hold →
  `UnsupportedPackage`;
- a tokenizer type `NFKTokenizer` does not read, or a tokenizer file it names (with its defaults) the
  listing does not hold → `UnsupportedTokenizer`.

The descriptor is engine `coreml`: a text decoder (prompt, and messages where the tokenizer states a chat
template) with one variant reading the manifest, every file under the package, and the tokenizer files.
Its memory facts are one stage of the package's bytes, with the manifest's `contextLength` and
`vocabSize`. Its requirements are macOS 15, iOS 18, and tvOS 18, where Core ML state holds the cache.

The local-server route (`serverResolutionOfRepository:gated:reasons:`) asks no server while
`allowsUnverifiedRoutes` is NO: it adds an `UnverifiedRouteNotAllowed` refusal and sends no request. With
the opt-in it skips a server that requires a key and asks the rest with
`+[NFKRemoteProvider availableProvidersAmong:timeout:]` (two seconds):

1. A server that answers and holds the repository → `use`. An Ollama or LM Studio runner lists its
   installed models, another server its served list; a model named the repository, or `hf.co/` or
   `huggingface.co/` with it and a tag, compared without case, matches.
2. Else, for a repository holding a GGUF: gated → `Gated` (a server pulls without the account's token);
   no answering server whose runner pulls → `NoServer`; else → `pull` of `hf.co/<repository>:<tag>`.
   The tag is the file name's quantization (`Q4_K_M`, `IQ3_XXS`, `Q8_0`, `F16`), preferring `Q4_K_M`,
   `Q4_K_S`, `Q5_K_M`, `Q4_0`, `Q6_K`, `Q8_0`, else the smallest. A file split into parts is skipped.
3. A repository without a GGUF that no server holds adds no refusal.
4. A server whose listing fails adds a `NoServer` refusal holding the listing's error, and the route
   asks the next server.

The descriptor is engine `local-server` with a source of kind `none`: nothing downloads through the hub.
Its `localServer` extension names the preset, the address the resolver asked (`baseURL`), and the
model, and its registry name is the model. The action carries the same address, and its `provider` is the
preset re-pointed there, so the build and a pull reach the server the resolver read. Where
the size is known, its memory facts are one stage of the model's bytes.

The core registers a builder for each engine in its builder registry (`NFKModelEngineBuilders.m`):
`NFKCoreMLEngineBuilder` builds `NFKCoreMLLanguageBackend` from the located directory, and
`NFKLocalServerEngineBuilder` builds `+[NFKRemoteProvider backendForProvider:apiKey:modelName:]` for the
extension's preset at its address and model, and refuses a preset that requires a key. A memory fit is the resolution's estimate on every route; no route refuses
for size.

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

`NFKMLXModelDescriptors.descriptor(for:matching:)` (`NFKMLXRepositoryMatching.swift`) answers for an
entry that lists `modelTypes` and has a family judge (`NFKMLXFamilyJudge.judge(for:)`). Any other entry
refuses with `UnsupportedArchitecture`. `testTheReleasesListTheModelTypesTheirLoaderReads` fails where an
entry lists model types no judge reads.

The judgment is a shared core and one judge per family:

- The judge (`NFKMLXFamilyJudge`) states what its loader requires: the configuration and tokenizer read by
  the family's own readers (`verify`), the weight files the loader reads (`weightFiles`), the files beyond
  the template's it reads where a repository has them (`optionalFiles`), the tensors it places
  (`verifyTensors`), and the sibling entry whose head and tasks describe the repository (`template`).
- A judge whose readers take a directory reads the repository's small files from its snapshot in the hub
  cache (`NFKMLXRepositoryJudgment.snapshot(of:holding:)`), so the reader sees what a load would.
- The core checks every repository the same way: the packing, the header bounds, the measurement, and
  the descriptor.
- `descriptor(for:matching:)` refuses a repository whose `model_type` the asked entry does not list, so a
  direct caller cannot put a text-only repository on a multimodal entry or a dense one on a mixture's.
- The resolved variant downloads the weight files the judgment measured (`weightFiles`, with the shard
  index where the repository lists one), not the template's weight entries, which name the template's own
  format.
- The resolved descriptor keeps the template's load profile and plan, and names the template's family.
  A judge's `families` names another family for a `model_type` its loader also reads (`llama` on
  Mistral Small 3's loader).
- A family's fields a reader does not read are refused where stated at a value the output depends on
  (`verifyFixedFields`, `verifyUnreadFields`, `verifySeq2SeqUnreadFields`). A reader maps an unknown
  layer kind to a default in Granite 4.0-H and Nemotron-H, so those judges refuse an unknown kind.

| Judge | File | Entries | `modelTypes` |
|---|---|---|---|
| `language` | `NFKMLXRepositoryMatching.swift` | the dense and mixture releases `NFKMLXLanguage` loads | `qwen3`, `qwen3_moe`, `qwen2_moe`, `mixtral`, `gpt_oss`, `mistral` with `llama` on Mistral Small 3 |
| `wav2Vec2` | `…Audio.swift` | the Wav2Vec2 and HuBERT releases | `wav2vec2`, `hubert` |
| `w2vBert` | `…Audio.swift` | W2V-BERT 2.0 | `wav2vec2-bert` |
| `timesFM` | `…Forecasting.swift` | TimesFM 2.5 | `timesfm`, `timesfm2_5` |
| `gemma3` | `…Gemma.swift` | Gemma 3 | `gemma3_text` on the text sizes, `gemma3` on the multimodal ones |
| `gemma3n` | `…Gemma.swift` | Gemma 3n | `gemma3n` |
| `gemma4` | `…Gemma.swift` | Gemma 4 but the 12B (`gemma4_unified`) | `gemma4` |
| `qwen3VL` | `…Vision.swift` | Qwen3-VL | `qwen3_vl`, `qwen3_vl_moe` on the 30B-A3B |
| `smolVLM` | `…Vision.swift` | SmolVLM2 | `smolvlm` |
| `trocr` | `…Vision.swift` | TrOCR | `vision-encoder-decoder` |
| `marian` | `…Translation.swift` | OPUS-MT | `marian` |
| `m2m100` | `…Translation.swift` | M2M-100 and SMaLL-100 | `m2m_100` |
| `nllb` | `…Translation.swift` | NLLB-200 | `m2m_100` |
| `graniteH` | `…Hybrid.swift` | Granite 4.0-H | `granitemoehybrid` |
| `nemotronH` | `…Hybrid.swift` | Nemotron-H | `nemotron_h` |

The loaders below read one release, or a model type no judge can tell apart, and list no `modelTypes`:

- Codestral Mamba (`mamba2`): its `config.json` writes `Infinity`, which the core resolver's JSON reader
  refuses before any judge. `NFKMLXMamba2Configuration.configuration(fromDirectory:)` patches the literal.
- The Gemma 4 12B (`gemma4_unified`), TranslateGemma (a chat template's language table), EmbeddingGemma
  (fixed geometry), DeepSeek V4 (one release per version, FP8 block scales), Pixtral, Phi-4-multimodal,
  Florence-2, Voxtral, Granite Speech, Qwen4-Exp, Qwen3.5, MADLAD-400 (`t5` with the gated-GELU fixed),
  the Qwen3-VL embedder and reranker, ModernBERT, DeBERTa and Open-Jev, Table Transformer, V-JEPA 2, SAM 3,
  Sa2VA, and ViTPose.
- The diffusers pipelines state `_class_name`, not `model_type`, so no repository matches them by type.

`testEveryStoredReleaseIsReadByItsOwnFamilysJudgment` judges each shipped release whose directory the
validation store holds with safetensors weights against its own entry: the Qwen3 sizes, gpt-oss,
Wav2Vec2, HuBERT, W2V-BERT, both TimesFM 2.5 layouts, Gemma 3, 3n, and 4, Qwen3-VL, SmolVLM2, SMaLL-100,
TrOCR, Granite 4.0-H, and Nemotron Nano 2. Each family's hermetic judgments are in
`NFKMLXRepositoryMatching{Audio,Forecasting,Gemma,Vision,Translation,Hybrid}Tests`.

### The language decoders

The judgment refuses what the loader reads wrongly with no error:

- `NFKMLXLanguage.configuration(fromJSON:)` throws → `UnsupportedConfiguration`.
- A field the reader does not read, stated at a value the output depends on → `UnsupportedConfiguration`
  (`verifyUnreadFields`). `NFKMLXWeights.apply` refuses missing parameters and ignores extra ones, so
  each of these loads with no error and runs wrong:
  - `mlp_bias` true: the MLP biases are dropped;
  - `hidden_act` other than `silu`;
  - no `rope_theta` at the root or under `rope_parameters`: the reader takes 10^6, transformers 10^4;
  - a numeric `sliding_window` narrower than `max_position_embeddings`, unless `use_sliding_window` is
    false or the model is gpt-oss, whose alternating windows the reader builds. Qwen releases state a
    window with `use_sliding_window` false, which transformers ignores.
- A quantization `NFKMLXQuantization.releaseQuantization(configJSON:)` refuses (a `quant_method` other
  than `mxfp4`, an mlx_lm block without `bits` and `group_size`, a mode MLX does not name) →
  `UnsupportedQuantization`. So are:
  - MXFP4 on a family whose experts the module does not fuse; the loader places MXFP4 words only in
    gpt-oss's routed experts;
  - a tensor in a dtype outside the floats and the `I64` and `BOOL` buffers (`U32`, `U8`, `I8`, `I32`,
    `F8_E4M3`, `F8_E5M2`) when no configuration or save describes the packing;
  - an InferKit save (`inferkit.quantization` metadata) split across files;
  - mlx_lm's gpt-oss layout: a configuration whose experts the module fuses (`clampedSwiGLU`) with an
    mlx_lm quantization and a separate expert `gate_proj`, the case
    `NFKMLXLanguage.packedReleaseEntries(inDirectory:net:)` refuses at load.
- The tokenizer → `UnsupportedTokenizer` when `tokenizer.json` or `tokenizer_config.json` is not listed,
  or `NFKMLXLanguage.tokenizerScheme(tokenizerJSON:tokenizerConfiguration:)`, the loader's own reader,
  refuses the pair.
- Weights outside `model.safetensors` or the shards a `model.safetensors.index.json` names →
  `UnsupportedWeightFormat`.
- A header shape or offset that is negative, malformed, or past 2^53 alone or in total
  (`NFKMLXRepositoryJudgment.measurableLimit`), and a listed size past it → `UnsupportedWeightFormat`.
  The headers are remote input, and the bound keeps every sum the measurement forms from overflowing.

`mlx-community/Qwen3-0.6B-4bit` and `mlx-community/Llama-3.2-1B-Instruct-4bit` resolve from their own
files in the validation store (`NFKMLXRepositoryMatchingTests`), with stored bytes equal to the
`total_size` mlx_lm wrote in their weight index. A small Llama 3 decoder (llama3 rotary scaling, tied
embeddings) resolves through `NFKModelResolver`'s family route on Mistral Small 3's loader against a
stubbed hub (`testALlamaRepositoryResolvesThroughTheFamilyRoute`).

### The speech encoders

`NFKMLXWav2Vec2`'s loader refuses a missing or an extra parameter, so a repository it would read wrongly
fails loudly at load. The judgment refuses earlier what the resolver can see:

- A configuration `NFKMLXWav2Vec2Configuration` or `NFKMLXWav2Vec2BertConfiguration` refuses (an activation
  other than the release's, an adapter, a position batch norm) → `UnsupportedConfiguration`.
- An `architectures` head the loader does not build (a sequence classifier, an x-vector head) →
  `UnsupportedArchitecture`. The loader builds `…ForCTC`, `…Model`, and Wav2Vec2's `…ForPreTraining`.
- A CTC head with no `vocab.json`, a `tokenizer_class` other than `Wav2Vec2CTCTokenizer` (a phoneme
  tokenizer), or a vocabulary past the head's width → `UnsupportedTokenizer`.
- Weights outside `model.safetensors`, which the loader reads whole → `UnsupportedWeightFormat`. Most of
  the shipped Wav2Vec2 and HuBERT releases hold only `pytorch_model.bin`, which the resolver refuses
  before any judge, since it measures from safetensors headers.
- A stated quantization → `UnsupportedQuantization`.

A repository takes its family's entry with the head it carries: a `…ForCTC` repository a CTC release
(transcription and embedding), another an encoder release (embedding). A fine-tuned W2V-BERT's
`vocab.json` and `tokenizer_config.json` join its source, since the W2V-BERT entry lists none.

### TimesFM 2.5

TimesFM 1.0 and 2.0 share the `timesfm` model type and are another network. The judgment refuses a
configuration stating their frequency embedding (`freq_size`, `use_positional_embedding`) and one
stating no continuous quantile head (`quantile_horizon_length` or `output_quantile_len`). The network
fixes its activation (`swish`), biases, key-value heads (one per query head), and rotary base (10000),
so another stated value is refused. The weights are one `model.safetensors`.

### Gemma

Every Gemma generation ships one tokenizer form, which `NFKMLXGemmaTokenizer` reproduces: a space
normalized to `▁`, the whole text one pre-token, byte-fallback BPE, decoded through the same steps
(`NFKMLXFamilyJudge.gemmaTokenizerForm`). The reader takes any BPE vocabulary and merges, so the
judgment refuses another form. Each generation refuses an activation other than `gelu_pytorch_tanh` and
attention biases. Each refuses a head untied from the embedding (`tie_word_embeddings` false at the top
or in `text_config`), since the loaders drop `lm_head.weight`. Gemma 3 refuses a tower other than SigLIP
and a rotary scaling other than linear (its reader's). Gemma 3n refuses a vision tower other than MobileNet-V5, which the loader builds at the
released geometry, and reads its audio configuration through the audio reader. Gemma 4 refuses a
rotary kind other than proportional (full attention) or default (sliding), any `rope_scaling`, and
weights outside `model.language_model.`. A Gemma 4 repository with routed experts takes the 26B-A4B
entry.

### Qwen3-VL, SmolVLM2, and TrOCR

Qwen3-VL's `text_config` takes the language decoders' refusals, and its tower the tower reader's. The
integration fixes the image token (151655), the 2×2 merge, the interleaved M-RoPE sections
`[24, 20, 20]` (under `rope_scaling`, or `rope_parameters` in transformers 5.x), and three DeepStack
layers, so another stated value is refused.

SmolVLM2's `text_config` takes the language decoders' refusals, its tokenizer the release reader's
(SmolLM's digit split), and its tower must be `smolvlm_vision` with the tanh GELU the loader runs.

TrOCR states the generic `vision-encoder-decoder` model type. The judgment requires a ViT or DeiT
encoder and a `trocr` decoder, refuses another pair (Donut, ViT-GPT2), and requires RoBERTa's
`vocab.json` and `merges.txt` or XLM-R's `sentencepiece.bpe.model`.

### The translators

Marian, M2M-100, and NLLB-200 read through `NFKMLXSeq2SeqConfiguration`, which takes the encoder's head
count for both stacks and shares the embeddings. The judgment refuses a decoder head count other than
the encoder's, unshared embeddings, and a `decoder_vocab_size` other than `vocab_size`. Marian refuses a
separate target vocabulary. M2M-100 and NLLB-200 both state `m2m_100`: M2M-100 requires its
`vocab.json` and a tokenizer class of `M2M100Tokenizer` or none, NLLB-200 requires `NllbTokenizer`, so
each family refuses the other's repositories and the resolver asks the next. A repository shipping
`tokenization_small100.py` takes the SMaLL-100 entry, and any other M2M-100 repository the M2M-100 entry.
Most shipped translator releases hold only
`pytorch_model.bin`, which the resolver refuses before any judge.

### The hybrid decoders

Granite 4.0's attention-only releases share `granitemoehybrid` and rotate their positions. The Granite
4.0-H network attends without positions, so the judgment refuses a `position_embedding_type` other than
`nope`, an activation other than `silu`, attention biases, and a layer kind other than `mamba` and
`attention`. A repository with routed experts takes the Tiny entry. Nemotron-H refuses an MLP activation
other than `relu2`, attention biases, and any layer outside Mamba-2, attention, and MLP (a pattern letter
other than `M`, `*`, and `-`, or a `layers_block_type` the reader does not name).

### The measurement

The measurement ports `Tools/catalog/measure.py`'s hub path to Swift: download bytes over the template's
files plus the shards, safetensors tensor bytes by dtype from the headers, parameters, the repeated
layers (`NFKMLXTensorStructure`), routed-expert bytes, quantizable elements (rank-2 `.weight`, input width
divisible by 64, outside embeddings and norms), and the geometry (`NFKMLXMeasuredGeometry(configuration:)`).
The resolved entry is the template entry with the new identifier, license, and variant, run through
`NFKMLXCatalogEntry.measured(by:plan:)`, the path the shipped entries take. The family's load plan and load
profile are keyed by family, so the resolved descriptor gets the same load options and memory model.
A repository the loader reads as stored whatever the request (an mlx_lm quantized release, an InferKit
quantized save) is measured under the `.stored` plan, and its entry's `load` is the family profile without
the precision and weight quantization options: the loader packs no layer the release already packs, and
a float32 figure would describe a load that never happens. Its parameter count reads each packed layer's
`U32` words at the layer's width (mlx_lm's per-layer override in `config.json`, else its default, else
the save's recorded width) and counts the packing's `scales` and `biases` as none, so a 4-bit release
reports the parameters of its float original. It records no quantizable elements.
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
  `resolutionWithDictionary:error:` reads it back. Tiers 2 and 3 add `NoServer` and `UnsupportedPackage`.
  A model too large for the budget is the estimate's verdict, as on the family route.
- **External routes.** A local-server resolution carries its `action`: `use` builds at once, and `pull`
  downloads into the server's own store. The host shows a pull and performs it only after the user
  confirms. The action's `dictionaryRepresentation` rides in the resolution's.

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
SmolVLM2's loader (`NFKMLXSmolVLM.tokenizer(inDirectory:)`) reads its tokenizer through the same reader.
The judgment reads what the loader reads through the loader's own predicates.

## Phasing

1. Metadata fetch, tier 0, the language-family judgment with the refusals above, the result type, and
   the catalog `modelTypes` field. Done.
2. Judgments for the other config-reading families, each listing its `modelTypes`. Done: the families
   in the judge table. Codestral Mamba waits on a core JSON reader that takes `Infinity`.
3. Tiers 2 and 3, with the server probes recorded. Done.
4. Tier 4 hints, and an optional post-load check: a short greedy generation tested for degenerate
   output. It is reported as a sanity check and never as parity. Done.
