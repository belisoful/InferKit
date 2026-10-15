# Model catalog

`NFKModelCatalog` lists the models that the engines linked into a process run, as `NFKModelDescriptor`
values an app shows, downloads, and configures. A catalog's JSON form is a **manifest**. The core,
each companion package, and a plugin all describe their models in the same schema, so an app reads
every source the same way.

This document is the schema reference: the manifest envelope, every field of every object, the
vocabulary strings, the memory model, and the build entry. The [examples](examples.md#model-catalog)
show the API in use.

- [The catalog](#the-catalog)
- [Manifest](#manifest)
- [Reading rules](#reading-rules)
- [Model](#model)
- [Port](#port)
- [Option](#option)
- [Choice](#choice)
- [License](#license)
- [Variant](#variant)
- [Source](#source)
- [File](#file)
- [Requirements](#requirements)
- [Memory](#memory)
- [Presets](#presets)
- [Load options and the build](#load-options-and-the-build)
- [Vocabulary](#vocabulary)
- [Errors](#errors)
- [The MLX catalog's measurements](#the-mlx-catalogs-measurements)
- [The snapshot and the tools](#the-snapshot-and-the-tools)

## The catalog

- `NFKModelCatalog.sharedCatalog` loads its descriptors once, on first use, from each provider class
  linked into the process. The core names four: `NFKAppleModelDescriptors`, `NFKMLXModelDescriptors`,
  `NFKFoundationModelsDescriptors`, and `NFKAppleSwiftModelDescriptors`.
- A provider is a class that adopts `NFKModelDescriptorProvider` (`+modelDescriptors`). The catalog
  finds it with `NSClassFromString`, so a package that is not linked contributes nothing and causes
  no link error.
- `registerProviderClassName:` adds a provider after the known ones. A later provider's descriptor
  replaces an earlier one with the same identifier.
- `addDescriptorsFromManifestData:error:` adds a manifest, all of its descriptors or none.
- `manifestDataWithError:` writes the catalog as a manifest: pretty-printed, keys sorted, and models
  sorted by identifier, so the same catalog always writes the same bytes.
- `visibleDescriptors` leaves out stand-ins. `descriptorsForTask:` and `descriptorsInFamily:` include
  them.

`NFKRemoteModelCatalog` is a different class: it lists what a remote provider serves at the moment.

Each package describes what it ships:

- `NFKAppleModelDescriptors` (the core): the Apple-framework engines, with engine `apple`, each built by
  `+backendNamed:` from its identifier. Their ports are read from the backend that name builds.
- `NFKMLXModelDescriptors` (InferKitMLX): the MLX models, built by `NFKMLXModelRegistry` under their
  `registryName`.
- `NFKFoundationModelsDescriptors` and `NFKAppleSwiftModelDescriptors`: the system language model and the
  Swift-only Apple engines.

A backend's outputs are its `supportedOutputKeys`, the result keys it can write. `NFKInferenceServer`
lists them as `outputs` in a model's description, and `NFKRemoteInferKitBackend` reads them back.

## Manifest

```json
{
  "schemaVersion": 1,
  "kind": "inferkit.model-catalog",
  "models": [ … ]
}
```

| Field | Type | Meaning |
|---|---|---|
| `schemaVersion` | integer | The schema the manifest follows. This release reads and writes 1 (`NFKModelCatalog.schemaVersion`). A newer version is refused. |
| `kind` | string | Always `"inferkit.model-catalog"` (`NFKModelCatalogManifestKind`). A missing or different kind is refused. |
| `models` | array of [model](#model) | The descriptors. |

## Reading rules

- **Unknown fields are kept.** Every object keeps the fields this release does not read in its
  `extensions` dictionary, and writing the object puts them back beside the known fields. A manifest
  a newer writer made passes through an older reader unchanged, and a plugin carries fields of its
  own without a schema change.
- **Absent values are left out.** A writer leaves out an optional field that has no value. It never
  writes `null`.
- **Strings stay strings.** Tasks, value types, engines, residencies, option kinds, stages, file
  roles, source kinds, and layouts are open string sets. The [vocabulary](#vocabulary) lists the
  values the core uses, and a reader passes any other string through, so a plugin extends a set by
  using a new string.
- Booleans are JSON `true` and `false`. Byte counts and parameter counts are JSON integers.

## Model

`NFKModelDescriptor`.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | The identifier, unique within a catalog, such as `"qwen3"`. A model an engine builds by name uses that name (its `registryName`), which is also the name a server hosts it under. |
| `displayName` | string | no | The name a user interface shows. The identifier when absent. |
| `family` | string | yes | The architecture shared by related models, such as `"qwen3"` or `"whisper"`. |
| `summary` | string | no | A sentence on what the model does. |
| `engine` | string | yes | The engine that runs the model: an [engine](#engines). |
| `registryName` | string | no | The name the model registers under in its engine's registry. |
| `providerClass` | string | no | The `NFKDynamicBackendProvider` class that builds the model's backend. |
| `tasks` | array of string | no | What the model does, primary first: [tasks](#tasks). |
| `inputModalities` | array of string | no | The [value types](#value-types) the model reads. The value types of `inputs`, in order without repeats, when absent. |
| `outputModalities` | array of string | no | The value types the model writes. The value types of `outputs` when absent. |
| `inputs` | array of [port](#port) | no | The inputs a request carries. |
| `outputs` | array of [port](#port) | no | The outputs a result carries. |
| `options` | array of [option](#option) | no | The settings the model reads, at load and on each request. A manifest that names the field `parameters` reads the same. |
| `license` | [license](#license) | no | The license of the model's weights. |
| `variants` | array of [variant](#variant) | no | The releases. The default variant is the one with `isDefault` true, else the first. |
| `requirements` | [requirements](#requirements) | no | What a machine needs. No requirements when absent. |
| `residencies` | array of string | no | How the model can hold its weights: [residencies](#residencies). |
| `memoryModel` | [memory model](#memory-model) | no | Which options change the model's memory. |
| `presets` | array of [preset](#presets) | no | Named sets of load options and request defaults. |
| `standIn` | boolean | no | True for a reference or test stand-in a user interface hides. False when absent. |
| `introducedVersion` | string | no | The InferKit version that introduced the model, such as `"0.4.0"`. |

A writer always writes `id`, `displayName`, `family`, `engine`, the arrays (empty when there is
nothing in them), `requirements`, and `standIn`. It writes `presets` only when there are some.

A descriptor with more than one variant derives a load option under `variant`
(`NFKModelLoadOptionVariant`): a choice of its variants' identifiers, the default variant by default.
The derived option is in `loadOptions` and `optionWithKey:` and is not written to the manifest. A
descriptor that declares an option under `variant` keeps its own.

`+[NFKModelDescriptor descriptorWithIdentifier:derivedFromBackend:]` builds a model from a backend's
`supportedInputKeys`, `supportedParameterKeys`, and `supportedOutputKeys`, typed from the core table.
Its engine is `"plugin"`, and its family is the backend's `NFKModelInfoArchitecture`, else its
`backendIdentifier`.

## Port

`NFKModelPort`, an entry of `inputs` or `outputs`.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `key` | string | yes | The request or result key: an `NFKInput*` or `NFKOutput*` key, or the backend's own. |
| `valueType` | string | no | The [value type](#value-types) under the key. The core table's type for a core key when absent. |
| `required` | boolean | no | On an input, true when every request carries it. On an output, true when every result carries it. False when absent. |
| `title` | string | no | A short name for a user interface. |
| `help` | string | no | A sentence that explains the port. |

`NFKModelValueTypeForKey` types a request key and `NFKModelValueTypeForOutputKey` a result key. The
two differ only where an input or parameter key and an output key share a string: `"citations"` is
a boolean as `NFKParameterCitations` and JSON as `NFKOutputCitations`.

## Option

`NFKModelOption`, an entry of `options`.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `key` | string | yes | The parameter key: an `NFKParameter*` key, or the backend's own. |
| `kind` | string | no | The [option kind](#option-kinds), which picks the control. When absent, the kind follows the key's value type: `boolean`, `integer`, `number`, `text`, and `text-list` give the kind of the same name, and anything else gives `json`. |
| `title` | string | no | A short name for a user interface. |
| `help` | string | no | A sentence that explains the option. |
| `default` | string, number, boolean, array, or object | no | The value the backend uses when a request leaves the option out. |
| `minimum` | number | no | The least value of a number or an integer. |
| `maximum` | number | no | The greatest value of a number or an integer. |
| `step` | number | no | The increment a control steps by. |
| `choices` | array of [choice](#choice) | no | The values of a `choice` option, in display order. A writer leaves the field out when there are none. |
| `stage` | string | no | When the option applies: an [option stage](#option-stages). `"request"` when absent. |
| `group` | string | no | The group a user interface shows the option in. |
| `requires` | [requirements](#requirements) | no | What a machine needs to use the option. |
| `property` | string | no | On a load option, the backend property the build sets to the value by key-value coding, after the backend is constructed and before it prepares. A load option without one is a construction argument the engine's builder reads. |
| `memoryBytesPerUnit` | integer | no | On an integer option, the bytes each unit adds to the model's memory, such as one context position's key-value cache. |

`loadOptions` lists the load-stage options, the derived `variant` option first, and `requestOptions`
the request-stage ones. The memory model's context length, cache bits, and cache group size are
load-stage options that set the backend's default, and a request may still set them, up to the
option's maximum.

## Choice

`NFKModelOptionChoice`, an entry of an option's `choices`.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `value` | string or number | yes | The value a request carries when the choice is selected. |
| `title` | string | no | The name a user interface shows. The value's text when absent. |
| `requires` | [requirements](#requirements) | no | What a machine needs to take the choice. |
| `memory` | [memory figure](#memory-figure) | no | The memory the model holds with the choice selected, at the default variant and the default options. |
| `memoryByVariant` | object of [memory figure](#memory-figure) | no | The same figure per variant identifier. |

## License

`NFKModelLicense`. The license of the weights a variant downloads. InferKit's own code has its own
license, which this does not describe.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | no | The SPDX identifier, such as `"Apache-2.0"`. |
| `name` | string | no | The license's name. The `id` when absent; one of the two is required. |
| `url` | string | no | Where the license text is published. |
| `gated` | boolean | no | True when the publisher asks a user to accept terms before the download, so the download needs an access token whose account accepted them. |
| `restrictsCommercialUse` | boolean | no | True when the license forbids or limits commercial use. |
| `note` | string | no | A term a user should know, such as an attribution requirement. |

## Variant

`NFKModelVariant`, an entry of `variants`: one release of the model, such as a size, a precision, or
a fine-tune.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | The identifier, unique within its model. |
| `displayName` | string | no | The name a user interface shows. The `id` when absent. |
| `isDefault` | boolean | no | True for the variant a caller gets without choosing one. |
| `registryName` | string | no | The name the variant registers under in its engine's registry. |
| `source` | [source](#source) | no | Where the variant's files come from. |
| `downloadBytes` | integer | no | The bytes a download transfers. |
| `residentBytes` | integer | no | The bytes the weights hold in memory once loaded. |
| `peakBytes` | integer | no | The most bytes the variant holds at once while it loads or runs at its default options, which can exceed `residentBytes`. |
| `parameterCount` | integer | no | The parameter count. |
| `precision` | string | no | The type the weights are stored in, such as `"float32"`, `"float16"`, `"bfloat16"`, or `"int4"`. |
| `minimumMemoryBytes` | integer | no | The least memory the variant runs in, as [requirements](#requirements) defines it. |
| `license` | [license](#license) | no | The license of this variant's weights where it differs from the model's `license`. The model's license applies when absent. `-[NFKModelDescriptor licenseForVariant:]` reads the one that applies. |
| `memoryFacts` | [memory facts](#memory-facts) | no | What the variant's stages and caches weigh, from which every memory estimate is computed. |

## Source

`NFKModelSource`, a variant's `source`.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | Where the files come from: a [source kind](#source-kinds). |
| `repository` | string | no | The repository of a Hugging Face source, such as `"org/name"`. |
| `revision` | string | no | The commit or tag the files are read at. A commit makes the download reproducible. |
| `url` | string | no | The download URL of a URL source, or the page that describes the release. |
| `layout` | string | no | How the model loads its files: a [source layout](#source-layouts). `"none"` when absent. |
| `files` | array of [file](#file) | no | The files the source holds. |
| `requiresConversion` | boolean | no | True when the original release needs a converter before InferKit loads it. |

`NFKHFHub` downloads a source into its cache (`downloadModelSource:progressHandler:completionHandler:`),
reports whether it is downloaded (`downloadStatusForModelSource:`), and removes it
(`removalPlanForModelSource:error:`, `removeFilesInPlan:error:`). A `huggingface` source that lists
files fetches those, an `optional` one only when the repository holds it; one that lists none fetches
the whole repository. A `url` source caches its file under the name its first file gives. A `system`
or `none` source holds no files. A source with `requiresConversion` is refused with
`NFKDownloadErrorRequiresConversion`. The [examples](examples.md#downloading-with-progress-then-removing-nfkdownload)
show the calls.

## File

`NFKModelFile`, an entry of a source's `files`.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `path` | string | yes | The path relative to the source: inside the repository or directory, or the file name of a single-file source. |
| `bytes` | integer | no | The file's size. |
| `sha256` | string | no | The SHA-256 of the contents, as lowercase hexadecimal digits. |
| `role` | string | no | What the file holds: a [file role](#file-roles). `"other"` when absent. |
| `optional` | boolean | no | True when the model loads without the file. |
| `repository` | string | no | The Hugging Face repository the file comes from when it is not its source's, such as a vocoder or the base release a ControlNet steers. `path` is then relative to that repository. |
| `revision` | string | no | The commit or tag the file is read at. With `repository`, the revision of that repository, whose default branch applies when absent. Alone, the source's repository at that revision. |
| `url` | string | no | The download URL of a file that does not come from its source. `path` is then the file name. |

A file that names `repository`, `revision`, or `url` reads `overridesSource` as `YES`. Every other file
comes from its variant's source.

## Requirements

`NFKModelRequirements`, a model's `requirements`, an option's `requires`, or a choice's `requires`.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `minimumOS` | object of string | no | The least operating-system version per platform: `"macOS"`, `"iOS"`, `"tvOS"`, `"visionOS"`, such as `{"macOS": "14.0", "iOS": "17.0"}`. A platform the object leaves out does not run the model. An empty object runs everywhere InferKit runs. |
| `requiresAppleSilicon` | boolean | no | True when the model runs only on Apple silicon. |
| `minimumMemoryBytes` | integer | no | The least memory the model runs in. |
| `requiredFeatures` | array of string | no | The [hardware features](#hardware-features) a machine needs. A feature the reader does not know is never met. |
| `unmetReason` | string | no | The sentence a user interface shows when the requirements are not met. |

`minimumMemoryBytes` is InferKit's residency budget for the model. It is the smallest recommended
Metal working set (`NFKHardwareProfile.recommendedWorkingSetSize`) whose budget admits the model at its
default options. `NFKModelMemoryBudget` budgets 0.85 of the working set and keeps a 4 GiB reserve inside
that budget. The figure is therefore the peak bytes at the defaults plus the reserve, divided by 0.85
(`+[NFKModelMemoryBudget leastWorkingSetHoldingByteCount:]`). A model that stages, pages, or streams
counts only what one stage holds at once.

`-[NFKModelRequirements problemForHardwareProfile:]` names the first requirement a machine does not
meet, or returns nil. The checks run in this order, and `unmetReason` replaces the sentence:

| Check | Sentence |
|---|---|
| The platform is not in `minimumOS` | `This does not run on macOS.` |
| The version is below the platform's | `This needs macOS 15.0 or later.` |
| `requiresAppleSilicon`, or the `apple-silicon` feature | `This needs Apple silicon.` |
| The `neural-accelerators` feature | `This needs a GPU with neural accelerators.` |
| A feature the reader does not know | `This needs the <feature> feature.` |

Memory is not checked here; the [memory estimate](#the-estimate) answers it. The machine is an
`NFKHardwareProfile`: the current one, or a snapshot of another (`+profileWithDictionary:`) written by
`dictionaryRepresentation`.

## Memory

`NFKModelMemory.h`. A variant's [memory facts](#memory-facts) state what its stages weigh. The
descriptor's [memory model](#memory-model) names the options that change those figures. The
[estimate](#the-estimate) combines the two for one combination of load options, and the
[constraints](#constraints) run the estimate over every combination to say which choices fit a
machine. Every figure is in bytes.

Activation memory depends on the request, such as an image's size, and no descriptor states it. The
budget's reserve covers it.

### Memory facts

`NFKModelMemoryFacts`, a variant's `memoryFacts`.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `stages` | array of [stage](#stage) | yes | The stages, in the order a staged load runs them. A model without stages has one. |
| `keyValueBytesPerToken` | integer | no | The key-value cache one position adds, across the layers whose cache grows, at 16 bits per element. |
| `boundedKeyValueBytes` | integer | no | The cache of the layers whose cache stops growing, such as sliding-window layers, at 16 bits per element. |
| `maximumContextLength` | integer | no | The longest context the model reads. The context length when no option sets it. |
| `vocabularySize` | integer | no | The tokens in the vocabulary, which a draft model shares with the model it drafts for. |
| `loadOverheadBytes` | integer | no | The bytes a load holds beyond the resident figure for a moment. 0 where it is not measured. |
| `basis` | string | no | `"measured"` (the tensor bytes at the precisions the loader holds them in, the default) or `"stored-dtype"` (the stored bytes, because the loader's precision is not known). |

### Stage

`NFKModelStageFacts`, an entry of a memory facts' `stages`. The `…ByPrecision` objects key a
[precision](#precisions) to bytes, and `checkpoint` holds the bytes as stored.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | The stage's identifier, such as `"text_encoder"` or `"decoder"`. |
| `title` | string | no | The stage's name. The `id` when absent. |
| `bytesByPrecision` | object of integer | yes | The stage's bytes per precision. |
| `quantizableElements` | integer | no | The elements a runtime weight quantization packs: the stage's eligible matrices. |
| `pageableBytesByPrecision` | object of integer | no | The routed-expert bytes a paged load leaves in the release. |
| `layerCount` | integer | no | The repeated layers a streamed load may leave in the release. |
| `largestLayerBytesByPrecision` | object of integer | no | The bytes of the largest repeated layer. |
| `widenedPrecision` | string | no | A wider precision the stage takes where the budget holds it, such as a text encoder stored at bfloat16 and run at float32. |

A precision a stage does not list converts from its `checkpoint` bytes by the ratio of element sizes,
taking `checkpoint` as the variant's `precision`. Integer and packed tensors convert with the floating
ones, so a converted figure is an approximation.

### Memory model

`NFKModelMemoryModel`, a descriptor's `memoryModel`. Each option field names the key of a declared
option. The variant comes from the `variant` option.

| Field | Type | Meaning |
|---|---|---|
| `precision` | string | The option whose value is each stage's [precision](#precisions). The checkpoint's when absent. |
| `quantizationBits` | string | The option whose value is the bits of a runtime weight quantization: 0 for none, 8, or 4. |
| `quantizationGroupSize` | string | The option whose value is the quantization's group size. 64 when absent. |
| `contextLength` | string | The option whose value is the positions the key-value cache holds. The facts' `maximumContextLength` when absent. |
| `cacheBits` | string | The option whose value is the bits of a quantized key-value cache: 0 for none, 8, or 4. |
| `cacheGroupSize` | string | The option whose value is the cache quantization's group size. 64 when absent. |
| `draft` | string | The model-reference option whose value is the draft model's identifier. |
| `draftBits` | string | The option whose value is the bits the draft model's weights are quantized to. |
| `residency` | string | The option whose value is a [residency](#residencies). The model's only residency when absent, else `automatic`. |
| `cacheStage` | string | The stage that holds the key-value cache and the draft model. The last stage when absent. |
| `draftStage` | string | The stage of this model a draft for another model loads. Every stage when absent. |
| `stagePrecisions` | object of string | The precision of each stage the precision option does not change, keyed by stage. |

### The estimate

`-[NFKModelDescriptor estimatedMemoryForOptions:memoryBudget:error:]` resolves the load options, fills
their defaults, and computes per stage, at precision p, with e(p) the bytes of one element:

- **Weights** w: the stage's bytes at p, where p is `stagePrecisions`' entry, else the precision option's
  value, else `checkpoint`. A quantization of b bits in groups of g replaces the stage's Q quantizable
  elements: w − Q·e(p) + Q·(b/8 + 4/g), rounded up.
- **Key-value cache**, on the cache stage, for c positions:
  - unquantized → (perToken·c + bounded)·e(p)/2;
  - quantized to b bits in groups of g → (perToken·c + bounded)/2·(b/8 + 4/g).
- **Draft**, on the cache stage: the draft descriptor's draft stage at the precision option's value and
  the draft bits, plus the draft's own key-value cache from its default variant's facts at the same c
  positions, cache precision, and cache bits: (perToken_draft·c + bounded_draft), converted as the cache
  above.
- **Linear options**, on the cache stage: each integer option with `memoryBytesPerUnit`, its value
  times that figure. The context length counts here only when the facts state no per-position cache.

The residency then decides what the model holds once loaded:

| Residency | Resident bytes |
|---|---|
| `resident` | The sum of the stages. |
| `staged` | The largest stage. |
| `paged` | The largest stage without its routed experts, plus an expert cache of min(pageable, (budget − reserve − unpaged) / 2), or 4 GiB without a budget. |
| `streamed` | A streamed stage's bytes outside its layers, its two largest layers, and the first layers that fit the rest of the budget. |
| `automatic` | Whichever of the above the [residency planner](#the-residency-planner) picks. |

The peak adds `loadOverheadBytes`. The estimate fits when the budget admits the peak: the peak plus the
4 GiB reserve is at most the budget, or the machine reports no budget.
`minimumWorkingSetByteCount` is the least working set that admits the peak. A variant without facts
estimates its `residentBytes` and `peakBytes` as one stage, and a variant with neither estimates 0 with
basis `"unknown"`. An estimate that does not fit says why:
`The model needs 9.4 GiB plus a 4.0 GiB reserve, against a 6.8 GiB budget.`

An estimate's dictionary holds `residentBytes`, `peakBytes`, `minimumWorkingSetBytes`,
`requestedResidency`, `residency`, `components` (`weights`, `keyValueCache`, `draft` (the draft's weights and cache), `expertCache`,
`linearOptions`, and `loadOverhead`, which sum to the resident bytes with the overhead aside),
`stageBytes`, `options`, `budget`, `fits`, `reason`, and `basis`.

### The budget

`NFKModelMemoryBudget` is the memory a machine plans against: 0.85 of the recommended Metal working set,
rounded down, with a 4 GiB reserve for activations and request buffers inside it. A machine that reports
no working set has a budget of 0, which admits every stage (`admitsByteCount:`) and holds none
(`holdsByteCount:`, the test for keeping stages resident or taking a wider precision).
`+currentBudget` reads this machine, `+budgetForHardwareProfile:` a profile, and
`+budgetWithWorkingSetByteCount:` a working set. Its dictionary holds `workingSetBytes`, `budgetBytes`,
`reserveBytes`, and `workingSetFraction`.

### The residency planner

`+[NFKModelResidencyPlan planForStages:residency:budget:error:]` decides how a model of stages
(`NFKModelStageFootprint`: bytes, pageable bytes, widened bytes, and layer bytes) holds them:

- `resident` → every stage held; refused where the stages are known not to fit.
- `staged` → stages take turns.
- `paged` → stages take turns and routed experts are paged; refused where a stage does not fit even
  paged. A model without routed experts is staged.
- `streamed` → stages take turns, and a stage known not to fit whole streams its repeated layers;
  refused where such a stage does not fit even streamed.
- `automatic` → resident where the total fits, paged where the largest stage does not fit whole and
  has experts, streamed where a stage with layers does not fit whole, else staged. With no budget,
  staged.

A stage with a wider precision takes it where that fits: in order while the total still fits when
resident, alone when staged or paged, never without a budget. A refusal is
`NFKModelCatalogErrorDoesNotFit`, and its description is the planner's sentence:

| Refusal | Sentence |
|---|---|
| Resident | `the stages need 41.0 GiB together plus a 4.0 GiB reserve, against a 27.2 GiB working set; hold them staged or paged` |
| Paged | `the largest stage needs 30.0 GiB with its routed experts paged, plus a 4.0 GiB reserve, against a 24.0 GiB working set` |
| Streamed | `stage 0 needs 22.0 GiB streamed, two layers in flight, plus a 4.0 GiB reserve, against a 24.0 GiB working set` |
| A staged stage alone | `transformer alone needs 30.0 GiB plus a 4.0 GiB reserve, against a 24.0 GiB working set` |

The resident sentence names `or paged` and `or streamed` only where the stages have experts or
layers. The last row is `+verifyEachStageLoads:plan:budget:names:error:`, which checks each stage of a
staged plan alone.

### Constraints

`constraintsForHardwareProfile:`, `constraintsForMemoryBudget:`, and
`constraintsForOptions:memoryBudget:` return `NFKModelMemoryConstraints`: what a budget leaves of each
option.

- The memory-affecting choice options (the variant and each choice option the memory model names)
  vary over their choices. A draft option without choices varies over no draft and the
  [compatible drafts](#draft-models).
- Up to 4096 combinations are enumerated in full, and `exhaustive` is true. Past that, each option's
  choices are evaluated one at a time with every other option at the caller's selection, else its
  default, so every choice still has a status, and `exhaustive` is false: a choice ruled out alone
  may still fit beside other changes.
- A linear integer option is solved for: the largest value that fits is (budget − reserve − peak at 0)
  / bytes per unit, clipped to the option's range and rounded down to its step. The context length's
  bytes per position include a selected draft model's cache.
- A combination fits when it fits with each linear option at its given value, else its minimum.
- `constraintsForOptions:memoryBudget:` fixes the given options. Each option's constraint fixes the
  other given options and lets the rest vary, so an app passes its current selections and reads what
  each control still allows.
- `constraintsForHardwareProfile:` also rules out each choice and option whose requirements the
  machine does not meet.

Each option's `NFKModelOptionConstraint` lists its choices' statuses (`allowed`, `required`,
`ruled-out`), `requiredValue` where exactly one choice fits (`automatic` is not counted, since it picks
one of the others), and `maximumValue` where the largest value that fits is below the option's own
maximum. Each reason is a sentence from a fixed template, and its numbers ride beside it
(`fullyResidentBytes`, `budgetBytes`, `techniqueBytes`) so an app can write its own:

| Constraint | Sentence |
|---|---|
| A required residency | `The model needs 41.0 GiB fully resident plus a 4.0 GiB reserve, against a 27.2 GiB budget. Paged, it needs 16.1 GiB.` |
| A required choice that is not the default | `With None, the model needs 9.4 GiB plus a 4.0 GiB reserve, against a 6.8 GiB budget. With 4-bit, it needs 2.6 GiB.` |
| A required choice that is the default | `Only Checkpoint fits: it needs 1.9 GiB plus a 4.0 GiB reserve, against a 6.8 GiB budget.` |
| A ruled-out choice | `Resident needs at least 41.0 GiB plus a 4.0 GiB reserve, against a 27.2 GiB budget.` |
| A choice the machine cannot take | `Turbo is unavailable here. This needs Apple silicon.` |
| A maximum | `Context length fits up to 9728 against a 6.8 GiB budget.` |
| No combination | `No combination of options fits: the smallest needs 9.0 GiB plus a 4.0 GiB reserve, against a 3.4 GiB budget.` |

`fittingOptions` is the fitting combination that changes the fewest options from their defaults, with
each linear option at its default or the largest value that fits, and `fittingEstimate` its estimate.
Every constraint type reads and writes its dictionary; the constraints' dictionary carries `exhaustive`,
which reads as true when absent.

### Memory figure

`NFKModelMemoryFigure`, a choice's `memory`: `residentBytes` and `peakBytes` (the resident bytes when
absent). `-[NFKModelDescriptor descriptorByDerivingMemoryFigures]` fills each memory-affecting choice's
figure from the facts, without a budget and with every other option at its default, and
`memoryByVariant` for a descriptor with several variants. The residency choice `automatic` takes no
figure, because its memory depends on the machine. It gives the context-length option the cache
bytes of one position as `memoryBytesPerUnit`, and a variant with facts its `residentBytes`, `peakBytes`,
and `minimumMemoryBytes` at the defaults. A figure the manifest states wins over the derived one. Draft
models are looked up in the shared catalog; `descriptorByDerivingMemoryFiguresInCatalog:` looks them up
in another, which a descriptor provider uses inside `+modelDescriptors`, since the shared catalog calls
that method while it loads.

### Draft models

A draft model is compatible with a model when it has the same `family`, the same `vocabularySize` in
its default variant's facts, and a smaller `parameterCount`.
`-[NFKModelDescriptor draftDescriptorsForVariant:inCatalog:]` lists them. A draft option that lists
choices allows exactly those, whatever the rule says.

## Presets

`NFKModelPreset` (`NFKModelPreset.h`), an entry of a model's `presets`: a named set of load options and
request defaults, reached as `model@preset`, such as `whisper@fast`.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | The identifier, unique within its model. |
| `title` | string | no | The name a user interface shows. The `id` when absent. |
| `help` | string | no | A sentence that explains the preset. |
| `base` | string | no | The model the preset applies to. The descriptor it is listed under when absent. |
| `loadOptions` | object | no | The load options the preset sets. |
| `requestDefaults` | object | no | The request parameters the preset fills where a request leaves them out. |
| `suggested` | boolean | no | True for a preset a server offers without being told to. |

A preset that sets only request defaults runs on a backend loaded for its base model; one that sets
load options needs a load of its own (`changesLoadOptions`).

- `NFKModelPresetSeparator` is `"@"`. `NFKModelPresetModelName(@"whisper", @"fast")` is
  `"whisper@fast"`.
- `NFKModelNameSplitAtPreset` splits at the last separator, so `a@b@fast` is the preset `fast` of
  `a@b`. The split happens only where the predicate it is given accepts the two parts, so
  `claude-3@20240620` passes whole where no model `claude-3` has a preset `20240620`.
- `NFKModelToolName` writes a name as a tool name: the separator becomes `"__"`
  (`NFKModelToolNameSeparator`), every character other than a letter, a digit, `_`, and `-` becomes `_`,
  and the result is cut to 64 characters. `whisper@fast` is `whisper__fast`.

## Load options and the build

### Option values

`NFKModelOptions.h`. `-[NFKModelDescriptor resolvedOptions:stage:error:]` converts each value to its
option's kind and fills each absent option's default; `problemsWithOptions:stage:` lists what it cannot
use. A value converts as follows:

| Kind | Takes |
|---|---|
| `integer` | A whole number, or a string of one such as `"8192"`. |
| `number` | A number, or a numeric string. |
| `boolean` | A boolean, or `"true"`, `"false"`, `"yes"`, `"no"`, `"1"`, `"0"`. |
| `choice` | A value equal to a choice's, or with the same text, such as `"8"` for `8`. The result is the choice's value. |
| `text` | A string. |
| `text-list` | An array of strings, or one string. |
| `file-or-folder` | A path or a file URL. The path has to exist. |
| `model-reference` | A catalog model's identifier. A draft option takes one of its choices, else a [compatible draft](#draft-models), or `""` for none. |
| `json` | Any JSON value. |

Each problem is an `NFKModelOptionProblem` with the key, a code, a message, and the value:

| Code | Message |
|---|---|
| `unknown-key` | `example.chat has no option "nope".` |
| `wrong-type` | `"steps" takes an integer, and "many" is not one.` |
| `below-minimum` | `"steps" takes 1 or more, and 0 is less.` |
| `above-maximum` | `"steps" takes 50 or less, and 80 is more.` |
| `off-step` | `"width" takes steps of 16 from 256, and 260 is between them.` |
| `not-a-choice` | `"precision" takes one of "checkpoint", "float32", and "int2" is none of them.` |
| `unavailable` | `"precision" cannot be "turbo" on this machine. This needs Apple silicon.` |
| `wrong-stage` | `"precision" is set when the model loads.` or `"temperature" is set on each request.` |
| `unknown-model` | `"draftModel" names "ghost", which is not in the catalog.` |
| `missing-file` | `"adapter" names "/path", which does not exist.` |

A request may set the memory model's context length, cache bits, and cache group size without a
`wrong-stage` problem. `ignoreMemoryBudget` is accepted by every load and left out of the result.
`canonicalLoadOptionsKey:` writes the resolved load options as sorted JSON, so two sets of options that
build the same backend give the same string.

### The build entry

`NFKModelBuilding.h`. `-[NFKModelCatalog backendForModelIdentifier:options:location:error:]` takes a
model's identifier, or `model@preset`, and the option values an app collected:

1. The descriptor is found, and a preset's load options and request defaults are merged under the
   given options (`NFKModelCatalogErrorUnknownModel`).
2. The options resolve at both stages (`NFKModelCatalogErrorInvalidOptions`, the problems under
   `NFKModelOptionProblemsKey`).
3. The model's requirements and the chosen choices' are checked against this machine
   (`NFKModelCatalogErrorUnavailable`).
4. The estimate is checked against this machine's budget (`NFKModelCatalogErrorDoesNotFit`, the
   estimate under `NFKModelMemoryEstimateKey`). `ignoreMemoryBudget` YES skips the refusal; an app sets
   it only when its user asks. The refusal carries the way out:
   - another combination fits → the one nearest the requested options is under
     `NFKModelMemoryFittingOptionsKey`, its estimate under `NFKModelMemoryFittingEstimateKey`, and the
     description names the options it changes and its peak:
     `The model needs 41.0 GiB plus a 4.0 GiB reserve, against a 27.2 GiB budget. Choosing residency paged fits: it holds 16.1 GiB.`
   - nothing fits → the description adds the smallest combination's need:
     `… against a 6.8 GiB budget. No combination of options fits: the smallest needs 9.0 GiB plus a 4.0 GiB reserve, against a 6.8 GiB budget.`
5. The variant's files are located: the given URL, else the catalog's `locator`. The default locator,
   `hubCacheLocator`, finds a Hugging Face source in `NFKHFHub`'s default cache: the snapshot directory
   of a directory layout, the first weights file of a file layout. A download store sets its own.
6. A builder builds the backend from an `NFKModelBuildRequest`: the descriptor's provider class when it
   implements `+backendForBuildRequest:error:`, else the `NFKModelBackendBuilding` registered for the
   engine (`+registerBackendBuilder:forEngine:`), else the descriptor's `providerClass` through
   `NFKDynamicBackend` with the property options set (`NFKModelCatalogErrorNoBuilder` when there is
   none).
7. Request defaults wrap the backend in an `NFKDefaultedBackend`.

The build request carries `loadOptions` (every load option, defaults filled), `argumentOptions` (those
without a `property`), `propertyOptions` (those with one), `requestDefaults`, `location`, and the
`memoryEstimate`. A builder calls `applyPropertyOptionsToBackend:error:` on what it constructs, so a
build that runs again sets the properties again; a key path without a setter is
`NFKModelCatalogErrorInvalidOptions` and sets nothing. `locationForModelIdentifier:variant:error:` finds
another model's files, such as a draft's.

`requestDefaults` holds the request options the given options and the preset set, and the load-stage
context length and cache settings, which a request may still override. `NFKDefaultedBackend` fills them
into each request that leaves them out and forwards the rest to the backend it wraps, `unload`,
`isLoaded`, and the byte counts included. A backend reports the load options it was built with as
`loadOptionValues`, read by `NFKInferenceLoadOptionValues`; the defaulted backend reports the build's
when the backend reports none. The values are the requested options with defaults filled, `automatic`
included, so they compare equal to the options an app would build with. A backend that knows how its
load held the weights reports it in `modelInfo` under `NFKModelInfoResidency`.

`NFKAppleModelDescriptors` builds its engines through `+backendNamed:`.

## Vocabulary

Each set is a list of string constants in `NFKModelVocabulary.h`. A reader passes a string it does
not know through unchanged.

### Value types

| String | Constant | The value |
|---|---|---|
| `text` | `NFKModelValueTypeText` | A string. |
| `messages` | `NFKModelValueTypeMessages` | A chat conversation in the OpenAI shape. |
| `image` | `NFKModelValueTypeImage` | One image. |
| `images` | `NFKModelValueTypeImages` | Several images. |
| `mask` | `NFKModelValueTypeMask` | A mask or matte over an image. |
| `video` | `NFKModelValueTypeVideo` | One clip. |
| `videos` | `NFKModelValueTypeVideos` | Several clips. |
| `audio` | `NFKModelValueTypeAudio` | One recording. |
| `audios` | `NFKModelValueTypeAudios` | Several recordings. |
| `document` | `NFKModelValueTypeDocument` | One document for a text model to read. |
| `documents` | `NFKModelValueTypeDocuments` | Several documents. |
| `number` | `NFKModelValueTypeNumber` | A real number. |
| `integer` | `NFKModelValueTypeInteger` | A whole number. |
| `boolean` | `NFKModelValueTypeBoolean` | A boolean. |
| `text-list` | `NFKModelValueTypeTextList` | A list of strings. |
| `json` | `NFKModelValueTypeJSON` | Any JSON value. |
| `embedding` | `NFKModelValueTypeEmbedding` | A feature vector. |
| `detections` | `NFKModelValueTypeDetections` | Detected objects (`NFKDetection`). |
| `keypoints` | `NFKModelValueTypeKeypoints` | Located landmarks (`NFKKeypoint`). |
| `classifications` | `NFKModelValueTypeClassifications` | Predicted classes (`NFKClassification`). |
| `segments` | `NFKModelValueTypeSegments` | Time spans (`NFKAudioSegment`). |
| `words` | `NFKModelValueTypeWords` | Timed words (`NFKAudioSegment`). |
| `midi` | `NFKModelValueTypeMIDI` | A note sequence (`NFKMIDISequence`). |
| `beats` | `NFKModelValueTypeBeats` | The beats of a track (`NFKMusicBeat`). |
| `questions` | `NFKModelValueTypeQuestions` | Typed questions for a decision model (`NFKDecisionQuestion`). |
| `answers` | `NFKModelValueTypeAnswers` | Typed answers from a decision model (`NFKDecisionAnswer`). |
| `tool-calls` | `NFKModelValueTypeToolCalls` | The tool calls a text model made. |
| `structured` | `NFKModelValueTypeStructured` | Named readings keyed by field name. |

### Tasks

A task a capability of `NFKDynamicBackend` already names uses the capability's string, so a task
resolves through `+[NFKDynamicBackend backendForCapability:error:]` as it is.

| String | Constant | The task |
|---|---|---|
| `text-generation` | `NFKModelTaskTextGeneration` | Writing text from a prompt or a conversation. |
| `vision-question-answering` | `NFKModelTaskVisionQuestionAnswering` | Answering a question about an image or a clip. |
| `transcription` | `NFKModelTaskTranscription` | Turning speech into text. |
| `translation` | `NFKModelTaskTranslation` | Turning text from one language into another. |
| `text-recognition` | `NFKModelTaskTextRecognition` | Reading the text in an image. |
| `speech-synthesis` | `NFKModelTaskSpeechSynthesis` | Speaking text aloud. |
| `text-to-image` | `NFKModelTaskTextToImage` | Making an image from a prompt. |
| `image-to-image` | `NFKModelTaskImageToImage` | Making an image from a source image and a prompt. |
| `inpainting` | `NFKModelTaskInpainting` | Filling a masked region of an image. |
| `controlnet` | `NFKModelTaskControlNet` | Generating an image under a control map. |
| `video-generation` | `NFKModelTaskVideoGeneration` | Making a clip from a prompt, an image, or both. |
| `frame-interpolation` | `NFKModelTaskFrameInterpolation` | Making the frames between two frames. |
| `upscaling` | `NFKModelTaskUpscaling` | Enlarging an image or a frame. |
| `image-restoration` | `NFKModelTaskImageRestoration` | Removing noise, blur, or damage from an image. |
| `depth-estimation` | `NFKModelTaskDepthEstimation` | Estimating the distance of each pixel. |
| `segmentation` | `NFKModelTaskSegmentation` | A mask over a region or a subject of an image. |
| `matting` | `NFKModelTaskMatting` | A soft alpha matte that separates a subject from its background. |
| `object-detection` | `NFKModelTaskObjectDetection` | Boxes around the objects in an image. |
| `pose` | `NFKModelTaskPoseEstimation` | Body or hand joints. |
| `face-detection` | `NFKModelTaskFaceDetection` | Faces and their landmarks. |
| `optical-flow` | `NFKModelTaskOpticalFlow` | Motion between two frames. |
| `classification` | `NFKModelTaskClassification` | Labels for an image, a clip, or a recording. |
| `image-embedding` | `NFKModelTaskImageEmbedding` | A vector for comparing images. |
| `embedding` | `NFKModelTaskEmbedding` | A vector for comparing text, or text with images. |
| `reranking` | `NFKModelTaskReranking` | Ordering documents by relevance to a query. |
| `decision` | `NFKModelTaskDecision` | Answering typed questions about a state: a choice, a score, or whether a statement holds (`NFKDecisionQuestion`). |
| `forecasting` | `NFKModelTaskForecasting` | Predicting the next values of a time series. |
| `source-separation` | `NFKModelTaskSourceSeparation` | Splitting a recording into its sources. |
| `speech-enhancement` | `NFKModelTaskSpeechEnhancement` | Removing noise and reverberation from speech. |
| `voice-activity-detection` | `NFKModelTaskVoiceActivityDetection` | Finding the spans of a recording that hold speech. |
| `audio-codec` | `NFKModelTaskAudioCodec` | Encoding audio into tokens and decoding it back. |
| `music-generation` | `NFKModelTaskMusicGeneration` | Making music from a prompt or lyrics. |
| `music-transcription` | `NFKModelTaskMusicTranscription` | Turning a performance into notes. |
| `music-structure` | `NFKModelTaskMusicStructure` | The beats and sections of a track. |

### Engines

| String | Constant | The engine |
|---|---|---|
| `mlx` | `NFKModelEngineMLX` | The InferKitMLX companion. |
| `apple` | `NFKModelEngineApple` | The core's Apple-framework engines. |
| `foundation-models` | `NFKModelEngineFoundationModels` | The InferKitFoundationModels companion. |
| `apple-swift` | `NFKModelEngineAppleSwift` | The InferKitAppleSwift companion. |
| `plugin` | `NFKModelEnginePlugin` | A backend a plugin or an app brings. |

### Residencies

| String | Constant | How the weights are held |
|---|---|---|
| `resident` | `NFKModelResidencyResident` | Every stage stays loaded between runs. |
| `staged` | `NFKModelResidencyStaged` | Each stage loads for its turn and is released after it. |
| `paged` | `NFKModelResidencyPaged` | Stages take turns, and each stage's routed experts load as a step routes to them. |
| `streamed` | `NFKModelResidencyStreamed` | Stages take turns, and a stage's repeated layers that do not fit load in their turn on each pass. |
| `automatic` | `NFKModelResidencyAutomatic` | The residency planner picks one of the others for the machine. The default; its memory depends on the machine. |

### Hardware features

| String | Constant | The machine has |
|---|---|---|
| `apple-silicon` | `NFKModelFeatureAppleSilicon` | Apple silicon: a Mac with an M-series chip, or any iPhone, iPad, or Apple TV. |
| `neural-accelerators` | `NFKModelFeatureNeuralAccelerators` | A GPU with the neural accelerators of the M5 on (`NFKHardwareProfile.hasNeuralAccelerators`). |

### Precisions

| String | Constant | The type |
|---|---|---|
| `checkpoint` | `NFKModelPrecisionCheckpoint` | Each tensor's stored type. |
| `float32` | `NFKModelPrecisionFloat32` | 32-bit floating point. |
| `float16` | `NFKModelPrecisionFloat16` | 16-bit IEEE floating point. |
| `bfloat16` | `NFKModelPrecisionBFloat16` | 16-bit brain floating point. |

### Load options

The keys an engine's load options share, so an app configures every engine the same way.

| String | Constant | The value |
|---|---|---|
| `variant` | `NFKModelLoadOptionVariant` | The variant's identifier. |
| `residency` | `NFKModelLoadOptionResidency` | A [residency](#residencies). |
| `precision` | `NFKModelLoadOptionPrecision` | A [precision](#precisions). |
| `quantizationBits` | `NFKModelLoadOptionQuantizationBits` | The bits of a runtime weight quantization: 0 for none, 8, or 4. |
| `quantizationGroupSize` | `NFKModelLoadOptionQuantizationGroupSize` | The weights that share one scale and bias. |
| `contextLength` | `NFKModelLoadOptionContextLength` | The positions the key-value cache holds by default. |
| `cacheBits` | `NFKModelLoadOptionCacheBits` | The bits of a quantized key-value cache: 0 for none, 8, or 4. |
| `cacheGroupSize` | `NFKModelLoadOptionCacheGroupSize` | The cache elements that share one scale and bias. |
| `draftModel` | `NFKModelLoadOptionDraftModel` | The draft model's identifier. |
| `draftBits` | `NFKModelLoadOptionDraftBits` | The bits of the draft model's weights. |
| `computeUnits` | `NFKModelLoadOptionComputeUnits` | The Core ML compute units, an `MLComputeUnits` value. |
| `ignoreMemoryBudget` | `NFKModelLoadOptionIgnoreMemoryBudget` | YES builds a combination the memory estimate says does not fit. |

### Option kinds

| String | Constant | The control edits |
|---|---|---|
| `boolean` | `NFKModelOptionKindBoolean` | An on-off switch. |
| `integer` | `NFKModelOptionKindInteger` | A whole number. |
| `number` | `NFKModelOptionKindNumber` | A real number. |
| `choice` | `NFKModelOptionKindChoice` | One of the option's choices. |
| `text` | `NFKModelOptionKindText` | A string. |
| `text-list` | `NFKModelOptionKindTextList` | A list of strings. |
| `json` | `NFKModelOptionKindJSON` | Any JSON value. |
| `file-or-folder` | `NFKModelOptionKindFileOrFolder` | A path to a file or a folder. |
| `model-reference` | `NFKModelOptionKindModelReference` | The identifier of another model in the catalog. |

### Option stages

| String | Constant | The option |
|---|---|---|
| `request` | `NFKModelOptionStageRequest` | Rides on each request as a parameter. |
| `load` | `NFKModelOptionStageLoad` | Is fixed when the backend is built, and changing it builds the backend again. |

### File roles

| String | Constant | The file holds |
|---|---|---|
| `weights` | `NFKModelFileRoleWeights` | The model's weights. |
| `config` | `NFKModelFileRoleConfig` | The model's configuration. |
| `tokenizer` | `NFKModelFileRoleTokenizer` | The model's tokenizer or vocabulary. |
| `other` | `NFKModelFileRoleOther` | Anything else the model reads. |

### Source kinds

| String | Constant | The files come from |
|---|---|---|
| `huggingface` | `NFKModelSourceKindHuggingFace` | A Hugging Face repository. |
| `url` | `NFKModelSourceKindURL` | A download URL. |
| `system` | `NFKModelSourceKindSystem` | The operating system. |
| `none` | `NFKModelSourceKindNone` | Nowhere: the model needs no files, or the app supplies them. |

### Source layouts

| String | Constant | The model loads from |
|---|---|---|
| `file` | `NFKModelSourceLayoutFile` | One file. |
| `directory` | `NFKModelSourceLayoutDirectory` | A directory of files. |
| `none` | `NFKModelSourceLayoutNone` | No files of its own. |

## Errors

Every failure is in `NFKModelCatalogErrorDomain`.

| Code | When |
|---|---|
| `NFKModelCatalogErrorUnsupportedSchemaVersion` | The manifest's `schemaVersion` is newer than this release reads. |
| `NFKModelCatalogErrorMalformedManifest` | The data is not a JSON object, or `schemaVersion`, `kind`, or `models` is missing or of the wrong type, or `kind` is not `"inferkit.model-catalog"`. |
| `NFKModelCatalogErrorMalformedDescriptor` | An object is missing a required field, has a field of the wrong type, or holds a value JSON cannot carry. The description names the field by its path, such as `models[2].variants[0].source.kind is missing`. |
| `NFKModelCatalogErrorInvalidOptions` | An option value does not resolve, or a build's property option names a key path the backend has no setter for. `NFKModelOptionProblemsKey` holds the problems. |
| `NFKModelCatalogErrorUnknownModel` | The catalog has no model with the identifier: `The catalog has no model "ghost".` |
| `NFKModelCatalogErrorNoBuilder` | Nothing builds the model's engine: `No builder builds models of the engine "x".` |
| `NFKModelCatalogErrorUnavailable` | This machine does not meet the model's requirements, or a chosen choice's: `Example Chat cannot run on this machine. This needs Apple silicon.` |
| `NFKModelCatalogErrorDoesNotFit` | The estimate does not fit the budget, or the residency planner refuses the residency. `NFKModelMemoryEstimateKey` holds the estimate where there is one; a build's refusal adds `NFKModelMemoryFittingOptionsKey` and `NFKModelMemoryFittingEstimateKey` where another combination fits. |

## The MLX catalog's measurements

The MLX companion's descriptors carry measured figures. `Tools/catalog/measure.py` reads every variant
source the catalog lists and writes them to `InferKitMLX/Sources/InferKitMLX/NFKMLXCatalogMeasurements.swift`,
a generated table the descriptors read when they are built.

| Field | Where it comes from |
|---|---|
| file `bytes` | The file's size in the repository's file tree, or a URL's `Content-Length`. |
| file `sha256` | The SHA-256 Git LFS records for the file. A file Git stores directly has none. |
| `downloadBytes` | The files the package's download fetches: each required file, each optional file the repository serves, the first weight candidate a directory serves, and the shards a `*.index.json` names. |
| `parameterCount` | The elements of the safetensors tensors the download holds, read from each file's header. A variant with a weight file in another format (a pickle, a GGUF, an archive), or a header the repository refuses, has none. A packed format counts its stored elements. |
| `memoryFacts` | Each stage's tensor bytes at the checkpoint's types, at float32, and at bfloat16, with the figures the model's loader acts on. [Memory facts and load options](#memory-facts-and-load-options) lists them. |
| `residentBytes`, `peakBytes` | The [estimate](#the-estimate) at the default options: the weights at the precision the loader holds them in, and a decoder's key-value cache at its default context length. A variant without memory facts takes the tensor bytes at the loader's precision. The load overhead is not measured, so the peak is the resident figure. |
| `minimumMemoryBytes` | The least working set whose residency budget admits the peak: the peak plus the 4 GiB reserve, divided by 0.85. A staged model counts its largest stage. |
| `precision` | The floating type that holds the most stored tensor bytes, where the entry states none. |

A measured variant can carry these extension fields:

| Field | On | Meaning |
|---|---|---|
| `measuredRevision` | source | The commit the source's revision resolved to when it was measured. The file digests are that commit's. |
| `note` | source | A sentence a user should read about the source, such as a release that needs a converter none ships yet. |
| `publiclyServed` | source | `false` for a repository its publisher no longer serves to the public. A local copy of the release still builds the model; the measurement records it as `"unmeasurable"`. |
| `measurement` | variant | `"gated"` (the repository refused the listing), `"unmeasurable"` (a release page or a host that serves no size), or `"missing"` (a required file is not served). Absent for a measured variant. |
| `measurementNote` | variant | Why a measurement is incomplete, such as the tensor headers a gated repository refused. |
| `residentEstimate` | variant | `"stored-dtype"` where `residentBytes` counts the stored bytes because the loader's precision or the stored type is not known. |

A model's `requirements.minimumMemoryBytes` is the least of its variants' figures.

### Memory facts and load options

`measure.py` also reads the structure of each safetensors header and the release's `config.json`:

| Measured | Where it comes from |
|---|---|
| Repeated layers | The tensor names under `layers.N.`, `h.N.`, `blocks.N.`, or `transformer_blocks.N.`: the count and the largest layer's bytes, of the container that holds the most bytes in each top-level directory. |
| Routed experts | The bytes of the tensors whose names hold `.experts.`. |
| Quantizable elements | The elements of each rank-2 `.weight` whose input width divides 64, outside the embeddings and the norms. |
| Geometry | `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `hidden_size`, `max_position_embeddings`, `sliding_window`, `sliding_window_pattern`, `num_kv_shared_layers`, `global_head_dim`, `num_global_key_value_heads`, `vocab_size`, and the layer kinds (`layer_types`, `layers_block_type`, or `hybrid_override_pattern`), from the root of `config.json` and from `text_config`. A gated repository refuses the file without an accepted token, and its geometry stays unmeasured. |

Each family's factory reads a fixed set of load options, and the MLX catalog declares them with their
memory model:

| Families | Load options |
|---|---|
| Qwen3, Mistral Small 3, Qwen3-30B-A3B, Qwen1.5-MoE, Mixtral | `precision` (`checkpoint`, `float32`), `quantizationBits`, `contextLength`, `cacheBits`, `cacheGroupSize`, and `draftModel` with `draftBits` where a draft exists. A mixture adds `residency` (`automatic`, `resident`, `paged`). |
| gpt-oss | The same without `precision`: its MXFP4 experts load as stored. |
| Gemma 3, TranslateGemma | `residency` (`automatic`, `resident`, `streamed`) and `precision`. |
| MADLAD-400 | `residency` (`automatic`, `resident`, `streamed`) and `precision` (`bfloat16`, `float32`). |
| Gemma 3n | `precision`. |
| Gemma 4, Granite 4.0-H, Qwen3-VL, Qwen3.8-Flash-Next, DeepSeek V4 | `residency` (`automatic`, `resident`, `paged`) on a mixture. A dense release declares none. |
| FLUX.1, FLUX.2, Stable Diffusion 3, SANA, Z-Image, Qwen-Image, Wan, LTX-Video, MiniMax Music 3 | `residency` (`automatic`, `resident`, `staged`). |
| Stable Diffusion 1.5, 2.1, SDXL-Turbo | `precision`, set on the backend's `precision` property. |
| Whisper | `precision`. |

`residency` defaults to `automatic`, `precision` to the precision the factory loads at without it,
`quantizationBits`, `cacheBits`, and `draftBits` to 0 (none, 8, or 4), `cacheGroupSize` to 64 (32, 64, or
128), and `contextLength` to 4096, at most the release's `max_position_embeddings`. An 8-bit or 4-bit
weight quantization is an approximation of the published weights, and its help text says so.

A variant's memory facts carry only what its loader acts on:

- Each stage's bytes: the stages the load plan names, or one stage. A stage whose directories load at
  different precisions states its loaded bytes under `"loaded"`, and the memory model's
  `stagePrecisions` names it. A stage the budget may widen states `widenedPrecision`.
- `quantizableElements` where the factory quantizes, `pageableBytesByPrecision` where the model pages its
  routed experts, and `layerCount` and `largestLayerBytesByPrecision` where it streams its layers.
- `keyValueBytesPerToken` where a context-length option sets the cache: the Qwen3, Mistral, and mixture
  decoders and gpt-oss, every layer at the key-value head count. gpt-oss keeps every key of its sliding
  layers and masks the window. A family whose backend reads no cache parameter declares no cache option
  and states no cache, because the estimate would otherwise count it at the maximum context.
- `vocabularySize` from the geometry, which the draft rule compares.

A decoder's `draftModel` lists as choices the catalog's decoders of its family with its vocabulary
size and fewer parameters, smallest first. The package's descriptors derive every choice's figure in a
catalog of their own (`descriptorByDerivingMemoryFiguresInCatalog:`), where the drafts are found.

The suggested [presets](#presets):

| Preset | On | Sets |
|---|---|---|
| `fast` | A decoder with a weight and a cache quantization | Load options: `quantizationBits` 4 and `cacheBits` 8. |
| `fast` | A generator of 8 or more default steps | Request defaults: half the default `steps`. |
| `precise` | A model whose `precision` offers `float32` and defaults to another | Load options: `precision` `float32`. |

## The snapshot and the tools

`Docs/model-catalog.json` is the manifest of the core's Apple engines and the MLX companion's models, the
catalog an app that links the core and InferKitMLX loads. A test in the MLX package builds the same
manifest in process and compares it with the file byte for byte.

```bash
# Rewrite the snapshot after a catalog change (from InferKitMLX/)
swift run inferkit-catalog export --output ../Docs/model-catalog.json

# List the variant sources the measurement reads, as JSON
swift run inferkit-catalog sources

# Measure every source again and regenerate the table (from the repository root)
python3 -I Tools/catalog/measure.py
```

- `inferkit-catalog` is an executable target of the MLX package that no product names, so a consumer of
  the package does not build it. It builds descriptors and evaluates no MLX array, so it runs without
  MLX's Metal library.
- `measure.py` uses the Python standard library. It reads `/api/models/{repo}/tree/{revision}?recursive=1`
  and `/api/models/{repo}/revision/{revision}` for each Hugging Face source, two HTTP range requests per
  safetensors file for its header, and a HEAD request for each URL file. It fetches small JSON files
  whole (the shard indexes and `config.json`) and no weight file. It sends `HF_TOKEN` when the
  environment sets it.
- The manifest writer sorts keys and models, and the measurements carry no date or machine path, so the
  same catalog writes the same bytes on every machine.
