# Model catalog

`NFKModelCatalog` lists the models that the engines linked into a process run, as `NFKModelDescriptor`
values an app shows, downloads, and configures. A catalog's JSON form is a **manifest**. The core,
each companion package, and a plugin all describe their models in the same schema, so an app reads
every source the same way.

This document is the schema reference: the manifest envelope, every field of every object, and the
vocabulary strings. The [examples](examples.md#model-catalog) show the API in use.

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
| `parameters` | array of [option](#option) | no | The settings the model reads. |
| `license` | [license](#license) | no | The license of the model's weights. |
| `variants` | array of [variant](#variant) | no | The releases. The default variant is the one with `isDefault` true, else the first. |
| `requirements` | [requirements](#requirements) | no | What a machine needs. No requirements when absent. |
| `residencies` | array of string | no | How the model can hold its weights: [residencies](#residencies). |
| `standIn` | boolean | no | True for a reference or test stand-in a user interface hides. False when absent. |
| `introducedVersion` | string | no | The InferKit version that introduced the model, such as `"0.4.0"`. |

A writer always writes `id`, `displayName`, `family`, `engine`, the arrays (empty when there is
nothing in them), `requirements`, and `standIn`.

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

`NFKModelOption`, an entry of `parameters`.

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

## Choice

`NFKModelOptionChoice`, an entry of an option's `choices`.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `value` | string or number | yes | The value a request carries when the choice is selected. |
| `title` | string | no | The name a user interface shows. The value's text when absent. |

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

`NFKModelRequirements`, a model's `requirements`.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `minimumOS` | object of string | no | The least operating-system version per platform: `"macOS"`, `"iOS"`, `"tvOS"`, `"visionOS"`, such as `{"macOS": "14.0", "iOS": "17.0"}`. A platform the object leaves out does not run the model. An empty object runs everywhere InferKit runs. |
| `requiresAppleSilicon` | boolean | no | True when the model runs only on Apple silicon. |
| `minimumMemoryBytes` | integer | no | The least memory the model runs in. |

`minimumMemoryBytes` is InferKit's residency budget for the model. It is the smallest recommended
Metal working set (`NFKHardwareProfile.recommendedWorkingSetSize`) whose budget holds the model's
largest resident stage. The residency planner budgets a fixed fraction of the working set and keeps
a reserve for activations inside that budget. The figure is therefore the bytes the largest resident
stage holds plus the reserve, divided by the fraction. A model that stages, pages, or streams counts
only what one stage holds at once. The MLX companion computes the figures for its models.

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
| `residentBytes` | An estimate: the tensor bytes at the precision the model's loader holds them in. |
| `minimumMemoryBytes` | The residency budget of the largest stage for a staged model whose code states the stage split, else of the resident bytes: the bytes plus the 4 GiB reserve, divided by 0.85. |
| `precision` | The floating type that holds the most stored tensor bytes, where the entry states none. |
| `peakBytes` | Not measured. |

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
  safetensors file for its header, and a HEAD request for each URL file. It fetches small JSON shard
  indexes whole and no weight file. It sends `HF_TOKEN` when the environment sets it.
- The manifest writer sorts keys and models, and the measurements carry no date or machine path, so the
  same catalog writes the same bytes on every machine.
