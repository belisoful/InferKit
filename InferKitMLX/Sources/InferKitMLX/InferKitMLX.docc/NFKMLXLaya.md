# ``NFKMLXLaya``

## Overview

Laya is the open reproduction of TypeSafe's Jev. It answers typed questions about a state on device:
a choice among named options, a score on an ordered scale, or a noul, the probability that a
statement holds. It takes the same `NFKDecisionQuestion` objects as the hosted `NFKTypeSafeBackend`
and returns the same `NFKDecisionAnswer` objects, so moving a feature between the hosted model and
the device means swapping one object.

The weights are published on Hugging Face as `convaiinnovations/laya` under Apache 2.0, ungated. No
account or access token is needed.

### Choose a variant

The repository holds three releases. ``NFKMLXLayaVariant`` names them.

| Variant | Folder | Encoder | Languages | Download | Memory once loaded |
| --- | --- | --- | --- | --- | --- |
| `.root` | repository root | ModernBERT-large, 421M parameters | English | 846 MB | about 1.7 GB |
| `.typedDecisions` | `typed-decisions/` | the root's geometry, fine-tuned on the typed-decisions benchmark | English | 846 MB | about 1.7 GB |
| `.multilingual` | `multilingual/` | mmBERT-base, 322M parameters, Gemma's tokenizer | 100-plus | 678 MB | about 1.3 GB |

The weights ship in half precision and load at float32, which doubles their size in memory. The
factory measures the release against the machine before loading and throws rather than exhausting
memory.

Each variant is five files: `rl_agent_config.json`, `encoder/config.json`,
`tokenizer/tokenizer_config.json`, `tokenizer/tokenizer.json`, and `model.safetensors`.
``NFKMLXLaya/releaseFiles(for:)`` lists them with the variant's folder prefixed.

### Download and build

One call downloads a variant into the hub cache and builds the model. A file already in the cache is
read from disk, so the second launch makes no network request.

```swift
import InferKit
import InferKitMLX

let laya = try NFKMLXLaya.laya(variant: .typedDecisions,
                               revision: NFKMLXLaya.measuredRevision,
                               cacheDirectoryURL: nil)
let answers = laya.decide(state: "I was charged twice for one order.", questions: [
    "department": .choiceQuestion(withInstructions: "Which team should handle this?",
                                  options: ["billing", "technical", "sales"]),
    "urgent": .noulQuestion(withInstructions: "The customer needs an answer today."),
])
answers["department"]?.choice          // the most probable option
answers["urgent"]?.probability         // the probability the statement holds
```

```objc
NSError *error = nil;
NFKMLXLaya *laya = [NFKMLXLaya layaWithVariant:NFKMLXLayaVariantTypedDecisions
                                      revision:NFKMLXLaya.measuredRevision
                             cacheDirectoryURL:nil
                                         error:&error];
```

The call blocks on the network and on loading. Run it off the main thread, or use the
completion-handler form, which does both on a background queue and calls the handler there:

```swift
NFKMLXLaya.laya(variant: .multilingual, revision: NFKMLXLaya.measuredRevision,
                cacheDirectoryURL: nil) { laya, error in
    guard let laya else { return }      // report `error`
    // Keep `laya`; building it again reads 1.3 GB from disk.
}
```

``NFKMLXLaya/backend(variant:revision:cacheDirectoryURL:)`` builds the same model as an
`NFKInferenceBackend`. It reads `NFKInputState` and `NFKInputQuestions` and answers under
`NFKOutputAnswers`, which is the request `NFKTypeSafeBackend` reads.

### Pin the revision

`NFKMLXLaya.measuredRevision` is the repository commit this package's reference-parity measurements
were taken at. Passing it pins the download to those weights. A nil revision follows `main` and
caches under `main`. A cached `main` is not checked for updates, so a device keeps whatever `main`
held on its first download. Pin a revision whenever two devices must answer identically.

### Manage the cache

With a nil `cacheDirectoryURL`, files land under `NFKHFHub.defaultCacheDirectoryURL()`, which is
Application Support's `InferKit/models`. The layout is
`<cache>/convaiinnovations/laya/<revision>/<folder>/`, and the download returns that folder.

- Backup → the hub excludes its cache folder from Time Machine and iCloud backup by default, so a
  device does not back up gigabytes it can download again.
- Size limit → a hub with a `cacheSizeLimit` evicts whole repository snapshots, least recently used
  first. All three variants at one revision are one snapshot, so they are evicted together.
- Keeping a release → `pinCachedRepo(_:revision:)` protects a snapshot from eviction, and can pin
  it before the first download. `removeCachedRepo(_:revision:)` deletes it.

```swift
let hub = NFKHFHub(cacheDirectoryURL: NFKHFHub.defaultCacheDirectoryURL())
try hub.pinCachedRepo(NFKMLXLaya.repository, revision: NFKMLXLaya.measuredRevision)
```

A sandboxed app that must store models in a folder the user chose passes that folder as
`cacheDirectoryURL`, with its security scope started.

### Load without the network

An app that bundles the release, or copies it into place itself, builds from the folder directly:

```swift
let laya = try NFKMLXLaya.laya(directoryURL: variantFolder)     // holds the five files
```

``NFKMLXLaya/download(variant:revision:cacheDirectoryURL:)`` fetches without building, for an app
that downloads during onboarding and builds the model later.

### Requirements

- Apple Silicon, macOS 14 or iOS 17, like the rest of InferKitMLX.
- MLX's Metal library placed beside the executable. An Xcode build places it; a `swift test` run
  needs `Tools/mlx-metallib.sh` first.
- Memory for the loaded variant, listed in the table above. A question costs one encoder pass over at
  most 512 tokens (1,024 for `.multilingual`).

### Fine-tune and reload

The release's README reports that the base checkpoints sit near chance on a new decision task until
they are fine-tuned. ``NFKMLXLaya/fineTune(examples:steps:learningRate:trainable:objective:observer:)`` trains on
labeled examples and ``NFKMLXLaya/fineTune(episodes:steps:learningRate:trainable:lambda:objective:observer:)`` on conversations.
Save the tuned network with `NFKMLXWeights.save`, then rebuild from the downloaded folder with the
tuned file in place of the release's weights:

```swift
let folder = try NFKMLXLaya.download(variant: .root, revision: NFKMLXLaya.measuredRevision,
                                     cacheDirectoryURL: nil)
let laya = try NFKMLXLaya.laya(directoryURL: folder)
_ = try laya.fineTune(examples: examples, steps: 200, learningRate: 1e-4, trainable: .head)
try NFKMLXWeights.save(laya.net, to: tunedURL)

let tuned = try NFKMLXLaya.laya(directoryURL: folder, weightsURL: tunedURL)
```

## Topics

### Downloading

- ``NFKMLXLayaVariant``
- ``NFKMLXLaya/repository``
- ``NFKMLXLaya/measuredRevision``
- ``NFKMLXLaya/releaseFiles(for:)``
- ``NFKMLXLaya/download(variant:revision:cacheDirectoryURL:)``
- ``NFKMLXLaya/laya(variant:revision:cacheDirectoryURL:)``
- ``NFKMLXLaya/backend(variant:revision:cacheDirectoryURL:)``

### Building from a folder

- ``NFKMLXLaya/laya(directoryURL:)``
- ``NFKMLXLaya/laya(directoryURL:weightsURL:)``
- ``NFKMLXLaya/backend(directoryURL:)``

### Deciding

- ``NFKMLXLaya/decide(state:questions:)``
- ``NFKMLXLaya/answer(state:question:)``
- ``NFKMLXLayaBackend``

### Customizing

- ``NFKMLXLayaNet``
- ``NFKMLXLayaObjective``
- ``NFKMLXLayaTrainable``
- ``NFKMLXLayaExample``
- ``NFKMLXLayaEpisode``
