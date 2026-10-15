# Model catalog

Describe the models a process can run, so an app lists, downloads, and configures them from one source.

## Overview

``NFKModelCatalog`` holds ``NFKModelDescriptor`` values. A descriptor names the model's engine and
how that engine builds it, the tasks it does, the inputs and outputs it uses, the options it reads,
its releases with their download and memory figures, the license of its weights, and what a machine
needs to run it. ``NFKRemoteModelCatalog`` is a different class: it lists what a remote provider
serves at the moment.

### Where descriptors come from

The shared catalog loads once, on first use, from each provider class linked into the process. A
provider adopts ``NFKModelDescriptorProvider`` and is looked up by name with `NSClassFromString`, as
``NFKDynamicBackend`` looks up a backend provider. A package that is not linked contributes nothing.
The core names four providers, one per engine, and
``NFKModelCatalog/registerProviderClassName:`` adds more.

``NFKAppleModelDescriptors`` is the core's own provider: one descriptor per Apple-framework engine,
each a system model with nothing to download. Its ``NFKAppleModelDescriptors/backendNamed:`` builds
the backend a descriptor's registry name names.

A plugin ships its descriptors as a JSON manifest, and an app adds it:

```objc
NSError *error = nil;
if (![NFKModelCatalog.sharedCatalog addDescriptorsFromManifestData:manifest error:&error]) {
    // NFKModelCatalogErrorUnsupportedSchemaVersion: the manifest comes from a newer InferKit
}
NSArray<NFKModelDescriptor *> *upscalers =
    [NFKModelCatalog.sharedCatalog descriptorsForTask:NFKModelTaskUpscaling];
```

An app describes a backend it builds itself from the keys the backend declares:

```objc
NFKModelDescriptor *described =
    [NFKModelDescriptor descriptorWithIdentifier:@"app.captioner" derivedFromBackend:captioner];
```

### The schema

The manifest is `{"schemaVersion": 1, "kind": "inferkit.model-catalog", "models": [...]}`. Every
object keeps the fields this release does not read in its `extensions` and writes them back, so a
manifest from a newer writer passes through an older reader unchanged. Tasks, value types, engines,
and the other vocabularies are plain strings (`NFKModelVocabulary.h`), so a plugin extends a set by
using a new string. `Docs/model-catalog.md` documents every field.

### Files from elsewhere, and per-variant licenses

A variant's files come from its source. A file that comes from another Hugging Face repository or a
URL names it in ``NFKModelFile/repository``, ``NFKModelFile/revision``, and ``NFKModelFile/URL``: a
vocoder a speech model reads, or the base release a ControlNet steers. A variant whose weights carry
another license than its model names it in ``NFKModelVariant/license``, and
``NFKModelDescriptor/licenseForVariant:`` reads the license that applies.

### Memory figures

A variant carries the bytes its download transfers, the bytes its weights hold once loaded, and the
most bytes it holds at once. ``NFKModelRequirements/minimumMemoryByteCount`` is InferKit's residency
budget: the bytes the model holds at its defaults plus the budget's reserve, as a fraction of the
recommended Metal working set.

### Load options and memory

A descriptor's ``NFKModelDescriptor/options`` hold the settings it reads at load and on each request.
A variant's ``NFKModelVariant/memoryFacts`` record what its stages weigh at each precision, and the
descriptor's ``NFKModelDescriptor/memoryModel`` names the options that change the figure: precision,
weight quantization, context length, cache quantization, a draft model, and residency. The estimate
combines them for one set of options:

```objc
NFKModelMemoryEstimate *estimate = [chat estimatedMemoryForOptions:@{ NFKModelLoadOptionQuantizationBits: @4 }
                                                      memoryBudget:[NFKModelMemoryBudget currentBudget]
                                                             error:&error];
```

``NFKModelMemoryBudget`` plans against 0.85 of the recommended Metal working set with a 4 GiB reserve,
and ``NFKModelResidencyPlan`` decides whether the stages stay resident, take turns, page their routed
experts, or stream their layers. ``NFKModelDescriptor/constraintsForHardwareProfile:`` runs the estimate
over every combination and states, for each option, which choices fit, which one is required, and the
longest context that fits, each with a sentence and its numbers.

### Building a backend

``NFKModelCatalog/backendForModelIdentifier:options:error:`` resolves the options
(``NFKModelDescriptor/resolvedOptions:stage:error:``), checks the machine and the memory estimate, finds
the variant's files, and asks a builder for the backend: the descriptor's provider class, else the
``NFKModelBackendBuilding`` registered for its engine. Request options become defaults an
``NFKDefaultedBackend`` fills into each request. A combination that does not fit is refused with
`NFKModelCatalogErrorDoesNotFit` unless the options set `NFKModelLoadOptionIgnoreMemoryBudget`, which an
app sets only when its user asks. The refusal names the nearest combination that fits, under
`NFKModelMemoryFittingOptionsKey`. A model with ``NFKModelPreset`` entries is also reached as
`model@preset`.

## Topics

### The catalog

- ``NFKModelCatalog``
- ``NFKModelDescriptorProvider``
- ``NFKModelCatalogError``
- ``NFKAppleModelDescriptors``

### Options and building

- ``NFKModelOptionProblem``
- ``NFKModelBuildRequest``
- ``NFKModelBackendBuilding``
- ``NFKDefaultedBackend``
- ``NFKModelPreset``

### Memory

- ``NFKModelMemoryBudget``
- ``NFKModelMemoryFacts``
- ``NFKModelStageFacts``
- ``NFKModelMemoryModel``
- ``NFKModelMemoryFigure``
- ``NFKModelMemoryEstimate``
- ``NFKModelStageFootprint``
- ``NFKModelResidencyPlan``
- ``NFKModelMemoryConstraints``
- ``NFKModelOptionConstraint``
- ``NFKModelChoiceConstraint``
- ``NFKModelChoiceStatus``

### Descriptors

- ``NFKModelDescriptor``
- ``NFKModelPort``
- ``NFKModelOption``
- ``NFKModelOptionChoice``
- ``NFKModelVariant``
- ``NFKModelSource``
- ``NFKModelFile``
- ``NFKModelLicense``
- ``NFKModelRequirements``
