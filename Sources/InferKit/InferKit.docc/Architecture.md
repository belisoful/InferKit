# Architecture

A dependency-free core, optional companion packages, and any engine you bring.

## Overview

InferKit's core is Objective-C with no third-party dependencies — only Apple frameworks. It holds the
contract, a handful of shipped backends, the download layer, and the runtime-discovery mechanism.
Heavier engines live outside the core and link in only when a consumer wants them.

![A dependency-free core; heavier engines link in optionally, discovered at runtime.](architecture)

### The core

- **The contract** — ``NFKInferenceRequest``, ``NFKInferenceResult``, ``NFKInferenceJob`` and the
  shared key vocabulary.
- **Shipped backends** — ``NFKPassthroughBackend`` (the mock), ``NFKCoreMLBackend`` and
  ``NFKCoreMLLanguageBackend`` (in-process Core ML), ``NFKRemoteBackend`` and
  ``NFKRemoteTranscriptionBackend`` (OpenAI-compatible), and ``NFKAsyncGenerationBackend`` (a
  submit-poll-fetch base). Every one is pure Apple frameworks.
- **Download / cache** — ``NFKHFHub`` resolves and caches Hugging Face files. See <doc:DownloadingModels>.
- **Discovery** — ``NFKDynamicBackend`` activates an engine only when its classes are linked. See
  <doc:DynamicDiscovery>.

### The companions

Two optional Swift packages build on the core without raising its platform floor or adding
dependencies to it:

- **InferKitMLX** (Apple Silicon) — 31 real MLXNN models (upscaling, depth, matting, segmentation,
  detection, pose, restoration, colorization, embeddings, audio) plus a bundled Stable Diffusion.
- **InferKitFoundationModels** (macOS 26 / iOS 26) — a bridge to Apple's on-device system language
  model.

Linking a companion ships its provider classes, so the core's discovery lights up the matching
capability automatically.

### Your engine

Anything that adopts ``NFKInferenceBackend`` is a backend. A heavier or licensed engine
(whisper.cpp, a ControlNet pipeline, a Rust runtime) is brought by the consumer and either used
directly or exposed through ``NFKDynamicBackendProvider`` for name-based discovery — the core never
references the engine's symbols.

## Topics

### Related

- <doc:Backends>
- <doc:DynamicDiscovery>
- <doc:DownloadingModels>
