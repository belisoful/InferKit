# Dynamic discovery

Activate a heavier engine only when its classes are linked into the build.

## Overview

InferKit ships only zero-dependency backends. A heavier engine — Stable Diffusion, an on-device LLM,
whisper.cpp, a ControlNet pipeline — is brought by the consumer. ``NFKDynamicBackend`` resolves such an
engine at runtime **by class name**, so the core never references its symbols. Linked in, the engine is
available; absent, resolution returns nil with no link error and no crash.

![A capability maps to a provider class name, looked up at runtime; present classes build a backend, absent ones return nil.](dynamic-discovery)

### Providers

A consumer adds a small class conforming to ``NFKDynamicBackendProvider`` — one method,
`+makeInferenceBackend` — that builds a backend around their engine. InferKit finds it with
`NSClassFromString`.

### Capabilities

Providers register under a capability string; the first present provider wins. Each built-in capability
also has a default provider class name, so a companion that ships that class activates the capability
with no registration:

| Capability | Constant | Default provider (shipped by) |
| --- | --- | --- |
| `stable-diffusion` | ``NFKCapabilityStableDiffusion`` | `NFKStableDiffusionProvider` — InferKitMLX |
| `transcription` | ``NFKCapabilityTranscription`` | `NFKMLXWhisperProvider` — InferKitMLX |
| `text-generation` | ``NFKCapabilityTextGeneration`` | `NFKFoundationModelsProvider` — InferKitFoundationModels |
| `controlnet` | ``NFKCapabilityControlNet`` | `NFKControlNetProvider` — you |

```objc
// Available because a companion (or your provider) is linked; nil otherwise.
if ([NFKDynamicBackend isCapabilityAvailable:NFKCapabilityStableDiffusion]) {
    NSError *error = nil;
    id<NFKInferenceBackend> sd = [NFKDynamicBackend stableDiffusionBackendWithError:&error];
}

// Register your own engine under a capability:
[NFKDynamicBackend registerProviderClassName:@"MyControlNetProvider" forCapability:NFKCapabilityControlNet];
```

## Topics

### The types

- ``NFKDynamicBackend``
- ``NFKDynamicBackendProvider``
