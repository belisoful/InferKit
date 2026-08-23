# ``InferKit/NFKDynamicBackend``

## Overview

Resolves an optional engine at runtime by class name, with no build dependency on it. A capability maps
to a provider class name; if that class is linked into the process it builds a backend, otherwise
resolution returns nil.

![A capability maps to a provider class name; present classes build a backend, absent ones return nil.](dynamic-discovery)

```objc
// Linking InferKitMLX ships NFKStableDiffusionProvider and NFKMLXWhisperProvider.
if ([NFKDynamicBackend isCapabilityAvailable:NFKCapabilityStableDiffusion]) {
    id<NFKInferenceBackend> sd = [NFKDynamicBackend stableDiffusionBackendWithError:&error];
}
id<NFKInferenceBackend> stt =
    [NFKDynamicBackend backendForCapability:NFKCapabilityTranscription error:&error];

// Bring your own engine under any capability:
[NFKDynamicBackend registerProviderClassName:@"MyControlNetProvider"
                               forCapability:NFKCapabilityControlNet];
```

See <doc:DynamicDiscovery> for the built-in capabilities and their default providers.

## Topics

### Provider protocol

- ``NFKDynamicBackendProvider``
