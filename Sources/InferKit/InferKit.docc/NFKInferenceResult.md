# ``InferKit/NFKInferenceResult``

## Overview

The immutable output of one run: a dictionary of named outputs, with type-checked convenience getters
for the keys that have a single natural type. A getter returns `nil` on a type mismatch rather than
crashing.

![Result output keys, their typed values, and the convenience accessors.](outputs)

```objc
NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];

result.text;             // NFKOutputText           → NSString
result.embedding;        // NFKOutputEmbedding      → NSArray<NSNumber *>
result.detections;       // NFKOutputDetections     → NSArray<NFKDetection *>
result.pose;             // NFKOutputPose           → NSArray<NFKKeypoint *>
result.classifications;  // NFKOutputClassifications→ NSArray<NFKClassification *>
result.segments;         // NFKOutputSegments       → NSArray<NFKAudioSegment *>

[result outputForKey:NFKOutputImage];  // CVPixelBuffer / texture / CGImage — backend's choice
```

Image, mask, and video outputs stay on `outputForKey:` because their representation is chosen by the
backend or caller.

## Topics

### Value types you may read back

- ``NFKDetection``
- ``NFKKeypoint``
- ``NFKClassification``
- ``NFKAudioSegment``
