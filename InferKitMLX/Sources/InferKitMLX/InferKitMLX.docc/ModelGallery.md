# Model gallery

Every shipped MLX model, grouped by task, with its registered name and the value type it returns.

## Overview

Each model is a real MLX implementation (in `MLXNN`), not a stand-in, run through one of the base
backends. Two access paths reach the same model:

- **Direct factory** — `SomeModel.backend(…weightsURL:)` builds it from local weights (`nil` → random
  init). A download companion, `SomeModel.backend(…repo:weightsPath:revision:cacheDirectoryURL:)`,
  fetches the checkpoint from Hugging Face first. Variant models take an `@objc` enum
  (``NFKMLXRealESRGANVariant``, ``NFKMLXDepthVariant``, ``NFKMLXU2NetVariant``).
- **By name** — ``NFKMLXReferenceModels/registerAll()`` registers every model, then
  ``NFKMLXModelRegistry/backend(named:weightsURL:)`` (local) or
  ``NFKMLXHub/backend(named:repo:weightsPath:revision:cacheDirectoryURL:)`` (download) builds by its
  registered name.

```swift
// By name, after registerAll():
NFKMLXReferenceModels.registerAll()
let depth = try NFKMLXModelRegistry.backend(named: "depth-anything-v2-small", weightsURL: url)
```

Every backend runs `runInference(for:)` synchronously and multi-second — call it off the render thread,
or submit a job for progress and cancellation.

### Image → image

| Model | Name | Task |
| --- | --- | --- |
| ``NFKMLXRealESRGAN`` | `real-esrgan-x4` · `-anime` · `-x2` | ×4 / ×2 super-resolution |
| ``NFKMLXSwinIR`` | `swinir-x4` | transformer super-resolution |
| ``NFKMLXNAFNet`` | `nafnet` | denoise / deblur |
| ``NFKMLXZeroDCE`` | `zero-dce` | low-light enhancement |
| ``NFKMLXStyleTransfer`` | `fast-style-transfer` | one baked style per checkpoint |
| ``NFKMLXColorizer`` | `colorizer-eccv16` | grayscale → color |
| ``NFKMLXLaMa`` | `lama-inpaint` | mask-guided inpainting |
| ``NFKMLXStableDiffusionInpaint`` | `sd-inpaint` | latent-diffusion inpainting |
| ``NFKMLXSDUNet`` / ``NFKMLXSDAutoencoder`` | — | the shared Stable Diffusion networks |

Each writes its result under `NFKOutputImage`.

### Image → map

`NFKMLXDepthAnything` and `NFKMLXMarigold` emit a grayscale depth map; the segmenters
(`NFKMLXSegFormer`, `NFKMLXDeepLab`, `NFKMLXBiSeNet`) emit a grayscale class-label map under
`NFKOutputImage` — recover the class index as `round(gray · (classCount − 1))`.

| Model | Name | Task |
| --- | --- | --- |
| ``NFKMLXDepthAnything`` | `depth-anything-v2-small` · `-base` · `-large` | monocular depth |
| ``NFKMLXMarigold`` | `marigold-depth` | diffusion depth |
| ``NFKMLXSegFormer`` | `segformer-b0` | transformer segmentation |
| ``NFKMLXDeepLab`` | `deeplabv3` | CNN segmentation |
| ``NFKMLXBiSeNet`` | `bisenet` | real-time segmentation |

### Matting & segmentation

`NFKMLXU2Net`, `NFKMLXRVM`, and `NFKMLXMODNet` produce a straight foreground plus an alpha matte under
`NFKOutputMask`; `NFKMLXSAM` segments from a point prompt; `NFKMLXCodeFormer` restores faces.

| Model | Name | Task |
| --- | --- | --- |
| ``NFKMLXU2Net`` | `u2net` · `u2netp` | salient-object matting |
| ``NFKMLXRVM`` | `robust-video-matting` | recurrent video matting |
| ``NFKMLXMODNet`` | `modnet` | trimap-free portrait matting |
| ``NFKMLXSAM`` | `sam` | promptable segmentation |
| ``NFKMLXSAM2`` | — | SAM 2: Hiera encoder, mask decoder, and video memory (weight loading; masks stay on ``NFKMLXSAM``) |
| ``NFKMLXCodeFormer`` | `codeformer` | face restoration |

### Detection, pose & embeddings

`NFKMLXYOLO` returns `[NFKDetection]` under `NFKOutputDetections`; `NFKMLXPose` returns `[NFKKeypoint]`
under `NFKOutputPose`; `NFKMLXCLIP` returns an L2-normalized vector under `NFKOutputEmbedding`.

| Model | Name | Task |
| --- | --- | --- |
| ``NFKMLXYOLO`` | `yolo` | object detection |
| ``NFKMLXPose`` | `pose-simplebaseline` | top-down pose |
| ``NFKMLXCLIP`` | `clip-vit-b-32` | image + text embeddings |

### Video

| Model | Name | Task |
| --- | --- | --- |
| ``NFKMLXRIFE`` | `rife` | frame interpolation |
| ``NFKMLXRAFT`` | `raft` | optical flow |
| ``NFKMLXVideoSR`` | `video-super-resolution` | recurrent video super-resolution |
| ``NFKMLXSDUpscaler`` | `sd-x4-upscaler` | diffusion ×4 upscaler |

The recurrent models (`NFKMLXRVM`, `NFKMLXVideoSR`) thread a hidden state across frames through their
`*Net.forward` / `upscaleSequence` entry points.

### Audio

`NFKMLXDemucs`, `NFKMLXHTDemucs`, `NFKMLXConvTasNet`, and `NFKMLXDenoiser` read `NFKInputAudio` and return one or more
`NFKAudioAsset`s; `NFKMLXWhisper` returns text; `NFKMLXVAD` returns `[NFKAudioSegment]` under
`NFKOutputSegments`; `NFKMLXAudioTagger` returns `[NFKClassification]` under `NFKOutputClassifications`;
`NFKMLXTTS` turns text into a WAV `NFKAudioAsset`. ``NFKMLXMusicBackend`` generates music: a
description under `NFKInputPrompt` and lyrics under `NFKInputLyrics` become a stereo 44.1 kHz
`NFKAudioAsset`, built with ``NFKMLXMusic3`` from the downloaded MiniMax Music 3 release directory
(the weights are separately licensed; the download is ~27 GB).

| Model | Name | Task |
| --- | --- | --- |
| ``NFKMLXDemucs`` | `demucs` | 4-stem music separation |
| ``NFKMLXHTDemucs`` | `htdemucs` | 4-stem music separation (hybrid transformer) |
| ``NFKMLXConvTasNet`` | `conv-tasnet` | speech separation |
| ``NFKMLXDenoiser`` | `denoiser` | speech noise suppression |
| ``NFKMLXWhisper`` | `whisper-tiny` | speech → text |
| ``NFKMLXVAD`` | `vad-marblenet` | voice-activity detection |
| ``NFKMLXAudioTagger`` | `audio-tagger-panns` | audio tagging |
| ``NFKMLXTTS`` | — | text → speech |
| ``NFKMLXMusicBackend`` | `minimax-music3` | text + lyrics → music |

## Topics

### Related

- <doc:BringYourOwnBackends>
- <doc:WeightsAndConversion>
