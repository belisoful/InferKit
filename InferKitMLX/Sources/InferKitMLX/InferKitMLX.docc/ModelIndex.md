# Model index

Every shipped model with its entry class, network class, the configuration preset or variant that matches
the released weights, its registered name, and a construction line to copy.

## Overview

One row per implemented model; <doc:ModelGallery> describes what each one does. A `*Configuration` struct is Swift-only; the `@objc` `*Variant` enum or the
`directoryURL:` factory is the Objective-C knob. Every `backend(…weightsURL:)` factory also has a
`backend(…repo:weightsPath:revision:cacheDirectoryURL:)` download peer and an asynchronous
`…completionHandler:` peer, and `nil` weights build a randomly initialized network. Every registered name
also builds through ``NFKMLXModelRegistry/backend(named:weightsURL:)`` after
``NFKMLXReferenceModels/registerAll()``, or downloads through
``NFKMLXHub/backend(named:repo:weightsPath:revision:cacheDirectoryURL:)``.

After each table, the construction lines for its rows: `url` and `dir` stand for the local weights or the
release directory, `error` for an `NSError *`. A model marked *no public constructor yet* has its networks
at reference parity but a pipeline or loader that is still internal to the package.

The measured parity of every row is recorded in `Docs/model-parity.md` in the repository.

### Image to image


| Model | Entry class | Network | Configuration for the released weights | Registered name | Base backend |
| --- | --- | --- | --- | --- | --- |
| Real-ESRGAN | ``NFKMLXRealESRGAN`` | `NFKRealESRGANNet` · `NFKRealESRGANCompactNet` | ``NFKMLXRealESRGANVariant`` `.x4` (23 blocks) · `.anime` (6 blocks) · `.x2` · `.generalX4V3` (compact, 32 convolutions) · `.animeVideoV3` (compact, 16) | `real-esrgan-x4` · `real-esrgan-x4-anime` · `real-esrgan-x2` · `real-esrgan-general-x4v3` · `real-esrgan-anime-video-x4v3` | ``NFKMLXModuleBackend`` |
| SwinIR | ``NFKMLXSwinIR`` | `NFKMLXSwinIRNet` | ``NFKMLXSwinIRVariant`` `.classicalX2` / `.classicalX3` / `.classicalX4` / `.classicalX8` / `.lightweightSRX2` / `.lightweightSRX3` / `.lightweightSRX4` / `.realWorldX4Medium` / `.realWorldX4Large` (`NFKMLXSwinIRConfiguration.classicalSRx4` …; `.lightweightX2` is the small test geometry) | `swinir-x4` | ``NFKMLXModuleBackend`` |
| HAT | ``NFKMLXHAT`` | `NFKMLXHATNet` | ``NFKMLXHATVariant`` `.base` / `.large` / `.realWorld` (`NFKMLXHATConfiguration.base` — 6 groups of 6 blocks at 180 channels — `.large` — 12 groups) | `hat-x4` · `hat-l-x4` · `real-hat-gan-x4` | ``NFKMLXModuleBackend`` |
| NAFNet | ``NFKMLXNAFNet`` | ``NFKMLXNAFNetNet`` | ``NFKMLXNAFNetVariant`` `.sidd` / `.goPro` / `.reds` / `.siddWidth64` / `.goProWidth64` (`NFKMLXNAFNetConfiguration.sidd` …); `network(weightsURL:configuration:)` and `fineTune` with an ``NFKMLXNAFNetTrainingRecipe`` | `nafnet` | ``NFKMLXModuleBackend`` |
| Zero-DCE | ``NFKMLXZeroDCE`` | ``NFKMLXZeroDCENet`` | fixed geometry (seven 3×3 convolutions) | `zero-dce` | ``NFKMLXModuleBackend`` |
| Zero-DCE++ | ``NFKMLXZeroDCEPlus`` | ``NFKMLXZeroDCEPlusNet`` | fixed geometry (seven depthwise-separable convolutions, one shared curve, scale factor 12) | `zero-dce-plus` | ``NFKMLXModuleBackend`` |
| Fast style transfer | ``NFKMLXStyleTransfer`` | ``NFKMLXStyleTransferNet`` | fixed geometry; one style per checkpoint; `network(weightsURL:)` and `fineTune` with an ``NFKMLXStyleTransferObjective`` | `fast-style-transfer` | ``NFKMLXModuleBackend`` |
| AdaIN | ``NFKMLXAdaIN`` | ``NFKMLXAdaINNet`` | fixed geometry (VGG-19 through relu4_1 and its mirrored decoder); style image under `NFKInputControl`; `network(encoderURL:decoderURL:)` and `fineTune` with an ``NFKMLXAdaINObjective`` | `adain` | ``NFKMLXModuleBackend`` |
| Colorizer ECCV-16 | ``NFKMLXColorizer`` | ``NFKMLXColorizerNet`` | `NFKMLXColorizerConfiguration.eccv16` | `colorizer-eccv16` | ``NFKMLXModuleBackend`` |
| Colorizer SIGGRAPH-17 | ``NFKMLXSiggraphColorizer`` | ``NFKMLXSiggraphNet`` | fixed geometry (four-channel input, hints optional) | `colorizer-siggraph17` | ``NFKMLXModuleBackend`` |
| DDColor | ``NFKMLXDDColor`` | `NFKMLXDDColorNet` | ``NFKMLXDDColorVariant`` `.modelscope` / `.paper` / `.artistic` (`NFKMLXDDColorConfiguration.large`; `.tiny` is the small test geometry) | `ddcolor` · `ddcolor-paper` · `ddcolor-artistic` | ``NFKMLXModuleBackend`` |
| LaMa | ``NFKMLXLaMa`` | `NFKMLXLaMaNet` | `NFKMLXLaMaConfiguration()` = big-lama (64 channels, 3 downsamples, 18 blocks) | `lama-inpaint` | ``NFKMLXMattingBackend`` |
| CodeFormer | ``NFKMLXCodeFormer`` | `NFKMLXCodeFormerNet` | `NFKMLXCodeFormerConfiguration.base`; fidelity `w` per backend (`backend(fidelity:weightsURL:)`); ``NFKMLXPhotoFaceBackend`` for whole photographs | `codeformer` | ``NFKMLXModuleBackend`` |
| TAESD | ``NFKMLXTAESD`` | `NFKMLXTAESDNet` | fixed geometry (64-wide, 8× down/up) | `taesd` | ``NFKMLXTAESDBackend`` |
| SD inpainting | ``NFKMLXStableDiffusionInpaint`` | ``NFKMLXSDUNet`` + ``NFKMLXSDAutoencoder`` | `NFKMLXSDInpaintConfiguration.stableDiffusion15` (`NFKMLXSDUNetConfiguration.inpainting`) | `sd-inpaint` | ``NFKMLXDiffusionBackend`` |

```swift
// Real-ESRGAN
let backend = try NFKMLXRealESRGAN.backend(variant: .x4, weightsURL: url)
// .anime / .x2 · …VariantAnime / …VariantX2
// SwinIR
let backend = try NFKMLXSwinIR.backend(variant: .classicalX4, weightsURL: url)
// .classicalX3 / .classicalX8 / .lightweightSRX2 · …VariantClassicalX3 / …ClassicalX8 / …LightweightSRX2
// HAT
let backend = try NFKMLXHAT.backend(variant: .large, weightsURL: url)
// .base / .realWorld · …VariantBase / …VariantRealWorld
// NAFNet
let backend = try NFKMLXNAFNet.backend(variant: .sidd, weightsURL: url)
// .goPro / .reds · …VariantGoPro / …VariantReds
// Zero-DCE
let backend = try NFKMLXZeroDCE.backend(weightsURL: url)
// Zero-DCE++
let backend = try NFKMLXZeroDCEPlus.backend(weightsURL: url)
// Fast style transfer
let backend = try NFKMLXStyleTransfer.backend(weightsURL: url)
// AdaIN (arbitrary style; the style image travels under NFKInputControl)
let backend = try NFKMLXAdaIN.backend(encoderURL: vgg, decoderURL: decoder)
// Colorizer ECCV-16
let backend = try NFKMLXColorizer.backend(weightsURL: url)
// Colorizer SIGGRAPH-17
let backend = try NFKMLXSiggraphColorizer.backend(weightsURL: url)
// DDColor
let backend = try NFKMLXDDColor.backend(variant: .modelscope, weightsURL: url)
// .paper / .artistic · …VariantPaper / …VariantArtistic
// LaMa
let backend = try NFKMLXLaMa.backend(weightsURL: url)
// CodeFormer
let backend = try NFKMLXCodeFormer.backend(fidelity: 0.5, weightsURL: url)
// whole photograph: try NFKMLXCodeFormer.photoBackend(fidelity: 0.5, weightsURL: url, detectorWeightsURL: retinaURL) · [NFKMLXCodeFormer photoBackendWithFidelity:0.5 weightsURL:url detectorWeightsURL:retinaURL error:&error]
// TAESD
let backend = try NFKMLXTAESD.backend(weightsURL: url)
// SD inpainting
let backend = try NFKMLXStableDiffusionInpaint.backend(unetWeightsURL: unetURL, vaeWeightsURL: vaeURL, textContextURL: contextURL)
```

```objc
// Real-ESRGAN
[NFKMLXRealESRGAN backendWithVariant:NFKMLXRealESRGANVariantX4 weightsURL:url error:&error]
// SwinIR
[NFKMLXSwinIR backendWithVariant:NFKMLXSwinIRVariantClassicalX4 weightsURL:url error:&error]
// HAT
[NFKMLXHAT backendWithVariant:NFKMLXHATVariantLarge weightsURL:url error:&error]
// NAFNet
[NFKMLXNAFNet backendWithVariant:NFKMLXNAFNetVariantSidd weightsURL:url error:&error]
// Zero-DCE
[NFKMLXZeroDCE backendWithWeightsURL:url error:&error]
// Zero-DCE++
[NFKMLXZeroDCEPlus backendWithWeightsURL:url error:&error]
// Fast style transfer
[NFKMLXStyleTransfer backendWithWeightsURL:url error:&error]
// AdaIN
[NFKMLXAdaIN backendWithEncoderURL:vgg decoderURL:decoder error:&error]
// Colorizer ECCV-16
[NFKMLXColorizer backendWithWeightsURL:url error:&error]
// Colorizer SIGGRAPH-17
[NFKMLXSiggraphColorizer backendWithWeightsURL:url error:&error]
// DDColor
[NFKMLXDDColor backendWithVariant:NFKMLXDDColorVariantModelscope weightsURL:url error:&error]
// LaMa
[NFKMLXLaMa backendWithWeightsURL:url error:&error]
// CodeFormer
[NFKMLXCodeFormer backendWithFidelity:0.5 weightsURL:url error:&error]
// TAESD
[NFKMLXTAESD backendWithWeightsURL:url error:&error]
// SD inpainting
[NFKMLXStableDiffusionInpaint backendWithUNetWeightsURL:unetURL vaeWeightsURL:vaeURL textContextURL:contextURL error:&error]
```


### Image to map


| Model | Entry class | Network | Configuration for the released weights | Registered name | Base backend |
| --- | --- | --- | --- | --- | --- |
| Depth Anything V2 | ``NFKMLXDepthAnything`` | ``NFKMLXDepthAnythingNet`` (a metric head with `maxDepth`) | ``NFKMLXDepthVariant`` `.small` / `.base` / `.large` (`NFKMLXDepthConfiguration.small` …) | `depth-anything-v2-small` · `-base` · `-large` | ``NFKMLXModuleBackend`` |
| Depth Anything 3 | ``NFKMLXDepthAnything3``; ``NFKMLXDepth3Estimator`` for the camera and the ray map | `NFKMLXDepthAnything3Net` | ``NFKMLXDepth3Variant`` `.small` / `.base` / `.large` (`NFKMLXDepth3Configuration.small` …) | `depth-anything-3-small` · `-base` · `-large` | ``NFKMLXModuleBackend``; ``NFKMLXGenerationBackend`` for the estimator (depth, camera, and rays) |
| Marigold depth | ``NFKMLXMarigold`` | ``NFKMLXSDUNet`` + ``NFKMLXSDAutoencoder`` | `NFKMLXSDUNetConfiguration.marigold` | `marigold-depth` | ``NFKMLXDiffusionBackend`` |
| SegFormer | ``NFKMLXSegFormer`` | ``NFKMLXSegFormerNet`` | `NFKMLXSegFormerConfiguration.mitB0`; `network(weightsURL:classCount:)` for a custom head | `segformer-b0` | ``NFKMLXModuleBackend`` |
| DeepLabV3 | ``NFKMLXDeepLab`` | ``NFKMLXDeepLabNet`` over `NFKMLXResNetBackbone` | `NFKMLXDeepLabConfiguration.base` (`NFKMLXResNetConfiguration.deepLab`); `network(weightsURL:configuration:)` for a custom head | `deeplabv3` | ``NFKMLXModuleBackend`` |
| BiSeNet V1 | ``NFKMLXBiSeNet`` | ``NFKMLXBiSeNetNet`` | `NFKMLXBiSeNetConfiguration.base` (ResNet-18 context path); `network(weightsURL:configuration:)` for a custom head | `bisenet` | ``NFKMLXModuleBackend`` |
| BiSeNet V2 | ``NFKMLXBiSeNetV2`` | `NFKMLXBiSeNetV2Net` | fixed geometry (older pixel-shuffle head) | `bisenet-v2` | ``NFKMLXModuleBackend`` |

```swift
// Depth Anything V2
let backend = try NFKMLXDepthAnything.backend(variant: .small, weightsURL: url)
// .base / .large · …VariantBase / …VariantLarge
// Depth Anything 3 (DA3-SMALL)
let backend = try NFKMLXDepthAnything3.backend(variant: .small, weightsURL: url)
// camera and rays: NFKMLXDepth3Estimator.estimator(variant: .small, weightsURL: url)
// depth, camera, and rays as a backend (depth under NFKOutputImage, the camera and the ray map under NFKOutputStructured):
let estimatorBackend = try NFKMLXDepth3Estimator.estimator(variant: .small, weightsURL: url).inferenceBackend()
// Marigold depth
let backend = try NFKMLXMarigold.backend(unetWeightsURL: unetURL, vaeWeightsURL: vaeURL, textContextURL: contextURL)
// SegFormer
let backend = try NFKMLXSegFormer.backend(weightsURL: url)
// custom classes: try NFKMLXSegFormer.network(weightsURL: url, classCount: 4)
// DeepLabV3
let backend = try NFKMLXDeepLab.backend(weightsURL: url)
// custom classes: try NFKMLXDeepLab.network(weightsURL: url, configuration: NFKMLXDeepLabConfiguration(classCount: 4))
// BiSeNet V1
let backend = try NFKMLXBiSeNet.backend(weightsURL: url)
// custom classes: try NFKMLXBiSeNet.network(weightsURL: url, configuration: NFKMLXBiSeNetConfiguration(classCount: 4))
// BiSeNet V2
let backend = try NFKMLXBiSeNetV2.backend(weightsURL: url)
```

```objc
// Depth Anything V2
[NFKMLXDepthAnything backendWithVariant:NFKMLXDepthVariantSmall weightsURL:url error:&error]
// Depth Anything 3
[NFKMLXDepthAnything3 backendWithVariant:NFKMLXDepth3VariantSmall weightsURL:url error:&error]
// camera and rays: [NFKMLXDepth3Estimator estimatorWithVariant:NFKMLXDepth3VariantSmall weightsURL:url error:&error]
// depth, camera, and rays as a backend:
[[NFKMLXDepth3Estimator estimatorWithVariant:NFKMLXDepth3VariantSmall weightsURL:url error:&error] inferenceBackend]
// Marigold depth
[NFKMLXMarigold backendWithUNetWeightsURL:unetURL vaeWeightsURL:vaeURL textContextURL:contextURL error:&error]
// SegFormer
[NFKMLXSegFormer backendWithWeightsURL:url error:&error]
// DeepLabV3
[NFKMLXDeepLab backendWithWeightsURL:url error:&error]
// BiSeNet V1
[NFKMLXBiSeNet backendWithWeightsURL:url error:&error]
// BiSeNet V2
[NFKMLXBiSeNetV2 backendWithWeightsURL:url error:&error]
```


### Image to 3D

| Model | Entry class | Network | Configuration for the released weights | Registered name | Base backend |
| --- | --- | --- | --- | --- | --- |
| TripoSR | ``NFKMLXTripoSR`` | ``NFKMLXTripoSRNet`` | `NFKMLXTripoSRConfiguration.released`; `network(weightsURL:configuration:)` builds it | `triposr` | ``NFKMLXTripoSRBackend`` |
| Stable Fast 3D | ``NFKMLXStableFast3D`` | ``NFKMLXStableFast3DNet`` | `NFKMLXStableFast3DConfiguration.released`; `network(weightsURL:configuration:)` builds it | `stable-fast-3d` | ``NFKMLXStableFast3DBackend`` |

```swift
let backend = try NFKMLXTripoSR.backend(weightsURL: url)
let mesh = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: image])).mesh   // NFKMesh under NFKOutputMesh
```

```objc
[NFKMLXTripoSR backendWithWeightsURL:url error:&error]
// NFKMLXTripoSRParameterKey.resolution / .threshold / .foregroundRatio / .vertexColors set the grid, the surface, the framing, and the colors
```

```swift
let backend = try NFKMLXStableFast3D.backend(directoryURL: releaseDirectory)   // model.safetensors beside 160_tets.npz
let mesh = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: image])).mesh   // with mesh.material
```

```objc
[NFKMLXStableFast3D backendWithDirectoryURL:releaseDirectory error:&error]
// NFKMLXStableFast3DParameterKey.textureResolution / .threshold / .foregroundRatio / .textureFormat / .estimateIllumination
```

### Matting, segmentation, and faces


| Model | Entry class | Network | Configuration for the released weights | Registered name | Base backend |
| --- | --- | --- | --- | --- | --- |
| U²-Net | ``NFKMLXU2Net`` | `NFKMLXU2NetNet` | ``NFKMLXU2NetVariant`` `.full` / `.light` | `u2net` · `u2netp` | ``NFKMLXMattingBackend`` |
| IS-Net (DIS) | ``NFKMLXISNet`` | `NFKMLXISNetNet` | fixed geometry (stride-2 stem, six Residual U-block stages, six side maps) | `isnet` | ``NFKMLXMattingBackend`` |
| Robust Video Matting | ``NFKMLXRVM`` | `NFKMLXRVMNet` | ``NFKMLXRVMVariant`` `.mobileNetV3` / `.resNet50` (`NFKMLXRVMConfiguration.large` / `.resNet50`); `downsampleRatio` for the guided-filter path | `robust-video-matting` · `-resnet50` · `robust-video-matting-clip` · `robust-video-matting-resnet50-clip` | ``NFKMLXMattingBackend`` (single frame) / ``NFKMLXVideoBackend`` (`clipBackend(variant:output:weightsURL:)`) |
| MODNet | ``NFKMLXMODNet`` | `NFKMLXMODNetNet` | `NFKMLXMODNetConfiguration.base` | `modnet` | ``NFKMLXMattingBackend`` |
| BiRefNet | ``NFKMLXBiRefNet`` | `NFKMLXBiRefNetModel` (Swin-v1-L backbone + ASPPDeformable decoder) | released `ZhengPeng7/BiRefNet` (`model.safetensors`, MIT); resizes to 1024 | `birefnet` | ``NFKMLXMattingBackend`` |
| SAM | ``NFKMLXSAM`` | `NFKMLXSAMNet` | ``NFKMLXSAMVariant`` `.vitB` / `.vitL` / `.vitH` (`NFKMLXSAMConfiguration.vitB` …); `.compact` for tests | `sam` | ``NFKMLXMattingBackend`` (point under `NFKSAMPointKey`) |
| SAM 3 / SAM 3.1 | ``NFKMLXSAM3`` | ``NFKMLXSAM3ImageModel`` | `NFKMLXSAM3Configuration.base`, `NFKMLXSAM3TextConfiguration.base`, `NFKMLXSAM3DetectorConfiguration.base` | `sam3`; `backend(directoryURL:)` | ``NFKMLXGenerationBackend`` (text prompts only; box prompts and video are not ported) |
| SAM 2 / SAM 2.1 | ``NFKMLXSAM2`` | ``NFKMLXSAM2TrackerNet`` | `NFKMLXSAM2Variant.tiny` / `.small` / `.basePlus` / `.large`; `NFKMLXSAM2Release.sam2` / `.sam21` | `sam2` · `sam2-video` | ``NFKMLXMattingBackend`` (image) / ``NFKMLXGenerationBackend`` (video) |
| RetinaFace | ``NFKMLXRetinaFace`` | ``NFKMLXRetinaFaceNet`` | `NFKMLXRetinaFaceConfiguration()` = mobile0.25; `network(weightsURL:)` for a fine-tuned file | `retinaface-mobile025` | detection backend; `detector(weightsURL:)` for landmarks; per request: `NFKMLXDetectionParameterKey.confidenceThreshold` and `.overlapThreshold` |
| Face alignment | ``NFKMLXFaceAlignment`` | — | ``NFKMLXRetinaFaceDetector`` (default) or ``NFKMLXVisionFaceDetector`` | — | used by ``NFKMLXPhotoFaceBackend`` |

```swift
// U²-Net
let backend = try NFKMLXU2Net.backend(variant: .full, weightsURL: url)
// IS-Net (DIS)
let backend = try NFKMLXISNet.backend(weightsURL: url)
// .light · …VariantLight
// Robust Video Matting
let backend = try NFKMLXRVM.backend(weightsURL: url)
// clip (NFKInputVideo → NFKOutputVideo): .foreground / .composition write the other outputs
let clipBackend = try NFKMLXRVM.clipBackend(variant: .resNet50, output: .alpha, weightsURL: url)
// MODNet
let backend = try NFKMLXMODNet.backend(weightsURL: url)
// BiRefNet
let backend = try NFKMLXBiRefNet.backend(weightsURL: url)
// SAM
let backend = try NFKMLXSAM.backend(variant: .vitB, weightsURL: url)
// SAM 3 / SAM 3.1
let backend = try NFKMLXSAM3.backend(directoryURL: dir)
// then NFKInputImage and a concept under NFKInputPrompt; NFKOutputDetections and the union mask under NFKOutputMask;
// thresholds under NFKMLXSAM3ParameterKey.scoreThreshold / .maskThreshold
// the model alone:
let sam3 = try NFKMLXSAM3.makeImageModel(fromHuggingFace: configURL)
try NFKMLXSAM3.loadWeights(into: sam3, from: url)
let found = sam3.detect(image: plate, tokens: ids, valid: mask)
// fine-tune: let detector = try NFKMLXSAM3.network(weightsURL: url, configuration: config)
//            try NFKMLXSAM3.fineTune(detector, examples: myImages, trainable: .detector, steps: 300)

// SAM 2 / SAM 2.1
let backend = try NFKMLXSAM2.backend(variant: .tiny, release: .sam21, weightsURL: url)
// video: a click under NFKSAMPointKey or a box under NFKMLXSAM2ParameterKey.box on the first frame of
// NFKInputVideo; the mask clip under NFKOutputVideo
let videoBackend = try NFKMLXSAM2.videoBackend(variant: .tiny, release: .sam21, weightsURL: url)
// fine-tune: let net = try NFKMLXSAM2.network(weightsURL: url, variant: .tiny, release: .sam21)
//            try NFKMLXSAM2.fineTune(net, examples: myFrames, trainable: .maskDecoder, steps: 300)
// the tracker alone: let net = NFKMLXSAM2.makeTracker(variant: .tiny, release: .sam21)
//                    try NFKMLXSAM2.loadWeights(into: net, from: url)
//                    let session = NFKMLXSAM2TrackerSession(frameCount: frames.count)
// RetinaFace
let backend = try NFKMLXRetinaFace.backend(weightsURL: url)
// landmarks: let detector = try NFKMLXRetinaFace.detector(weightsURL: url) then detector.faces(in: image) · [NFKMLXRetinaFace detectorWithWeightsURL:url confidenceThreshold:0.8 suppressionThreshold:0.4 error:&error]
// Face alignment
let crop = try NFKMLXFaceAlignment.alignedCrop(from: image, face: face)
// detectors: NFKMLXRetinaFace.detector(weightsURL:confidenceThreshold:suppressionThreshold:) / NFKMLXVisionFaceDetector()
```

```objc
// U²-Net
[NFKMLXU2Net backendWithVariant:NFKMLXU2NetVariantFull weightsURL:url error:&error]
// IS-Net (DIS)
[NFKMLXISNet backendWithWeightsURL:url error:&error]
// Robust Video Matting
[NFKMLXRVM backendWithWeightsURL:url error:&error]
// clip
[NFKMLXRVM clipBackendWithVariant:NFKMLXRVMVariantResNet50 output:NFKMLXRVMClipOutputAlpha weightsURL:url error:&error]
// MODNet
[NFKMLXMODNet backendWithWeightsURL:url error:&error]
// BiRefNet
[NFKMLXBiRefNet backendWithWeightsURL:url error:&error]
// SAM
[NFKMLXSAM backendWithVariant:NFKMLXSAMVariantVitB weightsURL:url error:&error]
// SAM 3 / SAM 3.1
[NFKMLXSAM3 backendWithDirectoryURL:dir error:&error]
// SAM 2 / SAM 2.1
[NFKMLXSAM2 backendWithVariant:NFKMLXSAM2VariantTiny release:NFKMLXSAM2ReleaseSam21 weightsURL:url error:&error]
// video
[NFKMLXSAM2 videoBackendWithVariant:NFKMLXSAM2VariantTiny release:NFKMLXSAM2ReleaseSam21 weightsURL:url error:&error]
// RetinaFace
[NFKMLXRetinaFace backendWithWeightsURL:url error:&error]
```


### Detection and pose


| Model | Entry class | Network | Configuration for the released weights | Registered name | Base backend |
| --- | --- | --- | --- | --- | --- |
| YOLOv8 | ``NFKMLXYOLO`` | `NFKMLXYOLONet` | ``NFKMLXYOLOVariant`` `.nano` / `.small` / `.medium` / `.large` / `.extraLarge` (`NFKMLXYOLOConfiguration.base`, `.small` …) | `yolo` | ``NFKMLXYOLOBackend`` (`labels:`); per request: `NFKMLXDetectionParameterKey.confidenceThreshold` and `.overlapThreshold` |
| YOLOv9 · v10 · 11 · v12 · 26 | ``NFKMLXYOLOGenerations`` | `NFKMLXYOLOGenerationNet` | ``NFKMLXYOLORelease`` — every released size of each generation | the checkpoint stem: `yolov9t` … `yolo26x` | ``NFKMLXYOLOGenerationBackend`` (`labels:`); per request: `NFKMLXDetectionParameterKey.confidenceThreshold` and `.overlapThreshold` (v10 and 26: the confidence alone) |
| RT-DETR | ``NFKMLXRTDetr`` | ``NFKMLXRTDetrNet`` | ``NFKMLXRTDetrVariant`` `.r18vd` / `.r34vd` / `.r50vd` / `.r101vd` | `rtdetr` · `rtdetr-r18vd` · `-r34vd` · `-r101vd` | ``NFKMLXRTDetrBackend`` (`labels:`); per request: `NFKMLXDetectionParameterKey.confidenceThreshold` |
| RT-DETRv2 | ``NFKMLXRTDetr`` | ``NFKMLXRTDetrNet`` | ``NFKMLXRTDetrVariant`` `.v2R18VD` / `.v2R34VD` / `.v2R50VD` / `.v2R101VD` | `rtdetr-v2-r18vd` · `-r34vd` · `-r50vd` · `-r101vd` | ``NFKMLXRTDetrBackend`` (`labels:`); `decoderMethod` (``NFKMLXRTDetrSamplingMethod``) and `decoderOffsetScale` carry v2's sampling; per request: `NFKMLXDetectionParameterKey.confidenceThreshold` |
| RF-DETR | ``NFKMLXRFDetr`` | ``NFKMLXRFDetrNet`` | ``NFKMLXRFDetrVariant`` `.nano` / `.small` / `.medium` / `.base` / `.large` | `rf-detr` · `rf-detr-nano` · `-small` · `-medium` · `-large` | ``NFKMLXRFDetrBackend`` (`labels:`); Roboflow naming converted on device; per request: `NFKMLXDetectionParameterKey.confidenceThreshold` |
| RF-DETR segmentation | ``NFKMLXRFDetrSegmentation`` | ``NFKMLXRFDetrSegmentationNet`` | ``NFKMLXRFDetrSegmentationVariant`` `.nano` / `.small` / `.preview` / `.medium` / `.large` / `.extraLarge` / `.extraExtraLarge` | `rf-detr-seg` · `rf-detr-seg-small` · `-preview` · `-medium` · `-large` · `-xlarge` · `-xxlarge` | ``NFKMLXRFDetrSegmentationBackend`` (`labels:`); instances under `NFKOutputDetections`, their combined mask under `NFKOutputMask`; per request: `NFKMLXDetectionParameterKey.confidenceThreshold` |
| SimpleBaseline pose | ``NFKMLXPose`` | `NFKMLXPoseNet` over `NFKMLXResNetBackbone` | `NFKMLXPoseConfiguration.simpleBaseline` (ResNet-50, 256×192) | `pose-simplebaseline` | ``NFKMLXPoseBackend`` (`jointNames:`) |
| ViTPose | ``NFKMLXVitPose`` | `NFKMLXVitPoseNet` | ``NFKMLXVitPoseVariant`` `.baseSimple` / `.base` (ViT-B 256×192; ``NFKMLXVitPoseDecoder`` `.simple` / `.classic`) | `vitpose-base-simple` · `vitpose-base` | ``NFKMLXVitPoseBackend`` (`jointNames:`); `backend(directoryURL:jointNames:)` reads a release's own `config.json` |
| Table Transformer | ``NFKMLXTableTransformer`` | `NFKMLXTableTransformerNet` | read from the release's `config.json` (geometry + `id2label`) | `backend(directoryURL:)`; retargeted: `network(directoryURL:labels:)`, `fineTune`, `save(_:toDirectoryURL:release:)` | ``NFKMLXTableTransformerBackend``; table-structure recognition (Table Transformer, Microsoft), no NMS; per request: `NFKMLXDetectionParameterKey.confidenceThreshold` |

```swift
// YOLOv8
let backend = try NFKMLXYOLO.backend(variant: .nano, weightsURL: url, labels: nil)
// fine-tuned: NFKMLXYOLO.network(variant:classCount:weightsURL:), then fineTune
// YOLOv9 / v10 / 11 / v12 / 26
let backend = try NFKMLXYOLOGenerations.backend(release: .v26Nano, weightsURL: url, labels: nil)
// fine-tuned: NFKMLXYOLOGenerations.network(release:classCount:weightsURL:), then fineTune
// .small / .medium / .large / .extraLarge · …VariantSmall … …VariantExtraLarge
// RT-DETR
let backend = try NFKMLXRTDetr.backend(weightsURL: url, labels: nil)
// RT-DETRv2 · .v2R18VD / .v2R34VD / .v2R50VD / .v2R101VD
let backend = try NFKMLXRTDetr.backend(variant: .v2R50VD, weightsURL: url, labels: nil)
// fine-tuned: NFKMLXRTDetr.network(variant:classCount:weightsURL:), then fineTune(_:variant:…)
// RF-DETR
let backend = try NFKMLXRFDetr.backend(weightsURL: url, labels: nil)
// RF-DETR segmentation · .small / .preview / .medium / .large / .extraLarge / .extraExtraLarge
let backend = try NFKMLXRFDetrSegmentation.backend(variant: .nano, weightsURL: url, labels: nil)
// SimpleBaseline pose
let backend = try NFKMLXPose.backend(weightsURL: url, jointNames: nil)
// ViTPose · .baseSimple / .base
let backend = try NFKMLXVitPose.backend(variant: .base, weightsURL: url, jointNames: nil)
// Table Transformer (table structure) · reads the release's config.json
let backend = try NFKMLXTableTransformer.backend(directoryURL: dir)
```

```objc
// YOLOv8
[NFKMLXYOLO backendWithVariant:NFKMLXYOLOVariantNano weightsURL:url labels:nil error:&error]
// YOLOv9 / v10 / 11 / v12 / 26
[NFKMLXYOLOGenerations backendWithRelease:NFKMLXYOLOReleaseV26Nano weightsURL:url labels:nil error:&error]
// RT-DETR
[NFKMLXRTDetr backendWithWeightsURL:url labels:nil error:&error]
// RT-DETRv2 · …VariantV2R18VD … …VariantV2R101VD
[NFKMLXRTDetr backendWithVariant:NFKMLXRTDetrVariantV2R50VD weightsURL:url labels:nil error:&error]
// RF-DETR
[NFKMLXRFDetr backendWithWeightsURL:url labels:nil error:&error]
// RF-DETR segmentation
[NFKMLXRFDetrSegmentation backendWithVariant:NFKMLXRFDetrSegmentationVariantNano weightsURL:url labels:nil error:&error]
// SimpleBaseline pose
[NFKMLXPose backendWithWeightsURL:url jointNames:nil error:&error]
// ViTPose · …VariantBaseSimple / …VariantBase
[NFKMLXVitPose backendWithVariant:NFKMLXVitPoseVariantBase weightsURL:url jointNames:nil error:&error]
// Table Transformer (table structure) · reads the release's config.json
[NFKMLXTableTransformer backendWithDirectoryURL:dir error:&error]
```


### Embeddings, reranking, and vision-language


| Model | Entry class | Network | Configuration for the released weights | Registered name / factory | Base backend |
| --- | --- | --- | --- | --- | --- |
| CLIP (ViT) | ``NFKMLXCLIP`` | ``NFKMLXCLIPNet`` | ``NFKMLXCLIPVariant`` `.vitB32` / `.vitB16` / `.vitL14` / `.vitL14At336` | `clip-vit-b-32` · `-b-16` · `-l-14` · `-l-14-336` | ``NFKMLXCLIPBackend`` |
| CLIP probe | ``NFKMLXCLIPProbe`` | linear head over frozen CLIP | trained on device (`trainProbe`) | `clip-probe`; `probeBackend(variant:weightsURL:probeURL:labels:)` | ``NFKMLXCLIPProbeBackend`` |
| SigLIP 2 | ``NFKMLXSigLIP2`` | `NFKMLXSigLIP2Net` (`NFKSigLIP2VisionNet`, `NFKSigLIP2TextNet`) | ``NFKMLXSigLIP2Variant`` (every fixed-resolution release: base / large / so400m / giant-opt at their patch sizes and resolutions) | `siglip2-base-patch16-224` and each release's own name | ``NFKMLXSigLIP2Backend`` |
| Qwen3-Embedding-0.6B | ``NFKMLXQwen3Embedding`` | ``NFKMLXLanguageNet`` | ``NFKMLXLanguageConfiguration`` read from `config.json`; ``NFKMLXTextEmbedderConfiguration`` (`.lastToken`, appended `<\|endoftext\|>`, Matryoshka `dimensions`) | `backend(directoryURL:)` | ``NFKMLXTextEmbeddingBackend``; per request: `NFKParameterEmbeddingDimensions` (32–1024) |
| EmbeddingGemma-300M | ``NFKMLXEmbeddingGemma`` | ``NFKMLXGemma3EncoderNet`` | `NFKMLXGemma3EncoderConfiguration.embeddingGemma300M` + the `2_Dense` / `3_Dense` projections | `backend(directoryURL:)` (`embeddinggemma-300m`) | ``NFKMLXTextEmbeddingBackend``; per request: `NFKParameterEmbeddingDimensions` (768, 512, 256, 128) |
| gte-reranker-modernbert-base | ``NFKMLXModernBERTReranker`` | ``NFKMLXModernBertRerankerNet`` | `NFKMLXModernBertConfiguration.gteReranker` | `gte-reranker-modernbert-base`; `backend(directoryURL:)`, `reranker(directoryURL:)` | ``NFKMLXGenerationBackend`` (``NFKMLXRerankKey``); `scores(query:documents:)` on the scorer |
| Laya (typed decisions) | ``NFKMLXLaya`` | ``NFKMLXLayaNet`` | `NFKMLXLayaConfiguration.large` / `.multilingual`, or the variant directory | Swift API | ``NFKMLXLayaBackend`` (`backendWithDirectoryURL:error:`, or `backendWithVariant:revision:cacheDirectoryURL:error:` to download) |
| open-jev-deberta (typed decisions) | ``NFKMLXOpenJevDeBERTa`` | ``NFKMLXOpenJevDeBERTaNet`` (``NFKMLXDeBERTaV2Net`` + scoring head) | `NFKMLXOpenJevDeBERTaConfiguration.v3Large`, or the release directory | Swift API | ``NFKMLXDecisionBackend`` (`backendWithDirectoryURL:error:`, or `backendWithRevision:cacheDirectoryURL:error:` to download) |
| Open-Jev 2B / 9B / 27B (typed decisions) | ``NFKMLXOpenJev`` | ``NFKMLXOpenJevNet`` (``NFKMLXHybridLanguageNet`` + LoRA + scalar head) | read from the release's `checkpoint` directory and its base | Swift API | ``NFKMLXDecisionBackend`` (`backendWithVariant:revision:cacheDirectoryURL:error:`) |
| pplx-decider v1.2 Flash / 27B (typed decisions) | ``NFKMLXPplxDecider`` | ``NFKMLXPplxDeciderNet`` (``NFKMLXHybridLanguageNet`` with bidirectional full attention + readout) | read from the release directory | `pplx-decider-flash-v1.2` · `pplx-decider-v1.2` | ``NFKMLXDecisionBackend`` (`backendWithDirectoryURL:error:`, or `backendWithVariant:revision:cacheDirectoryURL:quantizationBits:error:` to download) |
| chronos-bolt-base | ``NFKMLXChronos`` | ``NFKMLXChronosNet`` | `NFKMLXChronosConfiguration()` | `chronos-bolt-base`; `backend(weightsURL:)` | ``NFKMLXGenerationBackend`` (``NFKMLXForecastKey``); `forecast(context:horizon:)` on the forecaster |
| timesfm-2.5-200m | ``NFKMLXTimesFM`` | ``NFKMLXTimesFMNet`` | `NFKMLXTimesFMConfiguration.v2_5`; ``NFKMLXTimesFMForecastOptions`` | `timesfm-2.5-200m`; `backend(directoryURL:)`, `timesFM(directoryURL:)` | ``NFKMLXGenerationBackend`` (``NFKMLXForecastKey``); `forecast(context:horizon:options:)` on the forecaster |
| Qwen3-VL-Embedding-2B (text + image) | ``NFKMLXQwen3VLEmbedder`` | ``NFKMLXQwen3VLVisionNet`` + ``NFKMLXLanguageNet`` | read from the release's `config.json` and `preprocessor_config.json` | `embedder(directoryURL:)` / `backend(directoryURL:)` | ``NFKMLXTextEmbeddingBackend`` (text path) |
| Qwen3-VL-Reranker-2B (text + image) | ``NFKMLXQwen3VLReranker`` | ``NFKMLXQwen3VLVisionNet`` + ``NFKMLXLanguageNet`` | read from the release's `config.json` and `1_LogitScore/config.json` | `qwen3-vl-reranker-2b`; `backend(directoryURL:)`, `reranker(directoryURL:)` | ``NFKMLXGenerationBackend`` (``NFKMLXRerankKey``); `scores(query:documents:)` on the scorer |
| SmolVLM2-500M | ``NFKMLXSmolVLM`` | ``NFKMLXSmolVLMNet`` (``NFKMLXSigLIPNet`` + ``NFKMLXSmolVLMConnector`` + ``NFKMLXLanguageNet``) | `NFKMLXSigLIPConfiguration.smolVLM`, `NFKMLXLanguageConfiguration.smolVLM2Decoder` | `smolvlm2-500m`; `backend(directoryURL:)`, `load(directoryURL:)` | ``NFKMLXGenerationBackend``; `answer(image:question:)` on the model |
| Gemma 3 4B (image + text) | ``NFKMLXGemma3`` | ``NFKMLXGemma3Model`` (``NFKMLXGemma3VisionNet`` + ``NFKMLXGemma3MultimodalProjector`` + ``NFKMLXGemma3Net``) | read from the release's `config.json` (`NFKMLXSigLIPConfiguration.gemma3`, `NFKMLXGemma3Configuration.gemma3_4B`) | `load(directoryURL:)` / `backend(directoryURL:)` | ``NFKMLXGemma3Backend`` |
| Gemma 3n E2B / E4B (image + audio + text) | ``NFKMLXGemma3n`` | ``NFKMLXGemma3nModel`` (``NFKMLXGemma3nVisionNet`` + ``NFKMLXGemma3nAudioNet`` + ``NFKGemma3nMultimodalEmbedder`` + ``NFKMLXGemma3nNet``) | read from the release's `config.json` (``NFKMLXGemma3nConfiguration``, ``NFKMLXGemma3nAudioConfiguration``, ``NFKMLXGemma3nTokens``) | `load(directoryURL:)` / `backend(directoryURL:)` | ``NFKMLXGemma3nBackend`` |
| Qwen3-VL-2B vision tower | ``NFKMLXQwen3VL`` | ``NFKMLXQwen3VLVisionNet`` | `NFKMLXQwen3VLVisionConfiguration.qwen3VL2B` | `qwen3-vl-2b` (the whole VLM); `backend(directoryURL:)` | ``NFKMLXGenerationBackend`` (the whole VLM); the tower is Swift API |
| Pixtral 12B (vision + text) | ``NFKMLXPixtral`` | ``NFKMLXPixtralVisionNet`` + ``NFKMLXPixtralConnector`` + ``NFKMLXLanguageNet`` | read from the release's `config.json` | `pixtral-12b`; `backend(directoryURL:)` | ``NFKMLXGenerationBackend``; `answer(image:question:maxTokens:)` on the model |
| Florence-2-base / -large (unified vision) | ``NFKMLXFlorence2`` | ``NFKMLXFlorence2Net`` (``NFKMLXFlorence2VisionNet`` + ``NFKMLXFlorence2Projector`` + ``NFKMLXSeq2SeqNet``) | read from the release's `config.json` | `backend(directoryURL:)`; adapted: `network(directoryURL:)`, `fineTune`, `save(_:toDirectoryURL:release:)` | ``NFKMLXFlorence2Backend`` |
| TrOCR (handwriting reader) | ``NFKMLXTrOCR`` | ``NFKMLXTrOCRNet`` (``NFKMLXTrOCRVisionNet`` + ``NFKMLXSeq2SeqNet`` decoder-only) | read from the release's `config.json` (every small, base, and large release) | `backend(directoryURL:)`; fine-tuned: `network(directoryURL:)`, `fineTune`, `save(_:toDirectoryURL:release:)` | ``NFKMLXTrOCRBackend``; per request: `NFKParameterMaxTokens` (64) |
| Sa2VA (segmentation VLM) | ``NFKMLXSa2VA`` | ``NFKMLXSa2VANet`` (InternVL), ``NFKMLXSa2VAQwenNet`` (Qwen-VL), ``NFKMLXSa2VALLaVANet`` (LLaVA) | read from the release's `config.json` | `backend(directoryURL:)`; fine-tuned: `network(directoryURL:)`, `fineTune`, `save(_:toDirectoryURL:release:)` | ``NFKMLXSa2VABackend``; per request: `NFKParameterMaxTokens` (256) |
| Phi-4-multimodal (image + speech + text) | ``NFKMLXPhi4MM`` | ``NFKMLXPhi4MMModel`` (``NFKMLXLanguageNet`` with a mixture of LoRAs + ``NFKMLXPhi4MMImageNet`` + ``NFKMLXPhi4MMAudioNet``) | read from the release's `config.json` | `backend(directoryURL:)`; `register()` registers `phi-4-multimodal` | ``NFKMLXPhi4MMBackend`` |
| Gemma 4 vision / audio / fusion | ``NFKMLXGemma4ConditionalGeneration`` | ``NFKMLXGemma4VisionNet``, ``NFKMLXGemma4AudioNet``, ``NFKMLXGemma4MultimodalEmbedder``, ``NFKMLXGemmaNet`` | ``NFKMLXGemma4VisionConfiguration`` / ``NFKMLXGemma4AudioConfiguration`` read from the release (`useClippedLinears` on) | `gemma4-multimodal`; `NFKMLXGemmaLanguage.gemmaMultimodalBackend(directoryURL:)` | ``NFKMLXGenerationBackend`` (``NFKMLXGemma4ImageProcessor``, ``NFKMLXGemma4AudioFeatureExtractor``) |

```swift
// CLIP ViT-B/32
let backend = try NFKMLXCLIP.backend(weightsURL: url)
// CLIP probe (a saved probe over CLIP; registered as clip-probe)
let backend = try NFKMLXCLIP.probeBackend(variant: .vitB32, weightsURL: url, probeURL: probeURL, labels: nil)
// training: let net = try NFKMLXCLIP.network(weightsURL: url), then trainProbe and NFKMLXWeights.save;
// probeBackend(net:probe:labels:) serves the probe in memory
// SigLIP 2
let backend = try NFKMLXSigLIP2.backend(weightsURL: url)
// probe: model(variant:weightsURL:), imageEmbeddings(for:), NFKMLXEmbeddingProbe.train,
// then probeBackend(probeURL:labels:) · probeBackendWithProbeURL:labels:error:
// Qwen3-Embedding-0.6B
let backend = try NFKMLXQwen3Embedding.backend(directoryURL: dir)
// Matryoshka: backend(directoryURL: dir, dimensions: 256) · backendWithDirectoryURL:dir outputDimensions:256 error:&error
// adapter: embeddings(for:), fineTune(adapter:queries:documents:steps:), then loadAdapter(from:) · loadAdapterFromURL:error:
// EmbeddingGemma-300M
let backend = try NFKMLXEmbeddingGemma.backend(directoryURL: dir)
// backend(directoryURL: dir, dimensions: 256) · backendWithDirectoryURL:dir outputDimensions:256 error:&error
// adapter: as Qwen3-Embedding · loadAdapterFromURL:error:
// gte-reranker-modernbert-base
let backend = try NFKMLXModernBERTReranker.backend(directoryURL: dir)
// then the query under NFKInputPrompt and the documents under NFKInputDocuments;
// NFKMLXRerankKey.scores and .rankedIndices under NFKOutputStructured
// the scorer: reranker(directoryURL:), then scores(query:documents:) · scoresForQuery:documents:
// Laya (typed decisions)
let laya = try NFKMLXLaya.laya(directoryURL: dir)
// or download a variant first: NFKMLXLaya.laya(variant: .typedDecisions, revision: NFKMLXLaya.measuredRevision, cacheDirectoryURL: nil)
// then decide(state:questions:); fine-tune: fineTune(examples:steps:), reloaded by laya(directoryURL:weightsURL:)
// open-jev-deberta-v3-large (typed decisions)
let deberta = try NFKMLXOpenJevDeBERTa.openJev(directoryURL: dir)   // or openJev(revision:cacheDirectoryURL:)
// then decide(state:questions:) in order; fine-tune: fineTune(examples:steps:), reloaded by openJev(directoryURL:weightsURL:)
// Open-Jev-2B / -9B (typed decisions)
let openJev = try NFKMLXOpenJev.openJev(variant: .twoB, revision: nil, cacheDirectoryURL: nil)
// or openJev(checkpointDirectoryURL:baseDirectoryURL:); fine-tune: fineTune(examples:steps:) then save(to:)
// pplx-decider v1.2 (typed decisions in one pass)
let decider = try NFKMLXPplxDecider.decider(directoryURL: dir)   // the 27B: decider(directoryURL:quantizationBits: 4)
// fine-tune: fineTune(examples:steps:) then saveAdapter(to:), reloaded by decider(directoryURL:adapterDirectoryURL:precision:)
// chronos-bolt-base (time-series forecasting)
let backend = try NFKMLXChronos.backend(weightsURL: url)
// then the series under NFKMLXForecastKey.series and the steps under NFKMLXForecastKey.horizon;
// the median, the quantile rows, and their levels under NFKOutputStructured
// the forecaster: chronos(weightsURL:), then forecast(context:horizon:) · medianForecastForContext:horizon:
// timesfm-2.5-200m (time-series forecasting; the release directory)
let backend = try NFKMLXTimesFM.backend(directoryURL: dir)
// then NFKMLXForecastKey.series and .horizon as for Chronos, the forecasting flags under NFKMLXTimesFMParameterKey
// the forecaster: timesFM(directoryURL:), then forecast(context:horizon:options:) · forecastForContext:horizon:options:error:
// fine-tune: fineTune(_:windows:steps:) then save(_:toDirectoryURL:)
// Qwen3-VL-Embedding-2B (text + image)
let embedder = try NFKMLXQwen3VLEmbedder.embedder(directoryURL: dir)
// then embedding(forText:) · embeddingForText:, embedding(forImage:text:instruction:) · embeddingForImage:text:instruction:
// Qwen3-VL-Reranker-2B (text + image)
let backend = try NFKMLXQwen3VLReranker.backend(directoryURL: dir)
// then the query under NFKInputPrompt and / or NFKInputImage and the documents (text or images) under NFKInputDocuments;
// NFKMLXRerankKey.scores and .rankedIndices under NFKOutputStructured; the task under NFKMLXRerankKey.instruction
// the scorer: reranker(directoryURL:), then scores(query:documents:) · scoresForQuery:documents:
// SmolVLM2-500M
let backend = try NFKMLXSmolVLM.backend(directoryURL: dir)
// then NFKInputImage and a question under NFKInputPrompt; NFKOutputText carries the answer
// the model: load(directoryURL:), then answer(image:question:) · answerForImage:question:
// Gemma 3 4B (image + text)
let gemma = try NFKMLXGemma3.load(directoryURL: dir)
// then answer(image:question:) · answerForImage:question:error:; or backend(directoryURL:) with NFKInputImage beside NFKInputMessages
// Gemma 3n E2B / E4B (image + audio + text)
let gemma3n = try NFKMLXGemma3n.load(directoryURL: dir)
// then answer(image:question:) · answerForImage:question:error:; or backend(directoryURL:) with NFKInputImage / NFKInputAudio beside NFKInputMessages
// Qwen3-VL-2B (the whole VLM)
let backend = try NFKMLXQwen3VL.backend(directoryURL: dir)
// then NFKInputImage and a question under NFKInputPrompt; NFKOutputText carries the answer
// the vision tower alone (Swift only, MLXArray in and out):
let vision = try NFKMLXQwen3VL.visionNet(directoryURL: dir)
// Pixtral 12B (vision + text)
let backend = try NFKMLXPixtral.backend(directoryURL: dir)
// then NFKInputImage and a question under NFKInputPrompt; NFKOutputText carries the answer
// the model: model(directoryURL:), then answer(image:question:maxTokens:) · answerForImage:question:maxTokens:

// Florence-2-base / -large (unified vision)
let florence = try NFKMLXFlorence2.backend(directoryURL: dir)
// then a task token (<OD>, <CAPTION>, …) under NFKInputPrompt beside NFKInputImage;
// NFKOutputText, plus NFKOutputDetections for the localization tasks
// TrOCR (handwriting reader)
let trocr = try NFKMLXTrOCR.backend(directoryURL: dir)
// then NFKInputImage; NFKOutputText carries the transcription
// Sa2VA-4B (segmentation VLM)
let sa2va = try NFKMLXSa2VA.backend(directoryURL: dir)
// then a referring prompt under NFKInputPrompt beside NFKInputImage;
// NFKOutputText carries the answer, NFKOutputMask the mask when it emits [SEG]
// Phi-4-multimodal (image + speech + text)
let phi = try NFKMLXPhi4MM.backend(directoryURL: dir)
// then NFKInputMessages or NFKInputPrompt, with pictures under NFKInputImage / NFKInputImages and clips
// under NFKInputAudio / NFKInputAudios; NFKOutputText carries the answer
// Gemma 4 vision / audio / fusion
let backend = try NFKMLXGemmaLanguage.gemmaMultimodalBackend(directoryURL: dir)
// then NFKInputPrompt or NFKInputMessages, with a picture under NFKInputImage and a clip under NFKInputAudio;
// NFKOutputText carries the answer; gemmaMultimodalBackend(directoryURL:residency:) holds the routed experts as residency says
// the chain alone:
let chain = NFKMLXGemma4ConditionalGeneration(decoder: decoder, visionTower: vision, visionEmbedder: visionEmbedder)
// then generate(promptTokens:image:waveform:maxTokens:)
```

```objc
// CLIP ViT-B/32
[NFKMLXCLIP backendWithWeightsURL:url error:&error]
// CLIP probe
[NFKMLXCLIP probeBackendWithVariant:NFKMLXCLIPVariantVitB32 weightsURL:url probeURL:probeURL labels:nil error:&error]
// SigLIP 2
[NFKMLXSigLIP2 backendWithWeightsURL:url error:&error]
// Qwen3-Embedding-0.6B
[NFKMLXQwen3Embedding backendWithDirectoryURL:dir error:&error]
// EmbeddingGemma-300M
[NFKMLXEmbeddingGemma backendWithDirectoryURL:dir error:&error]
// gte-reranker-modernbert-base
[NFKMLXModernBERTReranker backendWithDirectoryURL:dir error:&error]
// the scorer
[NFKMLXModernBERTReranker rerankerWithDirectoryURL:dir error:&error]
// Laya (typed decisions)
[NFKMLXLaya layaWithDirectoryURL:dir error:&error]
// open-jev-deberta-v3-large (typed decisions)
[NFKMLXOpenJevDeBERTa openJevWithDirectoryURL:dir error:&error]
// Open-Jev-2B / -9B (typed decisions)
[NFKMLXOpenJev openJevWithVariant:NFKMLXOpenJevVariantTwoB revision:nil cacheDirectoryURL:nil error:&error]
// pplx-decider v1.2 (typed decisions in one pass)
[NFKMLXPplxDecider deciderWithDirectoryURL:dir error:&error]
// Qwen3-VL-Embedding-2B (text + image)
[NFKMLXQwen3VLEmbedder embedderWithDirectoryURL:dir error:&error]
// Qwen3-VL-Reranker-2B (text + image)
[NFKMLXQwen3VLReranker backendWithDirectoryURL:dir error:&error]
// the scorer
[NFKMLXQwen3VLReranker rerankerWithDirectoryURL:dir error:&error]
// SmolVLM2-500M
[NFKMLXSmolVLM backendWithDirectoryURL:dir error:&error]
// the model
[NFKMLXSmolVLM smolVLMWithDirectoryURL:dir error:&error]
// Gemma 3 4B (image + text)
[NFKMLXGemma3 gemma3WithDirectoryURL:dir error:&error]
// Gemma 3n E2B / E4B (image + audio + text)
[NFKMLXGemma3n gemma3nWithDirectoryURL:dir error:&error]
// chronos-bolt-base (time-series forecasting)
[NFKMLXChronos backendWithWeightsURL:url error:&error]
// the forecaster
[NFKMLXChronos chronosWithWeightsURL:url error:&error]
// timesfm-2.5-200m (time-series forecasting)
[NFKMLXTimesFM backendWithDirectoryURL:dir error:&error]
// the forecaster
[NFKMLXTimesFM timesFMWithDirectoryURL:dir error:&error]
// Qwen3-VL-2B (the whole VLM)
[NFKMLXQwen3VL backendWithDirectoryURL:dir error:&error]
// Pixtral 12B (vision + text)
[NFKMLXPixtral backendWithDirectoryURL:dir error:&error]
// the model
[NFKMLXPixtral modelWithDirectoryURL:dir error:&error]
// Florence-2 (unified vision)
[NFKMLXFlorence2 backendWithDirectoryURL:dir error:&error]
// TrOCR (handwriting reader)
[NFKMLXTrOCR backendWithDirectoryURL:dir error:&error]
// Sa2VA (segmentation VLM)
[NFKMLXSa2VA backendWithDirectoryURL:dir error:&error]
// Phi-4-multimodal (image + speech + text)
[NFKMLXPhi4MM backendWithDirectoryURL:dir error:&error]
// Gemma 4 vision / audio / fusion
[NFKMLXGemmaLanguage gemmaMultimodalBackendWithDirectoryURL:dir error:&error]
```


### Language models


| Model | Entry class | Network | Configuration for the released weights | Factory | Base backend |
| --- | --- | --- | --- | --- | --- |
| Qwen3 (dense), Qwen2, Llama | ``NFKMLXLanguage`` | ``NFKMLXLanguageNet`` | ``NFKMLXLanguageConfiguration`` from `config.json` (presets `.qwen3_0_6B`, `.qwen3_1_7B`, `.qwen3_4B`, `.qwen3_8B`, `.qwen3_14B`, `.qwen3_32B`); ``NFKMLXGenerationOptions`` per request; `network(weightsURL:configuration:)` for a fine-tuned file | `let backend = try NFKMLXLanguage.backend(directoryURL: dir)`<br>`[NFKMLXLanguage backendWithDirectoryURL:dir error:&error]`<br>speculative: `backend(directoryURL: dir, draftDirectoryURL: draftDir)` · `backendWithDirectoryURL:dir draftDirectoryURL:draftDir error:&error` | ``NFKMLXLanguageBackend`` |
| Qwen3-MoE, Qwen2-MoE, Mixtral, gpt-oss | ``NFKMLXLanguage`` | ``NFKMLXLanguageNet`` + `NFKLMMixtureFeedForward` (+ `NFKLMFusedSwitchGLU` for gpt-oss) | the same reader (`qwen3_moe`, `qwen2_moe`, `mixtral`, `gpt_oss` model types); `.tinyMixture` for tests | `let backend = try NFKMLXLanguage.backend(directoryURL: dir)`<br>`[NFKMLXLanguage backendWithDirectoryURL:dir error:&error]` | ``NFKMLXLanguageBackend`` |
| Dense GGUF (`llama` / `qwen2` / `qwen3`) | ``NFKMLXLanguage`` | ``NFKMLXLanguageNet`` | `configuration(fromGGUF:)` from the file's metadata | `let backend = try NFKMLXLanguage.backend(ggufURL: url)`<br>`[NFKMLXLanguage backendWithGGUFURL:url error:&error]` | ``NFKMLXLanguageBackend`` |
| Qwen3.8-Flash-Next (Qwen4-Exp) | ``NFKMLXQwen4Exp`` | ``NFKMLXQwen4ExpNet`` | ``NFKMLXQwen4ExpConfiguration`` from `config.json` (`.qwen3_8FlashNext`, `.tiny` presets) | `let backend = try NFKMLXQwen4Exp.backend(directoryURL: dir)`<br>`[NFKMLXQwen4Exp backendWithDirectoryURL:dir error:&error]`<br>the decoder alone: `network(directoryURL:precision:residency:)`; `qwen4-exp` | ``NFKMLXDecoderBackend`` (prefill-only; renders the release's chat template) |
| Qwen3.5 / 3.6 / 3.8 | ``NFKMLXHybridLanguage`` | ``NFKMLXHybridLanguageNet`` | ``NFKMLXHybridConfiguration`` from `config.json` (`.qwen3_8_27B` preset); `network(weightsURL:configuration:)` for a fine-tuned file | `let backend = try NFKMLXHybridLanguage.backend(directoryURL: dir)`<br>`[NFKMLXHybridLanguage backendWithDirectoryURL:dir error:&error]`<br>the decoder alone: `network(directoryURL:precision:)`, or `makeNet` and `loadWeights(into:fromDirectory:precision:)`; `qwen3.5` | ``NFKMLXDecoderBackend`` (prefill-only; renders the release's chat template) |
| Gemma 3 270M / 1B / 4B | ``NFKMLXGemma3`` | ``NFKMLXGemma3Net`` | ``NFKMLXGemma3Configuration`` from `config.json` (`.gemma3_270M`, `.gemma3_1B`, `.gemma3_4B`; `gemma3_text` or the multimodal `gemma3`); `NFKMLXGemma3Language.network(weightsURL:configuration:)` for a fine-tuned file | `let backend = try NFKMLXGemma3.backend(directoryURL: dir)`<br>`[NFKMLXGemma3 backendWithDirectoryURL:dir error:&error]`<br>also reached through `NFKMLXGemmaLanguage.backend(directoryURL:)`, which dispatches on the model type | ``NFKMLXGemma3Backend`` (hybrid key-value cache, streaming, the release's chat template) |
| Gemma 3n E2B / E4B | ``NFKMLXGemma3n`` | ``NFKMLXGemma3nNet`` | ``NFKMLXGemma3nConfiguration`` from `config.json` (`gemma3n_text`, or the tri-modal wrapper's `text_config`) | `backend(directoryURL:)` | ``NFKMLXGemma3nBackend`` |
| Gemma 4 E2B / E4B / 26B-A4B | ``NFKMLXGemmaLanguage`` | ``NFKMLXGemmaNet`` | ``NFKMLXGemmaConfiguration`` from `config.json` (`.e2b`; `enable_moe_block` turns on the routed branch) | `let backend = try NFKMLXGemmaLanguage.backend(directoryURL: dir)`<br>`[NFKMLXGemmaLanguage gemmaBackendWithDirectoryURL:dir error:&error]`<br>image and audio (`gemma4-multimodal`): `try NFKMLXGemmaLanguage.gemmaMultimodalBackend(directoryURL: dir)` · `[NFKMLXGemmaLanguage gemmaMultimodalBackendWithDirectoryURL:dir error:&error]` | ``NFKMLXGemmaBackend`` |
| Gemma 4 12B unified | ``NFKMLXGemmaLanguage`` | ``NFKMLXGemma4UnifiedNet`` | `unifiedConfiguration(fromHuggingFace:)` (`.twelveB`) | `let backend = try NFKMLXGemmaLanguage.backend(directoryURL: dir)`<br>`[NFKMLXGemmaLanguage gemmaBackendWithDirectoryURL:dir error:&error]` | ``NFKMLXGemmaBackend`` |
| Gemma 2 | — | ``NFKMLXGemma2Net`` | `NFKMLXGemma2Configuration.gemma2_2B` / `.gemma2_9B` / `.gemma2_27B`; `configuration(fromHuggingFace:)` | `let net = try NFKMLXGemma2Net.load(directoryURL: dir)`; Swift only | SANA's text encoder |
| DeepSeek V4.1 Flash | ``NFKMLXDeepSeek`` | ``NFKMLXDeepSeekNet`` + ``NFKMLXDeepSeekCache`` | ``NFKMLXDeepSeekConfiguration`` `.v41Flash` from `config.json` | `let backend = try NFKMLXDeepSeek.backend(directoryURL: dir)`<br>`let paged = try NFKMLXDeepSeek.backend(directoryURL: dir, paging: .all)`<br>`[NFKMLXDeepSeek deepSeekBackendWithDirectoryURL:dir error:&error]`<br>`[NFKMLXDeepSeek deepSeekPagedBackendWithDirectoryURL:dir expertCacheBytes:1 << 32 error:&error]`<br>`[NFKMLXDeepSeek deepSeekBackendWithDirectoryURL:dir options:options error:&error]` (``NFKMLXDeepSeekLoadOptions``)<br>download: `try NFKMLXDeepSeek.deepSeekBackend(repo: "deepseek-ai/DeepSeek-V4.1-Flash", revision: nil, cacheDirectoryURL: nil)` · `[NFKMLXDeepSeek deepSeekBackendWithRepo:@"deepseek-ai/DeepSeek-V4.1-Flash" revision:nil cacheDirectoryURL:nil error:&error]`; `…residency:`, `…options:`, and `deepSeekPagedBackendWithRepo:revision:cacheDirectoryURL:expertCacheBytes:error:` take the directory factories' knobs<br>the directory supplies the collapsed id space its n-gram memory hashes over, derived from its own tokenizer | ``NFKMLXDeepSeekBackend`` (computes in bf16 as the release does, bit for bit against its own code; 763B decodes to 1.39 TiB of bf16 parameters, and with every group mapped out of the release the decoder holds 17.7 GiB) |
| DeepSeek V4 Flash / Pro | ``NFKMLXDeepSeek`` | ``NFKMLXDeepSeekNet`` + ``NFKMLXDeepSeekCache`` | `NFKMLXDeepSeekConfiguration.v4Flash` / `.v4Pro` from `config.json`; ``NFKMLXDeepSeekQuantization`` decodes the fp8 / fp4 storage | `let backend = try NFKMLXDeepSeek.backend(directoryURL: dir)`<br>`[NFKMLXDeepSeek deepSeekBackendWithDirectoryURL:dir options:options error:&error]`<br>download: `try NFKMLXDeepSeek.deepSeekBackend(repo: "deepseek-ai/DeepSeek-V4-Flash-0731", revision: nil, cacheDirectoryURL: nil)` · `[NFKMLXDeepSeek deepSeekBackendWithRepo:@"deepseek-ai/DeepSeek-V4-Flash-0731" revision:nil cacheDirectoryURL:nil error:&error]` | ``NFKMLXDeepSeekBackend`` (computes in bf16, bit for bit against the release's own code at prefill and decode) |
| Codestral-Mamba | ``NFKMLXMamba`` | ``NFKMLXMamba2Net`` | ``NFKMLXMamba2Configuration`` from `config.json` (`.codestral7B` preset) | `let backend = try NFKMLXMamba.backend(directoryURL: dir)`<br>`[NFKMLXMamba mambaBackendWithDirectoryURL:dir error:&error]` | ``NFKMLXMambaBackend`` (prefill-only, state-space scan) |
| Granite 4.0-H | ``NFKMLXGraniteHybrid`` | ``NFKMLXGraniteHybridNet`` | ``NFKMLXGraniteHybridConfiguration`` from `config.json` (dense h-1b, MoE h-tiny / h-small); `network(weightsURL:configuration:)` for a fine-tuned file | `let backend = try NFKMLXGraniteHybrid.backend(directoryURL: dir)`<br>`[NFKMLXGraniteHybrid graniteBackendWithDirectoryURL:dir error:&error]` | ``NFKMLXGraniteBackend`` (prefill-only, hybrid Mamba/attention) |
| Nemotron Nano 2 | ``NFKMLXNemotronH`` | ``NFKMLXNemotronHNet`` | ``NFKMLXNemotronHConfiguration`` from `config.json` (9B / 12B via `hybrid_override_pattern`); `network(weightsURL:configuration:)` for a fine-tuned file | `let backend = try NFKMLXNemotronH.backend(directoryURL: dir)`<br>`[NFKMLXNemotronH nemotronBackendWithDirectoryURL:dir error:&error]` | ``NFKMLXNemotronBackend`` (prefill-only, hybrid Mamba/attention/MLP) |
| T5 v1.1 / umT5 | ``NFKMLXT5Encoder`` | `NFKMLXT5EncoderNet` | `NFKMLXT5Configuration.xxl` / `.umt5XXL` | `let t5 = try NFKMLXT5Encoder.encoder(configuration: .xxl, directory: dir)`<br>umT5: `configuration: .umt5XXL`; Swift only | LTX and Wan text conditioning |
| Chat templates | ``NFKMLXChatTemplateRenderer`` | — | `NFKMLXChatTemplate.jinja(template:bosToken:eosToken:)` / `.chatML` | `options.chatTemplate = .jinja(template: template, bosToken: bos, eosToken: eos)`<br>ObjC: `NFKMLXGenerationParameterKey.chatTemplate` request parameter carrying the Jinja source | used by ``NFKMLXLanguageBackend`` |
| Constrained decoding | ``NFKMLXJSONConstraint``, ``NFKMLXJSONSchemaConstraint``, ``NFKMLXChoiceConstraint`` | ``NFKMLXVocabulary``, ``NFKMLXJSONSchema`` | `root`, `maximumWhitespaceRun`; `outputFormat` / `choices` request keys; the core `NFKParameterJSONSchema` | `options.jsonSchema = try NFKMLXJSONSchema(jsonText: schema)`<br>ObjC: `NFKParameterJSONSchema` dictionary, `outputFormat` = `"json-object"`, or `choices` request parameter | used by ``NFKMLXLanguageBackend``; JSON comes back parsed under `NFKOutputStructured` |

```swift
// Qwen3 (dense), Qwen2, Llama
// backend(directoryURL:), backend(directoryURL:draftDirectoryURL:)
// Qwen3-MoE, Qwen2-MoE, Mixtral, gpt-oss (MXFP4 experts stay packed)
backend(directoryURL:)
// Dense GGUF (`llama` / `qwen2` / `qwen3`)
backend(ggufURL:)
// Qwen3.5 / 3.6 / 3.8
NFKMLXHybridLanguage.backend(directoryURL:)
// Qwen3.8-Flash-Next (Qwen4-Exp)
NFKMLXQwen4Exp.backend(directoryURL:)
// Gemma 3 270M / 1B / 4B
backend(directoryURL:)
// Gemma 4 E2B / E4B / 26B-A4B
backend(directoryURL:)
// image and audio (gemma4-multimodal)
NFKMLXGemmaLanguage.gemmaMultimodalBackend(directoryURL:)
// Gemma 4 12B unified
backend(directoryURL:)
// Gemma 2
NFKMLXGemma2Net.load(directoryURL:)
// DeepSeek V4.1 Flash, V4 Flash / Pro
NFKMLXDeepSeek.backend(directoryURL:), backend(directoryURL:residency:), backend(directoryURL:paging:)
// download
NFKMLXDeepSeek.deepSeekBackend(repo:revision:cacheDirectoryURL:)
// Codestral-Mamba (Mamba-2 SSM)
NFKMLXMamba.backend(directoryURL:)
// Granite 4.0-H (hybrid Mamba/attention, dense + MoE)
NFKMLXGraniteHybrid.backend(directoryURL:)
// Nemotron Nano 2 (hybrid Mamba/attention/MLP)
NFKMLXNemotronH.backend(directoryURL:)
// T5 v1.1 / umT5
encoder(configuration:directory:)
// Chat templates
// chatTemplate request parameter
// Constrained decoding
// —
```

```objc
// Qwen3 (dense), Qwen2, Llama, Qwen3-MoE, Qwen2-MoE, Mixtral, gpt-oss
[NFKMLXLanguage backendWithDirectoryURL:dir error:&error]
[NFKMLXLanguage backendWithDirectoryURL:dir draftDirectoryURL:draftDir error:&error]
// Dense GGUF
[NFKMLXLanguage backendWithGGUFURL:url error:&error]
// Gemma 3 270M / 1B / 4B
[NFKMLXGemma3 backendWithDirectoryURL:dir error:&error]
// Gemma 3n E2B / E4B
[NFKMLXGemma3n backendWithDirectoryURL:dir error:&error]
// Gemma 4 E2B / E4B / 26B-A4B, 12B unified
[NFKMLXGemmaLanguage gemmaBackendWithDirectoryURL:dir error:&error]
// Gemma 4 image and audio (gemma4-multimodal)
[NFKMLXGemmaLanguage gemmaMultimodalBackendWithDirectoryURL:dir error:&error]
// DeepSeek V4.1 Flash, V4 Flash / Pro
[NFKMLXDeepSeek deepSeekBackendWithDirectoryURL:dir error:&error]
[NFKMLXDeepSeek deepSeekBackendWithDirectoryURL:dir options:options error:&error]
// download
[NFKMLXDeepSeek deepSeekBackendWithRepo:@"deepseek-ai/DeepSeek-V4.1-Flash" revision:nil cacheDirectoryURL:nil error:&error]
// Qwen3.5 / 3.6 / 3.8
[NFKMLXHybridLanguage backendWithDirectoryURL:dir error:&error]
// Qwen3.8-Flash-Next (Qwen4-Exp)
[NFKMLXQwen4Exp backendWithDirectoryURL:dir error:&error]
// Codestral-Mamba
[NFKMLXMamba mambaBackendWithDirectoryURL:dir error:&error]
// Granite 4.0-H
[NFKMLXGraniteHybrid graniteBackendWithDirectoryURL:dir error:&error]
// Nemotron Nano 2
[NFKMLXNemotronH nemotronBackendWithDirectoryURL:dir error:&error]
// Chat templates: the NFKMLXGenerationParameterKey.chatTemplate request parameter
// Constrained decoding: NFKParameterJSONSchema, or the outputFormat / choices request parameters
```


### Translation

| Model | Entry class | Network | Configuration for the released weights | Registered name | Base backend |
| --- | --- | --- | --- | --- | --- |
| OPUS-MT (Marian) | ``NFKMLXMarian`` | ``NFKMLXSeq2SeqNet`` | ``NFKMLXSeq2SeqConfiguration`` from the release's `config.json`; `.tinyMarian` for tests | `opus-mt` (the registry URL is the release directory) | ``NFKMLXTranslationBackend`` |
| NLLB-200 | ``NFKMLXNLLB`` | ``NFKMLXSeq2SeqNet`` | ``NFKMLXNLLBVariant`` `.distilled600M` / `.m1_3B` / `.distilled1_3B` / `.m3_3B`; the configuration from `config.json`; `.tinyM2M100` for tests | `nllb-200` | ``NFKMLXTranslationBackend`` |
| M2M-100 / SMaLL-100 | ``NFKMLXM2M100`` | ``NFKMLXSeq2SeqNet`` | ``NFKMLXM2M100Variant`` `.m418M` / `.m1_2B` / `.small100`; the configuration from `config.json`; `.tinyM2M100` for tests | `m2m100` · `small100` | ``NFKMLXTranslationBackend`` |
| TranslateGemma 4B / 12B / 27B | ``NFKMLXTranslateGemma`` | ``NFKMLXGemma3Net`` through ``NFKMLXGemma3Model`` | the release's `config.json` and its `chat_template.jinja` language table | `translategemma` | ``NFKMLXTranslationBackend`` |
| MADLAD-400 3B-MT | ``NFKMLXMADLAD`` | ``NFKMLXT5Seq2SeqNet`` | ``NFKMLXMADLADConfiguration`` `.mt3B` / `.mt7B` (`.tiny` for tests), read from `config.json` | `madlad400-3b-mt` | ``NFKMLXTranslationBackend`` |

```swift
// OPUS-MT, by pair (downloads Helsinki-NLP/opus-mt-en-de) or from a release directory
NFKMLXMarian.backend(sourceLanguage: "en", targetLanguage: "de", cacheDirectoryURL: nil)
NFKMLXMarian.backend(directoryURL:)
// M2M-100 418M / 1.2B, SMaLL-100
NFKMLXM2M100.backend(variant: .m418M, directoryURL:)
NFKMLXM2M100.backend(variant: .m418M, revision: nil, cacheDirectoryURL: nil)
// NLLB-200, 202 languages
NFKMLXNLLB.backend(variant: .distilled600M, directoryURL:)
// MADLAD-400 3B-MT, bfloat16
NFKMLXMADLAD.backend(directoryURL:, half: true)
// MADLAD-400 7B-MT at float32, the blocks that do not fit streamed
NFKMLXMADLAD.backend(directoryURL:, half: false, residency: .automatic)
// TranslateGemma, at the checkpoint's bfloat16
NFKMLXTranslateGemma.backend(directoryURL:, precision: .checkpoint)
// Seed-X, at the checkpoint's bfloat16, or by variant
NFKMLXSeedX.backend(directoryURL:, precision: .checkpoint)
NFKMLXSeedX.backend(variant: .ppo, revision: nil, cacheDirectoryURL: nil, precision: .checkpoint)
// HY-MT1.5, Hy-MT2 (the 30B-A3B too), Hunyuan-MT, or Hunyuan-MT-Chimera, at the checkpoint's bfloat16, or by variant
NFKMLXHYMT.backend(release: .hyMT2, directoryURL:, precision: .checkpoint)
NFKMLXHYMT.backend(variant: .hyMT2_1_8B, revision: nil, cacheDirectoryURL: nil, precision: .checkpoint)
// Fine-tuning
// The release directory supplies its geometry from its config.json
NFKMLXMarian.network(directoryURL:) / NFKMLXM2M100.network(directoryURL:) / NFKMLXNLLB.network(directoryURL:) / NFKMLXMADLAD.network(directoryURL:)
```

```objc
// OPUS-MT, by pair or from a release directory
[NFKMLXMarian backendWithSourceLanguage:@"en" targetLanguage:@"de" cacheDirectoryURL:nil error:&error]
[NFKMLXMarian backendWithDirectoryURL:dir error:&error]
// M2M-100 418M / 1.2B, SMaLL-100
[NFKMLXM2M100 backendWithVariant:NFKMLXM2M100VariantM418M directoryURL:dir error:&error]
[NFKMLXM2M100 backendWithVariant:NFKMLXM2M100VariantM418M revision:nil cacheDirectoryURL:nil error:&error]
// NLLB-200, 202 languages
[NFKMLXNLLB backendWithVariant:NFKMLXNLLBVariantDistilled600M directoryURL:dir error:&error]
// MADLAD-400 3B-MT, bfloat16
[NFKMLXMADLAD backendWithDirectoryURL:dir halfPrecision:YES error:&error]
// MADLAD-400 7B-MT at float32, the blocks that do not fit streamed
[NFKMLXMADLAD backendWithDirectoryURL:dir halfPrecision:NO residency:NFKMLXResidencyAutomatic error:&error]
// TranslateGemma, at the checkpoint's bfloat16
[NFKMLXTranslateGemma backendWithDirectoryURL:dir precision:NFKMLXWeightPrecisionCheckpoint error:&error]
// Seed-X, at the checkpoint's bfloat16, or by variant
[NFKMLXSeedX backendWithDirectoryURL:dir precision:NFKMLXWeightPrecisionCheckpoint error:&error]
[NFKMLXSeedX backendWithVariant:NFKMLXSeedXVariantPpo revision:nil cacheDirectoryURL:nil precision:NFKMLXWeightPrecisionCheckpoint error:&error]
// HY-MT1.5, Hy-MT2 (the 30B-A3B too), Hunyuan-MT, or Hunyuan-MT-Chimera, at the checkpoint's bfloat16, or by variant
[NFKMLXHYMT backendWithRelease:NFKMLXHYMTReleaseHyMT2 directoryURL:dir precision:NFKMLXWeightPrecisionCheckpoint error:&error]
[NFKMLXHYMT backendWithVariant:NFKMLXHYMTVariantHyMT2_1_8B revision:nil cacheDirectoryURL:nil precision:NFKMLXWeightPrecisionCheckpoint error:&error]
```

### Video


| Model | Entry class | Network | Configuration for the released weights | Registered name | Base backend |
| --- | --- | --- | --- | --- | --- |
| RIFE HDv3 | ``NFKMLXRIFE`` | `NFKMLXRIFENet` | fixed geometry (three IFBlocks, width 90, scales 4/2/1) | `rife` · `rife-clip` | ``NFKMLXTensorBackend`` (`frame0` / `frame1`); `clipBackend(weightsURL:)` for a clip |
| RIFE v4 | ``NFKMLXRIFEv4`` | ``NFKMLXRIFEv4Net`` | fixed geometry (four blocks, timestep input, scales 8/4/2/1) | `rife-v4` | ``NFKMLXTensorBackend`` |
| RAFT | ``NFKMLXRAFT`` | `NFKMLXRAFTNet` | RAFT-large (feature 256, 4 levels, radius 4); `iterations` (default 6) | `raft` | ``NFKMLXTensorBackend`` |
| BasicVSR | ``NFKMLXVideoSR`` | `NFKMLXVideoSRNet` + `NFKVSRSPyNet` | `NFKMLXVideoSRConfiguration.base` (×4) | `video-super-resolution` · `video-super-resolution-clip` | ``NFKMLXModuleBackend`` (frame) / `clipBackend(weightsURL:)` |
| SD ×4 upscaler | ``NFKMLXSDUpscaler`` | ``NFKMLXSDUNet`` + ``NFKMLXSDAutoencoder`` | `NFKMLXSDUNetConfiguration.upscaler`, `NFKMLXSDVAEConfiguration.upscaler`; `noiseLevel` (20) | `sd-x4-upscaler` | ``NFKMLXDiffusionBackend`` |
| V-JEPA 2 (video features, video classification) | ``NFKMLXVJEPA2`` | ``NFKMLXVJEPA2Net`` | read from the release's `config.json` (ViT-L, ViT-H, or ViT-g; `id2label` for a classifier) | `backend(directoryURL:)`; fine-tuned: `network(directoryURL:labels:)`, `fineTune`, `save(_:toDirectoryURL:)` | ``NFKMLXVJEPA2Backend``; video / image → mean-pooled embedding, and ranked classes from a classification release (V-JEPA 2, Meta) |
| Cosmos Tokenizer (image and video tokens) | ``NFKMLXCosmosTokenizer`` | ``NFKMLXCosmosTokenizerNet`` | `NFKMLXCosmosTokenizerConfiguration.variant(_:)` for each ``NFKMLXCosmosTokenizerVariant`` (CI8x8 … DV8x16x16); fine-tuned: `network(variant:weightsURL:)` | `cosmos-tokenizer-<variant>` | ``NFKMLXCosmosTokenizerBackend``; image / clip → reconstruction, `codeForImage:error:` → latent or tokens |
| Clip backend | ``NFKMLXVideoBackend`` | — | ``NFKMLXVideoConfiguration`` (`frameRateMultiplier`, `outputFramesPerSecond`) | — | ``NFKMLXVideoFile`` under it |

```swift
// RIFE HDv3
let backend = try NFKMLXRIFE.backend(weightsURL: url)
// clip: try NFKMLXRIFE.clipBackend(weightsURL: url) · [NFKMLXRIFE clipBackendWithWeightsURL:url error:&error]
// clip download: clipBackend(repo:weightsPath:revision:cacheDirectoryURL:) · clipBackendWithRepo:weightsPath:revision:cacheDirectoryURL:error:
// RIFE v4
let backend = try NFKMLXRIFEv4.backend(weightsURL: url)
// RAFT
let backend = try NFKMLXRAFT.backend(weightsURL: url)
// BasicVSR
let backend = try NFKMLXVideoSR.backend(weightsURL: url)
// clip: try NFKMLXVideoSR.clipBackend(weightsURL: url) · [NFKMLXVideoSR clipBackendWithWeightsURL:url error:&error]
// clip download: clipBackend(repo:weightsPath:revision:cacheDirectoryURL:) · clipBackendWithRepo:weightsPath:revision:cacheDirectoryURL:error:
// SD ×4 upscaler
let backend = try NFKMLXSDUpscaler.backend(unetWeightsURL: unetURL, vaeWeightsURL: vaeURL, textContextURL: contextURL, noiseLevel: 20)
// V-JEPA 2 (video features) · reads the release's config.json
let backend = try NFKMLXVJEPA2.backend(directoryURL: dir)
// Cosmos Tokenizer · the release's autoencoder.jit
let backend = try NFKMLXCosmosTokenizer.backend(variant: .continuousVideo8x8x8, weightsURL: url)
// tokens: try NFKMLXCosmosTokenizer.tokenizer(variant: .discreteImage16x16, weightsURL: url).tokens(pixels)
// Clip backend
let backend = NFKMLXVideoBackend(identifier: "my-clip-model") { frames in frames }
// Swift only
```

```objc
// RIFE HDv3
[NFKMLXRIFE backendWithWeightsURL:url error:&error]
// clip
[NFKMLXRIFE clipBackendWithWeightsURL:url error:&error]
// RIFE v4
[NFKMLXRIFEv4 backendWithWeightsURL:url error:&error]
// RAFT
[NFKMLXRAFT backendWithWeightsURL:url error:&error]
// BasicVSR
[NFKMLXVideoSR backendWithWeightsURL:url error:&error]
// clip
[NFKMLXVideoSR clipBackendWithWeightsURL:url error:&error]
// SD ×4 upscaler
[NFKMLXSDUpscaler backendWithUNetWeightsURL:unetURL vaeWeightsURL:vaeURL textContextURL:contextURL noiseLevel:20 error:&error]
// V-JEPA 2 (video features)
[NFKMLXVJEPA2 backendWithDirectoryURL:dir error:&error]
// Cosmos Tokenizer
[NFKMLXCosmosTokenizer backendWithVariant:NFKMLXCosmosTokenizerVariantContinuousVideo8x8x8 weightsURL:url error:&error]
```


### Audio


| Model | Entry class | Network | Configuration for the released weights | Registered name | Base backend |
| --- | --- | --- | --- | --- | --- |
| Whisper | ``NFKMLXWhisper`` | ``NFKMLXWhisperNet`` | ``NFKMLXWhisperVariant`` `.tiny` / `.base` / `.small` / `.medium` / `.large` (v1 / v2) / `.largeV3` / `.largeV3Turbo` (`NFKMLXWhisperConfiguration.tiny` …); `emitsTimestamps` | `whisper-tiny` | ``NFKMLXWhisperBackend`` (`backend(variant:weightsURL:tokenizer:timestamps:)`) |
| Parakeet-TDT | ``NFKMLXParakeet`` | ``NFKMLXParakeetNet`` | `NFKMLXParakeetConfiguration.tdt06B` (0.6B v2: 24 rel-pos conformer layers, TDT durations 0…4) | `backend(directoryURL:)`; `register()` registers `parakeet-tdt` (a release directory or its `.nemo` archive) | ``NFKMLXParakeetBackend`` (text + per-token `NFKOutputSegments`) |
| Granite Speech 3.3-2b | ``NFKMLXGraniteSpeech`` | ``NFKMLXGraniteSpeechNet`` | `net(fromDirectory:)` (Conformer encoder + BLIP-2 Q-former + dense Granite decoder) | `backend(directoryURL:)`; `register()` registers `granite-speech-3.3-2b` | ``NFKMLXGraniteSpeechBackend`` (audio → text) |
| Voxtral-Mini 3B | ``NFKMLXVoxtral`` | ``NFKMLXVoxtralNet`` | `net(fromDirectory:)` (Whisper encoder + 2-linear projector + Llama decoder) | `backend(directoryURL:)`; `register()` registers `voxtral-mini-3b` | ``NFKMLXVoxtralBackend`` (audio → text) |
| Canary-1B-v2 | ``NFKMLXCanary`` | ``NFKMLXCanaryNet`` | `NFKMLXCanaryConfiguration.v2` (biased FastConformer encoder + Transformer attention encoder-decoder) | `backend(directoryURL:)`; `register()` registers `canary-1b-v2` | ``NFKMLXCanaryBackend`` (audio → text; `src>tgt` translates); per request: `NFKParameterSourceLanguage` and `NFKParameterTargetLanguage` |
| Wav2Vec2 / HuBERT | ``NFKMLXWav2Vec2`` | ``NFKMLXWav2Vec2Net`` | ``NFKMLXWav2Vec2Configuration`` read from the release's `config.json` (base post-norm, large pre-norm) | `backend(directoryURL:)` | ``NFKMLXWav2Vec2Backend`` (audio → text from a CTC release, and an embedding) |
| W2V-BERT 2.0 | ``NFKMLXWav2Vec2Bert`` | ``NFKMLXWav2Vec2BertNet`` | `NFKMLXWav2Vec2BertConfiguration.v2` (SeamlessM4T filterbanks + relative-key Conformer) | `backend(directoryURL:)` | ``NFKMLXWav2Vec2BertBackend`` (audio → embedding; text from a CTC fine-tune) |
| Demucs v2 | ``NFKMLXDemucs`` | ``NFKMLXDemucsNet`` | `NFKMLXDemucsConfiguration()` = music (stereo, depth 6, 4 stems, BLSTM, context 3); `network(weightsURL:)` for a fine-tuned file (the geometry read from it) | `demucs` | ``NFKMLXDemucsBackend`` |
| Speech denoiser | ``NFKMLXDenoiser`` | ``NFKMLXDemucsNet`` | ``NFKMLXDemucsConfiguration`` set to dns48 (mono, depth 5, 1 stem, causal, context 1); `network(weightsURL:)` for a fine-tuned file (the base width read from it) | `denoiser` | ``NFKMLXDenoiserBackend`` |
| MP-SENet | ``NFKMLXMPSENet`` | TS-transformer (bidirectional-GRU FFN) over compressed magnitude + phase | `NFKMLXMPSENetConfiguration()` (fftSize 400, hop 100, 4 blocks) | `mpsenet`; `NFKMLXMPSENetFactory.backend(weightsURL:)` | ``NFKMLXMPSENetBackend`` |
| GTCRN | ``NFKMLXGTCRN`` | ERB band merge/split, grouped-conv encoder/decoder, dual-path grouped RNN → complex ratio mask (~48K params) | `NFKMLXGTCRNConfiguration()` | `gtcrn`; `NFKMLXGTCRNFactory.backend(weightsURL:)` | ``NFKMLXGTCRNBackend`` |
| SGMSE+ | ``NFKMLXSGMSE`` | `NFKMLXNCSNppNet` score network + OUVE reverse-SDE predictor-corrector sampler | `NFKMLXSGMSEConfiguration()` (`ncsnpp` / `ncsnpp_48k`) | `sgmse`; `NFKMLXSGMSE.backend(weightsURL:)` | ``NFKMLXSGMSEBackend`` |
| StoRM | ``NFKMLXStoRM`` | discriminative predictor + `[noisy, denoised]`-conditioned NCSN++ score; SDE re-centered on the estimate | `NFKMLXStoRMConfiguration()` | `storm`; `NFKMLXStoRM.backend(weightsURL:)` | ``NFKMLXStoRMBackend`` |
| MossFormer2 SE 48K | ``NFKMLXMossFormer2SENet`` | `NFKMLXMossFormer2SENet` (FLASH + `Gated_FSMN`) over `NFKMLXKaldiFbank` | `NFKMLXMossFormer2Configuration()` (48 kHz, 24 blocks); `NFKMLXMossFormer2Factory.network(weightsURL:)` for a fine-tuned file | `mossformer2-se` | ``NFKMLXMossFormer2Backend`` |
| MossFormer2 SR 48K | ``NFKMLXMossFormer2SRNet`` | the SE backbone (80 → 80 mel) + ``NFKMLXMossFormer2SRGenerator`` (Snake HiFi-GAN, ×256) + the bandwidth substitution | `NFKMLXMossFormer2SRConfiguration()` (48 kHz, 1024/256, 80 mels to 8 kHz) | `mossformer2-sr`; `NFKMLXMossFormer2SRFactory.backend(directoryURL:)` | ``NFKMLXMossFormer2SRBackend`` |
| DeepFilterNet3 | ``NFKMLXDeepFilterNet`` | `DfNet` (`SqueezedGRU_S` encoder / ERB decoder / DF decoder) over a libdf-reproduced DSP | `NFKMLXDeepFilterNetConfiguration()` (48 kHz, 32 ERB bands, 96 DF bins) | `deepfilternet3` | ``NFKMLXDeepFilterNetBackend`` |
| VoiceRestore | ``NFKMLXVoiceRestore`` / ``NFKMLXBigVGAN`` | E2-TTS CFM transformer (gateloop + adaLN) + BigVGAN v2 vocoder | `NFKMLXVoiceRestoreConfiguration()` + `NFKMLXBigVGANConfiguration()` | `voicerestore` | ``NFKMLXVoiceRestoreBackend`` |
| Resemble Enhance | ``NFKMLXResembleEnhance`` | STFT-mask 2-D UNet denoiser + IRMAE/WaveNet-CFM latent flow matching + UnivNet LVC vocoder | `NFKMLXResembleConfiguration()` | `resemble-enhance` | ``NFKMLXResembleEnhanceBackend`` |
| MetricGAN+ | ``NFKMLXMetricGANPlus`` | ``NFKMLXMetricGANPlusNet`` (2-layer BLSTM 257→200 per direction, Linear 400→300, LeakyReLU 0.3, Linear 300→257, learnable sigmoid) over a zero-padded 512-point Hamming STFT | `NFKMLXMetricGANPlusConfiguration()` (16 kHz, hop 256) | `metricgan-plus`; `NFKMLXMetricGANPlus.backend(weightsURL:)` | ``NFKMLXMetricGANPlusBackend`` |
| CMGAN | ``NFKMLXCMGAN`` | ``NFKMLXCMGANNet`` (dense encoder, 4 two-stage conformer blocks with Shaw relative positions, mask + complex decoders) over a power-compressed 400/100 Hamming STFT | `NFKMLXCMGANConfiguration()` (16 kHz, 64 channels, compress 0.3) | `cmgan`; `NFKMLXCMGAN.backend(weightsURL:)` | ``NFKMLXCMGANBackend`` |
| FRCRN SE 16K | ``NFKMLXFRCRN`` | ``NFKMLXFRCRNNet`` (two complex UNets: frequency-recurrent FSMNs, complex squeeze-excites, a time FSMN bottleneck) over a 640/320 sqrt-Hann conv-STFT | `NFKMLXFRCRNConfiguration()` (16 kHz, 128 channels, order 20); `network(weightsURL:)` for a fine-tuned file | `frcrn`; `NFKMLXFRCRN.backend(weightsURL:)` | ``NFKMLXFRCRNBackend`` |
| NU-Wave 2 | ``NFKMLXNUWave2`` | ``NFKMLXNUWave2Net`` (15 short-time Fourier convolution blocks with BSFT band modulation) + an 8-step logSNR DDIM | `NFKMLXNUWave2Configuration()` (48 kHz, 1024/256, 64 channels) | `nuwave2`; `NFKMLXNUWave2.backend(weightsURL:)` | ``NFKMLXNUWave2Backend`` |
| Apollo | ``NFKMLXApollo`` | ``NFKMLXApolloNet`` (80-band split, 6 band-Roformer + ICB layers, GLU band heads) | `NFKMLXApolloConfiguration()` (44.1 kHz, 20 ms window, 256 features) | `apollo`; `NFKMLXApollo.backend(weightsURL:)` | ``NFKMLXApolloBackend`` |
| HT Demucs (v4) | ``NFKMLXHTDemucs`` | ``NFKMLXHTDemucsNet``, ``NFKMLXHTDemucsBag`` | ``NFKMLXHTDemucsVariant`` `.fourStem` / `.sixStem`; the fine-tuned release through `backend(fineTunedWeightsURLs:)`; `network(weightsURL:)` and `fineTune` with an ``NFKMLXHTDemucsTrainingRecipe`` | `htdemucs` · `htdemucs-6s` | ``NFKMLXHTDemucsBackend`` |
| Conv-TasNet | ``NFKMLXConvTasNet`` | `NFKMLXConvTasNetNet` | `NFKMLXConvTasNetConfiguration.libri2Mix16k`; `perChannelPReLU` optional | `conv-tasnet` | ``NFKMLXConvTasNetBackend`` |
| MarbleNet VAD | ``NFKMLXVAD`` | `NFKMLXVADNet` | `NFKMLXVADConfiguration.marbleNet` | `vad-marblenet` | ``NFKMLXVADBackend``; per request: `NFKMLXVoiceActivityParameterKey.speechThreshold` (0.5) |
| Silero VAD v6 | ``NFKMLXSileroVAD`` | ``NFKMLXSileroVADNet`` | `NFKMLXSileroVADConfiguration.v6`; `network(weightsURL:)` to fine-tune the decoder | `silero-vad` | ``NFKMLXSileroVADBackend``; per request: `NFKMLXVoiceActivityParameterKey.speechThreshold` (0.5) |
| PANNs Cnn14 | ``NFKMLXAudioTagger`` | ``NFKMLXAudioTaggerNet`` | `NFKMLXAudioTaggerConfiguration.panns`; `network(weightsURL:configuration:)` for a custom classifier | `audio-tagger-panns` | ``NFKMLXAudioTaggerBackend`` (`labels:`) |
| Descript Audio Codec | ``NFKMLXDAC`` | `NFKMLXDACNet` (`NFKDACEncoderNet`, `NFKDACDecoderNet`) | `NFKMLXDACConfiguration.dac44kHz` / `.dac24kHz` / `.dac16kHz` | `dac` | ``NFKMLXDACBackend``; `encode` / `decode` for the tokens |
| SNAC | ``NFKMLXSNAC`` | `NFKMLXSNACNet` (`NFKSNACEncoderNet`, `NFKSNACDecoderNet`) | ``NFKMLXSNACVariant`` `.speech24kHz` / `.music32kHz` / `.music44kHz` | `snac` · `snac-32khz` · `snac-44khz` | ``NFKMLXSNACBackend``; `decode(_:deterministic:)` |
| BigVGAN v2 | ``NFKMLXBigVGAN`` | SnakeBeta + anti-aliased `Activation1d` generator | `NFKMLXBigVGANConfiguration()` (24 kHz, 100-band) | `bigvgan-v2-24khz` | ``NFKMLXBigVGANBackend`` |
| Mimi | ``NFKMLXMimi`` | `NFKMLXMimiNet` (SEANet + RoPE transformers + split RVQ) | `NFKMLXMimiConfiguration()` (24 kHz, 12.5 Hz) | `mimi` | ``NFKMLXMimiBackend``; `encode` / `decode` for the tokens |
| Basic Pitch | ``NFKMLXBasicPitch`` | `NFKMLXBasicPitchNet` | ``NFKMLXBasicPitchConfiguration`` `.icassp2022`, `.icassp2022Trainable` | `basic-pitch` | ``NFKMLXBasicPitchBackend`` (``NFKMLXTranscriptionParameterKey``) |
| All-In-One | ``NFKMLXAllInOne`` | `NFKMLXAllInOneNet` + ``NFKMLXBarTracker`` | ``NFKMLXAllInOneConfiguration`` `.harmonix` | `allin1` | ``NFKMLXAllInOneBackend`` |
| MuScriptor | ``NFKMLXMuScriptor`` | `NFKMLXMuScriptorNet` | ``NFKMLXMuScriptorConfiguration`` `.small` / `.medium` / `.large` | `muscriptor` | ``NFKMLXMuScriptorBackend`` |
| hFT-Transformer | ``NFKMLXHFTTransformer`` | `NFKMLXHFTTransformerNet` | ``NFKMLXHFTTransformerConfiguration`` `.maestro` | `hft-transformer` | ``NFKMLXHFTTransformerBackend`` |

```swift
// Whisper
let backend = try NFKMLXWhisper.backend(variant: .tiny, weightsURL: url, tokenizer: tokenizer, timestamps: false)
// Parakeet-TDT (an unpacked .nemo directory)
let backend = try NFKMLXParakeet.backend(directoryURL: dir)
// Granite Speech 3.3-2b (a release directory; speech → text)
let speech = try NFKMLXGraniteSpeech.backend(directoryURL: dir)
// Voxtral-Mini 3B (a release directory; speech → text)
let voxtral = try NFKMLXVoxtral.backend(directoryURL: dir)

// Canary-1B-v2 (a release directory; speech → text, multitask ASR/translation)
let canary = try NFKMLXCanary.backend(directoryURL: dir)
// Wav2Vec2 / HuBERT (a release directory; speech → text from a CTC release, and features)
let wav2vec2 = try NFKMLXWav2Vec2.backend(directoryURL: dir)
// fine-tune: network(directoryURL:vocabulary:), fineTune(_:examples:steps:), save(_:tokenizer:toDirectoryURL:)
// W2V-BERT 2.0 (a release directory; speech → features)
let w2vBert = try NFKMLXWav2Vec2Bert.backend(directoryURL: dir)
// .small / .medium / .largeV3 · …VariantSmall / …VariantMedium / …VariantLargeV3
// Demucs v2
let backend = try NFKMLXDemucs.backend(weightsURL: url)
// Speech denoiser
let backend = try NFKMLXDenoiser.backend(weightsURL: url)
// MP-SENet (magnitude + phase speech enhancement)
let backend = try NFKMLXMPSENetFactory.backend(weightsURL: url)
// GTCRN (real-time speech enhancement, ~48K params)
let backend = try NFKMLXGTCRNFactory.backend(weightsURL: url)
// fine-tuned: let net = try NFKMLXGTCRNFactory.network(weightsURL: url), then fineTune and NFKMLXWeights.save
// SGMSE+ (score-based generative dereverberation / enhancement)
let backend = try NFKMLXSGMSE.backend(weightsURL: url)
// StoRM (few-step stochastic regeneration)
let backend = try NFKMLXStoRM.backend(weightsURL: url)
// MossFormer2 SE 48K (full-band enhancement)
let backend = try NFKMLXMossFormer2Factory.backend(weightsURL: url)
// MossFormer2 SR 48K (speech super-resolution; a directory holding the _m and _g checkpoints)
let backend = try NFKMLXMossFormer2SRFactory.backend(directoryURL: dir)
// DeepFilterNet3 (real-time 48 kHz denoiser)
let backend = try NFKMLXDeepFilterNetFactory.backend(weightsURL: url)
// VoiceRestore (flow-matching universal restorer: transformer + BigVGAN)
let backend = try NFKMLXVoiceRestoreFactory.backend(weightsURL: transformerURL, vocoderURL: bigvganURL, steps: 32, cfgStrength: 0.5)
// Resemble Enhance (five-network general restorer: denoiser + IRMAE/CFM + UnivNet LVC vocoder)
let backend = try NFKMLXResembleEnhanceFactory.backend(directoryURL: enhancerStage2Dir)
// MetricGAN+ (two-layer BLSTM magnitude mask)
let backend = try NFKMLXMetricGANPlus.backend(weightsURL: url)
// CMGAN (conformer metric GAN: mask + complex residual)
let backend = try NFKMLXCMGAN.backend(weightsURL: url)
// FRCRN (two complex UNets with frequency-recurrent FSMN memories)
let backend = try NFKMLXFRCRN.backend(weightsURL: url)
// NU-Wave 2 (diffusion bandwidth extension, 8-step DDIM)
let backend = try NFKMLXNUWave2.backend(weightsURL: url)
// fine-tuned: let net = try NFKMLXNUWave2.network(weightsURL: url), then fineTune and NFKMLXWeights.save
// Apollo (music codec-artifact restoration)
let backend = try NFKMLXApollo.backend(weightsURL: url)
// HT Demucs (v4)
let backend = try NFKMLXHTDemucs.backend(weightsURL: url)
// Conv-TasNet
let backend = try NFKMLXConvTasNet.backend(weightsURL: url)
// fine-tuned: let net = try NFKMLXConvTasNet.network(weightsURL: url), then fineTune and NFKMLXWeights.save
// MarbleNet VAD
let backend = try NFKMLXVAD.backend(weightsURL: url)
// fine-tuned: let net = try NFKMLXVAD.network(weightsURL: url), then fineTune and NFKMLXWeights.save
// Silero VAD v6
let backend = try NFKMLXSileroVAD.backend(weightsURL: url)
// fine-tuned: let net = try NFKMLXSileroVAD.network(weightsURL: url)
// PANNs Cnn14
let backend = try NFKMLXAudioTagger.backend(weightsURL: url, labels: nil)
// custom classes: try NFKMLXAudioTagger.network(weightsURL: url, configuration: myClasses)
// Descript Audio Codec
let backend = try NFKMLXDAC.backend(weightsURL: url)
// tokens: let codec = try NFKMLXDAC.codec(configuration: .dac44kHz, weightsURL: url) then codec.encode(samples) / codec.decode(codes)
// hFT-Transformer (piano)
let backend = try NFKMLXHFTTransformer.backend(weightsURL: url)

// MuScriptor (gated, CC BY-NC 4.0 weights)
let backend = try NFKMLXMuScriptor.backend(variant: .medium, weightsURL: url)

// All-In-One
let backend = try NFKMLXAllInOne.backend(weightsURL: url, demucsWeightsURL: demucs)
// stems: NFKMLXAllInOne.makeNet().analyze(stems: stems, barTracker: NFKMLXBarTracker())
// fine-tuned: let net = try NFKMLXAllInOne.network(weightsURL: url), then fineTune with NFKMLXAllInOneTargets

// Basic Pitch
let backend = try NFKMLXBasicPitch.backend(weightsURL: url)
// notes: let net = NFKMLXBasicPitch.makeNet() then net.transcribe(samples, sampleRate: 22050) -> NFKMIDISequence
// fine-tune: let net = try NFKMLXBasicPitch.network(weightsURL: trainableURL), then NFKMLXBasicPitch.fineTune

// SNAC
let backend = try NFKMLXSNAC.backend(weightsURL: url)
// tokens: let codec = try NFKMLXSNAC.codec(configuration: .snac24kHz, weightsURL: url) then codec.encode(samples) / codec.decode(codes, deterministic: true)
// BigVGAN v2 (anti-aliased SnakeBeta vocoder; copy-synthesis backend, or net(mel) for the generator)
let backend = try NFKMLXBigVGANFactory.backend(weightsURL: url)
// Mimi (transformer-in-codec; reconstruction backend, or codec.encode/decode for the tokens)
let backend = try NFKMLXMimi.backend(weightsURL: url)
```

```objc
// Whisper
[NFKMLXWhisper backendWithVariant:NFKMLXWhisperVariantTiny weightsURL:url tokenizer:tokenizer timestamps:NO error:&error]
// Parakeet-TDT
[NFKMLXParakeet backendWithDirectoryURL:dir error:&error]
// Canary-1B-v2
[NFKMLXCanary backendWithDirectoryURL:dir error:&error]
// Wav2Vec2 / HuBERT
[NFKMLXWav2Vec2 backendWithDirectoryURL:dir error:&error]
// W2V-BERT 2.0
[NFKMLXWav2Vec2Bert backendWithDirectoryURL:dir error:&error]
// Demucs v2
[NFKMLXDemucs backendWithWeightsURL:url error:&error]
// Speech denoiser
[NFKMLXDenoiser backendWithWeightsURL:url error:&error]
// HT Demucs (v4)
[NFKMLXHTDemucs backendWithWeightsURL:url error:&error]
// Conv-TasNet
[NFKMLXConvTasNet backendWithWeightsURL:url error:&error]
// MarbleNet VAD
[NFKMLXVAD backendWithWeightsURL:url error:&error]
// Silero VAD v6
[NFKMLXSileroVAD backendWithWeightsURL:url error:&error]
// PANNs Cnn14
[NFKMLXAudioTagger backendWithWeightsURL:url labels:nil error:&error]
// Descript Audio Codec
[NFKMLXDAC backendWithWeightsURL:url error:&error]
// SNAC
[NFKMLXSNAC backendWithWeightsURL:url error:&error]
// Granite Speech 3.3-2b
[NFKMLXGraniteSpeech graniteSpeechBackendWithDirectoryURL:dir error:&error]
// Voxtral-Mini 3B
[NFKMLXVoxtral voxtralBackendWithDirectoryURL:dir error:&error]
// BigVGAN v2
[NFKMLXBigVGAN_Factory backendWithWeightsURL:url error:&error]
// Mimi
[NFKMLXMimi backendWithWeightsURL:url error:&error]
```


### Text to speech and music


| Model | Entry class | Network | Configuration for the released weights | Registered name / factory | Base backend |
| --- | --- | --- | --- | --- | --- |
| FastSpeech2 conformer + paired HiFi-GAN | ``NFKMLXFastSpeech2`` (the backend factories), ``NFKMLXVoice`` | ``NFKMLXFastSpeech2Net``, `NFKMLXHiFiGANNet` | `NFKMLXFastSpeech2Configuration()` = espnet LJSpeech; `NFKMLXHiFiGANConfiguration()` = UNIVERSAL_V1 geometry (the paired `vocoder.` weights) | `fastspeech2-voice`; `voiceBackend(directoryURL:phonemizer:)`; `makeSpeechBackend(phonemize:)` | ``NFKMLXGenerationBackend``; `makeSpeechBackend(phonemize:)` returns an ``NFKMLXSpeechBackend`` |
| HiFi-GAN | ``NFKMLXHiFiGAN`` | `NFKMLXHiFiGANNet` | `NFKMLXHiFiGANConfiguration()` (80 mel bins, 512 channels, rates 8/8/2/2) | — | mel → waveform object |
| Kokoro-82M | ``NFKMLXKokoro`` | ``NFKMLXKokoroNet`` | `NFKMLXKokoroConfiguration.v1`; a voice from `loadVoice`, or per request under `NFKParameterVoice` | `backend(directoryURL:voiceName:)` | ``NFKMLXSpeechBackend`` (phonemes in) |
| Chatterbox | ``NFKMLXChatterbox`` | ``NFKMLXChatterboxTTS`` (``NFKMLXChatterboxVoiceEncoderNet``, ``NFKMLXS3TokenizerNet``, ``NFKMLXT3Net``, ``NFKMLXS3GenNet``) | `.released` on the VoiceEncoder (3×256), the S3 tokenizer (6×1280), and T3 (Llama 520M with llama3 rope); S3Gen flow + HiFT takes no configuration | `chatterbox`; `speechBackend(directoryURL:voiceURL:)`; a request's clip under `NFKInputVoiceReference` | ``NFKMLXSpeechBackend`` (24 kHz WAV; text → cloned voice) |
| Phonemizers | ``NFKMLXNeuralG2P``, ``NFKMLXEspeakPhonemizer`` | `NFKMLXG2PNet` | ``NFKMLXG2PConfiguration`` | — | ``NFKMLXPhonemizing`` protocol (Objective-C and Swift); ``NFKMLXPhonemizer`` protocol (Swift) |
| Hand-chained TTS | ``NFKMLXTTS`` | `NFKMLXAcousticNet` + `NFKMLXHiFiGANNet` | ``NFKMLXAcousticConfiguration`` | `ttsBackend(phonemizer:acousticURL:vocoderURL:symbols:)`; `makeSpeechBackend()` | ``NFKMLXGenerationBackend``; `makeSpeechBackend()` returns an ``NFKMLXSpeechBackend`` |
| MiniMax Music 3 | ``NFKMLXMusic3`` | `NFKMusic3VocoderNet`, `NFKMusic3DepthDecoderNet`, `NFKMusic3ConditionEncoderNet`, `NFKMusic3DiTNet`, ``NFKMLXLanguageNet`` | the release directory (bf16 LM, float32 DiT); `quantizeRelease(at:to:bits:transformerBits:groupSize:)` for the 7.7 GiB copy | `minimax-music3`; `backend(directoryURL:)` | ``NFKMLXMusicBackend`` |
| Stable Audio Open 1.0 | ``NFKMLXStableAudioGenerator`` | ``NFKMLXStableAudioDiTNet``, ``NFKMLXOobleckAutoencoderNet``, ``NFKMLXStableAudioProjectionNet``, `NFKMLXT5EncoderNet` | the release directory, float32 throughout; `NFKMLXStableAudio.network(directoryURL:)` for the fine-tunable transformer and conditioners | `stable-audio-open-1.0`; `backend(directoryURL:)` | ``NFKMLXGenerationBackend`` |

```swift
// FastSpeech2 conformer + paired HiFi-GAN (a release directory; g2p adopts NFKMLXPhonemizing)
let backend = try NFKMLXFastSpeech2.voiceBackend(directoryURL: dir, phonemizer: g2p)
// a G2P checkpoint: voiceBackend(directoryURL:g2pWeightsURL:); separate checkpoints: voiceBackend(acousticURL:vocoderURL:vocabularyURL:phonemizer:);
// download: voiceBackend(repo:revision:cacheDirectoryURL:phonemizer:) (espnet/fastspeech2_conformer_with_hifigan when nil)
// a closure phonemizer (Swift only):
let voice = try NFKMLXVoice.voice(acousticURL: acousticURL, vocoderURL: vocoderURL, vocabularyURL: vocabURL); let speech = voice.makeSpeechBackend(phonemize: phonemize)
// composed: NFKMLXVoice(acoustic: NFKMLXFastSpeech2.makeNet(), vocoder: NFKMLXHiFiGAN.makeNet(), vocabulary: symbols), each net filled by its loadWeights
// HiFi-GAN
// no standalone public entry; the net is built and loaded inside NFKMLXVoice.voice(acousticURL:vocoderURL:vocabularyURL:) and NFKMLXTTS.loadWeights(acousticURL:vocoderURL:)
// Kokoro-82M
let backend = try NFKMLXKokoro.backend(directoryURL: dir, voiceName: "af_heart")
// Kokoro per request: NFKParameterVoice names one of NFKMLXKokoro.releaseVoiceNames
// Chatterbox (a release directory; nil voice = the built-in conds.pt; a request's WAV under NFKInputVoiceReference clones that voice)
let backend = try NFKMLXChatterbox.speechBackend(directoryURL: dir, voiceURL: voiceWAV)
// per request: NFKMLXChatterboxParameterKey.exaggeration (0.5) beside NFKParameterTemperature, NFKParameterGuidanceScale, and NFKParameterSeed
// Phonemizers
let g2p = try NFKMLXNeuralG2P.g2p(weightsURL: url, phonemeSymbols: symbols)
// then phonemes(forText:); an application's own phonemizer adopts NFKMLXPhonemizing
// NFKMLXEspeakPhonemizer() when isInstalled; Swift only
// Hand-chained TTS
let backend = try NFKMLXFastSpeech2.ttsBackend(phonemizer: g2p, acousticURL: acousticURL, vocoderURL: vocoderURL, symbols: symbols)
// Swift: let tts = NFKMLXTTS(phonemizer: g2p); try tts.loadWeights(acousticURL: acousticURL, vocoderURL: vocoderURL); let speech = tts.makeSpeechBackend()
// MiniMax Music 3
let backend = try NFKMLXMusic3.backend(directoryURL: dir)
// quantize: try NFKMLXMusic3.quantizeRelease(at: dir, to: outDir)
// Stable Audio Open 1.0
let backend = try NFKMLXStableAudioGenerator.backend(directoryURL: dir)
```

```objc
// FastSpeech2 conformer + paired HiFi-GAN
[NFKMLXFastSpeech2 voiceBackendWithDirectoryURL:dir phonemizer:g2p error:&error]
[NFKMLXFastSpeech2 voiceBackendWithDirectoryURL:dir g2pWeightsURL:g2pURL error:&error]
// Kokoro-82M
[NFKMLXKokoro kokoroBackendWithDirectoryURL:dir voiceName:@"af_heart" error:&error]
// Chatterbox
[NFKMLXChatterbox chatterboxBackendWithDirectoryURL:dir voiceURL:voiceWAV error:&error]
// Phonemizers
[NFKMLXNeuralG2P g2pWithWeightsURL:url phonemeSymbols:symbols error:&error]
// Hand-chained TTS
[NFKMLXFastSpeech2 ttsBackendWithPhonemizer:g2p acousticURL:acousticURL vocoderURL:vocoderURL symbols:symbols error:&error]
// MiniMax Music 3
[NFKMLXMusic3 backendWithDirectoryURL:dir error:&error]
// Stable Audio Open 1.0
[NFKMLXStableAudioGenerator backendWithDirectoryURL:dir error:&error]
```


### Text to image and video


| Model | Entry class | Network | Configuration for the released weights | Registered name / factory | Base backend |
| --- | --- | --- | --- | --- | --- |
| Stable Diffusion 1.5 / 2.1 / SDXL / SDXL-Turbo, with LCM-LoRA, SDXL-Lightning, Hyper-SD, and LCM Dreamshaper | ``NFKMLXBackend``, ``NFKMLXTextToImage`` | ``NFKMLXSDUNet``, ``NFKMLXSDAutoencoder``, ``NFKMLXSDTextEncoderNet``; ``NFKMLXSDLoRA`` merges a released adapter | `NFKMLXSDTextToImageConfiguration.stableDiffusion15` / `.stableDiffusion21` / `.stableDiffusion21V` / `.sdxl` / `.sdxlTurbo` / `.sdxlLightning` / `.latentConsistency15`; `withSampler(_:steps:guidanceScale:)` for an adapter; ``NFKMLXWeightPrecision`` | `stable-diffusion`; `backend(model:directoryURL:)`; ``NFKMLXStableDiffusionModel`` release enum on ``NFKMLXBackend`` | ``NFKMLXDiffusionBackend`` + ``NFKDDIMScheduler`` / ``NFKLCMScheduler`` / ``NFKTCDScheduler`` |
| SD networks | ``NFKMLXSDPipeline`` | ``NFKMLXSDUNet`` + ``NFKMLXSDAutoencoder`` | `NFKMLXSDUNetConfiguration()` (SD 1.5) / `.sdxl` / `.inpainting` / `.marigold` / `.upscaler`; `NFKMLXSDVAEConfiguration.stableDiffusion` / `.upscaler` / `.flux` | — | Swift API |
| IP-Adapter | ``NFKMLXIPAdapterImageProjection``, ``NFKMLXIPAdapterAttention`` | — | `imageEmbedDim` 1024 → `crossAttentionDim` 768, 4 tokens | — | Swift API |
| Z-Image / Z-Image-Turbo | ``NFKMLXZImageGenerator``, ``NFKMLXZImagePipeline`` | ``NFKMLXZImageTransformerNet`` + ``NFKMLXSDAutoencoder`` (`.flux`) + Qwen3-4B | `NFKMLXZImageConfiguration`, the Flux VAE configuration, and `NFKMLXFlowMatchConfiguration.zImageTurbo` / `.zImage`, each read from the release | `z-image`; `backend(directoryURL:)`, `generator(directoryURL:residency:)` | ``NFKMLXGenerationBackend`` (``NFKMLXFlowMatchScheduler``) |
| SANA | ``NFKMLXSANAGenerator``, ``NFKMLXSANAPipeline`` | ``NFKMLXSANATransformerNet`` + ``NFKMLXDCAutoencoderNet`` | `NFKMLXSANAConfiguration.base`, or `configuration(fromHuggingFace:)`; `NFKMLXDCAEConfiguration.sana`; `NFKMLXDPMSolverConfiguration.sana`; caption from ``NFKMLXGemma2Net`` | `sana`; `backend(directoryURL:)`, `generator(directoryURL:residency:)`; the pipeline alone: `pipeline(directoryURL:)`, `init(transformer:vae:)` | ``NFKMLXGenerationBackend`` (``NFKMLXDPMSolverScheduler``) |
| LTX-Video 0.9.0 | ``NFKMLXLTXVideoGenerator``, ``NFKMLXLTXPipeline`` | ``NFKMLXLTXTransformerNet`` + ``NFKMLXLTXVideoVAENet`` | `NFKMLXLTXTransformerConfiguration.base`; `NFKMLXLTXVAEConfiguration.base`; `NFKMLXFlowMatchConfiguration.ltxVideo`; `NFKMLXT5Configuration.xxl` | `ltx-video-0.9.0`; `backend(directoryURL:)`, `generator(directoryURL:residency:)` | ``NFKMLXGenerationBackend`` (``NFKMLXFlowMatchScheduler``) |
| LTX-2 (audio-video transformer) | — | ``NFKMLXLTX2TransformerNet`` | `NFKMLXLTX2Configuration.ltx23` / `.ltx25` | Swift API (the forward takes `MLXArray`) | — |
| Wan 2.1 T2V / Wan 2.2 TI2V-5B | ``NFKMLXWanVideoGenerator``, ``NFKMLXWanPipeline`` | ``NFKMLXWanTransformerNet`` + ``NFKMLXWanVideoVAENet`` | `NFKMLXWanConfiguration.base`; `NFKMLXWanVAEConfiguration.wan22` / `.wan21`; `NFKMLXUniPCConfiguration.wan`; `NFKMLXT5Configuration.umt5XXL` | `wan`; `backend(directoryURL:)`, `generator(directoryURL:residency:)` | ``NFKMLXGenerationBackend`` (``NFKMLXUniPCScheduler``) |
| Qwen-Image 2.1 | ``NFKMLXQwenImageGenerator``, ``NFKMLXQwenImagePipeline`` | ``NFKMLXQwenImageNet`` + ``NFKMLXWanVideoVAENet`` (`.qwenImage21`) | `NFKMLXQwenImageConfiguration.base`, `NFKMLXWanVAEConfiguration.qwenImage21`, and `NFKMLXFlowMatchConfiguration.qwenImage21`, each read from the release | `qwen-image-2.1`; `backend(directoryURL:)`, `generator(directoryURL:residency:)` | ``NFKMLXGenerationBackend`` (``NFKMLXFlowMatchScheduler``) |
| Wan 2.2 Animate | ``NFKMLXWanAnimate`` | ``NFKMLXWanAnimateNet`` | `NFKMLXWanAnimateConfiguration.base` (14B) / `.tiny`; ``NFKMLXWanAnimateKVCache`` holds the reference pass | `let dit = NFKMLXWanAnimate.makeNet(.base); try NFKMLXWanAnimate.loadWeights(into: dit, fromDirectory: dir)`<br>`let cache = NFKMLXWanAnimateKVCache(layerCount: 40)`; Swift only — the configuration is a Swift struct | — (released weights exceed a workstation) |
| Stable Diffusion 3 / 3.5 | ``NFKMLXSD3Generator``, ``NFKMLXSD3Pipeline`` | ``NFKMLXSD3TransformerNet`` + ``NFKMLXSDAutoencoder`` + CLIP-L, CLIP-G and T5-XXL | `NFKMLXSD3Configuration`, the autoencoder configuration, and the flow shift, each read from the release | `sd3` · `sd3-medium` · `sd3.5-large` · `sd3.5-large-turbo` (4 steps, guidance 0); `backend(directoryURL:)`, `generator(directoryURL:residency:)` | ``NFKMLXGenerationBackend`` (``NFKMLXFlowMatchScheduler``) |
| FLUX.1 (text-to-image) | ``NFKMLXFlux`` | ``NFKMLXFluxTextEncoder`` + ``NFKMLXFluxPipeline`` | read from the diffusers release directory | `flux.1-schnell`; `backend(directoryURL:)`, `flux(directoryURL:)` | ``NFKMLXGenerationBackend`` (``NFKMLXFlowMatchScheduler``) |
| FLUX.1 (transformer + sampler) | ``NFKMLXFluxPipeline`` | ``NFKMLXFluxTransformerNet`` + ``NFKMLXSDAutoencoder`` (`.flux`) | `NFKMLXFluxConfiguration.dev` / `.schnell`; `NFKMLXFlowMatchConfiguration.flux` / `.fluxSchnell`; text from CLIP-L (pooled) + T5-XXL | Swift API (`generate(promptEmbeds:pooled:…)`) | ``NFKMLXFlowMatchScheduler`` |
| SD3 ControlNet | ``NFKMLXSD3ControlNetGenerator``, ``NFKMLXSD3ControlNetPipeline`` | ``NFKMLXSD3ControlNetNet`` + ``NFKMLXSD3TransformerNet`` + ``NFKMLXSDAutoencoder`` | `NFKMLXSD3ControlNetConfiguration.instantXMedium` / `.stabilitySD35Large`; a spatial control image | `sd3-controlnet`; `backend(directoryURL:controlNetDirectoryURL:)`, `generator(directoryURL:controlNetDirectoryURL:)`; the pipeline alone takes the embeddings from ``NFKMLXSD3Generator/promptEmbeddings(for:)`` | ``NFKMLXGenerationBackend`` (``NFKMLXFlowMatchScheduler``) |
| FLUX.2 [klein] (text-to-image, editing, inpainting) | ``NFKMLXFlux2`` | ``NFKMLXFlux2TransformerNet`` + ``NFKMLXSDAutoencoder`` (`.flux2`) + ``NFKMLXFlux2LatentCodec`` + a Qwen3 | `NFKMLXFlux2Configuration.klein4B` / `.klein9B` / `.dev`; `NFKMLXFlowMatchConfiguration.flux2` | `flux.2-klein-4b`; `backend(directoryURL:)`, `flux2(directoryURL:)` | ``NFKMLXGenerationBackend`` (``NFKMLXFlowMatchScheduler``) |
| FLUX.1 ControlNet | ``NFKMLXFluxControlNetGenerator``, ``NFKMLXFluxControlNetPipeline`` | ``NFKMLXFluxControlNetNet`` + ``NFKMLXFluxTransformerNet`` + ``NFKMLXSDAutoencoder`` (`.flux`) | `NFKMLXFluxControlNetConfiguration.unionPro` / `.single`; a spatial control image | `flux.1-controlnet`; `backend(directoryURL:controlNetDirectoryURL:)`, `generator(directoryURL:controlNetDirectoryURL:)` | ``NFKMLXGenerationBackend`` (``NFKMLXFlowMatchScheduler``) |
| Reference diffusion stand-ins | ``NFKMLXReferenceModels`` | oracle `denoise` closures | — | `diffusion-upscaler` · `diffusion-depth` · `diffusion-inpaint` · `diffusion-controlnet` | ``NFKMLXDiffusionBackend`` |

```swift
// Stable Diffusion 1.5 / 2.1 / SDXL / SDXL-Turbo
let backend = try NFKMLXTextToImage.backend(model: .stableDiffusion15, directoryURL: dir)
// download on first use: NFKMLXBackend(model: .stableDiffusion15) · [[NFKMLXBackend alloc] initWithModel:NFKMLXStableDiffusionModelStableDiffusion15]; .stableDiffusion21Base / .sdxlTurbo / .sdxlBase / .lcmDreamshaperV7
// a released LoRA merged into the UNet: LCM-LoRA on SD 1.5 under the latent consistency sampler, SDXL-Lightning on SDXL base under trailing DDIM
let lcm = try NFKMLXTextToImage.backend(configuration: .stableDiffusion15.withSampler(.lcm), directoryURL: dir, adapterURL: lcmLoRA)
let lightning = try NFKMLXTextToImage.backend(configuration: .sdxlLightning, directoryURL: sdxlDir, adapterURL: lightningLoRA)
// SD networks
let pipeline = NFKMLXSDPipeline(unet: .stableDiffusion, vae: .stableDiffusion); try pipeline.loadWeights(unetURL: unetURL, vaeURL: vaeURL)
// Swift only
// IP-Adapter
let projection = NFKMLXIPAdapterImageProjection(); let attention = NFKMLXIPAdapterAttention(queryDim: 320, crossAttentionDim: 768, heads: 8, headDim: 40)
// Stable Diffusion 3 / 3.5 (text-to-image; registered as sd3)
let backend = try NFKMLXSD3Generator.backend(directoryURL: releaseDirectory)
// the generator:
let sd3 = try NFKMLXSD3Generator.generator(directoryURL: releaseDirectory, residency: .automatic)
let image = try sd3.image(forPrompt: "a red fox in the snow", width: 1024, height: 1024, seed: 0)
// FLUX.1 (text-to-image; registered as flux.1-schnell; the [dev] schedule through backend(directoryURL:schedule:), Swift only)
let backend = try NFKMLXFlux.backend(directoryURL: dir)
// the model:
let flux = try NFKMLXFlux.flux(directoryURL: dir); let image = try flux.image(forPrompt: "an astronaut on the moon")
// FLUX.1 (transformer + sampler)
let dit = NFKMLXFluxTransformerNet(.dev); try NFKMLXFluxTransformerNet.loadWeights(into: dit, from: dir); let pipeline = NFKMLXFluxPipeline(transformer: dit, vae: vae)
// Swift only
// SD3 ControlNet (registered as sd3-controlnet; the control image under NFKInputControl)
let backend = try NFKMLXSD3ControlNetGenerator.backend(directoryURL: sd3Dir, controlNetDirectoryURL: cnDir)
// the generator: generator(directoryURL:controlNetDirectoryURL:), then cgImage(forPrompt:negativePrompt:controlImage:width:height:seed:)
// the pipeline alone (Swift only):
let cn = NFKMLXSD3ControlNetNet(.instantXMedium); try NFKMLXSD3ControlNetNet.loadWeights(into: cn, from: cnDir); let pipeline = NFKMLXSD3ControlNetPipeline(transformer: dit, controlnet: cn, vae: vae)
let (promptEmbeds, pooled) = try NFKMLXSD3Generator.generator(directoryURL: sd3Dir).promptEmbeddings(for: "a red fox")
// FLUX.1 ControlNet (registered as flux.1-controlnet; the control image under NFKInputControl, the mode under NFKMLXControlNetParameterKey.controlMode)
let backend = try NFKMLXFluxControlNetGenerator.backend(directoryURL: fluxDir, controlNetDirectoryURL: cnDir)
// the generator: generator(directoryURL:controlNetDirectoryURL:), then cgImage(forPrompt:controlImage:controlMode:width:height:seed:)
// the pipeline alone (Swift only):
let cn = NFKMLXFluxControlNetNet(.unionPro); try NFKMLXFluxControlNetNet.loadWeights(into: cn, from: cnDir); let pipeline = NFKMLXFluxControlNetPipeline(transformer: dit, controlnet: cn, vae: vae)
// FLUX.2 [klein] (text-to-image, editing, inpainting; registered as flux.2-klein-4b)
let backend = try NFKMLXFlux2.backend(directoryURL: releaseDirectory)
// the model:
let flux2 = try NFKMLXFlux2.flux2(directoryURL: releaseDirectory)
let image = try flux2.image(forPrompt: "a red fox in the snow", width: 1024, height: 1024, seed: 0)
// Z-Image (text-to-image; Turbo by default; registered as z-image)
let backend = try NFKMLXZImageGenerator.backend(directoryURL: releaseDirectory)
// the generator:
let zImage = try NFKMLXZImageGenerator.generator(directoryURL: releaseDirectory, residency: .automatic)
let image = try zImage.image(forPrompt: "a red fox in the snow", width: 1024, height: 1024, seed: 0)
// SANA (registered as sana)
let backend = try NFKMLXSANAGenerator.backend(directoryURL: dir)
// the generator: NFKMLXSANAGenerator.generator(directoryURL:), then image(forPrompt:…)
// the pipeline alone (Swift only, MLXArray in and out):
let sana = try NFKMLXSANAPipeline.pipeline(directoryURL: dir)
let image = sana.generate(promptEmbeds: caption, negativeEmbeds: nil, latentHeight: 32, latentWidth: 32)   // caption from NFKMLXGemma2Net
// LTX-Video (registered as ltx-video-0.9.0)
let backend = try NFKMLXLTXVideoGenerator.backend(directoryURL: dir)
let vae = try NFKMLXLTXVideoVAE.vae(configuration: .base, weightsURL: vaeURL); let t5 = try NFKMLXT5Encoder.encoder(configuration: .xxl, directory: t5Dir)
let ltx = try NFKMLXLTXVideoGenerator.generator(directoryURL: dir)   // loads the whole release, text encoder included
let pipeline = try NFKMLXLTXPipeline.pipeline(directoryURL: dir)     // the DiT and VAE alone; or NFKMLXLTXPipeline(transformer:vae:scalingFactor:) over NFKMLXLTXTransformer.makeNet / NFKMLXLTXVideoVAE.makeNet
// LTX-2 (audio-video transformer)
let dit = NFKMLXLTX2TransformerNet(.ltx25); try NFKMLXLTX2TransformerNet.loadWeights(into: dit, from: transformerDir)
// Swift only — the forward denoises the video and audio latents together and returns both
// Wan 2.2 Animate
let animate = NFKMLXWanAnimate.makeNet(.base); try NFKMLXWanAnimate.loadWeights(into: animate, fromDirectory: dir)
let cache = NFKMLXWanAnimateKVCache(layerCount: 40)
// Swift only — NFKMLXWanAnimateConfiguration is a Swift struct

// Qwen-Image 2.1 (text-to-image; registered as qwen-image-2.1)
let backend = try NFKMLXQwenImageGenerator.backend(directoryURL: releaseDirectory)
// the generator:
let qwenImage = try NFKMLXQwenImageGenerator.generator(directoryURL: releaseDirectory, residency: .automatic)
let picture = try qwenImage.image(forPrompt: "a red fox in the snow", width: 1024, height: 1024, seed: 0)

// Wan (registered as wan)
let backend = try NFKMLXWanVideoGenerator.backend(directoryURL: dir)
let dit = NFKMLXWanTransformerNet(.base); let vae = NFKMLXWanVideoVAENet(.wan22)
let wan = try NFKMLXWanVideoGenerator.generator(directoryURL: dir)   // loads the whole release, text encoder included
let pipeline = try NFKMLXWanPipeline.pipeline(directoryURL: dir)     // the DiT, VAE, and schedule alone; or NFKMLXWanPipeline(transformer:vae:latentsMean:latentsStd:schedule:)
// Reference diffusion stand-ins
NFKMLXReferenceModels.registerAll(); let backend = try NFKMLXModelRegistry.backend(named: "diffusion-controlnet", weightsURL: nil)
```

```objc
// Stable Diffusion 1.5 / 2.1 / SDXL / SDXL-Turbo
[NFKMLXTextToImage backendWithModel:NFKMLXStableDiffusionModelStableDiffusion15 directoryURL:dir error:&error]
// a released LoRA merged into the UNet, with the sampler it needs
[NFKMLXTextToImage backendWithModel:NFKMLXStableDiffusionModelStableDiffusion15 directoryURL:dir adapterURL:lcmLoRA adapterScale:1 sampler:NFKMLXSDSamplerLCM error:&error]
[NFKMLXTextToImage backendWithModel:NFKMLXStableDiffusionModelSdxlBase directoryURL:sdxlDir adapterURL:lightningLoRA adapterScale:1 sampler:NFKMLXSDSamplerDDIMTrailing error:&error]
// Stable Diffusion 3 / 3.5
[NFKMLXSD3Generator backendWithDirectoryURL:dir error:&error]
// the generator, then imageForPrompt:negativePrompt:width:height:seed:error:
[NFKMLXSD3Generator generatorWithDirectoryURL:dir residency:NFKMLXResidencyAutomatic error:&error]
// FLUX.1 (text-to-image)
[NFKMLXFlux backendWithDirectoryURL:dir error:&error]
// the model, then imageForPrompt:width:height:seed:error:
[NFKMLXFlux fluxWithDirectoryURL:dir error:&error]
// SD3 ControlNet
[NFKMLXSD3ControlNetGenerator backendWithDirectoryURL:sd3Dir controlNetDirectoryURL:cnDir error:&error]
// the generator, then imageForPrompt:negativePrompt:controlImage:width:height:seed:error:
[NFKMLXSD3ControlNetGenerator generatorWithDirectoryURL:sd3Dir controlNetDirectoryURL:cnDir error:&error]
// FLUX.1 ControlNet
[NFKMLXFluxControlNetGenerator backendWithDirectoryURL:fluxDir controlNetDirectoryURL:cnDir error:&error]
// the generator, then imageForPrompt:controlImage:controlMode:width:height:seed:error:
[NFKMLXFluxControlNetGenerator generatorWithDirectoryURL:fluxDir controlNetDirectoryURL:cnDir error:&error]
// FLUX.2 [klein]
[NFKMLXFlux2 backendWithDirectoryURL:dir error:&error]
// the model, then imageForPrompt:negativePrompt:width:height:seed:error:
[NFKMLXFlux2 flux2WithDirectoryURL:dir error:&error]
// Z-Image
[NFKMLXZImageGenerator backendWithDirectoryURL:dir error:&error]
// the generator, then imageForPrompt:negativePrompt:width:height:seed:error:
[NFKMLXZImageGenerator generatorWithDirectoryURL:dir residency:NFKMLXResidencyAutomatic error:&error]
// SANA
[NFKMLXSANAGenerator backendWithDirectoryURL:dir error:&error]
// the generator, then imageForPrompt:negativePrompt:width:height:seed:error:
[NFKMLXSANAGenerator generatorWithDirectoryURL:dir error:&error]
// LTX-Video
[NFKMLXLTXVideoGenerator backendWithDirectoryURL:dir error:&error]
// Wan
[NFKMLXWanVideoGenerator backendWithDirectoryURL:dir error:&error]
// Qwen-Image 2.1
[NFKMLXQwenImageGenerator backendWithDirectoryURL:dir error:&error]
// the generator, then imageForPrompt:negativePrompt:width:height:seed:error:
[NFKMLXQwenImageGenerator generatorWithDirectoryURL:dir residency:NFKMLXResidencyAutomatic error:&error]
// Reference diffusion stand-ins
[NFKMLXReferenceModels registerAll]; [NFKMLXModelRegistry backendNamed:@"diffusion-controlnet" weightsURL:nil error:&error]
```


### Bring-your-own bases and reference stand-ins

| Base | Closure | Registered reference |
| --- | --- | --- |
| ``NFKMLXModuleBackend`` | `(MLXArray) -> MLXArray` | — |
| ``NFKMLXMattingBackend`` | `(plate, hint) -> [H,W,4]`, ``NFKMattingConfiguration`` | `green-screen-keyer` |
| ``NFKMLXTensorBackend`` | `[String: MLXArray] -> [String: MLXArray]`, ``NFKMLXTensorConfiguration`` | — |
| ``NFKMLXSpeechBackend`` | `(String) -> MLXArray`, ``NFKMLXSpeechConfiguration`` | `tone-speech` |
| ``NFKMLXDiffusionBackend`` | encode / denoise / decode + ``NFKDiffusionScheduler``, ``NFKDiffusionConfiguration`` | the four `diffusion-*` pipelines |
| ``NFKMLXVideoBackend`` | `([MLXArray]) -> [MLXArray]`, ``NFKMLXVideoConfiguration`` | — |


## Topics

### Related

- <doc:DiffusionAndSchedulers>
- <doc:WeightsAndConversion>
