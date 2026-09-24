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
| NAFNet | ``NFKMLXNAFNet`` | `NFKMLXNAFNetNet` | ``NFKMLXNAFNetVariant`` `.sidd` / `.goPro` / `.reds` / `.siddWidth64` / `.goProWidth64` (`NFKMLXNAFNetConfiguration.sidd` …) | `nafnet` | ``NFKMLXModuleBackend`` |
| Zero-DCE | ``NFKMLXZeroDCE`` | ``NFKMLXZeroDCENet`` | fixed geometry (seven 3×3 convolutions) | `zero-dce` | ``NFKMLXModuleBackend`` |
| Zero-DCE++ | ``NFKMLXZeroDCEPlus`` | `NFKMLXZeroDCEPlusNet` | fixed geometry (seven depthwise-separable convolutions, one shared curve, scale factor 12) | `zero-dce-plus` | ``NFKMLXModuleBackend`` |
| Fast style transfer | ``NFKMLXStyleTransfer`` | `NFKStyleTransferNet` | fixed geometry; one style per checkpoint | `fast-style-transfer` | ``NFKMLXModuleBackend`` |
| AdaIN | ``NFKMLXAdaIN`` | `NFKMLXAdaINNet` | fixed geometry (VGG-19 through relu4_1 and its mirrored decoder); style image under `NFKInputControl` | `adain` | ``NFKMLXModuleBackend`` |
| Colorizer ECCV-16 | ``NFKMLXColorizer`` | `NFKMLXColorizerNet` | `NFKMLXColorizerConfiguration.eccv16` | `colorizer-eccv16` | ``NFKMLXModuleBackend`` |
| Colorizer SIGGRAPH-17 | ``NFKMLXSiggraphColorizer`` | `NFKMLXSiggraphNet` | fixed geometry (four-channel input, hints optional) | `colorizer-siggraph17` | ``NFKMLXModuleBackend`` |
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
| Depth Anything V2 | ``NFKMLXDepthAnything`` | `NFKMLXDepthAnythingNet` | ``NFKMLXDepthVariant`` `.small` / `.base` / `.large` (`NFKMLXDepthConfiguration.small` …) | `depth-anything-v2-small` · `-base` · `-large` | ``NFKMLXModuleBackend`` |
| Depth Anything 3 | ``NFKMLXDepthAnything3``; ``NFKMLXDepth3Estimator`` for the camera and the ray map | `NFKMLXDepthAnything3Net` | ``NFKMLXDepth3Variant`` `.small` / `.base` / `.large` (`NFKMLXDepth3Configuration.small` …) | `depth-anything-3-small` · `-base` · `-large` | ``NFKMLXModuleBackend`` |
| Marigold depth | ``NFKMLXMarigold`` | ``NFKMLXSDUNet`` + ``NFKMLXSDAutoencoder`` | `NFKMLXSDUNetConfiguration.marigold` | `marigold-depth` | ``NFKMLXDiffusionBackend`` |
| SegFormer | ``NFKMLXSegFormer`` | ``NFKMLXSegFormerNet`` | `NFKMLXSegFormerConfiguration.mitB0`; `network(weightsURL:classCount:)` for a custom head | `segformer-b0` | ``NFKMLXModuleBackend`` |
| DeepLabV3 | ``NFKMLXDeepLab`` | `NFKMLXDeepLabNet` over `NFKMLXResNetBackbone` | `NFKMLXDeepLabConfiguration.base` (`NFKMLXResNetConfiguration.deepLab`) | `deeplabv3` | ``NFKMLXModuleBackend`` |
| BiSeNet V1 | ``NFKMLXBiSeNet`` | `NFKMLXBiSeNetNet` | `NFKMLXBiSeNetConfiguration.base` (ResNet-18 context path) | `bisenet` | ``NFKMLXModuleBackend`` |
| BiSeNet V2 | ``NFKMLXBiSeNetV2`` | `NFKMLXBiSeNetV2Net` | fixed geometry (older pixel-shuffle head) | `bisenet-v2` | ``NFKMLXModuleBackend`` |

```swift
// Depth Anything V2
let backend = try NFKMLXDepthAnything.backend(variant: .small, weightsURL: url)
// .base / .large · …VariantBase / …VariantLarge
// Depth Anything 3 (DA3-SMALL)
let backend = try NFKMLXDepthAnything3.backend(variant: .small, weightsURL: url)
// camera and rays: NFKMLXDepth3Estimator.estimator(variant: .small, weightsURL: url)
// Marigold depth
let backend = try NFKMLXMarigold.backend(unetWeightsURL: unetURL, vaeWeightsURL: vaeURL, textContextURL: contextURL)
// SegFormer
let backend = try NFKMLXSegFormer.backend(weightsURL: url)
// custom classes: try NFKMLXSegFormer.network(weightsURL: url, classCount: 4)
// DeepLabV3
let backend = try NFKMLXDeepLab.backend(weightsURL: url)
// BiSeNet V1
let backend = try NFKMLXBiSeNet.backend(weightsURL: url)
// BiSeNet V2
let backend = try NFKMLXBiSeNetV2.backend(weightsURL: url)
```

```objc
// Depth Anything V2
[NFKMLXDepthAnything backendWithVariant:NFKMLXDepthVariantSmall weightsURL:url error:&error]
// Depth Anything 3
[NFKMLXDepthAnything3 backendWithVariant:NFKMLXDepth3VariantSmall weightsURL:url error:&error]
// camera and rays: [NFKMLXDepth3Estimator estimatorWithVariant:NFKMLXDepth3VariantSmall weightsURL:url error:&error]
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


### Matting, segmentation, and faces


| Model | Entry class | Network | Configuration for the released weights | Registered name | Base backend |
| --- | --- | --- | --- | --- | --- |
| U²-Net | ``NFKMLXU2Net`` | `NFKMLXU2NetNet` | ``NFKMLXU2NetVariant`` `.full` / `.light` | `u2net` · `u2netp` | ``NFKMLXMattingBackend`` |
| IS-Net (DIS) | ``NFKMLXISNet`` | `NFKMLXISNetNet` | fixed geometry (stride-2 stem, six Residual U-block stages, six side maps) | `isnet` | ``NFKMLXMattingBackend`` |
| Robust Video Matting | ``NFKMLXRVM`` | `NFKMLXRVMNet` | ``NFKMLXRVMVariant`` `.mobileNetV3` / `.resNet50` (`NFKMLXRVMConfiguration.large` / `.resNet50`); `downsampleRatio` for the guided-filter path | `robust-video-matting` · `-resnet50` | ``NFKMLXMattingBackend`` (single frame) / `NFKMLXRVMNet.forward` (video) |
| MODNet | ``NFKMLXMODNet`` | `NFKMLXMODNetNet` | `NFKMLXMODNetConfiguration.base` | `modnet` | ``NFKMLXMattingBackend`` |
| BiRefNet | ``NFKMLXBiRefNet`` | `NFKMLXBiRefNetModel` (Swin-v1-L backbone + ASPPDeformable decoder) | released `ZhengPeng7/BiRefNet` (`model.safetensors`, MIT); resizes to 1024 | `birefnet` | ``NFKMLXMattingBackend`` |
| SAM | ``NFKMLXSAM`` | `NFKMLXSAMNet` | ``NFKMLXSAMVariant`` `.vitB` / `.vitL` / `.vitH` (`NFKMLXSAMConfiguration.vitB` …); `.compact` for tests | `sam` | ``NFKMLXMattingBackend`` (point under `NFKSAMPointKey`) |
| SAM 3 / SAM 3.1 | ``NFKMLXSAM3`` | ``NFKMLXSAM3ImageModel`` | `NFKMLXSAM3Configuration.base`, `NFKMLXSAM3TextConfiguration.base`, `NFKMLXSAM3DetectorConfiguration.base` | — | Swift API (`MLXArray`) |
| SAM 2 / SAM 2.1 | ``NFKMLXSAM2`` | ``NFKMLXSAM2TrackerNet`` | `NFKMLXSAM2Variant.tiny` / `.small` / `.basePlus` / `.large`; `NFKMLXSAM2Release.sam2` / `.sam21` | `sam2` | ``NFKMLXMattingBackend`` |
| RetinaFace | ``NFKMLXRetinaFace`` | `NFKMLXRetinaFaceNet` | `NFKMLXRetinaFaceConfiguration()` = mobile0.25 | `retinaface-mobile025` | detection backend; `detector(weightsURL:)` for landmarks |
| Face alignment | ``NFKMLXFaceAlignment`` | — | ``NFKMLXRetinaFaceDetector`` (default) or ``NFKMLXVisionFaceDetector`` | — | used by ``NFKMLXPhotoFaceBackend`` |

```swift
// U²-Net
let backend = try NFKMLXU2Net.backend(variant: .full, weightsURL: url)
// IS-Net (DIS)
let backend = try NFKMLXISNet.backend(weightsURL: url)
// .light · …VariantLight
// Robust Video Matting
let backend = try NFKMLXRVM.backend(weightsURL: url)
// MODNet
let backend = try NFKMLXMODNet.backend(weightsURL: url)
// BiRefNet
let backend = try NFKMLXBiRefNet.backend(weightsURL: url)
// SAM
let backend = try NFKMLXSAM.backend(variant: .vitB, weightsURL: url)
// SAM 3 / SAM 3.1
let sam3 = try NFKMLXSAM3.makeImageModel(fromHuggingFace: configURL)
try NFKMLXSAM3.loadWeights(into: sam3, from: url)
let found = sam3.detect(image: plate, tokens: ids, valid: mask)
// fine-tune: let detector = try NFKMLXSAM3.network(weightsURL: url, configuration: config)
//            try NFKMLXSAM3.fineTune(detector, examples: myImages, trainable: .detector, steps: 300)

// SAM 2 / SAM 2.1
let sam2 = try NFKMLXSAM2.backend(variant: .tiny, release: .sam21, weightsURL: url)
// fine-tune: let net = try NFKMLXSAM2.network(weightsURL: url, variant: .tiny, release: .sam21)
//            try NFKMLXSAM2.fineTune(net, examples: myFrames, trainable: .maskDecoder, steps: 300)
// video: let net = NFKMLXSAM2.makeTracker(variant: .tiny, release: .sam21)
//        try NFKMLXSAM2.loadWeights(into: net, from: url)
//        let session = NFKMLXSAM2TrackerSession(frameCount: frames.count)
// RetinaFace
let backend = try NFKMLXRetinaFace.backend(weightsURL: url)
// landmarks: let detector = try NFKMLXRetinaFace.detector(weightsURL: url) then detector.faces(in: image) · [NFKMLXRetinaFace detectorWithWeightsURL:url confidenceThreshold:0.8 suppressionThreshold:0.4 error:&error]
// Face alignment
let crop = try NFKMLXFaceAlignment.alignedCrop(from: image, face: face)
// detectors: NFKMLXRetinaFaceDetector(weightsURL:) / NFKMLXVisionFaceDetector()
```

```objc
// U²-Net
[NFKMLXU2Net backendWithVariant:NFKMLXU2NetVariantFull weightsURL:url error:&error]
// IS-Net (DIS)
[NFKMLXISNet backendWithWeightsURL:url error:&error]
// Robust Video Matting
[NFKMLXRVM backendWithWeightsURL:url error:&error]
// MODNet
[NFKMLXMODNet backendWithWeightsURL:url error:&error]
// BiRefNet
[NFKMLXBiRefNet backendWithWeightsURL:url error:&error]
// SAM
[NFKMLXSAM backendWithVariant:NFKMLXSAMVariantVitB weightsURL:url error:&error]
// RetinaFace
[NFKMLXRetinaFace backendWithWeightsURL:url error:&error]
```


### Detection and pose


| Model | Entry class | Network | Configuration for the released weights | Registered name | Base backend |
| --- | --- | --- | --- | --- | --- |
| YOLOv8 | ``NFKMLXYOLO`` | `NFKMLXYOLONet` | ``NFKMLXYOLOVariant`` `.nano` / `.small` / `.medium` / `.large` / `.extraLarge` (`NFKMLXYOLOConfiguration.base`, `.small` …) | `yolo` | ``NFKMLXYOLOBackend`` (`labels:`) |
| YOLOv9 · v10 · 11 · v12 · 26 | ``NFKMLXYOLOGenerations`` | `NFKMLXYOLOGenerationNet` | ``NFKMLXYOLORelease`` — every released size of each generation | the checkpoint stem: `yolov9t` … `yolo26x` | ``NFKMLXYOLOGenerationBackend`` (`labels:`) |
| RT-DETR | ``NFKMLXRTDetr`` | ``NFKMLXRTDetrNet`` | ``NFKMLXRTDetrVariant`` `.r18vd` / `.r34vd` / `.r50vd` / `.r101vd` | `rtdetr` · `rtdetr-r18vd` · `-r34vd` · `-r101vd` | ``NFKMLXRTDetrBackend`` (`labels:`) |
| RT-DETRv2 | ``NFKMLXRTDetr`` | ``NFKMLXRTDetrNet`` | ``NFKMLXRTDetrVariant`` `.v2R18VD` / `.v2R34VD` / `.v2R50VD` / `.v2R101VD` | `rtdetr-v2-r18vd` · `-r34vd` · `-r50vd` · `-r101vd` | ``NFKMLXRTDetrBackend`` (`labels:`); `decoderMethod` (``NFKMLXRTDetrSamplingMethod``) and `decoderOffsetScale` carry v2's sampling |
| RF-DETR | ``NFKMLXRFDetr`` | ``NFKMLXRFDetrNet`` | ``NFKMLXRFDetrVariant`` `.nano` / `.small` / `.medium` / `.base` / `.large` | `rf-detr` · `rf-detr-nano` · `-small` · `-medium` · `-large` | ``NFKMLXRFDetrBackend`` (`labels:`); Roboflow naming converted on device |
| SimpleBaseline pose | ``NFKMLXPose`` | `NFKMLXPoseNet` over `NFKMLXResNetBackbone` | `NFKMLXPoseConfiguration.simpleBaseline` (ResNet-50, 256×192) | `pose-simplebaseline` | ``NFKMLXPoseBackend`` (`jointNames:`) |
| ViTPose | ``NFKMLXVitPose`` | `NFKMLXVitPoseNet` | ``NFKMLXVitPoseVariant`` `.baseSimple` / `.base` (ViT-B 256×192; ``NFKMLXVitPoseDecoder`` `.simple` / `.classic`) | `vitpose-base-simple` · `vitpose-base` | ``NFKMLXVitPoseBackend`` (`jointNames:`); `backend(directoryURL:jointNames:)` reads a release's own `config.json` |
| Table Transformer | ``NFKMLXTableTransformer`` | `NFKMLXTableTransformerNet` | read from the release's `config.json` (geometry + `id2label`) | `backend(directoryURL:)`; retargeted: `network(directoryURL:labels:)`, `fineTune`, `save(_:toDirectoryURL:release:)` | ``NFKMLXTableTransformerBackend``; table-structure recognition (Table Transformer, Microsoft), no NMS |

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
// RF-DETR
let backend = try NFKMLXRFDetr.backend(weightsURL: url, labels: nil)
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
| CLIP probe | ``NFKMLXCLIPProbe`` | linear head over frozen CLIP | trained on device (`trainProbe`) | `clip-probe` | ``NFKMLXCLIPProbeBackend`` |
| SigLIP 2 | ``NFKMLXSigLIP2`` | `NFKMLXSigLIP2Net` (`NFKSigLIP2VisionNet`, `NFKSigLIP2TextNet`) | ``NFKMLXSigLIP2Variant`` (every fixed-resolution release: base / large / so400m / giant-opt at their patch sizes and resolutions) | `siglip2-base-patch16-224` and each release's own name | ``NFKMLXSigLIP2Backend`` |
| Qwen3-Embedding-0.6B | ``NFKMLXQwen3Embedding`` | ``NFKMLXLanguageNet`` | ``NFKMLXLanguageConfiguration`` read from `config.json`; ``NFKMLXTextEmbedderConfiguration`` (`.lastToken`, appended `<\|endoftext\|>`, Matryoshka `dimensions`) | `backend(directoryURL:)` | ``NFKMLXTextEmbeddingBackend`` |
| EmbeddingGemma-300M | ``NFKMLXEmbeddingGemma`` | ``NFKMLXGemma3EncoderNet`` | `NFKMLXGemma3EncoderConfiguration.embeddingGemma300M` + the `2_Dense` / `3_Dense` projections | `backend(directoryURL:)` (`embeddinggemma-300m`) | ``NFKMLXTextEmbeddingBackend`` |
| gte-reranker-modernbert-base | ``NFKMLXModernBERTReranker`` | ``NFKMLXModernBertRerankerNet`` | `NFKMLXModernBertConfiguration.gteReranker` | `reranker(directoryURL:)` | scoring object (`scores(query:documents:)`) |
| Laya (typed decisions) | ``NFKMLXLaya`` | ``NFKMLXLayaNet`` | `NFKMLXLayaConfiguration.large` / `.multilingual`, or the variant directory | Swift API | ``NFKMLXLayaBackend`` (`backendWithDirectoryURL:error:`, or `backendWithVariant:revision:cacheDirectoryURL:error:` to download) |
| open-jev-deberta (typed decisions) | ``NFKMLXOpenJevDeBERTa`` | ``NFKMLXOpenJevDeBERTaNet`` (``NFKMLXDeBERTaV2Net`` + scoring head) | `NFKMLXOpenJevDeBERTaConfiguration.v3Large`, or the release directory | Swift API | ``NFKMLXDecisionBackend`` (`backendWithDirectoryURL:error:`, or `backendWithRevision:cacheDirectoryURL:error:` to download) |
| Open-Jev 2B / 9B / 27B (typed decisions) | ``NFKMLXOpenJev`` | ``NFKMLXOpenJevNet`` (``NFKMLXHybridLanguageNet`` + LoRA + scalar head) | read from the release's `checkpoint` directory and its base | Swift API | ``NFKMLXDecisionBackend`` (`backendWithVariant:revision:cacheDirectoryURL:error:`) |
| chronos-bolt-base | ``NFKMLXChronos`` | ``NFKMLXChronosNet`` | `NFKMLXChronosConfiguration()` | Swift API | forecasting object (`forecast(context:horizon:)`) |
| Qwen3-VL-Embedding-2B (text + image) | ``NFKMLXQwen3VLEmbedder`` | ``NFKMLXQwen3VLVisionNet`` + ``NFKMLXLanguageNet`` | read from the release's `config.json` and `preprocessor_config.json` | `embedder(directoryURL:)` / `backend(directoryURL:)` | ``NFKMLXTextEmbeddingBackend`` (text path) |
| Qwen3-VL-Reranker-2B (text + image) | ``NFKMLXQwen3VLReranker`` | ``NFKMLXQwen3VLVisionNet`` + ``NFKMLXLanguageNet`` | read from the release's `config.json` and `1_LogitScore/config.json` | `reranker(directoryURL:)` | scoring object (`scores(query:documents:)`) |
| SmolVLM2-500M | ``NFKMLXSmolVLM`` | ``NFKMLXSmolVLMNet`` (``NFKMLXSigLIPNet`` + ``NFKMLXSmolVLMConnector`` + ``NFKMLXLanguageNet``) | `NFKMLXSigLIPConfiguration.smolVLM`, `.smolVLM2Decoder` | `smolVLM(directoryURL:)` | object (`answer(image:question:)`) |
| Gemma 3 4B (image + text) | ``NFKMLXGemma3`` | ``NFKMLXGemma3Model`` (``NFKMLXGemma3VisionNet`` + ``NFKMLXGemma3MultimodalProjector`` + ``NFKMLXGemma3Net``) | read from the release's `config.json` (`NFKMLXSigLIPConfiguration.gemma3`, `NFKMLXGemma3Configuration.gemma3_4B`) | `load(directoryURL:)` / `backend(directoryURL:)` | ``NFKMLXGemma3Backend`` |
| Gemma 3n E2B / E4B (image + audio + text) | ``NFKMLXGemma3n`` | ``NFKMLXGemma3nModel`` (``NFKMLXGemma3nVisionNet`` + ``NFKMLXGemma3nAudioNet`` + ``NFKGemma3nMultimodalEmbedder`` + ``NFKMLXGemma3nNet``) | read from the release's `config.json` (``NFKMLXGemma3nConfiguration``, ``NFKMLXGemma3nAudioConfiguration``, ``NFKMLXGemma3nTokens``) | `load(directoryURL:)` / `backend(directoryURL:)` | ``NFKMLXGemma3nBackend`` |
| Qwen3-VL-2B vision tower | ``NFKMLXQwen3VL`` | ``NFKMLXQwen3VLVisionNet`` | `NFKMLXQwen3VLVisionConfiguration.qwen3VL2B` | — | Swift API |
| Pixtral 12B (vision + text) | ``NFKMLXPixtral`` | ``NFKMLXPixtralVisionNet`` + ``NFKMLXPixtralConnector`` + ``NFKMLXLanguageNet`` | read from the release's `config.json` | — | Swift API |
| Florence-2-base / -large (unified vision) | ``NFKMLXFlorence2`` | ``NFKMLXFlorence2Net`` (``NFKMLXFlorence2VisionNet`` + ``NFKMLXFlorence2Projector`` + ``NFKMLXSeq2SeqNet``) | read from the release's `config.json` | `backend(directoryURL:)`; adapted: `network(directoryURL:)`, `fineTune`, `save(_:toDirectoryURL:release:)` | ``NFKMLXFlorence2Backend`` |
| TrOCR (handwriting reader) | ``NFKMLXTrOCR`` | ``NFKMLXTrOCRNet`` (``NFKMLXTrOCRVisionNet`` + ``NFKMLXSeq2SeqNet`` decoder-only) | read from the release's `config.json` (every small, base, and large release) | `backend(directoryURL:)`; fine-tuned: `network(directoryURL:)`, `fineTune`, `save(_:toDirectoryURL:release:)` | ``NFKMLXTrOCRBackend`` |
| Sa2VA (segmentation VLM) | ``NFKMLXSa2VA`` | ``NFKMLXSa2VANet`` (InternVL), ``NFKMLXSa2VAQwenNet`` (Qwen-VL), ``NFKMLXSa2VALLaVANet`` (LLaVA) | read from the release's `config.json` | `backend(directoryURL:)`; fine-tuned: `network(directoryURL:)`, `fineTune`, `save(_:toDirectoryURL:release:)` | ``NFKMLXSa2VABackend`` |
| Phi-4-multimodal (image + speech + text) | ``NFKMLXPhi4MM`` | ``NFKMLXPhi4MMModel`` (``NFKMLXLanguageNet`` with a mixture of LoRAs + ``NFKMLXPhi4MMImageNet`` + ``NFKMLXPhi4MMAudioNet``) | read from the release's `config.json` | `backend(directoryURL:)` | ``NFKMLXPhi4MMBackend`` |
| Gemma 4 vision / audio / fusion | ``NFKMLXGemma4ConditionalGeneration`` | ``NFKMLXGemma4VisionNet``, ``NFKMLXGemma4AudioNet``, ``NFKMLXGemma4MultimodalEmbedder``, ``NFKMLXGemmaNet`` | ``NFKMLXGemma4VisionConfiguration`` / ``NFKMLXGemma4AudioConfiguration`` read from the release (`useClippedLinears` on) | — | Swift API (``NFKMLXGemma4ImageProcessor``, ``NFKMLXGemma4AudioFeatureExtractor``) |

```swift
// CLIP ViT-B/32
let backend = try NFKMLXCLIP.backend(weightsURL: url)
// CLIP probe
let net = try NFKMLXCLIP.network(weightsURL: url)
// then trainProbe and probeBackend(net:probe:labels:); Swift only
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
let reranker = try NFKMLXModernBERTReranker.reranker(directoryURL: dir)
// then scores(query:documents:) · scoresForQuery:documents:
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
// chronos-bolt-base (time-series forecasting)
let chronos = try NFKMLXChronos.chronos(weightsURL: url)
// then forecast(context:horizon:) · medianForecastForContext:horizon:
// Qwen3-VL-Embedding-2B (text + image)
let embedder = try NFKMLXQwen3VLEmbedder.embedder(directoryURL: dir)
// then embedding(forText:) · embeddingForText:, embedding(forImage:text:instruction:) · embeddingForImage:text:instruction:
// Qwen3-VL-Reranker-2B (text + image)
let multimodalReranker = try NFKMLXQwen3VLReranker.reranker(directoryURL: dir)
// then scores(query:documents:) · scoresForQuery:documents:
// SmolVLM2-500M
let vlm = try NFKMLXSmolVLM.load(directoryURL: dir)
// then answer(image:question:) · answerForImage:question:
// Gemma 3 4B (image + text)
let gemma = try NFKMLXGemma3.load(directoryURL: dir)
// then answer(image:question:) · answerForImage:question:error:; or backend(directoryURL:) with NFKInputImage beside NFKInputMessages
// Gemma 3n E2B / E4B (image + audio + text)
let gemma3n = try NFKMLXGemma3n.load(directoryURL: dir)
// then answer(image:question:) · answerForImage:question:error:; or backend(directoryURL:) with NFKInputImage / NFKInputAudio beside NFKInputMessages
// Qwen3-VL-2B vision tower
let vision = try NFKMLXQwen3VL.visionNet(directoryURL: dir)
// Swift only
// Pixtral 12B (vision + text)
let pixtral = try NFKMLXPixtral.model(directoryURL: dir)
// then answer(image:question:maxTokens:) · answerForImage:question:maxTokens:

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
let chain = NFKMLXGemma4ConditionalGeneration(decoder: decoder, visionTower: vision, visionEmbedder: visionEmbedder)
// then generate(promptTokens:image:waveform:maxTokens:); the decoder comes from NFKMLXGemmaLanguage's internal makeNet, so end to end this is reachable only inside the package today
```

```objc
// CLIP ViT-B/32
[NFKMLXCLIP backendWithWeightsURL:url error:&error]
// SigLIP 2
[NFKMLXSigLIP2 backendWithWeightsURL:url error:&error]
// Qwen3-Embedding-0.6B
[NFKMLXQwen3Embedding backendWithDirectoryURL:dir error:&error]
// EmbeddingGemma-300M
[NFKMLXEmbeddingGemma backendWithDirectoryURL:dir error:&error]
// gte-reranker-modernbert-base
[NFKMLXModernBERTReranker rerankerWithDirectoryURL:dir error:&error]
// Laya (typed decisions)
[NFKMLXLaya layaWithDirectoryURL:dir error:&error]
// open-jev-deberta-v3-large (typed decisions)
[NFKMLXOpenJevDeBERTa openJevWithDirectoryURL:dir error:&error]
// Open-Jev-2B / -9B (typed decisions)
[NFKMLXOpenJev openJevWithVariant:NFKMLXOpenJevVariantTwoB revision:nil cacheDirectoryURL:nil error:&error]
// Qwen3-VL-Embedding-2B (text + image)
[NFKMLXQwen3VLEmbedder embedderWithDirectoryURL:dir error:&error]
// Qwen3-VL-Reranker-2B (text + image)
[NFKMLXQwen3VLReranker rerankerWithDirectoryURL:dir error:&error]
// SmolVLM2-500M
[NFKMLXSmolVLM smolVLMWithDirectoryURL:dir error:&error]
// Gemma 3 4B (image + text)
[NFKMLXGemma3 gemma3WithDirectoryURL:dir error:&error]
// Gemma 3n E2B / E4B (image + audio + text)
[NFKMLXGemma3n gemma3nWithDirectoryURL:dir error:&error]
```


### Language models


| Model | Entry class | Network | Configuration for the released weights | Factory | Base backend |
| --- | --- | --- | --- | --- | --- |
| Qwen3 (dense), Qwen2, Llama | ``NFKMLXLanguage`` | ``NFKMLXLanguageNet`` | ``NFKMLXLanguageConfiguration`` from `config.json` (presets `.qwen3_0_6B`, `.qwen3_1_7B`, `.qwen3_4B`, `.qwen3_8B`, `.qwen3_14B`, `.qwen3_32B`); ``NFKMLXGenerationOptions`` per request | `let backend = try NFKMLXLanguage.backend(directoryURL: dir)`<br>`[NFKMLXLanguage backendWithDirectoryURL:dir error:&error]`<br>speculative: `backend(directoryURL: dir, draftDirectoryURL: draftDir)` · `backendWithDirectoryURL:dir draftDirectoryURL:draftDir error:&error` | ``NFKMLXLanguageBackend`` |
| Qwen3-MoE, Qwen2-MoE, Mixtral, gpt-oss | ``NFKMLXLanguage`` | ``NFKMLXLanguageNet`` + `NFKLMMixtureFeedForward` (+ `NFKLMFusedSwitchGLU` for gpt-oss) | the same reader (`qwen3_moe`, `qwen2_moe`, `mixtral`, `gpt_oss` model types); `.tinyMixture` for tests | `let backend = try NFKMLXLanguage.backend(directoryURL: dir)`<br>`[NFKMLXLanguage backendWithDirectoryURL:dir error:&error]` | ``NFKMLXLanguageBackend`` |
| Dense GGUF (`llama` / `qwen2` / `qwen3`) | ``NFKMLXLanguage`` | ``NFKMLXLanguageNet`` | `configuration(fromGGUF:)` from the file's metadata | `let backend = try NFKMLXLanguage.backend(ggufURL: url)`<br>`[NFKMLXLanguage backendWithGGUFURL:url error:&error]` | ``NFKMLXLanguageBackend`` |
| Qwen3.8-Flash-Next (Qwen4-Exp) | ``NFKMLXQwen4Exp`` | ``NFKMLXQwen4ExpNet`` | ``NFKMLXQwen4ExpConfiguration`` from `config.json` (`.qwen3_8FlashNext`, `.tiny` presets) | `let net = try NFKMLXQwen4Exp.backend(directoryURL: dir)`<br>Swift-only: the net is an `MLXNN.Module`, which does not bridge | — (prefill-only) |
| Qwen3.5 / 3.6 / 3.8 | ``NFKMLXHybridLanguage`` | ``NFKMLXHybridLanguageNet`` | ``NFKMLXHybridConfiguration`` from `config.json` (`.qwen3_8_27B` preset) | `let config = try NFKMLXHybridLanguage.configuration(fromHuggingFace: dir.appendingPathComponent("config.json"))`<br>no public constructor yet — `makeNet` and `loadWeights(into:fromDirectory:)` are internal, so there is no backend factory for the hybrid yet | — (prefill-only) |
| Gemma 3 270M / 1B / 4B | ``NFKMLXGemma3`` | ``NFKMLXGemma3Net`` | ``NFKMLXGemma3Configuration`` from `config.json` (`.gemma3_270M`, `.gemma3_1B`, `.gemma3_4B`; `gemma3_text` or the multimodal `gemma3`) | `let backend = try NFKMLXGemma3.backend(directoryURL: dir)`<br>`[NFKMLXGemma3 backendWithDirectoryURL:dir error:&error]`<br>also reached through `NFKMLXGemmaLanguage.backend(directoryURL:)`, which dispatches on the model type | ``NFKMLXGemma3Backend`` (hybrid key-value cache, streaming, the release's chat template) |
| Gemma 3n E2B / E4B | ``NFKMLXGemma3n`` | ``NFKMLXGemma3nNet`` | ``NFKMLXGemma3nConfiguration`` from `config.json` (`gemma3n_text`, or the tri-modal wrapper's `text_config`) | `backend(directoryURL:)` | ``NFKMLXGemma3nBackend`` |
| Gemma 4 E2B / E4B / 26B-A4B | ``NFKMLXGemmaLanguage`` | ``NFKMLXGemmaNet`` | ``NFKMLXGemmaConfiguration`` from `config.json` (`.e2b`; `enable_moe_block` turns on the routed branch) | `let backend = try NFKMLXGemmaLanguage.backend(directoryURL: dir)`<br>`[NFKMLXGemmaLanguage gemmaBackendWithDirectoryURL:dir error:&error]` | ``NFKMLXGemmaBackend`` |
| Gemma 4 12B unified | ``NFKMLXGemmaLanguage`` | ``NFKMLXGemma4UnifiedNet`` | `unifiedConfiguration(fromHuggingFace:)` (`.twelveB`) | `let backend = try NFKMLXGemmaLanguage.backend(directoryURL: dir)`<br>`[NFKMLXGemmaLanguage gemmaBackendWithDirectoryURL:dir error:&error]` | ``NFKMLXGemmaBackend`` |
| Gemma 2 | — | ``NFKMLXGemma2Net`` | `NFKMLXGemma2Configuration.gemma2_2B` / `.gemma2_9B` / `.gemma2_27B`; `configuration(fromHuggingFace:)` | `let net = try NFKMLXGemma2Net.load(directoryURL: dir)`; Swift only | SANA's text encoder |
| DeepSeek V4.1 Flash | ``NFKMLXDeepSeek`` | ``NFKMLXDeepSeekNet`` + ``NFKMLXDeepSeekCache`` | ``NFKMLXDeepSeekConfiguration`` `.v41Flash` from `config.json` | `let backend = try NFKMLXDeepSeek.backend(directoryURL: dir)`<br>`let paged = try NFKMLXDeepSeek.backend(directoryURL: dir, paging: .all)`<br>`[NFKMLXDeepSeek deepSeekBackendWithDirectoryURL:dir error:&error]`<br>`[NFKMLXDeepSeek deepSeekPagedBackendWithDirectoryURL:dir expertCacheBytes:1 << 32 error:&error]`<br>`[NFKMLXDeepSeek deepSeekBackendWithDirectoryURL:dir options:options error:&error]` (``NFKMLXDeepSeekLoadOptions``)<br>the directory supplies the collapsed id space its n-gram memory hashes over, derived from its own tokenizer | ``NFKMLXDeepSeekBackend`` (computes in bf16 as the release does, bit for bit against its own code; 763B decodes to 1.39 TiB of bf16 parameters, and with every group mapped out of the release the decoder holds 17.7 GiB) |
| DeepSeek V4 Flash / Pro | ``NFKMLXDeepSeek`` | ``NFKMLXDeepSeekNet`` + ``NFKMLXDeepSeekCache`` | `NFKMLXDeepSeekConfiguration.v4Flash` / `.v4Pro` from `config.json`; ``NFKMLXDeepSeekQuantization`` decodes the fp8 / fp4 storage | `let backend = try NFKMLXDeepSeek.backend(directoryURL: dir)`<br>`[NFKMLXDeepSeek deepSeekBackendWithDirectoryURL:dir options:options error:&error]` | ``NFKMLXDeepSeekBackend`` (computes in bf16, bit for bit against the release's own code at prefill and decode) |
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
// Swift makeNet / loadWeights
// Qwen3.8-Flash-Next (Qwen4-Exp)
NFKMLXQwen4Exp.backend(directoryURL:)
// Gemma 3 270M / 1B / 4B
backend(directoryURL:)
// Gemma 4 E2B / E4B / 26B-A4B
backend(directoryURL:)
// Gemma 4 12B unified
backend(directoryURL:)
// Gemma 2
NFKMLXGemma2Net.load(directoryURL:)
// DeepSeek V4 Flash / Pro
// Swift API
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


### Translation

| Model | Entry class | Network | Configuration for the released weights | Registered name | Base backend |
| --- | --- | --- | --- | --- | --- |
| OPUS-MT (Marian) | ``NFKMLXMarian`` | ``NFKMLXSeq2SeqNet`` | ``NFKMLXSeq2SeqConfiguration`` from the release's `config.json`; `.tinyMarian` for tests | `opus-mt` (the registry URL is the release directory) | ``NFKMLXTranslationBackend`` |
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
// MADLAD-400 3B-MT, bfloat16
NFKMLXMADLAD.backend(directoryURL:, half: true)
// TranslateGemma, at the checkpoint's bfloat16
NFKMLXTranslateGemma.backend(directoryURL:, precision: .checkpoint)
// Fine-tuning
NFKMLXMarian.network(directoryURL:) / NFKMLXM2M100.network(directoryURL:) / NFKMLXMADLAD.network(directoryURL:)
```

### Video


| Model | Entry class | Network | Configuration for the released weights | Registered name | Base backend |
| --- | --- | --- | --- | --- | --- |
| RIFE HDv3 | ``NFKMLXRIFE`` | `NFKMLXRIFENet` | fixed geometry (three IFBlocks, width 90, scales 4/2/1) | `rife` | ``NFKMLXTensorBackend`` (`frame0` / `frame1`); `clipBackend(weightsURL:)` for a clip |
| RIFE v4 | ``NFKMLXRIFEv4`` | `NFKMLXRIFEv4Net` | fixed geometry (four blocks, timestep input, scales 8/4/2/1) | `rife-v4` | ``NFKMLXTensorBackend`` |
| RAFT | ``NFKMLXRAFT`` | `NFKMLXRAFTNet` | RAFT-large (feature 256, 4 levels, radius 4); `iterations` (default 6) | `raft` | ``NFKMLXTensorBackend`` |
| BasicVSR | ``NFKMLXVideoSR`` | `NFKMLXVideoSRNet` + `NFKVSRSPyNet` | `NFKMLXVideoSRConfiguration.base` (×4) | `video-super-resolution` | ``NFKMLXModuleBackend`` (frame) / `clipBackend(weightsURL:)` |
| SD ×4 upscaler | ``NFKMLXSDUpscaler`` | ``NFKMLXSDUNet`` + ``NFKMLXSDAutoencoder`` | `NFKMLXSDUNetConfiguration.upscaler`, `NFKMLXSDVAEConfiguration.upscaler`; `noiseLevel` (20) | `sd-x4-upscaler` | ``NFKMLXDiffusionBackend`` |
| V-JEPA 2 (video features, video classification) | ``NFKMLXVJEPA2`` | ``NFKMLXVJEPA2Net`` | read from the release's `config.json` (ViT-L, ViT-H, or ViT-g; `id2label` for a classifier) | `backend(directoryURL:)`; fine-tuned: `network(directoryURL:labels:)`, `fineTune`, `save(_:toDirectoryURL:)` | ``NFKMLXVJEPA2Backend``; video / image → mean-pooled embedding, and ranked classes from a classification release (V-JEPA 2, Meta) |
| Cosmos Tokenizer (image and video tokens) | ``NFKMLXCosmosTokenizer`` | ``NFKMLXCosmosTokenizerNet`` | `NFKMLXCosmosTokenizerConfiguration.variant(_:)` for each ``NFKMLXCosmosTokenizerVariant`` (CI8x8 … DV8x16x16); fine-tuned: `network(variant:weightsURL:)` | `cosmos-tokenizer-<variant>` | ``NFKMLXCosmosTokenizerBackend``; image / clip → reconstruction, `codeForImage:error:` → latent or tokens |
| Clip backend | ``NFKMLXVideoBackend`` | — | ``NFKMLXVideoConfiguration`` (`frameRateMultiplier`, `outputFramesPerSecond`) | — | ``NFKMLXVideoFile`` under it |

```swift
// RIFE HDv3
let backend = try NFKMLXRIFE.backend(weightsURL: url)
// clip: try NFKMLXRIFE.clipBackend(weightsURL: url) · [NFKMLXRIFE clipBackendWithWeightsURL:url error:&error]
// RIFE v4
let backend = try NFKMLXRIFEv4.backend(weightsURL: url)
// RAFT
let backend = try NFKMLXRAFT.backend(weightsURL: url)
// BasicVSR
let backend = try NFKMLXVideoSR.backend(weightsURL: url)
// clip: try NFKMLXVideoSR.clipBackend(weightsURL: url) · [NFKMLXVideoSR clipBackendWithWeightsURL:url error:&error]
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
// RIFE v4
[NFKMLXRIFEv4 backendWithWeightsURL:url error:&error]
// RAFT
[NFKMLXRAFT backendWithWeightsURL:url error:&error]
// BasicVSR
[NFKMLXVideoSR backendWithWeightsURL:url error:&error]
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
| Parakeet-TDT | ``NFKMLXParakeet`` | ``NFKMLXParakeetNet`` | `NFKMLXParakeetConfiguration.tdt06B` (0.6B v2: 24 rel-pos conformer layers, TDT durations 0…4) | `parakeet-tdt`; `backend(directoryURL:)` | ``NFKMLXParakeetBackend`` (text + per-token `NFKOutputSegments`) |
| Granite Speech 3.3-2b | ``NFKMLXGraniteSpeech`` | ``NFKMLXGraniteSpeechNet`` | `net(fromDirectory:)` (Conformer encoder + BLIP-2 Q-former + dense Granite decoder) | `backend(directoryURL:)` | ``NFKMLXGraniteSpeechBackend`` (audio → text) |
| Voxtral-Mini 3B | ``NFKMLXVoxtral`` | ``NFKMLXVoxtralNet`` | `net(fromDirectory:)` (Whisper encoder + 2-linear projector + Llama decoder) | `backend(directoryURL:)` | ``NFKMLXVoxtralBackend`` (audio → text) |
| Canary-1B-v2 | ``NFKMLXCanary`` | ``NFKMLXCanaryNet`` | `NFKMLXCanaryConfiguration.v2` (biased FastConformer encoder + Transformer attention encoder-decoder) | `canary-1b-v2`; `backend(directoryURL:)` | ``NFKMLXCanaryBackend`` (audio → text; `src>tgt` translates) |
| Chatterbox | ``NFKMLXChatterbox`` | ``NFKMLXChatterboxTTS`` (``NFKMLXChatterboxVoiceEncoderNet``, ``NFKMLXS3TokenizerNet``, ``NFKMLXT3Net``, ``NFKMLXS3GenNet``) | `.released` on every stage (VoiceEncoder 3×256, S3 tokenizer 6×1280, T3 Llama 520M with llama3 rope, S3Gen flow + HiFT) | `chatterbox`; `speechBackend(directoryURL:voiceURL:)` | ``NFKMLXSpeechBackend`` (24 kHz WAV; text → cloned voice) |
| Demucs v2 | ``NFKMLXDemucs`` | `NFKMLXDemucsNet` | `NFKMLXDemucsConfiguration()` = music (stereo, depth 6, 4 stems, BLSTM, context 3) | `demucs` | ``NFKMLXDemucsBackend`` |
| Speech denoiser | ``NFKMLXDenoiser`` | `NFKMLXDemucsNet` | ``NFKMLXDemucsConfiguration`` set to dns48 (mono, depth 5, 1 stem, causal, context 1) | `denoiser` | ``NFKMLXDenoiserBackend`` |
| MP-SENet | ``NFKMLXMPSENet`` | TS-transformer (bidirectional-GRU FFN) over compressed magnitude + phase | `NFKMLXMPSENetConfiguration()` (fftSize 400, hop 100, 4 blocks) | `mpsenet`; `NFKMLXMPSENetFactory.backend(weightsURL:)` | ``NFKMLXMPSENetBackend`` |
| GTCRN | ``NFKMLXGTCRN`` | ERB band merge/split, grouped-conv encoder/decoder, dual-path grouped RNN → complex ratio mask (~48K params) | `NFKMLXGTCRNConfiguration()` | `gtcrn`; `NFKMLXGTCRNFactory.backend(weightsURL:)` | ``NFKMLXGTCRNBackend`` |
| SGMSE+ | ``NFKMLXSGMSE`` | `NFKMLXNCSNppNet` score network + OUVE reverse-SDE predictor-corrector sampler | `NFKMLXSGMSEConfiguration()` (`ncsnpp` / `ncsnpp_48k`) | `sgmse`; `NFKMLXSGMSE.backend(weightsURL:)` | ``NFKMLXSGMSEBackend`` |
| StoRM | ``NFKMLXStoRM`` | discriminative predictor + `[noisy, denoised]`-conditioned NCSN++ score; SDE re-centered on the estimate | `NFKMLXStoRMConfiguration()` | `storm`; `NFKMLXStoRM.backend(weightsURL:)` | ``NFKMLXStoRMBackend`` |
| MossFormer2 SE 48K | ``NFKMLXMossFormer2SENet`` | `NFKMLXMossFormer2SENet` (FLASH + `Gated_FSMN`) over `NFKMLXKaldiFbank` | `NFKMLXMossFormer2Configuration()` (48 kHz, 24 blocks) | `mossformer2-se` | ``NFKMLXMossFormer2Backend`` |
| MossFormer2 SR 48K | ``NFKMLXMossFormer2SRNet`` | the SE backbone (80 → 80 mel) + ``NFKMLXMossFormer2SRGenerator`` (Snake HiFi-GAN, ×256) + the bandwidth substitution | `NFKMLXMossFormer2SRConfiguration()` (48 kHz, 1024/256, 80 mels to 8 kHz) | `mossformer2-sr`; `NFKMLXMossFormer2SRFactory.backend(directoryURL:)` | ``NFKMLXMossFormer2SRBackend`` |
| DeepFilterNet3 | ``NFKMLXDeepFilterNet`` | `DfNet` (`SqueezedGRU_S` encoder / ERB decoder / DF decoder) over a libdf-reproduced DSP | `NFKMLXDeepFilterNetConfiguration()` (48 kHz, 32 ERB bands, 96 DF bins) | `deepfilternet3` | ``NFKMLXDeepFilterNetBackend`` |
| VoiceRestore | ``NFKMLXVoiceRestore`` / ``NFKMLXBigVGAN`` | E2-TTS CFM transformer (gateloop + adaLN) + BigVGAN v2 vocoder | `NFKMLXVoiceRestoreConfiguration()` + `NFKMLXBigVGANConfiguration()` | `voicerestore` | ``NFKMLXVoiceRestoreBackend`` |
| Resemble Enhance | ``NFKMLXResembleEnhance`` | STFT-mask 2-D UNet denoiser + IRMAE/WaveNet-CFM latent flow matching + UnivNet LVC vocoder | `NFKMLXResembleConfiguration()` | `resemble-enhance` | ``NFKMLXResembleEnhanceBackend`` |
| MetricGAN+ | ``NFKMLXMetricGANPlus`` | ``NFKMLXMetricGANPlusNet`` (2-layer BLSTM 257→200 per direction, Linear 400→300, LeakyReLU 0.3, Linear 300→257, learnable sigmoid) over a zero-padded 512-point Hamming STFT | `NFKMLXMetricGANPlusConfiguration()` (16 kHz, hop 256) | `metricgan-plus`; `NFKMLXMetricGANPlus.backend(weightsURL:)` | ``NFKMLXMetricGANPlusBackend`` |
| CMGAN | ``NFKMLXCMGAN`` | ``NFKMLXCMGANNet`` (dense encoder, 4 two-stage conformer blocks with Shaw relative positions, mask + complex decoders) over a power-compressed 400/100 Hamming STFT | `NFKMLXCMGANConfiguration()` (16 kHz, 64 channels, compress 0.3) | `cmgan`; `NFKMLXCMGAN.backend(weightsURL:)` | ``NFKMLXCMGANBackend`` |
| FRCRN SE 16K | ``NFKMLXFRCRN`` | ``NFKMLXFRCRNNet`` (two complex UNets: frequency-recurrent FSMNs, complex squeeze-excites, a time FSMN bottleneck) over a 640/320 sqrt-Hann conv-STFT | `NFKMLXFRCRNConfiguration()` (16 kHz, 128 channels, order 20) | `frcrn`; `NFKMLXFRCRN.backend(weightsURL:)` | ``NFKMLXFRCRNBackend`` |
| NU-Wave 2 | ``NFKMLXNUWave2`` | ``NFKMLXNUWave2Net`` (15 short-time Fourier convolution blocks with BSFT band modulation) + an 8-step logSNR DDIM | `NFKMLXNUWave2Configuration()` (48 kHz, 1024/256, 64 channels) | `nuwave2`; `NFKMLXNUWave2.backend(weightsURL:)` | ``NFKMLXNUWave2Backend`` |
| Apollo | ``NFKMLXApollo`` | ``NFKMLXApolloNet`` (80-band split, 6 band-Roformer + ICB layers, GLU band heads) | `NFKMLXApolloConfiguration()` (44.1 kHz, 20 ms window, 256 features) | `apollo`; `NFKMLXApollo.backend(weightsURL:)` | ``NFKMLXApolloBackend`` |
| HT Demucs (v4) | ``NFKMLXHTDemucs`` | ``NFKMLXHTDemucsNet``, ``NFKMLXHTDemucsBag`` | ``NFKMLXHTDemucsVariant`` `.fourStem` / `.sixStem`; the fine-tuned release through `backend(fineTunedWeightsURLs:)` | `htdemucs` · `htdemucs-6s` | ``NFKMLXHTDemucsBackend`` |
| Conv-TasNet | ``NFKMLXConvTasNet`` | `NFKMLXConvTasNetNet` | `NFKMLXConvTasNetConfiguration.libri2Mix16k`; `perChannelPReLU` optional | `conv-tasnet` | ``NFKMLXConvTasNetBackend`` |
| MarbleNet VAD | ``NFKMLXVAD`` | `NFKMLXVADNet` | `NFKMLXVADConfiguration.marbleNet` | `vad-marblenet` | ``NFKMLXVADBackend`` |
| Silero VAD v6 | ``NFKMLXSileroVAD`` | `NFKMLXSileroVADNet` | `NFKMLXSileroVADConfiguration.v6` | `silero-vad` | ``NFKMLXSileroVADBackend`` |
| PANNs Cnn14 | ``NFKMLXAudioTagger`` | `NFKMLXAudioTaggerNet` | `NFKMLXAudioTaggerConfiguration.panns` | `audio-tagger-panns` | ``NFKMLXAudioTaggerBackend`` (`labels:`) |
| Descript Audio Codec | ``NFKMLXDAC`` | `NFKMLXDACNet` (`NFKDACEncoderNet`, `NFKDACDecoderNet`) | `NFKMLXDACConfiguration.dac44kHz` / `.dac24kHz` / `.dac16kHz` | `dac` | ``NFKMLXDACBackend``; `encode` / `decode` for the tokens |
| SNAC | ``NFKMLXSNAC`` | `NFKMLXSNACNet` (`NFKSNACEncoderNet`, `NFKSNACDecoderNet`) | ``NFKMLXSNACVariant`` `.speech24kHz` / `.music32kHz` / `.music44kHz` | `snac` · `snac-32khz` · `snac-44khz` | ``NFKMLXSNACBackend``; `decode(_:deterministic:)` |
| BigVGAN v2 | ``NFKMLXBigVGAN`` | SnakeBeta + anti-aliased `Activation1d` generator | `NFKMLXBigVGANConfiguration()` (24 kHz, 100-band) | `bigvgan-v2-24khz` | ``NFKMLXBigVGANBackend`` |
| Mimi | ``NFKMLXMimi`` | `NFKMLXMimiNet` (SEANet + RoPE transformers + split RVQ) | `NFKMLXMimiConfiguration()` (24 kHz, 12.5 Hz) | `mimi` | ``NFKMLXMimiBackend``; `encode` / `decode` for the tokens |
| Basic Pitch | ``NFKMLXBasicPitch`` | `NFKMLXBasicPitchNet` | ``NFKMLXBasicPitchConfiguration`` `.icassp2022` | `basic-pitch` | ``NFKMLXBasicPitchBackend`` (``NFKMLXTranscriptionParameterKey``) |
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
// Chatterbox (a release directory; nil voice = the built-in conds.pt)
let backend = try NFKMLXChatterbox.speechBackend(directoryURL: dir, voiceURL: voiceWAV)
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
// PANNs Cnn14
let backend = try NFKMLXAudioTagger.backend(weightsURL: url, labels: nil)
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
// Chatterbox
[NFKMLXChatterbox chatterboxBackendWithDirectoryURL:dir voiceURL:voiceWAV error:&error]
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
```


### Text to speech and music


| Model | Entry class | Network | Configuration for the released weights | Registered name / factory | Base backend |
| --- | --- | --- | --- | --- | --- |
| FastSpeech2 conformer + paired HiFi-GAN | ``NFKMLXVoice`` | ``NFKMLXFastSpeech2Net``, `NFKMLXHiFiGANNet` | `NFKMLXFastSpeech2Configuration()` = espnet LJSpeech; `NFKMLXHiFiGANConfiguration()` = UNIVERSAL_V1 geometry (the paired `vocoder.` weights) | `fastspeech2-voice`; `makeSpeechBackend(phonemize:)` | ``NFKMLXSpeechBackend`` |
| HiFi-GAN | ``NFKMLXHiFiGAN`` | `NFKMLXHiFiGANNet` | `NFKMLXHiFiGANConfiguration()` (80 mel bins, 512 channels, rates 8/8/2/2) | — | mel → waveform object |
| Kokoro-82M | ``NFKMLXKokoro`` | ``NFKMLXKokoroNet`` | `NFKMLXKokoroConfiguration.v1`; a voice from `loadVoice` | `backend(directoryURL:voiceName:)` | ``NFKMLXSpeechBackend`` (phonemes in) |
| Phonemizers | ``NFKMLXNeuralG2P``, ``NFKMLXEspeakPhonemizer`` | `NFKMLXG2PNet` | ``NFKMLXG2PConfiguration`` | — | ``NFKMLXPhonemizer`` protocol |
| Hand-chained TTS | ``NFKMLXTTS`` | `NFKMLXAcousticNet` + `NFKMLXHiFiGANNet` | ``NFKMLXAcousticConfiguration`` | `makeSpeechBackend()` | ``NFKMLXSpeechBackend`` |
| MiniMax Music 3 | ``NFKMLXMusic3`` | `NFKMusic3VocoderNet`, `NFKMusic3DepthDecoderNet`, `NFKMusic3ConditionEncoderNet`, `NFKMusic3DiTNet`, ``NFKMLXLanguageNet`` | the release directory (bf16 LM, float32 DiT); `quantizeRelease(at:to:bits:transformerBits:)` for the 7.7 GiB copy | `minimax-music3`; `backend(directoryURL:)` | ``NFKMLXMusicBackend`` |

```swift
// FastSpeech2 conformer + paired HiFi-GAN
let voice = try NFKMLXFastSpeech2.voice(acousticURL: acousticURL, vocoderURL: vocoderURL, vocabularyURL: vocabURL); let backend = voice.makeSpeechBackend(phonemize: phonemize)
// Swift only
// HiFi-GAN
// no standalone public entry; the net is built and loaded inside NFKMLXFastSpeech2.voice(acousticURL:vocoderURL:vocabularyURL:) and NFKMLXTTS.loadWeights(acousticURL:vocoderURL:)
// Kokoro-82M
let backend = try NFKMLXKokoro.backend(directoryURL: dir, voiceName: "af_heart")
// Phonemizers
let g2p = NFKMLXNeuralG2P(); try g2p.loadWeights(from: url)
// NFKMLXEspeakPhonemizer() when isInstalled; Swift only
// Hand-chained TTS
let tts = NFKMLXTTS(phonemizer: g2p); try tts.loadWeights(acousticURL: acousticURL, vocoderURL: vocoderURL); let backend = tts.makeSpeechBackend()
// Swift only
// MiniMax Music 3
let backend = try NFKMLXMusic3.backend(directoryURL: dir)
// quantize: try NFKMLXMusic3.quantizeRelease(at: dir, to: outDir)
```

```objc
// Kokoro-82M
[NFKMLXKokoro kokoroBackendWithDirectoryURL:dir voiceName:@"af_heart" error:&error]
// MiniMax Music 3
[NFKMLXMusic3 backendWithDirectoryURL:dir error:&error]
```


### Text to image and video


| Model | Entry class | Network | Configuration for the released weights | Registered name / factory | Base backend |
| --- | --- | --- | --- | --- | --- |
| Stable Diffusion 1.5 / 2.1 / SDXL-Turbo | ``NFKMLXBackend``, ``NFKMLXTextToImage`` | ``NFKMLXSDUNet``, ``NFKMLXSDAutoencoder``, ``NFKMLXSDTextEncoderNet`` | `NFKMLXSDTextToImageConfiguration.stableDiffusion15` / `.stableDiffusion21` / `.stableDiffusion21V` / `.sdxlTurbo`; ``NFKMLXWeightPrecision`` | `stable-diffusion`; `backend(model:directoryURL:)`; ``NFKMLXStableDiffusionModel`` release enum on ``NFKMLXBackend`` | ``NFKMLXDiffusionBackend`` + ``NFKDDIMScheduler`` |
| SD networks | ``NFKMLXSDPipeline`` | ``NFKMLXSDUNet`` + ``NFKMLXSDAutoencoder`` | `NFKMLXSDUNetConfiguration.stableDiffusion` / `.sdxl` / `.inpainting` / `.marigold` / `.upscaler`; `NFKMLXSDVAEConfiguration.stableDiffusion` / `.upscaler` / `.flux` | — | Swift API |
| IP-Adapter | ``NFKMLXIPAdapterImageProjection``, ``NFKMLXIPAdapterAttention`` | — | `imageEmbedDim` 1024 → `crossAttentionDim` 768, 4 tokens | — | Swift API |
| Z-Image / Z-Image-Turbo | ``NFKMLXZImageGenerator``, ``NFKMLXZImagePipeline`` | ``NFKMLXZImageTransformerNet`` + ``NFKMLXSDAutoencoder`` (`.flux`) + Qwen3-4B | `NFKMLXZImageConfiguration`, the Flux VAE configuration, and `NFKMLXFlowMatchConfiguration.zImageTurbo` / `.zImage`, each read from the release | `let zImage = try NFKMLXZImageGenerator.generator(directoryURL: dir, residency: .automatic)`<br>`[NFKMLXZImageGenerator generatorWithDirectoryURL:dir residency:NFKMLXResidencyAutomatic error:&error]` | ``NFKMLXFlowMatchScheduler`` |
| SANA | ``NFKMLXSANAPipeline`` | ``NFKMLXSANATransformerNet`` + ``NFKMLXDCAutoencoderNet`` | `NFKMLXSANAConfiguration.base`; `NFKMLXDCAEConfiguration.sana`; `NFKMLXDPMSolverConfiguration.sana`; caption from ``NFKMLXGemma2Net`` | Swift API | ``NFKMLXDPMSolverScheduler`` |
| LTX-Video 0.9.0 | ``NFKMLXLTXVideoGenerator``, ``NFKMLXLTXPipeline`` | `NFKMLXLTXTransformerNet` + `NFKMLXLTXVideoVAENet` | `NFKMLXLTXTransformerConfiguration.base`; `NFKMLXLTXVAEConfiguration.base`; `NFKMLXFlowMatchConfiguration.ltxVideo`; `NFKMLXT5Configuration.xxl` | `let ltx = try NFKMLXLTXVideoGenerator.generator(directoryURL: dir, residency: .automatic)`<br>`[NFKMLXLTXVideoGenerator generatorWithDirectoryURL:dir residency:NFKMLXResidencyAutomatic error:&error]` | ``NFKMLXFlowMatchScheduler`` |
| LTX-2 (audio-video transformer) | — | ``NFKMLXLTX2TransformerNet`` | `NFKMLXLTX2Configuration.ltx23` / `.ltx25` | Swift API (the forward takes `MLXArray`) | — |
| Wan 2.1 T2V / Wan 2.2 TI2V-5B | ``NFKMLXWanVideoGenerator``, ``NFKMLXWanPipeline`` | ``NFKMLXWanTransformerNet`` + ``NFKMLXWanVideoVAENet`` | `NFKMLXWanConfiguration.base`; `NFKMLXWanVAEConfiguration.wan22` / `.wan21`; `NFKMLXUniPCConfiguration.wan`; `NFKMLXT5Configuration.umt5XXL` | `let wan = try NFKMLXWanVideoGenerator.generator(directoryURL: dir, residency: .automatic)`<br>`[NFKMLXWanVideoGenerator generatorWithDirectoryURL:dir residency:NFKMLXResidencyAutomatic error:&error]` | ``NFKMLXUniPCScheduler`` |
| Qwen-Image 2.1 | ``NFKMLXQwenImageGenerator``, ``NFKMLXQwenImagePipeline`` | ``NFKMLXQwenImageNet`` + ``NFKMLXWanVideoVAENet`` (`.qwenImage21`) | `NFKMLXQwenImageConfiguration.base`, `NFKMLXWanVAEConfiguration.qwenImage21`, and `NFKMLXFlowMatchConfiguration.qwenImage21`, each read from the release | `let qwen = try NFKMLXQwenImageGenerator.generator(directoryURL: dir, residency: .automatic)`<br>`[NFKMLXQwenImageGenerator generatorWithDirectoryURL:dir residency:NFKMLXResidencyAutomatic error:&error]` | ``NFKMLXFlowMatchScheduler`` |
| Wan 2.2 Animate | ``NFKMLXWanAnimate`` | ``NFKMLXWanAnimateNet`` | `NFKMLXWanAnimateConfiguration.base` (14B) / `.tiny`; ``NFKMLXWanAnimateKVCache`` holds the reference pass | `let dit = NFKMLXWanAnimate.makeNet(.base)`<br>`let cache = NFKMLXWanAnimateKVCache(layerCount: 40)`; Swift only — the configuration is a Swift struct | — (released weights exceed a workstation) |
| Stable Diffusion 3 / 3.5 | ``NFKMLXSD3Generator``, ``NFKMLXSD3Pipeline`` | ``NFKMLXSD3TransformerNet`` + ``NFKMLXSDAutoencoder`` + CLIP-L, CLIP-G and T5-XXL | `NFKMLXSD3Configuration`, the autoencoder configuration, and the flow shift, each read from the release | `let sd3 = try NFKMLXSD3Generator.generator(directoryURL: dir, residency: .automatic)`<br>`[NFKMLXSD3Generator generatorWithDirectoryURL:dir residency:NFKMLXResidencyAutomatic error:&error]` | ``NFKMLXFlowMatchScheduler`` |
| FLUX.1 (text-to-image) | ``NFKMLXFlux`` | ``NFKMLXFluxTextEncoder`` + ``NFKMLXFluxPipeline`` | read from the diffusers release directory | Swift API + `@objc` (`image(forPrompt:…)`) | ``NFKMLXFlowMatchScheduler`` |
| FLUX.1 (transformer + sampler) | ``NFKMLXFluxPipeline`` | ``NFKMLXFluxTransformerNet`` + ``NFKMLXSDAutoencoder`` (`.flux`) | `NFKMLXFluxConfiguration.dev` / `.schnell`; `NFKMLXFlowMatchConfiguration.flux` / `.fluxSchnell`; text from CLIP-L (pooled) + T5-XXL | Swift API (`generate(promptEmbeds:pooled:…)`) | ``NFKMLXFlowMatchScheduler`` |
| SD3 ControlNet | ``NFKMLXSD3ControlNetPipeline`` | ``NFKMLXSD3ControlNetNet`` + ``NFKMLXSD3TransformerNet`` + ``NFKMLXSDAutoencoder`` | `NFKMLXSD3ControlNetConfiguration.instantXMedium` / `.stabilitySD35Large`; a spatial control image | Swift API (`generate(promptEmbeds:pooled:…controlImage:…)`) | ``NFKMLXFlowMatchScheduler`` |
| FLUX.2 [klein] (text-to-image, editing, inpainting) | ``NFKMLXFlux2`` | ``NFKMLXFlux2TransformerNet`` + ``NFKMLXSDAutoencoder`` (`.flux2`) + ``NFKMLXFlux2LatentCodec`` + a Qwen3 | `NFKMLXFlux2Configuration.klein4B` / `.klein9B` / `.dev`; `NFKMLXFlowMatchConfiguration.flux2` | `@objc` | ``NFKMLXFlowMatchScheduler`` |
| FLUX.1 ControlNet | ``NFKMLXFluxControlNetPipeline`` | ``NFKMLXFluxControlNetNet`` + ``NFKMLXFluxTransformerNet`` + ``NFKMLXSDAutoencoder`` (`.flux`) | `NFKMLXFluxControlNetConfiguration.unionPro` / `.single`; a spatial control image | Swift API (`generate(promptEmbeds:pooled:…controlImage:…)`) | ``NFKMLXFlowMatchScheduler`` |
| Reference diffusion stand-ins | ``NFKMLXReferenceModels`` | oracle `denoise` closures | — | `diffusion-upscaler` · `diffusion-depth` · `diffusion-inpaint` · `diffusion-controlnet` | ``NFKMLXDiffusionBackend`` |

```swift
// Stable Diffusion 1.5 / 2.1 / SDXL-Turbo
let backend = try NFKMLXTextToImage.backend(model: .stableDiffusion15, directoryURL: dir)
// download on first use: NFKMLXBackend(model: .stableDiffusion15) · [[NFKMLXBackend alloc] initWithModel:NFKMLXStableDiffusionModelStableDiffusion15]; .stableDiffusion21Base / .sdxlTurbo
// SD networks
let pipeline = NFKMLXSDPipeline(unet: .stableDiffusion, vae: .stableDiffusion); try pipeline.loadWeights(unetURL: unetURL, vaeURL: vaeURL)
// Swift only
// IP-Adapter
let projection = NFKMLXIPAdapterImageProjection(); let attention = NFKMLXIPAdapterAttention(queryDim: 320, crossAttentionDim: 768, heads: 8, headDim: 40)
// Stable Diffusion 3 / 3.5 (text-to-image)
let sd3 = try NFKMLXSD3Generator.generator(directoryURL: releaseDirectory, residency: .automatic)
let image = try sd3.image(forPrompt: "a red fox in the snow", width: 1024, height: 1024, seed: 0)
// ObjC: [NFKMLXSD3Generator generatorWithDirectoryURL:dir residency:NFKMLXResidencyAutomatic error:&e], then imageForPrompt:negativePrompt:width:height:seed:error:
// Swift only
// FLUX.1 (text-to-image)
let flux = try NFKMLXFlux.flux(directoryURL: dir); let image = try flux.image(forPrompt: "an astronaut on the moon")
// FLUX.1 (transformer + sampler)
let dit = NFKMLXFluxTransformerNet(.dev); try NFKMLXFluxTransformerNet.loadWeights(into: dit, from: dir); let pipeline = NFKMLXFluxPipeline(transformer: dit, vae: vae)
// Swift only
// SD3 ControlNet
let cn = NFKMLXSD3ControlNetNet(.instantXMedium); try NFKMLXSD3ControlNetNet.loadWeights(into: cn, from: cnDir); let pipeline = NFKMLXSD3ControlNetPipeline(transformer: dit, controlnet: cn, vae: vae)
// Swift only
// FLUX.1 ControlNet
let cn = NFKMLXFluxControlNetNet(.unionPro); try NFKMLXFluxControlNetNet.loadWeights(into: cn, from: cnDir); let pipeline = NFKMLXFluxControlNetPipeline(transformer: dit, controlnet: cn, vae: vae)
// Swift only
// FLUX.2 [klein] (text-to-image, editing, inpainting)
let flux2 = try NFKMLXFlux2.flux2(directoryURL: releaseDirectory)
let image = try flux2.image(forPrompt: "a red fox in the snow", width: 1024, height: 1024, seed: 0)
// ObjC: [NFKMLXFlux2 flux2WithDirectoryURL:dir error:&e], then imageForPrompt:negativePrompt:width:height:seed:error:
// Z-Image (text-to-image; Turbo by default)
let zImage = try NFKMLXZImageGenerator.generator(directoryURL: releaseDirectory, residency: .automatic)
let image = try zImage.image(forPrompt: "a red fox in the snow", width: 1024, height: 1024, seed: 0)
// ObjC: [NFKMLXZImageGenerator generatorWithDirectoryURL:dir residency:NFKMLXResidencyAutomatic error:&e], then imageForPrompt:negativePrompt:width:height:seed:error:
// SANA
let dit = NFKMLXSANATransformerNet(.base); let vae = NFKMLXDCAutoencoderNet(.sana); try NFKMLXDCAutoencoderNet.loadWeights(into: vae, from: vaeURL)
// no public constructor yet — NFKMLXSANAPipeline's initializers are internal and the DiT has no public loader; generate(promptEmbeds:negativeEmbeds:latentHeight:…) is public
// LTX-Video
let vae = try NFKMLXLTXVideoVAE.vae(configuration: .base, weightsURL: vaeURL); let t5 = try NFKMLXT5Encoder.encoder(configuration: .xxl, directory: t5Dir)
// no public constructor yet — NFKMLXLTXTransformerNet and NFKMLXLTXPipeline's initializers are internal; generate(promptTokens:negativeTokens:frames:height:…) is public
// LTX-2 (audio-video transformer)
let dit = NFKMLXLTX2TransformerNet(.ltx25); try NFKMLXLTX2TransformerNet.loadWeights(into: dit, from: transformerDir)
// Swift only — the forward denoises the video and audio latents together and returns both
// Wan 2.2 Animate
let animate = NFKMLXWanAnimate.makeNet(.base)
let cache = NFKMLXWanAnimateKVCache(layerCount: 40)
// Swift only — NFKMLXWanAnimateConfiguration is a Swift struct

// Wan
let dit = NFKMLXWanTransformerNet(.base); let vae = NFKMLXWanVideoVAENet(.wan22)
// no public constructor yet — NFKMLXWanPipeline's initializers are internal and neither net has a public loader; generate(textEmbeds:negativeEmbeds:frames:height:…) is public
// Reference diffusion stand-ins
NFKMLXReferenceModels.registerAll(); let backend = try NFKMLXModelRegistry.backend(named: "diffusion-controlnet", weightsURL: nil)
```

```objc
// Stable Diffusion 1.5 / 2.1 / SDXL-Turbo
[NFKMLXTextToImage backendWithModel:NFKMLXStableDiffusionModelStableDiffusion15 directoryURL:dir error:&error]
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
