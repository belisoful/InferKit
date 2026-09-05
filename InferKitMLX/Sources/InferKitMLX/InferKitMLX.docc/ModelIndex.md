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
| Real-ESRGAN | ``NFKMLXRealESRGAN`` | `NFKRealESRGANNet` | ``NFKMLXRealESRGANVariant`` `.x4` (23 blocks) · `.anime` (6 blocks) · `.x2` | `real-esrgan-x4` · `real-esrgan-x4-anime` · `real-esrgan-x2` | ``NFKMLXModuleBackend`` |
| SwinIR | ``NFKMLXSwinIR`` | `NFKMLXSwinIRNet` | ``NFKMLXSwinIRVariant`` `.classicalX4` / `.classicalX3` / `.classicalX8` / `.lightweightSRX2` (`NFKMLXSwinIRConfiguration.classicalSRx4` …, `.lightweightSRx2`; `.lightweightX2` is the small test geometry) | `swinir-x4` | ``NFKMLXModuleBackend`` |
| NAFNet | ``NFKMLXNAFNet`` | `NFKMLXNAFNetNet` | ``NFKMLXNAFNetVariant`` `.sidd` / `.goPro` / `.reds` (`NFKMLXNAFNetConfiguration.sidd` …) | `nafnet` | ``NFKMLXModuleBackend`` |
| Zero-DCE | ``NFKMLXZeroDCE`` | ``NFKMLXZeroDCENet`` | fixed geometry (seven 3×3 convolutions) | `zero-dce` | ``NFKMLXModuleBackend`` |
| Fast style transfer | ``NFKMLXStyleTransfer`` | `NFKStyleTransferNet` | fixed geometry; one style per checkpoint | `fast-style-transfer` | ``NFKMLXModuleBackend`` |
| Colorizer ECCV-16 | ``NFKMLXColorizer`` | `NFKMLXColorizerNet` | `NFKMLXColorizerConfiguration.eccv16` | `colorizer-eccv16` | ``NFKMLXModuleBackend`` |
| Colorizer SIGGRAPH-17 | ``NFKMLXSiggraphColorizer`` | `NFKMLXSiggraphNet` | fixed geometry (four-channel input, hints optional) | `colorizer-siggraph17` | ``NFKMLXModuleBackend`` |
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
// NAFNet
let backend = try NFKMLXNAFNet.backend(variant: .sidd, weightsURL: url)
// .goPro / .reds · …VariantGoPro / …VariantReds
// Zero-DCE
let backend = try NFKMLXZeroDCE.backend(weightsURL: url)
// Fast style transfer
let backend = try NFKMLXStyleTransfer.backend(weightsURL: url)
// Colorizer ECCV-16
let backend = try NFKMLXColorizer.backend(weightsURL: url)
// Colorizer SIGGRAPH-17
let backend = try NFKMLXSiggraphColorizer.backend(weightsURL: url)
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
// NAFNet
[NFKMLXNAFNet backendWithVariant:NFKMLXNAFNetVariantSidd weightsURL:url error:&error]
// Zero-DCE
[NFKMLXZeroDCE backendWithWeightsURL:url error:&error]
// Fast style transfer
[NFKMLXStyleTransfer backendWithWeightsURL:url error:&error]
// Colorizer ECCV-16
[NFKMLXColorizer backendWithWeightsURL:url error:&error]
// Colorizer SIGGRAPH-17
[NFKMLXSiggraphColorizer backendWithWeightsURL:url error:&error]
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
| Marigold depth | ``NFKMLXMarigold`` | ``NFKMLXSDUNet`` + ``NFKMLXSDAutoencoder`` | `NFKMLXSDUNetConfiguration.marigold` | `marigold-depth` | ``NFKMLXDiffusionBackend`` |
| SegFormer | ``NFKMLXSegFormer`` | ``NFKMLXSegFormerNet`` | `NFKMLXSegFormerConfiguration.mitB0`; `network(weightsURL:classCount:)` for a custom head | `segformer-b0` | ``NFKMLXModuleBackend`` |
| DeepLabV3 | ``NFKMLXDeepLab`` | `NFKMLXDeepLabNet` over `NFKMLXResNetBackbone` | `NFKMLXDeepLabConfiguration.base` (`NFKMLXResNetConfiguration.deepLab`) | `deeplabv3` | ``NFKMLXModuleBackend`` |
| BiSeNet V1 | ``NFKMLXBiSeNet`` | `NFKMLXBiSeNetNet` | `NFKMLXBiSeNetConfiguration.base` (ResNet-18 context path) | `bisenet` | ``NFKMLXModuleBackend`` |
| BiSeNet V2 | ``NFKMLXBiSeNetV2`` | `NFKMLXBiSeNetV2Net` | fixed geometry (older pixel-shuffle head) | `bisenet-v2` | ``NFKMLXModuleBackend`` |

```swift
// Depth Anything V2
let backend = try NFKMLXDepthAnything.backend(variant: .small, weightsURL: url)
// .base / .large · …VariantBase / …VariantLarge
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
| Robust Video Matting | ``NFKMLXRVM`` | `NFKMLXRVMNet` | `NFKMLXRVMConfiguration.large` (MobileNetV3-Large); `downsampleRatio` for the guided-filter path | `robust-video-matting` | ``NFKMLXMattingBackend`` (single frame) / `NFKMLXRVMNet.forward` (video) |
| MODNet | ``NFKMLXMODNet`` | `NFKMLXMODNetNet` | `NFKMLXMODNetConfiguration.base` | `modnet` | ``NFKMLXMattingBackend`` |
| SAM | ``NFKMLXSAM`` | `NFKMLXSAMNet` | ``NFKMLXSAMVariant`` `.vitB` (`NFKMLXSAMConfiguration.vitB`); `.compact` for tests | `sam` | ``NFKMLXMattingBackend`` (point under `NFKSAMPointKey`) |
| SAM 2 | ``NFKMLXSAM2`` | `NFKMLXSAM2EncoderNet`, `NFKMLXSAM2MemoryEncoderNet`, `NFKMLXSAM2MemoryAttentionNet` | `NFKMLXSAM2Configuration.tiny` / `.basePlus` / `.large` | — | Swift API (`MLXArray`) |
| RetinaFace | ``NFKMLXRetinaFace`` | `NFKMLXRetinaFaceNet` | `NFKMLXRetinaFaceConfiguration()` = mobile0.25 | `retinaface-mobile025` | detection backend; `detector(weightsURL:)` for landmarks |
| Face alignment | ``NFKMLXFaceAlignment`` | — | ``NFKMLXRetinaFaceDetector`` (default) or ``NFKMLXVisionFaceDetector`` | — | used by ``NFKMLXPhotoFaceBackend`` |

```swift
// U²-Net
let backend = try NFKMLXU2Net.backend(variant: .full, weightsURL: url)
// .light · …VariantLight
// Robust Video Matting
let backend = try NFKMLXRVM.backend(weightsURL: url)
// MODNet
let backend = try NFKMLXMODNet.backend(weightsURL: url)
// SAM
let backend = try NFKMLXSAM.backend(variant: .vitB, weightsURL: url)
// SAM 2
// no public constructor yet — NFKMLXSAM2's makeEncoder / loadWeights(into:from:) and the decoder and memory loaders are internal; only NFKMLXSAM2Configuration is public
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
// Robust Video Matting
[NFKMLXRVM backendWithWeightsURL:url error:&error]
// MODNet
[NFKMLXMODNet backendWithWeightsURL:url error:&error]
// SAM
[NFKMLXSAM backendWithVariant:NFKMLXSAMVariantVitB weightsURL:url error:&error]
// RetinaFace
[NFKMLXRetinaFace backendWithWeightsURL:url error:&error]
```


### Detection and pose


| Model | Entry class | Network | Configuration for the released weights | Registered name | Base backend |
| --- | --- | --- | --- | --- | --- |
| YOLOv8 | ``NFKMLXYOLO`` | `NFKMLXYOLONet` | ``NFKMLXYOLOVariant`` `.nano` / `.small` / `.medium` / `.large` / `.extraLarge` (`NFKMLXYOLOConfiguration.base`, `.small` …) | `yolo` | ``NFKMLXYOLOBackend`` (`labels:`) |
| RT-DETR | ``NFKMLXRTDetr`` | ``NFKMLXRTDetrNet`` | `NFKMLXRTDetrConfiguration.r50vd` | `rtdetr` | ``NFKMLXRTDetrBackend`` (`labels:`) |
| SimpleBaseline pose | ``NFKMLXPose`` | `NFKMLXPoseNet` over `NFKMLXResNetBackbone` | `NFKMLXPoseConfiguration.simpleBaseline` (ResNet-50, 256×192) | `pose-simplebaseline` | ``NFKMLXPoseBackend`` (`jointNames:`) |

```swift
// YOLOv8
let backend = try NFKMLXYOLO.backend(variant: .nano, weightsURL: url, labels: nil)
// .small / .medium / .large / .extraLarge · …VariantSmall … …VariantExtraLarge
// RT-DETR
let backend = try NFKMLXRTDetr.backend(weightsURL: url, labels: nil)
// SimpleBaseline pose
let backend = try NFKMLXPose.backend(weightsURL: url, jointNames: nil)
```

```objc
// YOLOv8
[NFKMLXYOLO backendWithVariant:NFKMLXYOLOVariantNano weightsURL:url labels:nil error:&error]
// RT-DETR
[NFKMLXRTDetr backendWithWeightsURL:url labels:nil error:&error]
// SimpleBaseline pose
[NFKMLXPose backendWithWeightsURL:url jointNames:nil error:&error]
```


### Embeddings, reranking, and vision-language


| Model | Entry class | Network | Configuration for the released weights | Registered name / factory | Base backend |
| --- | --- | --- | --- | --- | --- |
| CLIP ViT-B/32 | ``NFKMLXCLIP`` | ``NFKMLXCLIPNet`` | `NFKMLXCLIPConfiguration.base` | `clip-vit-b-32` | ``NFKMLXCLIPBackend`` |
| CLIP probe | ``NFKMLXCLIPProbe`` | linear head over frozen CLIP | trained on device (`trainProbe`) | `clip-probe` | ``NFKMLXCLIPProbeBackend`` |
| SigLIP 2 | ``NFKMLXSigLIP2`` | `NFKMLXSigLIP2Net` (`NFKSigLIP2VisionNet`, `NFKSigLIP2TextNet`) | `NFKMLXSigLIP2Configuration.base` (base-patch16-224) | `siglip2-base-patch16-224` | ``NFKMLXSigLIP2Backend`` |
| Qwen3-Embedding-0.6B | ``NFKMLXQwen3Embedding`` | ``NFKMLXLanguageNet`` | ``NFKMLXLanguageConfiguration`` read from `config.json`; ``NFKMLXTextEmbedderConfiguration`` (`.lastToken`, appended `<\|endoftext\|>`, Matryoshka `dimensions`) | `backend(directoryURL:)` | ``NFKMLXTextEmbeddingBackend`` |
| EmbeddingGemma-300M | ``NFKMLXEmbeddingGemma`` | ``NFKMLXGemma3EncoderNet`` | `NFKMLXGemma3EncoderConfiguration.embeddingGemma300M` + the `2_Dense` / `3_Dense` projections | `backend(directoryURL:)` (`embeddinggemma-300m`) | ``NFKMLXTextEmbeddingBackend`` |
| gte-reranker-modernbert-base | ``NFKMLXModernBERTReranker`` | ``NFKMLXModernBertRerankerNet`` | `NFKMLXModernBertConfiguration.gteReranker` | `reranker(directoryURL:)` | scoring object (`scores(query:documents:)`) |
| SmolVLM2-500M | ``NFKMLXSmolVLM`` | ``NFKMLXSmolVLMNet`` (``NFKMLXSigLIPNet`` + ``NFKMLXSmolVLMConnector`` + ``NFKMLXLanguageNet``) | `NFKMLXSigLIPConfiguration.smolVLM`, `.smolVLM2Decoder` | `smolVLM(directoryURL:)` | object (`answer(image:question:)`) |
| Qwen3-VL-2B vision tower | ``NFKMLXQwen3VL`` | ``NFKMLXQwen3VLVisionNet`` | `NFKMLXQwen3VLVisionConfiguration.qwen3VL2B` | — | Swift API |
| Gemma 4 vision / audio / fusion | ``NFKMLXGemma4ConditionalGeneration`` | ``NFKMLXGemma4VisionNet``, ``NFKMLXGemma4AudioNet``, ``NFKMLXGemma4MultimodalEmbedder``, ``NFKMLXGemmaNet`` | ``NFKMLXGemma4VisionConfiguration`` / ``NFKMLXGemma4AudioConfiguration`` read from the release (`useClippedLinears` on) | — | Swift API (``NFKMLXGemma4ImageProcessor``, ``NFKMLXGemma4AudioFeatureExtractor``) |

```swift
// CLIP ViT-B/32
let backend = try NFKMLXCLIP.backend(weightsURL: url)
// CLIP probe
let net = try NFKMLXCLIPProbe.network(weightsURL: url, configuration: .base)
// then trainProbe and probeBackend(net:probe:labels:); Swift only
// SigLIP 2
let backend = try NFKMLXSigLIP2.backend(weightsURL: url)
// Qwen3-Embedding-0.6B
let backend = try NFKMLXQwen3Embedding.backend(directoryURL: dir)
// Matryoshka: backend(directoryURL: dir, dimensions: 256) · backendWithDirectoryURL:dir outputDimensions:256 error:&error
// EmbeddingGemma-300M
let backend = try NFKMLXEmbeddingGemma.backend(directoryURL: dir)
// backend(directoryURL: dir, dimensions: 256) · backendWithDirectoryURL:dir outputDimensions:256 error:&error
// gte-reranker-modernbert-base
let reranker = try NFKMLXModernBERTReranker.reranker(directoryURL: dir)
// then scores(query:documents:) · scoresForQuery:documents:
// SmolVLM2-500M
let vlm = try NFKMLXSmolVLM.load(directoryURL: dir)
// then answer(image:question:) · answerForImage:question:
// Qwen3-VL-2B vision tower
let vision = try NFKMLXQwen3VL.visionNet(directoryURL: dir)
// Swift only
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
// SmolVLM2-500M
[NFKMLXSmolVLM smolVLMWithDirectoryURL:dir error:&error]
```


### Language models


| Model | Entry class | Network | Configuration for the released weights | Factory | Base backend |
| --- | --- | --- | --- | --- | --- |
| Qwen3 (dense), Qwen2, Llama | ``NFKMLXLanguage`` | ``NFKMLXLanguageNet`` | ``NFKMLXLanguageConfiguration`` from `config.json` (presets `.qwen3_0_6B`, `.qwen3_1_7B`, `.qwen3_4B`, `.qwen3_8B`); ``NFKMLXGenerationOptions`` per request | `let backend = try NFKMLXLanguage.backend(directoryURL: dir)`<br>`[NFKMLXLanguage backendWithDirectoryURL:dir error:&error]`<br>speculative: `backend(directoryURL: dir, draftDirectoryURL: draftDir)` · `backendWithDirectoryURL:dir draftDirectoryURL:draftDir error:&error` | ``NFKMLXLanguageBackend`` |
| Qwen3-MoE, Mixtral | ``NFKMLXLanguage`` | ``NFKMLXLanguageNet`` + `NFKLMMixtureFeedForward` | the same reader (`qwen3_moe`, `mixtral` model types); `.tinyMixture` for tests | `let backend = try NFKMLXLanguage.backend(directoryURL: dir)`<br>`[NFKMLXLanguage backendWithDirectoryURL:dir error:&error]` | ``NFKMLXLanguageBackend`` |
| Dense GGUF (`llama` / `qwen2` / `qwen3`) | ``NFKMLXLanguage`` | ``NFKMLXLanguageNet`` | `configuration(fromGGUF:)` from the file's metadata | `let backend = try NFKMLXLanguage.backend(ggufURL: url)`<br>`[NFKMLXLanguage backendWithGGUFURL:url error:&error]` | ``NFKMLXLanguageBackend`` |
| Qwen3.5 / 3.6 / 3.8 | ``NFKMLXHybridLanguage`` | ``NFKMLXHybridLanguageNet`` | ``NFKMLXHybridConfiguration`` from `config.json` (`.qwen3_8_27B` preset) | `let config = try NFKMLXHybridLanguage.configuration(fromHuggingFace: dir.appendingPathComponent("config.json"))`<br>**no public constructor yet** — `makeNet` and `loadWeights(into:fromDirectory:)` are internal, so there is no backend factory for the hybrid yet | — (prefill-only) |
| Gemma 4 E2B / E4B / 26B-A4B | ``NFKMLXGemmaLanguage`` | ``NFKMLXGemmaNet`` | ``NFKMLXGemmaConfiguration`` from `config.json` (`.e2b`; `enable_moe_block` turns on the routed branch) | `let backend = try NFKMLXGemmaLanguage.backend(directoryURL: dir)`<br>`[NFKMLXGemmaLanguage gemmaBackendWithDirectoryURL:dir error:&error]` | ``NFKMLXGemmaBackend`` |
| Gemma 4 12B unified | ``NFKMLXGemmaLanguage`` | ``NFKMLXGemma4UnifiedNet`` | `unifiedConfiguration(fromHuggingFace:)` (`.twelveB`) | `let backend = try NFKMLXGemmaLanguage.backend(directoryURL: dir)`<br>`[NFKMLXGemmaLanguage gemmaBackendWithDirectoryURL:dir error:&error]` | ``NFKMLXGemmaBackend`` |
| Gemma 2 | — | ``NFKMLXGemma2Net`` | `NFKMLXGemma2Configuration.gemma2_2B` | `let net = NFKMLXGemma2Net(.gemma2_2B)`<br>no public weight loader yet (the parity test loads through the internal reader); Swift only | SANA's text encoder |
| DeepSeek V4 Flash / Pro | ``NFKMLXDeepSeek`` | ``NFKMLXDeepSeekNet`` | `NFKMLXDeepSeekConfiguration.v4Flash` / `.v4Pro`; ``NFKMLXDeepSeekQuantization`` decodes the fp8 / fp4 storage | `let config = try NFKMLXDeepSeek.configuration(fromHuggingFace: configURL)`<br>**no public constructor yet** — `makeNet` is internal; `NFKMLXDeepSeek.dequantized(_:shapes:)` decodes the fp8 / fp4 storage | — (released weights exceed a workstation) |
| T5 v1.1 / umT5 | ``NFKMLXT5Encoder`` | `NFKMLXT5EncoderNet` | `NFKMLXT5Configuration.xxl` / `.umt5XXL` | `let t5 = try NFKMLXT5Encoder.encoder(configuration: .xxl, directory: dir)`<br>umT5: `configuration: .umt5XXL`; Swift only | LTX and Wan text conditioning |
| Chat templates | ``NFKMLXChatTemplateRenderer`` | — | `NFKMLXChatTemplate.jinja(template:bosToken:eosToken:)` / `.chatML` | `options.chatTemplate = .jinja(template: template, bosToken: bos, eosToken: eos)`<br>ObjC: `NFKMLXGenerationParameterKey.chatTemplate` request parameter carrying the Jinja source | used by ``NFKMLXLanguageBackend`` |
| Constrained decoding | ``NFKMLXJSONConstraint``, ``NFKMLXChoiceConstraint`` | ``NFKMLXVocabulary`` | `root`, `maximumWhitespaceRun`; `outputFormat` / `choices` request keys | `options.constraint = NFKMLXJSONConstraint(root: .object)`<br>ObjC: `outputFormat` = `"json-object"` or `choices` request parameter | used by ``NFKMLXLanguageBackend`` |

```swift
// Qwen3 (dense), Qwen2, Llama
// backend(directoryURL:), backend(directoryURL:draftDirectoryURL:)
// Qwen3-MoE, Mixtral
backend(directoryURL:)
// Dense GGUF (`llama` / `qwen2` / `qwen3`)
backend(ggufURL:)
// Qwen3.5 / 3.6 / 3.8
// Swift makeNet / loadWeights
// Gemma 4 E2B / E4B / 26B-A4B
backend(directoryURL:)
// Gemma 4 12B unified
backend(directoryURL:)
// Gemma 2
// Swift API
// DeepSeek V4 Flash / Pro
// Swift API
// T5 v1.1 / umT5
encoder(configuration:directory:)
// Chat templates
// chatTemplate request parameter
// Constrained decoding
// —
```


### Video


| Model | Entry class | Network | Configuration for the released weights | Registered name | Base backend |
| --- | --- | --- | --- | --- | --- |
| RIFE HDv3 | ``NFKMLXRIFE`` | `NFKMLXRIFENet` | fixed geometry (three IFBlocks, width 90, scales 4/2/1) | `rife` | ``NFKMLXTensorBackend`` (`frame0` / `frame1`); `clipBackend(weightsURL:)` for a clip |
| RIFE v4 | ``NFKMLXRIFEv4`` | `NFKMLXRIFEv4Net` | fixed geometry (four blocks, timestep input, scales 8/4/2/1) | `rife-v4` | ``NFKMLXTensorBackend`` |
| RAFT | ``NFKMLXRAFT`` | `NFKMLXRAFTNet` | RAFT-large (feature 256, 4 levels, radius 4); `iterations` (default 6) | `raft` | ``NFKMLXTensorBackend`` |
| BasicVSR | ``NFKMLXVideoSR`` | `NFKMLXVideoSRNet` + `NFKVSRSPyNet` | `NFKMLXVideoSRConfiguration.base` (×4) | `video-super-resolution` | ``NFKMLXModuleBackend`` (frame) / `clipBackend(weightsURL:)` |
| SD ×4 upscaler | ``NFKMLXSDUpscaler`` | ``NFKMLXSDUNet`` + ``NFKMLXSDAutoencoder`` | `NFKMLXSDUNetConfiguration.upscaler`, `NFKMLXSDVAEConfiguration.upscaler`; `noiseLevel` (20) | `sd-x4-upscaler` | ``NFKMLXDiffusionBackend`` |
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
```


### Audio


| Model | Entry class | Network | Configuration for the released weights | Registered name | Base backend |
| --- | --- | --- | --- | --- | --- |
| Whisper | ``NFKMLXWhisper`` | ``NFKMLXWhisperNet`` | ``NFKMLXWhisperVariant`` `.tiny` / `.small` / `.medium` / `.largeV3` (`NFKMLXWhisperConfiguration.tiny` …); `emitsTimestamps` | `whisper-tiny` | ``NFKMLXWhisperBackend`` (`backend(variant:weightsURL:tokenizer:timestamps:)`) |
| Parakeet-TDT | ``NFKMLXParakeet`` | ``NFKMLXParakeetNet`` | `NFKMLXParakeetConfiguration.tdt06B` (0.6B v2: 24 rel-pos conformer layers, TDT durations 0…4) | `parakeet-tdt`; `backend(directoryURL:)` | ``NFKMLXParakeetBackend`` (text + per-token `NFKOutputSegments`) |
| Chatterbox | ``NFKMLXChatterbox`` | ``NFKMLXChatterboxTTS`` (``NFKMLXChatterboxVoiceEncoderNet``, ``NFKMLXS3TokenizerNet``, ``NFKMLXT3Net``, ``NFKMLXS3GenNet``) | `.released` on every stage (VoiceEncoder 3×256, S3 tokenizer 6×1280, T3 Llama 520M with llama3 rope, S3Gen flow + HiFT) | `chatterbox`; `speechBackend(directoryURL:voiceURL:)` | ``NFKMLXSpeechBackend`` (24 kHz WAV; text → cloned voice) |
| Demucs v2 | ``NFKMLXDemucs`` | `NFKMLXDemucsNet` | `NFKMLXDemucsConfiguration()` = music (stereo, depth 6, 4 stems, BLSTM, context 3) | `demucs` | ``NFKMLXDemucsBackend`` |
| Speech denoiser | ``NFKMLXDenoiser`` | `NFKMLXDemucsNet` | ``NFKMLXDemucsConfiguration`` set to dns48 (mono, depth 5, 1 stem, causal, context 1) | `denoiser` | ``NFKMLXDenoiserBackend`` |
| HT Demucs (v4) | ``NFKMLXHTDemucs`` | ``NFKMLXHTDemucsNet`` | `NFKMLXHTDemucsConfiguration.htdemucs` | `htdemucs` | ``NFKMLXHTDemucsBackend`` |
| Conv-TasNet | ``NFKMLXConvTasNet`` | `NFKMLXConvTasNetNet` | `NFKMLXConvTasNetConfiguration.libri2Mix16k`; `perChannelPReLU` optional | `conv-tasnet` | ``NFKMLXConvTasNetBackend`` |
| MarbleNet VAD | ``NFKMLXVAD`` | `NFKMLXVADNet` | `NFKMLXVADConfiguration.marbleNet` | `vad-marblenet` | ``NFKMLXVADBackend`` |
| Silero VAD v6 | ``NFKMLXSileroVAD`` | `NFKMLXSileroVADNet` | `NFKMLXSileroVADConfiguration.v6` | `silero-vad` | ``NFKMLXSileroVADBackend`` |
| PANNs Cnn14 | ``NFKMLXAudioTagger`` | `NFKMLXAudioTaggerNet` | `NFKMLXAudioTaggerConfiguration.panns` | `audio-tagger-panns` | ``NFKMLXAudioTaggerBackend`` (`labels:`) |
| Descript Audio Codec | ``NFKMLXDAC`` | `NFKMLXDACNet` (`NFKDACEncoderNet`, `NFKDACDecoderNet`) | `NFKMLXDACConfiguration.dac44kHz` / `.dac24kHz` / `.dac16kHz` | `dac` | ``NFKMLXDACBackend``; `encode` / `decode` for the tokens |
| SNAC | ``NFKMLXSNAC`` | `NFKMLXSNACNet` (`NFKSNACEncoderNet`, `NFKSNACDecoderNet`) | `NFKMLXSNACConfiguration.snac24kHz` | `snac` | ``NFKMLXSNACBackend``; `decode(_:deterministic:)` |

```swift
// Whisper
let backend = try NFKMLXWhisper.backend(variant: .tiny, weightsURL: url, tokenizer: tokenizer, timestamps: false)
// Parakeet-TDT (an unpacked .nemo directory)
let backend = try NFKMLXParakeet.backend(directoryURL: dir)
// Chatterbox (a release directory; nil voice = the built-in conds.pt)
let backend = try NFKMLXChatterbox.speechBackend(directoryURL: dir, voiceURL: voiceWAV)
// .small / .medium / .largeV3 · …VariantSmall / …VariantMedium / …VariantLargeV3
// Demucs v2
let backend = try NFKMLXDemucs.backend(weightsURL: url)
// Speech denoiser
let backend = try NFKMLXDenoiser.backend(weightsURL: url)
// HT Demucs (v4)
let backend = try NFKMLXHTDemucs.backend(weightsURL: url)
// Conv-TasNet
let backend = try NFKMLXConvTasNet.backend(weightsURL: url)
// MarbleNet VAD
let backend = try NFKMLXVAD.backend(weightsURL: url)
// Silero VAD v6
let backend = try NFKMLXSileroVAD.backend(weightsURL: url)
// PANNs Cnn14
let backend = try NFKMLXAudioTagger.backend(weightsURL: url, labels: nil)
// Descript Audio Codec
let backend = try NFKMLXDAC.backend(weightsURL: url)
// tokens: let codec = try NFKMLXDAC.codec(configuration: .dac44kHz, weightsURL: url) then codec.encode(samples) / codec.decode(codes)
// SNAC
let backend = try NFKMLXSNAC.backend(weightsURL: url)
// tokens: let codec = try NFKMLXSNAC.codec(configuration: .snac24kHz, weightsURL: url) then codec.encode(samples) / codec.decode(codes, deterministic: true)
```

```objc
// Whisper
[NFKMLXWhisper backendWithVariant:NFKMLXWhisperVariantTiny weightsURL:url tokenizer:tokenizer timestamps:NO error:&error]
// Parakeet-TDT
[NFKMLXParakeet backendWithDirectoryURL:dir error:&error]
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
| Z-Image | ``NFKMLXZImagePipeline`` | ``NFKMLXZImageTransformerNet`` + ``NFKMLXSDAutoencoder`` (`.flux`) | `NFKMLXZImageConfiguration.base`; `NFKMLXFlowMatchConfiguration.zImage`; caption from the Qwen3-4B penultimate layer | Swift API (`generate(promptEmbeds:…)`, `generate(image:strength:…)`) | flow scheduler |
| SANA | ``NFKMLXSANAPipeline`` | ``NFKMLXSANATransformerNet`` + ``NFKMLXDCAutoencoderNet`` | `NFKMLXSANAConfiguration.base`; `NFKMLXDCAEConfiguration.sana`; `NFKMLXDPMSolverConfiguration.sana`; caption from ``NFKMLXGemma2Net`` | Swift API | ``NFKMLXDPMSolverScheduler`` |
| LTX-Video | ``NFKMLXLTXPipeline`` | `NFKMLXLTXTransformerNet` + `NFKMLXLTXVideoVAENet` | `NFKMLXLTXTransformerConfiguration.base`; `NFKMLXLTXVAEConfiguration.base`; `NFKMLXFlowMatchConfiguration.ltxVideo`; `NFKMLXT5Configuration.xxl` | Swift API (`generate(promptTokens:…)`) | ``NFKMLXFlowMatchScheduler`` |
| Wan | ``NFKMLXWanPipeline`` | ``NFKMLXWanTransformerNet`` + ``NFKMLXWanVideoVAENet`` | `NFKMLXWanConfiguration.base`; `NFKMLXWanVAEConfiguration.wan22` / `.wan21`; `NFKMLXUniPCConfiguration.wan`; `NFKMLXT5Configuration.umt5XXL` | Swift API (`generate(textEmbeds:…)`) | ``NFKMLXUniPCScheduler`` |
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
// Swift only
// Z-Image
let dit = NFKMLXZImageTransformerNet(.base); let vae = NFKMLXSDAutoencoder(configuration: .flux)
// no public constructor yet — NFKMLXZImagePipeline's initializers are internal and the DiT has no public loader; generate(promptEmbeds:negativeEmbeds:latentHeight:…) and generate(image:…strength:…) are public once a pipeline exists
// SANA
let dit = NFKMLXSANATransformerNet(.base); let vae = NFKMLXDCAutoencoderNet(.sana); try NFKMLXDCAutoencoderNet.loadWeights(into: vae, from: vaeURL)
// no public constructor yet — NFKMLXSANAPipeline's initializers are internal and the DiT has no public loader; generate(promptEmbeds:negativeEmbeds:latentHeight:…) is public
// LTX-Video
let vae = try NFKMLXLTXVideoVAE.vae(configuration: .base, weightsURL: vaeURL); let t5 = try NFKMLXT5Encoder.encoder(configuration: .xxl, directory: t5Dir)
// no public constructor yet — NFKMLXLTXTransformerNet and NFKMLXLTXPipeline's initializers are internal; generate(promptTokens:negativeTokens:frames:height:…) is public
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
