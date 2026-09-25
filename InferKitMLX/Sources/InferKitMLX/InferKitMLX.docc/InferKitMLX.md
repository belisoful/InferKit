# ``InferKitMLX``

MLX-backed inference on Apple Silicon — a gallery of ready-to-run models across image, video, audio,
language, and generative diffusion, plus bring-your-own-model backends, all behind the InferKit contract.

@Metadata {
    @DisplayName("InferKitMLX")
}

## Overview

This companion adds [MLX](https://github.com/ml-explore/mlx-swift)-backed inference on top of the
InferKit core. It keeps MLX out of the core, so the core stays cross-platform and dependency-free while
this package targets Apple Silicon (macOS 14 / iOS 17). Every model adopts the same `NFKInferenceBackend`
protocol, so an InferKit consumer runs an MLX model exactly like any other backend.

Three ways to use it:

- **Shipped models** — sixty-plus real models, each implemented in `MLXNN` and measured against its
  reference implementation on the released weights, each with a public `@objc` factory. See
  <doc:ModelGallery> for what each does and <doc:ModelIndex> for the class, configuration, and a
  construction line to copy.
- **Generative pipelines** — Stable Diffusion, Z-Image, and SANA text-to-image; LTX-Video and Wan
  text-to-video; on-device language models (Qwen3, Qwen3.5, Gemma 3, Gemma 3n, Gemma 4, DeepSeek V4 and V4.1, any dense GGUF); and
  MiniMax Music 3 text-to-music. See <doc:DiffusionAndSchedulers> and ``NFKMLXLanguageBackend``.
- **Bring your own** — supply an MLX forward closure and let a base backend handle the InferKit contract
  and the image/audio bridge. See <doc:BringYourOwnBackends>.

```swift
import InferKit
import InferKitMLX

// A shipped model through its @objc factory (nil weights → random init, useful for wiring/tests):
let upscaler = try NFKMLXRealESRGAN.backend(variant: .x4, weightsURL: weightsURL)
let result = try upscaler.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: cgImage]))

// A language model from a downloaded Hugging Face release directory:
let llm = try NFKMLXLanguage.backend(directoryURL: releaseDirectory)
let reply = try llm.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: "Explain diffraction."])).text
```

Weights are downloaded at runtime, not bundled — for size, licensing, and update reasons. See
<doc:WeightsAndConversion>.

![The InferKitMLX model gallery grouped by modality: image, video, audio, generation, and language.](model-gallery)

### Activating core capabilities

Linking this package ships two dynamic-backend providers, so core capabilities light up with no
registration:

- ``NFKStableDiffusionProvider`` → the core's `stable-diffusion` capability (``NFKMLXBackend``, SD 1.5:
  the ungated release, so it activates with no credential).
- ``NFKMLXWhisperProvider`` → the core's `transcription` capability (`NFKMLXWhisper`).
- ``NFKMLXTranslationProvider`` → the core's `translation` capability (M2M-100 when its release is cached).

### Sizing a model against the machine

``NFKMLXModelSizing`` counts a decoder's parameters from its geometry before anything is allocated,
measures the machine's memory bandwidth, and answers whether a release fits the working set — and at
what context window. ``NFKMLXGPU`` reports what the machine has and sets standing cache and memory
limits.

## Topics

### Concepts

- <doc:ModelGallery>
- <doc:ModelIndex>
- <doc:BringYourOwnBackends>
- <doc:DiffusionAndSchedulers>
- <doc:WeightsAndConversion>

### Image → image

- ``NFKMLXRealESRGAN``
- ``NFKMLXSwinIR``
- ``NFKMLXHAT``
- ``NFKMLXNAFNet``
- ``NFKMLXZeroDCE``
- ``NFKMLXZeroDCEPlus``
- ``NFKMLXStyleTransfer``
- ``NFKMLXAdaIN``
- ``NFKMLXColorizer``
- ``NFKMLXSiggraphColorizer``
- ``NFKMLXDDColor``
- ``NFKMLXDDColorVariant``
- ``NFKMLXLaMa``
- ``NFKMLXStableDiffusionInpaint``
- ``NFKMLXTAESD``

### Image → map (depth & segmentation)

- ``NFKMLXDepthAnything``
- ``NFKMLXDepthAnything3``
- ``NFKMLXDepth3Estimator``
- ``NFKMLXDepth3Camera``
- ``NFKMLXMarigold``
- ``NFKMLXSegFormer``
- ``NFKMLXDeepLab``
- ``NFKMLXBiSeNet``
- ``NFKMLXBiSeNetV2``

### Matting, segmentation & faces

- ``NFKMLXU2Net``
- ``NFKMLXISNet``
- ``NFKMLXRVM``
- ``NFKMLXMODNet``
- ``NFKMLXBiRefNet``
- ``NFKMLXSAM``
- ``NFKMLXSAM2``
- ``NFKMLXSAM2TrackerNet``
- ``NFKMLXSAM2TrackerSession``
- ``NFKMLXSAM3``
- ``NFKMLXSAM3ImageModel``
- ``NFKMLXSAM3VisionNet``
- ``NFKMLXSAM3TextNet``
- ``NFKMLXSAM3DetectorNet``
- ``NFKMLXCodeFormer``
- ``NFKMLXPhotoFaceBackend``
- ``NFKMLXFaceAlignment``
- ``NFKMLXRetinaFace``
- ``NFKMLXRetinaFaceDetector``
- ``NFKMLXVisionFaceDetector``

### Detection & pose

- ``NFKMLXYOLO``
- ``NFKMLXYOLOGenerations``
- ``NFKMLXYOLOGenerationBackend``
- ``NFKMLXRTDetr``
- ``NFKMLXRTDetrSamplingMethod``
- ``NFKMLXRFDetr``
- ``NFKMLXTableTransformer``
- ``NFKMLXTableTransformerSizing``
- ``NFKMLXPose``
- ``NFKMLXVitPose``
- ``NFKMLXVitPoseBackend``
- ``NFKMLXVitPoseVariant``
- ``NFKMLXVitPoseDecoder``

### Embeddings & reranking

- ``NFKMLXCLIP``
- ``NFKMLXSigLIP2``
- ``NFKMLXQwen3Embedding``
- ``NFKMLXEmbeddingGemma``
- ``NFKMLXTextEmbeddingBackend``
- ``NFKMLXModernBERTReranker``
- ``NFKMLXLaya``
- ``NFKMLXLayaBackend``
- ``NFKMLXLayaVariant``
- ``NFKMLXOpenJevDeBERTa``
- ``NFKMLXOpenJev``
- ``NFKMLXOpenJevVariant``
- ``NFKMLXOpenJevRelease``
- ``NFKMLXDecisionBackend``
- ``NFKMLXDecisionModel``
- ``NFKMLXDecisionTokenizer``
- ``NFKMLXQwen3VLEmbedder``
- ``NFKMLXQwen3VLReranker``
- ``NFKMLXChronos``

### Language models

- ``NFKMLXLanguage``
- ``NFKMLXLanguageBackend``
- ``NFKMLXGenerationOptions``
- ``NFKMLXGenerationParameterKey``
- ``NFKMLXReasoningFormat``
- ``NFKMLXKeyValueCache``
- ``NFKMLXPromptCache``
- ``NFKMLXSpeculativeReport``
- ``NFKMLXJSONConstraint``
- ``NFKMLXJSONSchemaConstraint``
- ``NFKMLXJSONSchema``
- ``NFKMLXChoiceConstraint``
- ``NFKMLXVocabulary``
- ``NFKMLXQwen4Exp``
- ``NFKMLXHybridLanguage``
- ``NFKMLXGemma3``
- ``NFKMLXGemma3n``
- ``NFKMLXGemma3Backend``
- ``NFKMLXGemma3Language``
- ``NFKMLXGemma3Net``
- ``NFKMLXGemma3Configuration``
- ``NFKMLXGemma3Cache``
- ``NFKMLXGemmaLanguage``
- ``NFKMLXGemmaBackend``
- ``NFKMLXDeepSeek``
- ``NFKMLXDeepSeekBackend``
- ``NFKMLXDeepSeekCache``
- ``NFKMLXDeepSeekPaging``
- ``NFKMLXDeepSeekPagingMode``
- ``NFKMLXDeepSeekLoadOptions``
- ``NFKMLXDeepSeekExpertStore``
- ``NFKMLXMappedFile``
- ``NFKMLXSafetensors``
- ``NFKMLXDeepSeekImageProcessor``
- ``NFKMLXDeepSeekImageStack``
- ``NFKMLXDeepSeekDraftStack``
- ``NFKMLXDeepSeekVisionNet``
- ``NFKMLXDeepSeekAligner``
- ``NFKMLXMamba``
- ``NFKMLXMambaBackend``
- ``NFKMLXGraniteHybrid``
- ``NFKMLXGraniteBackend``
- ``NFKMLXNemotronH``
- ``NFKMLXNemotronBackend``
- ``NFKMLXGemma2Net``
- ``NFKMLXT5Encoder``
- ``NFKMLXRoPEScaling``
- ``NFKMLXModelSizing``
- ``NFKMLXModelFit``

### Translation

- ``NFKMLXMarian``
- ``NFKMLXMarianTranslator``
- ``NFKMLXM2M100``
- ``NFKMLXM2M100Variant``
- ``NFKMLXM2M100Translator``
- ``NFKMLXMADLAD``
- ``NFKMLXMADLADConfiguration``
- ``NFKMLXMADLADTranslator``
- ``NFKMLXTranslateGemma``
- ``NFKMLXTranslateGemmaTranslator``
- ``NFKMLXTranslationBackend``
- ``NFKMLXTranslationParameterKey``
- ``NFKMLXTranslator``
- ``NFKMLXSeq2SeqConfiguration``
- ``NFKMLXSeq2SeqCache``
- ``NFKMLXSeq2SeqDecoding``
- ``NFKMLXSeq2SeqDecoder``
- ``NFKMLXSeq2SeqDecodable``
- ``NFKMLXSentencePieceModel``
- ``NFKMLXSentencePieceSegmenter``
- ``NFKMLXSentencePieceTokenizer``

### Vision-language

- ``NFKMLXSmolVLM``
- ``NFKMLXSigLIPNet``
- ``NFKMLXSmolVLMConnector``
- ``NFKMLXFlorence2``
- ``NFKMLXFlorence2Net``
- ``NFKMLXFlorence2VisionNet``
- ``NFKMLXFlorence2Projector``
- ``NFKMLXFlorence2Backend``
- ``NFKMLXTrOCR``
- ``NFKMLXTrOCRNet``
- ``NFKMLXTrOCRVisionNet``
- ``NFKMLXTrOCRBackend``
- ``NFKMLXSa2VA``
- ``NFKMLXSa2VANet``
- ``NFKMLXSa2VAVisionNet``
- ``NFKMLXSa2VABackend``
- ``NFKMLXSa2VAConfiguration``
- ``NFKMLXSa2VATemplate``
- ``NFKMLXSa2VAQwen``
- ``NFKMLXSa2VAQwenNet``
- ``NFKMLXSa2VALLaVA``
- ``NFKMLXSa2VALLaVANet``
- ``NFKMLXInternLM2Tokenizer``
- ``NFKMLXPhi4MM``
- ``NFKMLXPhi4MMModel``
- ``NFKMLXPhi4MMImageNet``
- ``NFKMLXPhi4MMAudioNet``
- ``NFKMLXPhi4MMImageProcessor``
- ``NFKMLXPhi4MMAudioFeatures``
- ``NFKMLXPhi4MMImageInput``
- ``NFKMLXPhi4MMAudioInput``
- ``NFKMLXPhi4MMBackend``
- ``NFKMLXGemma3Model``
- ``NFKMLXGemma3VisionNet``
- ``NFKMLXGemma3MultimodalProjector``
- ``NFKMLXGemma3ImageProcessor``
- ``NFKMLXGemma3Tokens``
- ``NFKMLXQwen3VL``
- ``NFKMLXQwen3VLVisionNet``
- ``NFKMLXQwen25VLVisionNet``
- ``NFKMLXQwen25VLVisionConfiguration``
- ``NFKMLXMRoPELayout``
- ``NFKMLXPixtral``
- ``NFKMLXPixtralVisionNet``
- ``NFKMLXPixtralConnector``
- ``NFKMLXPixtralImageProcessor``
- ``NFKMLXGemma4VisionNet``
- ``NFKMLXGemma4UnifiedNet``
- ``NFKMLXGemma4AudioNet``
- ``NFKMLXGemma4ImageProcessor``
- ``NFKMLXGemma4AudioFeatureExtractor``
- ``NFKMLXGemma4MultimodalEmbedder``
- ``NFKMLXGemma4Fusion``
- ``NFKMLXGemma4ConditionalGeneration``

### Video

- ``NFKMLXRIFE``
- ``NFKMLXRIFEv4``
- ``NFKMLXRAFT``
- ``NFKMLXVideoSR``
- ``NFKMLXVJEPA2``
- ``NFKMLXVJEPA2Configuration``
- ``NFKMLXVJEPA2Backend``
- ``NFKMLXCosmosTokenizer``
- ``NFKMLXCosmosTokenizerVariant``
- ``NFKMLXCosmosTokenizerConfiguration``
- ``NFKMLXCosmosTokenizerNet``
- ``NFKMLXCosmosTokenizerBackend``
- ``NFKMLXCosmosTokenizerCode``
- ``NFKMLXCosmosFSQ``
- ``NFKMLXSDUpscaler``
- ``NFKMLXVideoBackend``
- ``NFKMLXVideoFile``

### Audio → audio

- ``NFKMLXDemucs``
- ``NFKMLXHTDemucs``
- ``NFKMLXConvTasNet``
- ``NFKMLXDenoiser``
- ``NFKMLXMPSENet``
- ``NFKMLXGTCRN``
- ``NFKMLXSGMSE``
- ``NFKMLXStoRM``
- ``NFKMLXMossFormer2SENet``
- ``NFKMLXMossFormer2SRNet``
- ``NFKMLXDeepFilterNet``
- ``NFKMLXVoiceRestore``
- ``NFKMLXResembleEnhance``
- ``NFKMLXMetricGANPlus``
- ``NFKMLXCMGAN``
- ``NFKMLXFRCRN``
- ``NFKMLXNUWave2``
- ``NFKMLXApollo``
- ``NFKMLXDAC``
- ``NFKMLXSNAC``
- ``NFKMLXBigVGAN``
- ``NFKMLXMimi``
- ``NFKMLXBasicPitch``
- ``NFKMLXAllInOne``
- ``NFKMLXMuScriptor``
- ``NFKMLXHFTTransformer``

### Audio → text & labels

- ``NFKMLXWhisper``
- ``NFKMLXParakeet``
- ``NFKMLXGraniteSpeech``
- ``NFKMLXGraniteSpeechBackend``
- ``NFKMLXVoxtral``
- ``NFKMLXVoxtralBackend``
- ``NFKMLXCanary``
- ``NFKMLXCanaryBackend``
- ``NFKMLXVAD``
- ``NFKMLXSileroVAD``
- ``NFKMLXAudioTagger``

### Text → speech

- ``NFKMLXTTS``
- ``NFKMLXVoice``
- ``NFKMLXFastSpeech2``
- ``NFKMLXHiFiGAN``
- ``NFKMLXKokoro``
- ``NFKMLXChatterbox``
- ``NFKMLXChatterboxTTS``
- ``NFKMLXPhonemizer``
- ``NFKMLXNeuralG2P``
- ``NFKMLXEspeakPhonemizer``

### Text → music

- ``NFKMLXMusic3``
- ``NFKMLXMusicBackend``

### Text → image

- ``NFKMLXBackend``
- ``NFKMLXTextToImage``
- ``NFKMLXSDPipeline``
- ``NFKMLXStableDiffusionModels``
- ``NFKMLXSDTextEncoder``
- ``NFKMLXSDPromptTokenizer``
- ``NFKMLXZImageGenerator``
- ``NFKMLXZImagePipeline``
- ``NFKMLXZImageTransformerNet``
- ``NFKMLXSANAPipeline``
- ``NFKMLXSANATransformerNet``
- ``NFKMLXDCAutoencoderNet``
- ``NFKMLXIPAdapterImageProjection``
- ``NFKMLXIPAdapterAttention``

### Text → video

- ``NFKMLXLTXVideoGenerator``
- ``NFKMLXWanVideoGenerator``
- ``NFKMLXLTXPipeline``
- ``NFKMLXLTXVideoVAE``
- ``NFKMLXLTXTransformer``
- ``NFKMLXLTX2TransformerNet``
- ``NFKMLXWanPipeline``
- ``NFKMLXWanTransformerNet``
- ``NFKMLXWanVideoVAENet``
- ``NFKMLXWanAnimate``
- ``NFKMLXWanAnimateNet``
- ``NFKMLXWanAnimateKVCache``
- ``NFKMLXQwenImageGenerator``
- ``NFKMLXQwenImage``
- ``NFKMLXQwenImageNet``
- ``NFKMLXQwenImageVAE``
- ``NFKMLXQwenImagePipeline``
- ``NFKMLXSD3Generator``
- ``NFKMLXSD3Pipeline``
- ``NFKMLXSD3TransformerNet``
- ``NFKMLXSD3ControlNetPipeline``
- ``NFKMLXSD3ControlNetNet``
- ``NFKMLXFlux``
- ``NFKMLXFluxTextEncoder``
- ``NFKMLXFluxPipeline``
- ``NFKMLXFluxTransformerNet``
- ``NFKMLXFlux2``
- ``NFKMLXFlux2TransformerNet``
- ``NFKMLXFlux2Pipeline``
- ``NFKMLXFlux2LatentCodec``
- ``NFKMLXFlux2TextEncoder``
- ``NFKMLXFluxControlNetPipeline``
- ``NFKMLXFluxControlNetNet``

### Bring-your-own backends

- ``NFKMLXModuleBackend``
- ``NFKMLXMattingBackend``
- ``NFKMLXTensorBackend``
- ``NFKMLXSpeechBackend``
- ``NFKMLXDiffusionBackend``

### Registry, download & discovery

- ``NFKMLXReferenceModels``
- ``NFKMLXModelRegistry``
- ``NFKMLXHub``
- ``NFKStableDiffusionProvider``
- ``NFKMLXTranslationProvider``
- ``NFKMLXWhisperProvider``

### Customizing a model

- ``NFKMLXFineTune``
- ``NFKMLXTrainer``
- ``NFKMLXTrainingCheckpoint``
- ``NFKMLXTrainingStep``
- ``NFKMLXTrainingCachePolicy``
- ``NFKMLXLearningRateSchedule``
- ``NFKMLXTrainingData``
- ``NFKMLXBatchSampler``
- ``NFKMLXLoRA``
- ``NFKMLXLoRALinear``
- ``NFKMLXZeroDCENet``
- ``NFKMLXZeroDCEObjective``
- ``NFKMLXZeroDCELoss``
- ``NFKMLXSegFormerNet``
- ``NFKMLXSegFormerObjective``
- ``NFKMLXSegFormerTrainable``
- ``NFKMLXSAM2Objective``
- ``NFKMLXSAM2Trainable``
- ``NFKMLXSAM3Objective``
- ``NFKMLXSAM3Trainable``
- ``NFKMLXCosmosTokenizerObjective``
- ``NFKMLXCosmosTokenizerTrainable``
- ``NFKMLXSa2VAObjective``
- ``NFKMLXSa2VAExample``
- ``NFKMLXFlorence2Objective``
- ``NFKMLXTableTransformerNet``
- ``NFKMLXTableTransformerObjective``
- ``NFKMLXTableTransformerTrainable``
- ``NFKMLXTrOCRObjective``
- ``NFKMLXTrOCRTrainable``
- ``NFKMLXVJEPA2Net``
- ``NFKMLXVJEPA2Objective``
- ``NFKMLXVJEPA2Trainable``
- ``NFKMLXVJEPA2Processor``
- ``NFKMLXVGG16Features``
- ``NFKMLXCLIPNet``
- ``NFKMLXCLIPProbe``
- ``NFKMLXCLIPProbeBackend``
- ``NFKMLXEmbeddingProbe``
- ``NFKMLXEmbeddingProbeBackend``
- ``NFKMLXEmbeddingAdapter``
- ``NFKMLXEmbeddingRankingObjective``
- ``NFKMLXGTCRNObjective``
- ``NFKMLXNUWave2Objective``
- ``NFKMLXAllInOneTargets``
- ``NFKMLXAllInOneObjective``
- ``NFKMLXConvTasNetNet``
- ``NFKMLXConvTasNetObjective``
- ``NFKMLXVADNet``
- ``NFKMLXVADObjective``
- ``NFKMLXVADSpecAugment``
- ``NFKMLXYOLONet``
- ``NFKMLXYOLOGenerationNet``
- ``NFKMLXYOLOBox``
- ``NFKMLXYOLOObjective``
- ``NFKMLXYOLOEndToEndObjective``
- ``NFKMLXYOLOTrainable``
- ``NFKMLXRTDetrObjective``
- ``NFKMLXRTDetrPredictions``
- ``NFKMLXRTDetrTarget``
- ``NFKMLXRTDetrDenoisingGroup``
- ``NFKMLXRTDetrTrainingOutputs``
- ``NFKMLXRTDetrTrainable``
- ``NFKMLXWhisperNet``
- ``NFKMLXWhisperObjective``
- ``NFKMLXSeq2SeqNet``
- ``NFKMLXT5Seq2SeqNet``
- ``NFKMLXTranslationObjective``
- ``NFKMLXTranslateGemmaObjective``
- ``NFKMLXLayaNet``
- ``NFKMLXLayaObjective``
- ``NFKMLXLayaTrainable``
- ``NFKMLXLayaExample``
- ``NFKMLXLayaEpisode``
- ``NFKMLXDeBERTaV2Net``
- ``NFKMLXOpenJevDeBERTaNet``
- ``NFKMLXOpenJevDeBERTaObjective``
- ``NFKMLXOpenJevDeBERTaTrainable``
- ``NFKMLXOpenJevDeBERTaExample``
- ``NFKMLXOpenJevNet``
- ``NFKMLXOpenJevObjective``
- ``NFKMLXOpenJevExample``
- ``NFKMLXGraniteHybridNet``
- ``NFKMLXGraniteObjective``
- ``NFKMLXNemotronHNet``
- ``NFKMLXNemotronObjective``

### Chat templates

- ``NFKMLXChatTemplateRenderer``
- ``NFKMLXChatTemplate``

### Weights & formats

- ``NFKMLXTorchCheckpoint``
- ``NFKMLXGGUF``
- ``NFKMLXGGUFTensorInfo``
- ``NFKMLXWeightPrecision``

### Runtime

- ``NFKMLXResidency``
- ``NFKMLXExpertStore``
- ``NFKMLXRandom``
- ``NFKMLXGPU``
- ``NFKMLXDevice``
- ``NFKMLXDeviceType``
