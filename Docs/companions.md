# Optional companion packages

InferKit's core ships only backends built on Apple frameworks. Two companion packages add heavier
engines without raising the core's platform floor or adding dependencies to it. Each is a separate
SwiftPM package in a subdirectory of this repository; see the README's "Adding a companion" for how
to consume one.

## InferKitFoundationModels (optional companion)

`InferKitFoundationModels/` is a separate SwiftPM package (macOS 26 / iOS 26, Apple Intelligence
hardware) that bridges InferKit and Apple's **Foundation Models** framework:

- **`NFKFoundationModelsBackend`** — wraps the on-device system language model
  (`LanguageModelSession`) as an `NFKInferenceBackend`. The same request that runs against
  `NFKCoreMLLanguageBackend` or `NFKRemoteBackend` runs here: `NFKInputPrompt` or
  `NFKInputMessages` (a system message becomes the session's instructions), the standard text
  parameters, streamed partial text through the job. `isReady` reflects the model's availability.

The reverse direction — adopting Apple's provider protocols (`LanguageModel` /
`LanguageModelExecutor`, WWDC26) so InferKit's local and remote backends stand behind
`LanguageModelSession` — needs the macOS 27 / iOS 27 SDK and follows when that SDK is the build
baseline.

## InferKitMLX (optional companion)

`InferKitMLX/` is a separate SwiftPM package (Apple Silicon, macOS 14 / iOS 17) that keeps MLX out
of the core. It ships five bring-your-own-model backends and a gallery of real models — upscaling,
depth, matting, segmentation, detection, pose, restoration, interpolation, optical flow, colorization,
diffusion, speech, and music — each implemented in MLXNN and validated numerically against its
reference implementation on the released weights:

- **`NFKMLXBackend`** — a bundled Stable Diffusion release: SD 1.5, SD 2.1 base, or SDXL-Turbo. A
  request with no image runs text-to-image; a request with a `CGImage` under `NFKInputImage` runs
  image-to-image, with `NFKParameterStrength` controlling how much of the source survives. Every part
  of the run is this package's own — the CLIP text tower, the UNet, the DDIM sampler, and the
  autoencoder — each measured against its reference implementation, so text-to-image builds for iOS
  as the rest of the package does.
- **`NFKMLXModuleBackend`** — a bring-your-own MLX image model. Supply a `forward` closure over
  `MLXArray`; the backend handles the InferKit contract and the RGB `CGImage ↔ MLXArray` bridge.
- **`NFKMLXMattingBackend`** — a bring-your-own MLX image-matting model (a green/blue-screen keyer, a
  background remover). The plate under `NFKInputImage` and an optional hint (trimap or coarse alpha)
  under `NFKInputMask` become tensors, a `(plate, hint) -> [H, W, 4]` closure runs, and the straight
  foreground plus alpha matte returns as an RGBA `CGImage` under `NFKOutputImage`. `NFKMattingConfiguration`
  adds the matte on its own under `NFKOutputMask`, premultiplication, color space, tiled inference for
  large plates, and `MTLTexture` output.
- **`NFKMLXTensorBackend`** — a general bring-your-own MLX backend over named image tensors: several
  inputs in, several outputs out (a compositing model reading a foreground and a background, a model
  returning both an image and a mask). Each port binds an InferKit key to a tensor name.
Every shipped real model has a direct Objective-C factory — `[Model backendWith[Variant:]weightsURL:error:]`
for local weights and `[Model backendWith[Variant:]repo:weightsPath:revision:cacheDirectoryURL:error:]`
to download from Hugging Face and build — so a consumer (e.g. MetalForge) constructs them without the
registry. The registry (`register()` / `registerAll()` / `NFKMLXHub`) remains for custom / bring-your-own
models.

- **`NFKMLXRealESRGAN`** — a real, shipped single-forward model: the Real-ESRGAN generator (RRDBNet)
  in MLXNN, run through `NFKMLXModuleBackend` for ×4 upscaling. Build directly from Objective-C via
  `backendWithVariant:weightsURL:error:`, or download and build via the `repo:` factory.
- **`NFKMLXDepthAnything`** — a real single-forward depth model: the Depth Anything V2 DINOv2 + DPT
  network in MLXNN, run through `NFKMLXModuleBackend` (image → grayscale depth). Register and build by
  name; a self-validating converter turns the release into a safetensors checkpoint.
- **`NFKMLXU2Net`** — a real single-forward background remover: the U²-Net nested-U saliency network
  in MLXNN, run through the matting backend (plate → foreground + alpha cutout). Full `u2net` + light `u2netp`.
- **`NFKMLXSAM`** — real promptable segmentation (Segment Anything): a ViT encoder, prompt encoder, and
  two-way-transformer mask decoder in MLXNN, run through the matting backend (plate + point → mask).
- **`NFKMLXNAFNet`** — a real single-forward restoration network (denoise / deblur): a U-shaped stack
  of NAFBlocks in MLXNN, run through the module backend (degraded image → restored image).
- **`NFKMLXRIFE`** / **`NFKMLXRIFEv4`** — real frame interpolation: the RIFE HDv3 and v4 IFNets in
  MLXNN with a flow-based warp (v4 adds an arbitrary-timestep midpoint), run
  through the tensor backend (two frames → the interpolated middle frame) for slow-motion / retiming.
- **`NFKMLXRAFT`** — real optical flow: the RAFT correlation-and-ConvGRU pipeline in MLXNN, run through
  the tensor backend (two frames → a dense flow field) for motion vectors, warping, retiming.
- **`NFKMLXLaMa`** — a real single-forward inpainter: the LaMa FFC-ResNet generator in MLXNN, with an
  FFT spectral branch (via MLXFFT), run through the matting backend (plate + mask → inpainted image).
- **`NFKMLXTextToImage`** — Stable Diffusion text-to-image, on the diffusion backend: a CLIP text
  tower encodes the prompt, the UNet denoises over the DDIM loop with classifier-free guidance, the
  autoencoder decodes. Three releases ship as configurations — Stable Diffusion 1.5, Stable Diffusion
  2.1, and SDXL-Turbo — and `NFKMLXBackend` is that model behind a release name. Each is measured end
  to end against the diffusers pipeline it comes from, sampling with DDIM at the release's own
  timestep spacing on both sides.
- **`NFKMLXStableDiffusionInpaint`** — a latent-diffusion inpainter (VAE + UNet in MLXNN) on the
  diffusion backend: VAE-encode the plate, denoise a 9-channel input over the DDIM loop, VAE-decode.
- **`NFKMLXMarigold`** / **`NFKMLXSDUpscaler`** — image-conditioned latent-diffusion models on the
  diffusion backend: Marigold depth (image → depth) and the SD ×4 latent upscaler (image → ×4 image).
- **`NFKMLXStyleTransfer`** — real fast neural style transfer: Johnson et al.'s `TransformerNet` in
  MLXNN, run through the module backend (image → stylized image). The style is baked into the weights.
- **`NFKMLXCLIP`** — real image+text embeddings (CLIP ViT-B/32): a ViT image tower and a text
  transformer projected into a shared space. `NFKMLXCLIPBackend` returns an embedding under
  `NFKOutputEmbedding` for semantic search, tagging, and diffusion guidance.
- **`NFKMLXRVM`** — real video matting (Robust Video Matting): an encoder, LR-ASPP, and a **recurrent
  ConvGRU decoder** that carries state across frames, run through the matting backend (single frame) or
  `NFKMLXRVMNet.forward` (video, state threaded) for background removal without a green screen.
- **`NFKMLXCodeFormer`** — real face restoration: a VQGAN encoder/generator with a Transformer that
  predicts codebook indices, run through the module backend (aligned face → restored face).
- **`NFKMLXZeroDCE`** — real low-light enhancement: the Zero-DCE DCE-Net estimates pixel-wise tone
  curves and iteratively brightens, run through the module backend (dark image → brightened image).
- **`NFKMLXMODNet`** — real trimap-free portrait matting: the three-branch (semantic / detail /
  fusion) MODNet, run through the matting backend (portrait → foreground + alpha).
- **`NFKMLXYOLO`** — real object detection: an anchor-free YOLO with box decode and non-max
  suppression, returning `NFKDetection`s (label, confidence, normalized box) under
  `NFKOutputDetections`. Build with class names via the `backendWith…labels:` factory.
- **`NFKMLXSegFormer`** — real semantic segmentation: the SegFormer MiT transformer + all-MLP head,
  run through the module backend, emitting a grayscale class-label map under `NFKOutputImage`.
- **`NFKMLXSwinIR`** — real transformer super-resolution: SwinIR with true shifted-window attention
  and pixel-shuffle upsampling, run through the module backend (low-res → high-res image).
- **`NFKMLXColorizer`** / **`NFKMLXSiggraphColorizer`** — real colorization (ECCV-16 and
  SIGGRAPH-17): predict ab chroma from the L channel in CIELAB space and recombine with the original
  luminance, run through the module backend (grayscale photo → color photo); the SIGGRAPH model also
  takes user color hints. The converter loads the reference releases directly.
- **`NFKMLXPose`** — real top-down pose estimation (SimpleBaseline): a residual backbone and a
  deconvolution head produce joint heatmaps, decoded to `NFKKeypoint`s (name, normalized position,
  confidence) under `NFKOutputPose`. Build with joint names via the `backendWith…jointNames:` factory.
- **`NFKMLXDeepLab`** — real semantic segmentation (DeepLabV3): a residual backbone and an ASPP head,
  run through the module backend, emitting a grayscale class-label map (a CNN counterpart to SegFormer).
- **`NFKMLXConvTasNet`** — real time-domain speech separation: a convolutional encoder, a masking
  temporal-convolution network, and a shared decoder, returning one `NFKAudioAsset` per speaker.
- **`NFKMLXDenoiser`** — real speech noise suppression: the Demucs time-domain U-Net with a single
  output channel, returning one cleaned `NFKAudioAsset` under `NFKOutputAudio`.
- **`NFKMLXVAD`** — real voice activity detection (MarbleNet): a small conv net over a log-mel
  spectrogram marks speech spans, returning `NFKAudioSegment`s under `NFKOutputSegments`.
- **`NFKMLXAudioTagger`** — real audio tagging (PANNs): a conv net over the log-mel spectrogram
  predicts the sounds present, returning top-K `NFKClassification`s under `NFKOutputClassifications`.
- **`NFKMLXBiSeNet`** / **`NFKMLXBiSeNetV2`** — real real-time semantic segmentation: a two-path
  (spatial detail + context)
  network with attention refinement and feature fusion, emitting a grayscale class-label map.
- **`NFKMLXVideoSR`** — real recurrent video super-resolution: a ConvGRU propagates a hidden state
  along the clip; single frame through the module backend, or a sequence via `upscaleSequence`.
- **`NFKMLXSpeechBackend`** — a bring-your-own MLX text-to-speech backend: a `(String) -> MLXArray`
  waveform closure, written to a WAV file and returned as an `NFKAudioAsset` (text → audio).
- **`NFKMLXMusicBackend`** — real music generation (MiniMax Music 3): a music description under
  `NFKInputPrompt` and lyrics under `NFKInputLyrics` become a stereo 44.1 kHz `NFKAudioAsset`. A
  Qwen3-8B autoregressive stage samples RVQ codes frame by frame, a flow-matching transformer
  denoises audio latents conditioned on its hidden states over overlapping windows, and a Snake
  vocoder decodes them — every network measured against the official diffusers implementation on
  the released weights. Build with `NFKMLXMusic3.backend(directoryURL:)` from the downloaded
  release tree (~27 GB); the weights are separately licensed — see "Model weight licenses" below.
  `NFKMLXMusic3.quantizeRelease(at:to:)` writes a quantized copy (4-bit language model including its
  untied input embedding, 8-bit DiT — the split is measured: the flow field is the
  quantization-sensitive stage) that the same factory takes unchanged at **7.7 GiB**, small enough
  that the backend keeps every stage loaded between runs instead of staging them from disk per
  request. Fallback precision, if a smaller stack matters more than the last of the quality:
  `transformerBits: 6` trades the DiT down to a measured velocity cosine of 0.99844 (0.99990 at
  8-bit) for roughly 0.6 GB more — do a listening A/B first, because the DiT's error compounds over
  the sampling loop.
- **`NFKMLXDiffusionBackend`** — a bring-your-own MLX diffusion model, for the iterative-sampler shape
  the single-forward backends cannot express. Supply `encode`, `denoise`, `decode`, and a scheduler;
  the backend runs the denoise loop with per-step progress and cancellation. No source latent runs
  text-to-image, a source latent runs image-to-image (`NFKParameterStrength`), and a source latent
  plus a mask runs inpainting. Reference pipelines register by name (upscale, depth, inpaint,
  **controlnet**); `NFKDDIMScheduler` and `NFKLCMScheduler` (few-step) ship, and `NFKDiffusionScheduler`
  is the seam for other samplers. **ControlNet and LCM need no full SD reimplementation** — LCM is a
  scheduler swap, ControlNet is a `denoise` closure over a control map (`NFKInputControl` →
  `conditioning["control"]`); the UNet is supplied by your `denoise` or a dynamically linked SD engine.

The image backends share `NFKMLXImageBridge`, which converts a **`CGImage` or an `MTLTexture`**
to and from `MLXArray` in either direction, preserving alpha — so a Metal render pipeline hands
textures straight in and gets textures back.

### Dynamic backend discovery (optional engines)

`NFKDynamicBackend` (core) activates a heavier engine **only when its classes are linked into your
build**, with no build dependency on it. Your engine's adapter conforms to `NFKDynamicBackendProvider`
and InferKit resolves it by name at runtime (`NSClassFromString`) — absent when unlinked, with no link
error. Built-in capabilities light up when you link a companion:

- **`stable-diffusion`** — linking **InferKitMLX** ships `NFKStableDiffusionProvider`, so
  `NFKDynamicBackend.stableDiffusionBackend()` returns the bundled SD backend.
- **`transcription`** — InferKitMLX also ships `NFKMLXWhisperProvider` (a native whisper.cpp can override).
- **`text-generation`** — linking **InferKitFoundationModels** ships `NFKFoundationModelsProvider` for
  on-device LLM.
- **`controlnet`** — no shipped default; bring a ControlNet engine and register its provider.

Without the companion, the capability is simply unavailable. Model **weights are downloaded at runtime**
(not bundled at build time) and cached under Application Support (`NFKHFHub.defaultCacheDirectoryURL`, or
a host-supplied security-scoped folder); the download blocks, so run it off the main thread or use the
async `downloadRepo:…completionHandler:` (`try await`).

### Model weight licenses

InferKit's code is MIT and the checkpoints the shipped models load are, with one exception, released
under MIT or Apache-2.0 terms. Weights are a separately licensed asset your app downloads at runtime;
the license travels with the checkpoint, not with InferKit.

The exception is **MiniMax Music 3** (`NFKMLXMusicBackend`). Its weights
(`MiniMaxAI/MiniMax-Music3` on Hugging Face) are under the **MiniMax-Music3 Community License**, which
is not a permissive license:

- A commercial product or service using the weights must prominently display "MiniMax-Music3" in its
  user interface (License §3.1).
- Aggregate yearly revenue above USD 20 million from such products requires separate prior written
  authorization from MiniMax (License §3.2, `api@minimax.io`).
- A product or hosted service that lets third parties generate outputs must implement and maintain
  safeguards against infringing uses and outputs (License §4).

The model's inference code carries no such terms (the architecture derives from Qwen3-8B, Apache-2.0;
Stable Audio tools, MIT; and the Descript Audio Codec, MIT). The obligations attach to the checkpoint.
An app that ships the MiniMax music backend accepts these terms on behalf of its own product; review
the LICENSE file in the weight repository before enabling it commercially.
