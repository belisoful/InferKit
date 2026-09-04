# Diffusion & schedulers

An iterative sampler where the consumer owns the model and the scheduler is a swappable seam.

## Overview

``NFKMLXDiffusionBackend`` runs a diffusion model that is not a single forward but an iterative loop. The
consumer supplies four pieces and the backend owns the loop, per-step progress and cancellation, and the
image bridge:

- `encode` — request plus bridged image/mask → a ``NFKDiffusionContext``.
- `denoise` — latent, timestep, context, guidance → a model prediction.
- `decode` — final latent → an image tensor.
- a ``NFKDiffusionScheduler`` — the sampler that turns a prediction into the next latent.

![The diffusion loop: encode once, denoise-and-step per timestep, decode once.](diffusion-loop)

The presence of a source latent selects the task: none runs text-to-image, a source latent runs
image-to-image (`NFKParameterStrength`), a source latent plus mask runs inpainting (the kept region is
held to the source each step).

### Stable Diffusion text-to-image

``NFKMLXTextToImage`` is that loop wired to this package's own networks: ``NFKMLXSDTextEncoderNet`` for
the prompt, ``NFKMLXSDUNet`` for the denoiser, ``NFKMLXSDAutoencoder`` for the picture. Three releases
ship as configurations — Stable Diffusion 1.5, Stable Diffusion 2.1, and SDXL-Turbo — and each is
measured end to end against the diffusers pipeline it comes from. ``NFKMLXBackend`` is that model
behind a release name, downloading the release's files on first use.

Above a guidance of 1 the conditional and unconditional predictions run as one batch of two, which
makes classifier-free guidance one forward pass rather than two.

### Schedulers

Five ship, and swapping them changes the sampler without touching `denoise`:

- ``NFKDDIMScheduler`` — DDIM with epsilon, v, and sample prediction types
  (``NFKDiffusionPredictionType``), at either timestep spacing
  (``NFKDiffusionTimestepSpacing``). A distilled few-step release needs `trailing`: at one step,
  `leading` visits training step 1, where almost no noise remains and the model has nothing to remove.
  It is at reference parity against diffusers' own `DDIMScheduler`, `steps_offset` and
  `set_alpha_to_one` included.
- ``NFKLCMScheduler`` — few-step latent consistency: a consistency boundary `c_out·x₀ + c_skip·latent`,
  then fresh step-keyed deterministic noise.
- ``NFKMLXFlowMatchScheduler`` — the rectified-flow Euler sampler LTX, Wan, Z-Image, SANA, Flux, and
  SD3 use: a sigma ramp with resolution-dependent shifting and an optional terminal stretch, one
  Euler update per step. ``NFKMLXFlowMatchConfiguration`` carries each release's shift.
- ``NFKMLXDPMSolverScheduler`` — DPM-Solver++ (second order, midpoint), SANA's released sampler.
- ``NFKMLXUniPCScheduler`` — UniPC (second order, predictor-corrector), Wan's released sampler.

The three flow samplers are value types with no weights, each verified step for step against diffusers
under `swift test`. Noise for the DDIM loop is a deterministic SplitMix64 + Box–Muller stream, so a run
repeats without the MLX random state.

### Latent previews

``NFKDiffusionLatentPreview`` gives the progress callback something to show: a 1×1 map from latent
channels to RGB, reported as the job's `partialResult` every `previewEverySteps` steps, at a fraction
of a decode's cost. The published Stable Diffusion and SDXL factors ship, and `fitted(…)` derives a
map for any decoder by least squares.

### The DiT pipelines

Four diffusion-transformer pipelines chain a text encoder, a denoising transformer, and an autoencoder
over the flow schedule, each network at reference parity against diffusers. The caller supplies the
text embedding and manages residency — the encoders are large and load and free in turn:

- ``NFKMLXZImagePipeline`` — Z-Image (single-stream S3-DiT, ``NFKMLXZImageTransformerNet``) with the
  Flux VAE (an ``NFKMLXSDVAEConfiguration/flux`` preset of the shared autoencoder) and a Qwen3-4B
  caption embedding read from the decoder's penultimate layer. Text-to-image, and image-to-image
  through `generate(image:strength:…)`.
- ``NFKMLXSANAPipeline`` — SANA (ReLU linear attention, ``NFKMLXSANATransformerNet``) with the 32×
  Deep-Compression Autoencoder ``NFKMLXDCAutoencoderNet`` and a Gemma 2 (``NFKMLXGemma2Net``) caption,
  sampled by DPM-Solver++.
- ``NFKMLXLTXPipeline`` — LTX-Video (``NFKMLXLTXTransformer``) with the causal 3-D
  ``NFKMLXLTXVideoVAE`` and a T5-XXL (``NFKMLXT5Encoder``) prompt, sampled by the flow scheduler.
- ``NFKMLXWanPipeline`` — Wan (``NFKMLXWanTransformerNet``) with the streaming 3-D causal
  ``NFKMLXWanVideoVAENet`` and a umT5 prompt, sampled by UniPC.

``NFKMLXIPAdapterImageProjection`` and ``NFKMLXIPAdapterAttention`` add IP-Adapter image conditioning
beside a text prompt: a CLIP image embedding becomes a few extra tokens read through a second,
image-conditioned cross-attention.

### ControlNet and LCM need no full SD reimplementation

LCM is a scheduler swap; ControlNet is a `denoise` closure that reads `context.conditioning["control"]`.
The UNet itself is brought by the consumer's `denoise` or a dynamically linked SD engine.
``NFKMLXReferenceModels`` registers CI-runnable oracle pipelines — `registerDiffusionUpscaler`,
`registerDiffusionDepth`, `registerDiffusionInpainter`, and `registerControlNet` (`diffusion-controlnet`,
reading a control map under the core key `NFKInputControl`). Their `denoise` is an oracle that drives the
loop to a target derived from the input; a real integration swaps the oracle for a trained UNet forward.

## Topics

### Backend

- ``NFKMLXDiffusionBackend``
- ``NFKDiffusionLatentPreview``

### Scheduler seam

- ``NFKDiffusionScheduler``
- ``NFKDDIMScheduler``
- ``NFKLCMScheduler``
- ``NFKMLXFlowMatchScheduler``
- ``NFKMLXFlowMatchConfiguration``
- ``NFKMLXDPMSolverScheduler``
- ``NFKMLXUniPCScheduler``
- ``NFKDiffusionPredictionType``
- ``NFKDiffusionTimestepSpacing``

### Context

- ``NFKDiffusionContext``

### Stable Diffusion

- ``NFKMLXTextToImage``
- ``NFKMLXSDPipeline``
- ``NFKMLXSDUNet``
- ``NFKMLXSDAutoencoder``
- ``NFKMLXSDTextEncoderNet``

### Diffusion transformers

- ``NFKMLXZImagePipeline``
- ``NFKMLXSANAPipeline``
- ``NFKMLXLTXPipeline``
- ``NFKMLXWanPipeline``
