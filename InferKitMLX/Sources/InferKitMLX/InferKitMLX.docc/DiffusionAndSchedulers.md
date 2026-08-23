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

Two ship, and swapping them changes the sampler without touching `denoise`:

- ``NFKDDIMScheduler`` — DDIM with epsilon, v, and sample prediction types
  (``NFKDiffusionPredictionType``), at either timestep spacing
  (``NFKDiffusionTimestepSpacing``). A distilled few-step release needs `trailing`: at one step,
  `leading` visits training step 1, where almost no noise remains and the model has nothing to remove.
- ``NFKLCMScheduler`` — few-step latent consistency: a consistency boundary `c_out·x₀ + c_skip·latent`,
  then fresh step-keyed deterministic noise.

Noise is a deterministic SplitMix64 + Box–Muller stream, so a run repeats without the MLX random state.

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

### Scheduler seam

- ``NFKDiffusionScheduler``
- ``NFKDDIMScheduler``
- ``NFKLCMScheduler``
- ``NFKDiffusionPredictionType``

### Context

- ``NFKDiffusionContext``
