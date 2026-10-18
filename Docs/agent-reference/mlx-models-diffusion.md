<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX models: Stable Diffusion and latent diffusion

The diffusion backend seam and schedulers, the SD networks and pipelines, IP-Adapter, TAESD.

- `NFKMLXDiffusionBackend` — a bring-your-own MLX diffusion model (iterative sampler, not a single
  forward). The consumer supplies `encode` (request + bridged image/mask → `NFKDiffusionContext`),
  `denoise` (latent + timestep + context + guidance → prediction), `decode` (latent → image tensor),
  and a scheduler; the backend owns the loop, per-step progress/cancellation, and the image bridge.
  No source latent runs text-to-image; a source latent runs image-to-image (`NFKParameterStrength`);
  a source latent + mask runs inpainting (kept region held to the source each step).
  `windowedContinuation(...)` produces output longer than the model's window, the `NFKMLXMusic3`
  mechanism lifted out for reuse: it tiles one axis into overlapping windows and keeps continuity
  Inside the sampler — each window's overlap is held to the previous window's finished latent,
  re-noised to the current step's level through `scheduler.addNoise` (the same primitive the inpaint
  path uses, so it is scheduler-agnostic rather than flow-specific), locked to it after the loop,
  and the windows stitched at the hop stride. It is a static helper over the shared-conditioning
  case (text-to-long-image, an extendable texture); a per-window `condition` encoder like the music
  path's keeps its own specialized loop. `windowStarts` pulls the final window back to end exactly
  at the total. Tested by single-window equivalence to the plain loop and that the overlap hold
  changes the result (`NFKMLXDiffusionWindowedTests`). `NFKDiffusionLatentPreview` gives the
  progress callback something to show. A full decode per step costs more than the sampling, so the
  preview is a 1×1 convolution over the channel axis — twelve weights and three biases for a
  four-channel latent — reported as the job's `partialResult`, the same mechanism a streaming text
  backend uses. `previewEverySteps` thins it. `.passthrough` suits a latent already in image space;
  `.stableDiffusion` and `.stableDiffusionXL` are the published latent-RGB factors, and what is
  measured about them is how closely they track a real decode: against the released SD 1.5
  autoencoder, on a latent encoded from a real photograph, the shipped map reproduces the decode's
  structure at a mean-removed correlation of **0.93**. Raw cosine is the wrong measure here and the
  first version of that test used it — two nearly constant grey images score 0.98, so cosine cannot
  tell a preview showing the picture from a flat rectangle of the right brightness. It also fed the
  map a latent at the autoencoder's scale rather than the sampler's; the pipeline divides by
  `scaleFactor` before the autoencoder, so the two differ by 5.5× and the earlier numbers were
  artifacts of that. `fitted(latentChannels:decode:sample:)` derives a map from any decoder by least
  squares, which is the route for a model with no published factors — though a map fitted on noise
  scores 0.917 where the shipped coefficients score 0.931, so fit on latents encoded from real
  images. A preview is a progress indicator, so a mismatched map returns nil rather than failing the
  run it is only reporting on. `NFKDiffusionScheduler` is the sampler seam. `NFKDDIMScheduler` is at
  reference parity against diffusers' own `DDIMScheduler` (worst per-step latent cosine
  0.9999999999999941, add-noise 0.9999999999999983, and the schedule it visits matches exactly).
  Reaching it corrected two things the sampler had wrong: it divided the training range evenly (999,
  949, … 49) where the reference walks a fixed stride and adds **`steps_offset`** (951, 901, … 1),
  and its final step denoised against a signal ratio of 1 where the released configurations set
  **`set_alpha_to_one: false`** and use the ratio at training step 0, which deliberately leaves a
  little noise. Both are now init parameters defaulting to the released Stable Diffusion values. The
  four `diffusion-*` reference stand-ins pass `setsAlphaToOne: true`, because an oracle that drives
  the loop to an exact target only lands on it when the last step denoises fully. `NFKDDIMScheduler`
  (epsilon/v/sample prediction types), `NFKLCMScheduler` (few-step latent-consistency: consistency
  boundary `c_out·x₀ + c_skip·latent`, then fresh step-keyed deterministic noise from the same
  SplitMix64 stream), and `NFKTCDScheduler` (trajectory consistency, diffusers' `TCDScheduler`: the
  LCM schedule, but the "previous" timestep after the last step is 0, not −1, so the final step's
  signal ratio is training step 0's; each step re-noises the clean estimate to `floor((1 − eta) ·
  next)` and, when not final, carries it to the next level with fresh noise; eta 1 is the Hyper-SD
  recipe, eta 0 deterministic; `NFKDDIMScheduler.train(withAlphaBar:)` recovers the next training
  step from the ratio the schedule recorded) ship. Noise is a deterministic SplitMix64 + Box–Muller
  stream, so a run is repeatable without the MLX random state. Reference pipelines register by name
  via `NFKMLXReferenceModels`: `registerDiffusionUpscaler` (2× upscale), `registerDiffusionDepth`
  (Marigold-style depth), `registerDiffusionInpainter`, `registerControlNet`
  (`diffusion-controlnet`: a control map under the core key `NFKInputControl` →
  `context.conditioning["control"]`, the slot a real ControlNet `denoise` reads to inject
  residuals). To stay CI-runnable without real weights, their `denoise` is an oracle epsilon that
  drives the loop to a target derived from the input; a real integration swaps the oracle for a
  trained UNet forward. ControlNet/LCM need no full SD reimplementation: LCM is a scheduler swap,
  ControlNet is a `denoise` closure over `conditioning["control"]`; the UNet is brought by the
  consumer's `denoise` or a dynamically linked SD engine.
- `NFKMLXSDUNet` / `NFKMLXSDAutoencoder` (`NFKMLXStableDiffusionModels.swift`) — the real
  `UNet2DConditionModel` and `AutoencoderKL` in `MLXNN`, in the diffusers layout. One implementation
  serves every latent-diffusion model here, sharing one structure and differing only in the scalars a
  configuration carries (channel widths, which levels attend, the cross-attention width,
  convolution-versus-linear transformer projections, a class embedding, how many transformer blocks an
  attention runs, whether a pooled embedding joins the timestep). Reference parity against diffusers on
  every released configuration: SD-1.5-inpainting UNet 0.9999999999993, Marigold UNet 0.99999999999,
  ×4-upscaler UNet 0.9999999999973, SD 2.1 UNet 0.9999999993, SDXL UNet 0.9999999999955, SD autoencoder
  0.99999999982 latent / 0.99999999917 decoded, upscaler autoencoder 0.99999999998 / 0.99999999992.
  Every one covered on the first triage run.
  SDXL adds two axes and no new structure: `transformerLayers` (`transformer_layers_per_block`,
  `[1, 2, 10]` — the coarsest level runs ten transformer blocks where every earlier release runs one)
  and `additionEmbedding` (`addition_embed_type: "text_time"` — six `time_ids`, the original size, the
  crop's top-left corner, and the target size, each embedded at 256 and run together into 1536, joined
  by the second tower's 1280-wide pooled embedding and projected to the timestep's width).
  Three details are load-bearing. `only_cross_attention` (the ×4 upscaler sets it on three levels)
  makes a block's first attention a cross-attention too, so its keys and values take the context's
  width; nothing in the checkpoint's key names says so, only the tensor shapes, and MLX adopts a
  checkpoint's shapes wholesale, so getting it wrong loads cleanly and fails later. The transformer's
  input normalization uses epsilon 1e-6 where every resnet uses 1e-5. The autoencoder's downsampling
  convolution pads asymmetrically (right and bottom only), where the UNet's pads evenly. The SD 1.5
  autoencoders also predate the diffusers attention rename, so `remapVAEKey` accepts
  `query`/`key`/`value`/`proj_attn` as well as `to_q`/`to_k`/`to_v`/`to_out.0`.
  Oracle: diffusers cannot be installed beside the other oracles here (it needs a newer transformers
  than the 4.33.3 the Whisper / CLIP / SegFormer records were measured against), so it lives in its own
  virtual environment and every `sd_*` mode of `run_reference.py` runs under that interpreter. The
  manifest's `oracle_environments` records where it is and how to rebuild it. Weights are the released
  diffusers layout, one checkpoint per network.
- `NFKMLXSDPipeline` — a UNet and an autoencoder together, plus the text conditioning the released
  checkpoints cross-attend to. The tower is not on this path: the caller brings the embedding
  (`loadTextContext(from:)` reads a `[tokens, dimensions]` tensor), which is what the image-conditioned
  models here do (a model that takes no prompt still expects the embedding of an empty one).
  `NFKMLXTextToImage` is the path that owns a tower, because a prompt is its input.
  `loadWeights(unetURL:vaeURL:)` reads the released layout; `loadWeights(from:)` reads a single file
  holding both under `unet.`/`vae.`, which is what `NFKMLXWeights.save` writes, so a fine-tuned
  pipeline reloads through one path.
- `NFKMLXStableDiffusionInpaint` (`@objc`) — Stable Diffusion inpainting on `NFKMLXDiffusionBackend`,
  over those real networks. `encode` VAE-encodes the plate and the masked plate and builds the
  nine-channel conditioning (noisy latent, then mask, then masked-image latent, in that order);
  `denoise` is the UNet (epsilon); `decode` is the VAE. The backend runs the DDIM loop and the per-step
  inpaint compositing. The masked-image latent is the plate with the hole blanked and then encoded, not
  the plate's latent with the hole blanked; the encoder is not local, so the two differ. `+register`
  under `sd-inpaint`. Pipeline and offline round-trip tested under xcodebuild.
- `NFKMLXTextToImage` (`@objc`) — Stable Diffusion **text-to-image**, on `NFKMLXDiffusionBackend`, and
  what `NFKMLXBackend` is built from. `encode` turns the prompt into a conditioning sequence, `denoise`
  is the UNet with classifier-free guidance, `decode` is the autoencoder; the backend owns the loop.
  Above a guidance of 1 the conditional and unconditional predictions run as one batch of two, which
  makes guidance one forward pass rather than two, ordered unconditional first as the reference reads it
  back. `NFKMLXSDTextToImageConfiguration` carries the three releases
  (`.stableDiffusion15`, `.stableDiffusion21` / `.stableDiffusion21V`, `.sdxlTurbo`), and
  `NFKMLXSDReleaseFiles(directoryURL:)` resolves a downloaded release's tree (`unet/`, `vae/`,
  `text_encoder/`, `tokenizer/`, and for SDXL `text_encoder_2/` and `tokenizer_2/`), accepting the
  `.fp16.safetensors` spelling a half-precision-only release uses.
  Reference parity end to end against diffusers' own pipelines, starting from the reference's own
  initial latent (matching a random source across two implementations proves nothing about either):
  SD 1.5 image cosine 0.999998948, SD 2.1 (v-prediction) 0.9999986, SDXL-Turbo 0.9999973, SDXL with
  guidance 0.9999990, SDXL with no negative prompt 0.9999988. The record also carries the per-step
  latents and the first guided prediction, so a whole-picture mismatch says which stage diverged.
  The sampler is DDIM in every case, including SDXL-Turbo, whose release names
  `EulerAncestralDiscreteScheduler`; both sides run DDIM at the release's own `timestep_spacing`, so
  the comparison measures this port rather than two different samplers. A caller wanting the released
  sampler exactly brings one through `NFKDiffusionScheduler`. Three details are load-bearing, and each
  one first showed up as a wrong picture. Stable Diffusion 2.x pads a prompt with `!` (id 0), not the
  end marker, and `special_tokens_map.json` overrides `tokenizer_config.json`, which is where that is
  written, so reading only the config pads with 75 end markers and the model reads a different sentence
  (that scored 0.825). A bfloat16 release turns a float32 module into a bfloat16 one, because MLX's
  `update(parameters:)` adopts a checkpoint's element type along with its values; the SD 2.1 text tower
  and autoencoder are published that way, and it cost three orders of magnitude (0.9999956 against
  0.9999999999841). `NFKMLXWeightPrecision` makes that a choice rather than an accident.
  `force_zeros_for_empty_prompt` acts on an absent negative prompt, not an empty one: the reference's
  condition is `negative_prompt is None`, and an empty string is a sentence the model is asked to encode.
  At float16 the SD 1.5 UNet with the IP-Adapter is held to diffusers at float16. Its transformer blocks
  sit at 0.3 of the accumulated floor from summation order alone. MLX's GPU matmul splits the inner
  dimension into parts summed as running float32 sums (the first transformer's `to_q` differs from torch's
  in 498 of 327,680 elements, exactly as torch's own linear summed that way does), and its layer norm sums
  its moments apart from torch's Welford update. Each transformer block is held to its order floor
  (`mlx-parity-checklist.md`). SD 2.1's UNet at float16 carries an end-to-end order floor: one reduction
  changed alone moves the reference's own end-to-end distance to as much as 4.21 times its floor. In
  PyTorch's arithmetic (`NFKMLXArithmeticReference`) every SD 1.5 transformer block sits within its order
  floor, the farthest at 0.89 of it, and SD 2.1's end-to-end distance from float32 is 1.37 of its floor,
  inside its widest rerun's 4.21.
  A custom GPU kernel that copies torch's summation order is written only if a repository Issue requests
  it.
- `NFKMLXSDTextEncoderNet` / `NFKMLXSDTextEncoder` — the CLIP text tower the releases cross-attend to,
  built from `NFKMLXCLIP`'s own blocks (one implementation, four configurations). `NFKSDTextOutput`
  spells the difference between the releases: SD 1.x and 2.x read the last hidden state after the final
  layer normalization (2.x drops the tower's 24th layer in its own configuration rather than skipping it
  here), while SDXL reads the penultimate one, before that normalization. Only SDXL's second tower
  carries a projection, and it is the pooled embedding SDXL's UNet conditions on; the pooled path runs
  the whole stack through the final normalization even when the sequence stops a layer short, and reads
  the position of the highest token id (the reference's rule, which the padding repeats, so the first
  occurrence is the one). The releases are a `transformers` CLIPTextModel, whose attention stores
  separate `q_proj`/`k_proj`/`v_proj` where the module keeps the reference's fused projection, so the
  remap concatenates three tensors into one, which a 1:1 key map cannot express, as SegFormer's `kv` is
  a two-into-one. The activation is `quick_gelu` for SD 1.x and the exact error-function GELU for the
  OpenCLIP towers, not the tanh approximation its neighbour `gelu_new` selects. Reference parity on
  every released tower: SD 1.5 0.9999999999986,
  SD 2.1 0.9999999999841, SDXL primary 0.9999999999986, SDXL secondary 0.9999999999257 (pooled
  0.9999999999820).
- `NFKMLXSDLoRA` — released Stable Diffusion adapters merged into `NFKMLXSDUNet`: the LCM-LoRA and
  SDXL-Lightning distillations, and fine-tunes published in the same layouts. Two key layouts: kohya
  over diffusers module names (`lora_unet_<path with underscores>`, `.lora_down.weight` /
  `.lora_up.weight` / a scalar `.alpha`) and PEFT (`unet.<path>`, `.lora_A.weight` /
  `.lora_B.weight`, no alpha tensor, so the strength is 1; an alpha in the file's metadata is not
  read). A kohya file over the original model's block names (`lora_unet_input_blocks_…`) is not
  translated and is refused. The underscore spelling is ambiguous against module names that also
  carry underscores (`to_out_0`, `ff_net_0_proj`, `time_emb_proj`), so the reader never splits a
  key: it spells every linear and convolution of the UNet the kohya way (`diffusersPath`, the
  inverse of `remapUNetKey`, then dots to underscores) and looks the file's targets up. A
  convolution target stores `lora_down` as `[rank, in, kH, kW]` and `lora_up` as `[out, rank, 1,
  1]`; the delta is `up · down` over the rank axis, reshaped to `[out, in, kH, kW]` and transposed
  to the module's `[out, kH, kW, in]`. A 1×1 kernel on a linear target (`proj_in` under
  `use_linear_projection`) reshapes to the matrix. The strength is `scale · alpha / rank`; the
  product forms in float32 and takes the weight's type, which is diffusers' `fuse_lora`. A file that
  also carries text-tower keys (`lora_te…`, `text_encoder…`) is refused unless the caller passes
  `ignoringTextEncoderKeys: true`, and the report then counts the keys left alone; a UNet key that
  names no layer is refused before any weight changes. The LCM-LoRA for SD 1.5 adapts 278 targets
  (attention, feed-forward, projections, resnet convolutions, time projections), the SDXL adapters
  788. `NFKMLXSDSampler` / `withSampler` give an adapter the sampler it was distilled for: `.lcm`
  runs `NFKLCMScheduler` at 4 steps, guidance 1; `.ddimTrailing` counts DDIM down from the end of
  the training range at 4 steps, no guidance (SDXL-Lightning, SDXL-Turbo). A configuration already
  on the requested sampler keeps its own steps and guidance, so `.latentConsistency15` keeps
  guidance 8 through the Objective-C factory. The backend seeds the sampler from the request
  (`NFKDiffusionScheduler.seeded`), so `NFKParameterSeed` changes the LCM re-noising draws along
  with the initial latent. Hyper-SD's SD adapters (ByteDance/Hyper-SD, UNet-only kohya files, rank
  64, no text-tower keys; the SD 1.5 files adapt 278 targets, the SDXL files 788, downsampler
  convolutions included) merge through the same reader: the N-step files run `.ddimTrailing` at
  guidance 0, the CFG-preserved 8- and 12-step files `.ddimTrailing` at 8 or 12 steps with guidance
  5 to 8, and the unified `1step` files `.tcd` (`NFKTCDScheduler`, eta 1, one to eight steps;
  `NFKMLXTextToImageParameterKey.trajectoryConsistencyEta` sets eta per request, through the
  backend's `schedulerForRequest` seam). Their SD half is under the CreativeML Open RAIL++-M License
  (LICENSE.md, "other SD-related models", Copyright (c) 2024 Bytedance Inc.); the FLUX half carries
  FLUX.1 [dev] non-commercial terms and the SD3 half the Stability Community license, and neither is
  taken. Catalog names `hyper-sd15-4steps-lora`, `hyper-sdxl-4steps-lora`,
  `hyper-sd15-8steps-cfg-lora`, `hyper-sdxl-8steps-cfg-lora`, `hyper-sd15-1step-lora`,
  `hyper-sdxl-1step-lora` (`NFKMLXCatalogImage.hyperSDAdapters`; `StableDiffusionAdapter` carries
  the trained steps and guidance where they differ from the sampler's defaults). Parity, with the
  adapter fused in diffusers (records `IK_PARITY_HYPER_{SD15,SDXL}_{4STEP,8STEP_CFG,UNIFIED}_T2I`,
  the unified records at 2 steps so the re-noising step and the final-step rule both run): SD 1.5
  4-step image cosine 0.99999862 (mean abs 0.00073), SDXL 4-step 0.99999953 (0.00035), SD 1.5 8-step
  CFG 0.99999558 (0.00117), SDXL 8-step CFG 0.99999947 (0.00038), SD 1.5 unified 0.99999725
  (0.00097), SDXL unified 0.99999938 (0.00053); every context ≥ 0.9999999999969, every first
  prediction ≥ 0.9999999999934, every per-step latent ≥ 0.9999896575. Five of the six image figures
  sit within 1e-9 of their picture's own 8-bit floor; the SD 1.5 8-step CFG figure sits 3e-6 below
  its floor (0.99999859) because its eight guided steps compound a 1e-12 step-0 difference to 1e-5
  by step 7 (the run's own final latent decodes at 0.9999970 against the reference picture), a
  trajectory seam to explore; the decode and the merge measure clean. The first SD 1.5 trailing
  records sampled leading with offset (751, 501, 251, 1) because diffusers' `from_config` drops an
  overridden `timestep_spacing` the stored SD 1.5 config lists in `_use_default_values`; the oracle
  strips the key, and the records were re-made. Parity, with the adapter fused in diffusers through
  `load_lora_weights` + `fuse_lora` (the diffusers venv carries `peft` for it), from the reference's
  own initial latent: LCM-LoRA on SD 1.5 (`IK_PARITY_SD15_LCM_LORA_T2I`) image cosine 0.99999858,
  mean abs 0.00089; context 0.9999999999984, first prediction 0.9999999999997, per-step latents ≥
  0.99999999596. SDXL-Lightning four-step on SDXL base (`IK_PARITY_SDXL_LIGHTNING_T2I`, DDIM
  trailing both sides through `IK_SD_SPACING`) image cosine 0.99999917, mean abs 0.00056; context
  0.9999999999969, first prediction 0.9999999999995, per-step latents ≥ 0.9999999998. Two reference
  facts carry the LCM path: `LCMScheduler` spaces its schedule over `original_inference_steps` (50)
  as `floor(i · 50 / n)` into `19, 39, …, 999` reversed (4 steps: 999, 759, 499, 259), each step's
  previous ratio is the next step's, and the last step returns `denoised` without re-noising
  whatever `set_alpha_to_one` says; and the sampler re-noises every other step with a fresh draw, so
  the record carries the draws (`noises`, a `randn_tensor` hook in the oracle) and the port takes
  them through `NFKLCMScheduler.noise`.
- `NFKMLXSDTextToImageConfiguration.latentConsistency15` — LCM Dreamshaper v7, a whole latent
  consistency model (release `.lcmDreamshaperV7`): SD 1.5 geometry plus `time_cond_proj_dim` 256
  (`NFKMLXSDUNetConfiguration.timeConditionProjectionDimensions`, `time_embedding.cond_proj`, a
  `Linear(256, 320, bias: false)` added to the time embedding's input;
  `NFKMLXSDUNet.guidanceEmbedding(scale:channels:)` forms the input). The pipeline's
  `get_guidance_scale_embedding` embeds `(guidance − 1) · 1000` sines first over frequencies `i · step`
  with `step = −log(10000) / (half − 1)`, formed in that order (the timestep embedding divides by `half`
  and puts cosines first), and a UNet that carries the projection runs no classifier-free guidance
  whatever the scale (`do_classifier_free_guidance` requires `time_cond_proj_dim is None`), so `denoise`
  runs one pass with the embedding. The release ships float32 only (`SimianLuo/LCM_Dreamshaper_v7`, MIT).
  Parity (`IK_PARITY_LCM_DREAMSHAPER_T2I`, 4 steps, guidance 8) image cosine 0.99999278, mean abs
  0.00098; context 0.9999999999986, first prediction 0.9999999999998, per-step latents ≥ 0.99999999978.
  The image figure sits below the other SD end-to-end figures because the test reads the public path's
  picture back through `CGImage`, which rounds to eight bits; the reference picture rounded to eight bits
  against itself measures cosine 0.9999927803, mean abs 0.00098, the end-to-end figures exactly. This
  picture is dark (RMS 0.297, no saturated pixel), so the same
  rounding noise is a larger share of its signal than of the brighter, partly saturated SD 1.5, SD 2.1,
  LCM-LoRA, and Lightning pictures (their own floors: 0.99999894, 0.99999862, 0.99999858, 0.99999917).
  The uniform-rounding model predicts the floor from the picture alone: `1 − cos ≈ (1/255)² / (24 ·
  mean(x²))`. `verifyTextToImage` prints each record's floor, the image cosine against the eight-bit
  reference, and the decode seam in float (the reference's final latent and the run's own final latent
  through `NFKMLXSDAutoencoder`, against the reference picture before rounding). For LCM Dreamshaper the
  reference's final latent decodes to the reference picture at 0.999999999997 (mean abs 3.2e-7), the
  run's own final latent at 0.99999999968 (mean abs 2.3e-6), and the public path's picture matches the
  8-bit reference at 0.99999995. The other four records measure the same way: each image figure sits
  within 6e-8 of its own floor, and the reference final latent of each record that carries one
  (LCM-LoRA, SD 2.1, Lightning; the SD 1.5 record has no per-step trace) decodes at ≥ 0.99999999999.
  Customization: offline. The model is a consistency distillation of SD 1.5, and its objective needs
  the teacher UNet and the ODE solver of the distillation; the SD 1.5 LoRA recipe does not describe it.
  SDXL base 1.0 (`.sdxl`, release `.sdxlBase`) joins as the base the Lightning adapters merge into;
  the Turbo directory does not stand in for it. Its download takes the half-precision files
  and the two network `config.json`s, which `upcast_attention` and `force_upcast` are read from; a VAE
  config without `force_upcast` upcasts, the reference's default.
- `NFKMLXSDPromptTokenizer` — the release's `tokenizer/` directory driving the core's `NFKTokenizer`
  CLIP variant, plus the padding to the tower's context length. Token-for-token agreement with
  `transformers`' CLIPTokenizer over five prompts, including punctuation, an empty prompt, a multi-byte
  one, and the markers written out literally.
- `NFKMLXMarigold` / `NFKMLXSDUpscaler` (`@objc`) — two image-conditioned latent-diffusion models on
  `NFKMLXDiffusionBackend`, over the same real networks. Marigold (`marigold-depth`, image → depth) is
  Stable Diffusion 2 geometry and denoises a depth latent concatenated with the image latent; the ×4
  upscaler (`sd-x4-upscaler`, image → ×4 image) denoises a high-resolution latent conditioned on the
  low-resolution image itself, with a **noise level** joining the timestep through a class embedding
  (`noiseLevel`, the release's own default 20). The upscaler's autoencoder is one level shallower than
  the others, which is where its ×4 comes from — narrowing a test configuration must keep the level
  count or the model silently becomes a ×2. Output size and round-trip tested.
  Customization: trainable at `LoRA`, with no recipe written yet. prs-eth/Marigold at 2bfbdea
  (`src/trainer/marigold_depth_trainer.py`) trains the whole UNet with a masked latent MSE on the
  v-target, the autoencoder and the empty-text embedding frozen and no teacher; Adam at 3e-5 under its
  `IterExponential` schedule, an effective batch of 32 over 30k iterations. The affine invariance is in
  the target normalization, which maps the 2nd and 98th percentiles to [−1, 1]. On a device the level is
  LoRA on the UNet, as for the SD 1.5 UNet. The upscaler's objective is in
  Stability-AI/stablediffusion's `ddpm.py`: L2 on v, the low-resolution image noised at a level drawn
  uniformly from [0, 350) and passed as the class label, AdamW at 1e-4. That repository is deleted and
  was read from a mirror whose commit could not be pinned, which makes this the weakest reading behind
  the ruling; its launcher and degradation pipeline are not published, so a recipe supplies the
  degradation.
- `NFKMLXIPAdapterImageProjection` / `NFKMLXIPAdapterAttention` — IP-Adapter, lightweight image
  conditioning for a diffusion model (steer a Stable Diffusion generation with a reference image, not
  only text). Two pieces: the image projection maps a CLIP image embedding to a short sequence of
  image-text tokens (`image_embeds` Linear → reshape → LayerNorm), and the decoupled cross-attention
  adds a second, image-conditioned attention beside the text cross-attention — `text_attn + scale·ip_attn`,
  sharing the query, through its own `to_k_ip` / `to_v_ip` projections (the reference stores those under a
  `processor.` prefix, stripped on load). The projection and the extra key/value weights are the only
  trained parameters; the base UNet is frozen, so an adapter is a small file over a shipped SD model.
  Reference parity against diffusers (`run_reference.py ip_adapter`, `ltx` env): the ImageProjection
  against `diffusers.models.embeddings.ImageProjection`, and the decoupled attention against
  `IPAdapterAttnProcessor2_0` — both cosine 0.9999999999999997. The adapter is wired into the shipped
  `NFKMLXSDUNet`: an optional `NFKSDImageConditioning` (tokens + scale, nil by default so the base UNet is
  byte-identical) threads down → mid → up → transformer block → `attn2` Only, and `NFKSDAttention` gains
  optional `to_k_ip`/`to_v_ip` attached through `update(modules:)` (assigning a `@ModuleInfo` optional
  after init does not register it). `NFKMLXIPAdapter.load(from:into:)` reads the released
  `ip-adapter_sd15.safetensors`; the adapter's sorted indices map onto the cross-attentions in
  down → up → mid order, because `UNet2DConditionModel` registers `mid_block` last. Validated in the real
  SD 1.5 UNet with the real adapter against diffusers (`run_reference.py ip_adapter_unet`,
  `IK_PARITY_IP_ADAPTER_UNET`): cosine 0.99999999999989, with the base UNet and SD 1.5 text-to-image still
  at parity. `NFKMLXTextToImage.imageAdapterBackend(configuration:directoryURL:adapterURL:scale:)` is the
  consumer path (a precomputed CLIP-ViT-H image embedding under `NFKMLXInputImageEmbedding`; under CFG the
  unconditional row takes zero image tokens, the reference's `negative_image_embeds`).
- `NFKMLXTAESD` (`@objc`) — TAESD (Tiny AutoEncoder for Stable Diffusion), the fast preview decoder a
  latent-diffusion pipeline uses: a small distilled autoencoder mapping an image to a four-channel latent
  and back (8× down/up). The encoder and decoder are flat `nn.Sequential` stacks of 3×3 convolutions and
  residual blocks; modeling them as `[Module]` arrays makes the numeric Sequential keys (`0.weight`,
  `1.conv.0.weight`, …) match with no remap — the case the MLX-runtime note calls out as the legitimate
  use of numeric keys. `NFKTAESDBlock` is three convs (ReLU between) added back to the input then a fusing
  ReLU (skip identity, all 64→64); the downsample convs are stride-2 bias-free; the decoder clamps its
  input `tanh(x/3)·3`, upsamples nearest ×2, and ends in a `conv→3`. Parameter-free ops (ReLU, the clamp,
  the upsample) are empty marker `Module`s occupying their Sequential index; the net's forward dispatches
  by type. `NFKMLXTAESDBackend` reads `NFKInputImage` → the reconstruction under `NFKOutputImage`;
  `encode`/`decode` are the object accessors (the preview-decode use). `+register` under `taesd`. The
  release is two `.pth` files (encoder + decoder, GitHub, ~5 MB each); `Tools/taesd-to-safetensors`
  combines them into `encoder.*`/`decoder.*` keys and the loader transposes the 4-D convs. Reference
  parity against madebyollin's own `taesd.py` on the first numeric run: latent cosine 0.9999999999996,
  decode cosine 0.9999999999998, mean |difference| 1.9e-7.
  Customization: untrainable here. madebyollin/taesd at 401ce45 publishes the network and its weights
  only, with no training script, loss, or discriminator.
