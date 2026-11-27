<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX models: frame interpolation, optical flow, and video

- `NFKMLXRIFE` (`@objc`) — real frame interpolation: the released HDv3 IFNet in `MLXNN`, run through
  `NFKMLXTensorBackend` (two frames under keys `frame0`/`frame1` → the middle frame under
  `NFKOutputImage`). Three identical IFBlocks (11 input channels, width 90) run coarse-to-fine at scales
  4/2/1. Each block's trunk is four groups of two convolutions, each group added back to its own input
  (not one residual over eight), and flow and mask leave through separate heads (`conv1`/`conv2`), each
  two transposed convolutions undoing `conv0`'s ×4 stride. The net applies every block twice per scale:
  once as given and once with the frames swapped, the mask negated, and the flow halves exchanged,
  averaging the two, because the network is trained symmetric in its inputs. The bilinear backward warp
  is `grid_sample(align_corners=True, padding_mode='border')` built from `take` gather (MLX has no
  grid_sample); the per-scale resampling is bilinear, as the reference interpolates. `+register` under
  `rife`. `remapReferenceKey` strips the training wrapper's `module.` prefix, maps `blockN` → `blocks.N`,
  and names the `Sequential` entries of `conv0`/`convblock0…3` (convolution, PReLU) and the heads
  (`up1`, `prelu`, `up2`); the checkpoint's `block_tea` teacher is ignored as an extra key. Reference
  parity against the released HDv3 `flownet.pkl` (interpolated cosine 0.9999999999993, mean |difference|
  2.7e-7). Warp, interpolate, remap, and round-trip tested. Weights: `huggingface.co/yow46228/RIFE`
  ships the checkpoint together with its own `IFNet_HDv3.py`, which pins the architecture.
- `NFKMLXRIFEv4` (`@objc`) — the third IFNet generation, a separate architecture from HDv3. Reference
  parity on the released `rife-flownet-4.13.2` weights (interpolated cosine 0.9999999999995, mean
  |difference| 2.3e-7). Four blocks instead of three (widths 192/128/96/64), a learned frame `encode`
  module (`Head`: three convolutions and a transposed one to eight feature channels) whose features are
  warped alongside the frames, `ResConv` trunk entries that scale their convolution by a learned
  per-channel `beta` before the residual add, an upsampling convolution emitting `4 × 6` channels that
  pixel-shuffles ×2, and a timestep channel that is what v4 adds: `interpolate(_:_:timestep:)` lands
  anywhere between the frames, not only the midpoint. The backend reads it as an `NSNumber` under
  `timestepKey` (`NFKParameterTimestep`), clamped to `0...1`, defaulting to 0.5; it is the first
  model on `NFKMLXTensorBackend`'s request-aware init. Its convolutions activate with a parameter-free
  leaky ReLU where HDv3 used a PReLU; that difference surfaced as exactly eight uncovered parameters,
  the coverage guard naming a structural mistake rather than a number going quietly wrong. `+register`
  under `rife-v4`; pads to a multiple of 64 and runs scales `[8, 4, 2, 1]`. Oracle: the architecture
  vendored by ComfyUI-Frame-Interpolation (`rife_arch.py`, `arch_ver="4.17"`), whose own `IFNet.py`
  ships inside the model zip rather than in the repository; only one ComfyUI device helper needs
  stubbing.
  **Customization ships** at `full` (`NFKMLXRIFEv4Training.swift`), measured against the v4.15 training code
  hzwer/Practical-RIFE links from its README as Google Drive archives (kept under `<validation
  root>/reference-sources/practical-rife-train`). `run_reference.py rife_v4_training` runs that code's own
  `Model.update` on the CPU: `.cuda()` is an identity while it builds, `torchstat` is stubbed,
  `torchvision.models.vgg19` loads the local ImageNet weights, and `FlownetCas.module` returns the network
  the `DistributedDataParallel` wrapper would hold.
  - `NFKMLXRIFEv4Net.trainingPass` is `FlownetCas.forward` with the teacher. It takes a timestep per example
    and a scale per block, neither clamps nor pads, and blends the teacher's flow and raw mask by each block's
    sigmoid confidence over their sum plus 1e-3. The blocks emit that confidence as their sixth channel, and
    the warp runs a batch.
  - `NFKMLXRIFEv4Objective` is `Model.update`'s `loss_G`:
    - the VGG-19 term over `relu1_1`…`relu5_1` at weights 1/32…1 (`NFKMLXVGG19Features`), less 0.1 × SSIM
      (window 3, σ 1.5, stride 3);
    - plus each block's `encode_loss`, decayed by 0.8 a block;
    - plus 0.1 × the teacher term, the consistency term, and 0.05 × the decayed L1.
  - `encode_loss` compares the four features of a moving-average copy of the frame encoder. The reference's
    leaky ReLU runs in place, so the first three features it compares are activated. The objective copies
    the network's encoder when built and moves it 1% toward the network's after every update, as the
    reference's `soft_update` does. The reference copies the encoder before a checkpoint loads, so a
    fine-tune that follows its order scores features with a random encoder; the objective copies the
    loaded one.
  - `referenceOptimizer()` is AdamW at 1e-4 with decay 0.01 on every parameter. `referenceSchedule(steps:
    warmupSteps:)` is `get_learning_rate`: 2,000 updates of linear warm-up, then a half cosine to zero, plus
    1e-7. `fineTune` clips at 1, doubles each batch with its mirror, and draws the block scales per update
    (`[4, 2, 1, 1]` below 0.3, `[2, 1, 1, 1]` below 0.6, `[8, 4, 2, 1]` otherwise) from `scaleSeed:`.
  - Measured on the 4.13.2 release, two 96-pixel triplets at timesteps 0.5 and 0.25, mirror-doubled, the
    blocks at `[4, 2, 1, 1]`:
    - the frame, the teacher, and every loss term within float32 rounding of float64;
    - every gradient at the reference's float32 floor (1.3e-4 to 1.5e-3 from float64);
    - our forward on float64's updated weights and target 4.6e-8 from float64's total.
  - Adam's first update moves each weight by about the rate in its gradient's direction. 1,456 of the
    weights have gradients within float32 rounding of zero and step the other way from float64's. The tensors
    that move most hold one each, biases first, since a bias's gradient sums over every position. Those flips
    alone put the total after one update 2.7e-4 from float64; with float64's signs pinned it lands 7.4e-5
    away.
- `NFKMLXRAFT` (`@objc`) — real optical flow: the RAFT pipeline in `MLXNN` (shared feature encoder,
  all-pairs correlation volume + pyramid + bilinear lookup via `take` gather, context encoder, an
  iterative ConvGRU update). Run through `NFKMLXTensorBackend` (two frames `frame0`/`frame1` → a packed
  flow map under `NFKOutputImage`; raw `[H,W,2]` flow via `NFKMLXRAFTNet.flow`, the eighth-resolution
  field via `flowLow`). `+register` under `raft`. Faithful to RAFT-large (feature 256, 4 levels, radius
  4), including the convex-mask upsampling: each output pixel is a combination of its coarse 3×3
  neighborhood weighted by the mask the last update predicts (scaled ×0.25 as the reference does), over
  a zero-padded unfold. The default iteration count is low (6). Normalization is per encoder, as the
  reference `norm_fn` is: `fnet` uses a parameter-free InstanceNorm (`affine: false`, so the checkpoint
  carries nothing for it) and `cnet` uses BatchNorm, with `makeNet` setting eval mode for the running
  statistics. `flow` takes images in `0...1` and rescales to the trained `-1...1`; the correlation
  neighborhood is emitted in the reference's plane order (outer index shifts x, inner shifts y), which
  the trained 1×1 `convc1` depends on. The correlation lookup samples like the reference's
  `grid_sample(padding_mode: "zeros")`: a corner outside the map contributes nothing, where an edge
  clamp costs real accuracy (the lookup radius is 4 and the coarsest pyramid level is a few cells wide,
  so most of that neighborhood is outside). Converter `Tools/raft-to-safetensors/convert.py` renames
  `update_block`/`downsample`/`flow_head`/`mask`. Reference parity against princeton-vl's own RAFT on
  raft-things (eighth-resolution flow cosine 0.9999999999989, full-resolution 0.9999999999998); both
  sides run the same iteration count. Flow + round-trip tested under xcodebuild.
  `NFKRAFTCorrelation` carries the package's first custom Metal kernel, written as a source string
  through `MLXFast.metalKernel` (no `.metal` file, nothing for a consumer's build to link, compiled and
  cached by MLX on first use). The elementwise path walks 81 planes per level doing four gathers each,
  so one lookup was over thirteen hundred dispatches, and the update runs it once per GRU iteration; the
  kernel does the same arithmetic with one thread per (pixel, plane). Measured at RAFT's own
  eighth-resolution geometry, 2753 ms against 3.98 ms at 60×80, about 730×. `gatherLookup` stays as the
  CPU-stream path, since a Metal kernel cannot dispatch there and the package lets a caller select the
  CPU; the kernel is held to it bit for bit, including the zero-padding at the edges where most of the
  neighborhood lies. A silent fallback to the gathers would fail nothing and make the model unusable, so
  `testTheDispatchChoosesTheFusedPathOnTheGPU` compares the dispatched result to the fused one exactly
  rather than timing anything.
- `NFKMLXVideoBackend` / `NFKMLXVideoFile` — the first backend that produces video, and the AVFoundation
  decode/encode layer under it (the video counterpart of `NFKMLXWaveFile`). `NFKModalityVideo` and the
  `NFKInputVideo` / `NFKOutputVideo` keys were in the core's vocabulary with nothing emitting a clip.
  The backend reads an `NFKVideoAsset`, hands every frame to a `([MLXArray]) -> [MLXArray]` transform
  as `[H, W, 3]` in `0...1`, encodes what comes back, and returns a new `NFKVideoAsset`. The transform
  takes the whole sequence, not one frame, because that is what the models need: frame interpolation
  reads pairs and returns more frames than it took, and BasicVSR propagates state backward and forward
  through time, so upscaling a clip is not upscaling its frames independently. A per-frame model simply
  maps. `frameRateMultiplier` / `outputFramesPerSecond` carry the rate change a frame count change
  implies — a doubled clip written at the source rate is slow motion, not smoother footage, so the
  duration is what stays fixed. `NFKMLXRIFE.clipBackend` (`n` → `2n - 1` frames at twice the rate) and
  `NFKMLXVideoSR.clipBackend` (×4, `upscaleSequence`) are the shipped users. AVFoundation's synchronous
  property accessors are deprecated, so the reads go through a semaphore-blocked `loadTracks` /
  `load(.nominalFrameRate)`; the contract is synchronous and the caller is already off the render
  thread. H.264 needs even dimensions and one frame size per clip, and both are rejected explicitly
  rather than cropped or scaled where the change would be invisible in the result.
  Trained models are carried through the whole path in tests, not only the shape: a clip is built
  by translating a real photograph (a synthetic gradient measures nothing — the models are trained on
  photographs), and the assertions are about the result. RIFE's synthesized frame must correlate
  better with the true midpoint than with either neighbour (0.9981 against 0.9145 / 0.9104), which is
  what separates interpolation from copying a frame; BasicVSR's output must still be the source frame
  enlarged (0.9891). Both comparisons resize to a common size first — correlating a ×4 output against
  its small source compares the output's first rows and measures nothing, which read as a model
  failure at 0.805 until the comparison was fixed.
- `NFKMLXVideoSR` (`@objc`) — real video super-resolution: the complete **BasicVSR** (mmediting
  `BasicVSRNet`, ×4). **SPyNet** estimates flow between neighbors (coarse-to-fine pyramid, six
  five-conv modules; its ImageNet normalization tensors load from the checkpoint), and two
  propagation branches — backward and forward through time — warp their hidden features along that
  flow (`flowWarp`, a bilinear `grid_sample(align_corners=True)`: `zeros` padding for propagation,
  `border` inside SPyNet; the flow upsampling between pyramid levels is `align_corners=True` bilinear
  ×2, a separate grid from `resizeBilinear`). Fusion + two `PixelShufflePack` stages + `conv_hr`/
  `conv_last` reconstruct over a bilinear ×4 base. Bidirectionality means a frame draws on frames
  after it, so a clip goes through `NFKMLXVideoSRNet.upscaleSequence` whole; the backend upscales a
  single frame. `+register` under `video-super-resolution`. `remapReferenceKey` strips the
  checkpoint's `generator.` wrapper and maps the branches' positional `main.0`/`main.2` Sequential.
  Reference parity against mmediting's own BasicVSRNet on the released REDS4 checkpoint, over a
  three-frame translated clip (clip cosine 0.9999999999997946, mean |difference| 1.1e-7).
  Forward, bidirectional propagation, flow-warp identity/shift/padding-mode, remap, sequence, and
  round-trip tested.
- `NFKMLXWanAnimate` — the Wan 2.2 Animate 2 denoising transformer
  (`Wan-AI/Wan2.2-Animate-2-14B`, Apache-2.0), which drives a reference character image with the
  motion of a driving video. The block is the Wan adaLN block this package already ports — a
  `[1, 6, dim]` modulation added to the timestep projection and chunked six ways, a non-affine
  LayerNorm scaled and shifted before self-attention with a gated residual, an affine cross-attention
  norm, and a gated feed-forward — plus an image cross-attention branch in every block
  (`add_k_proj` / `add_v_proj` / `norm_added_k`) fed by the `img_emb` projector over CLIP embeddings.
  Its naming inverts the text-to-video port's: `norm3` is the affine cross-attention norm and `norm2`
  the non-affine pre-feed-forward one.
  The port is the reference mechanism, not the block. Generation runs two passes over one
  `NFKMLXWanAnimateKVCache`. `extractReference` runs the reference latents and stores every block's
  PRE-rotary keys and values, modulated at a fixed timestep of 1 and rotated on a grid the
  `referOffset*` values place away from the generation grid (`t` 1, `h` 0, `w` the reference grid's
  own width, which is what the released `refer_offset_w = -1` resolves to). `generate` runs a chunk of
  the video, rotates the cached keys onto that offset grid, and attends per frame over the whole
  video's generation buffer plus the reference tokens at that frame's index: generation frame `f`
  reads reference frame `f - 1`, and frame 0 reads none.
  The buffer is the full video's, so a chunk shorter than the video leaves zero-filled key positions
  inside it, and the reference's own buffer leaves zero-filled frames past what the cache holds.
  Those positions are NOT masked out. A zero key scores zero against every query, so each one adds
  `exp(0)` to the softmax denominator while contributing nothing to the numerator, and a port that
  drops them is right only at full length. `maskedAttention` reproduces the dilution by counting the
  zero positions instead of materializing them, which is also what keeps the memory to the real keys;
  the reference expresses the same pattern as a flex `BlockMask` over a 128-aligned buffer and needs
  `torch.compile` for it.
  The released 14B model is 32.8 GB in bf16 and its pipeline about 50 GB (DiT 32.8, umT5-XXL 11.4,
  CLIP 4.8, VAE 0.8), so it does not run on a 32 GiB machine and the path is DeepSeek V4's: numeric
  parity at a tiny random configuration, plus the released file held to the module by shape. Reference
  parity against diffusers' `WanAnimate2Transformer3DModel` across both passes — cached reference keys
  0.9999999999999837, reference pass 0.9999999999999781, generation pass 0.99999999999998, chunked
  generation pass 0.9999999999999801 — and the released `wan_animate_2_bf16.safetensors` header holds
  exactly: 1303 tensors consumed, none missing, none mismatched, none unaccounted.
  `moduleKey(forRelease:)` and `releaseKey(forModule:)` convert the original Wan naming the release
  ships (every block under an extra `block.` level, single-letter attention projections, `k_img` /
  `v_img` / `norm_k_img` for the image branch) to and from the module's. Two of the four pipeline
  stages are already at parity here: `videomodel/Wan-AI/vae.pth` is the Wan VAE and
  `models_t5_umt5-xxl-enc-bf16.pth` the umT5-XXL encoder; the remaining stage is an
  `open-clip-xlm-roberta-large-vit-huge-14` image encoder (4.8 GB). There is no `registerAll` entry
  and no backend, because neither can run what the machine cannot hold.
  **Customization is ruled out here, not skipped.** A training path needs an optimizer step over the
  weights, and the smallest released form is 32.8 GB against 32 GiB of unified memory; the tiny
  configuration a test can train is not the released model. The oracle is `wananimatevenv` (Homebrew
  3.12, diffusers 0.40.0, torch 2.14.0), which is the only environment here carrying
  `WanAnimate2Transformer3DModel` — the 3.9 environments top out at diffusers 0.36.
- `NFKMLXWanAnimateGenerator` (`@objc`) — Wan 2.2 Animate 2 end to end from its diffusers release
  (`Wan-AI/Wan2.2-Animate-2-14B-Diffusers` as `wan2.2-animate-14b`, and the distilled
  `…-Distilled-Diffusers` as `wan2.2-animate-14b-distilled`; Apache-2.0), measured against diffusers'
  modular `WanAnimate2Blocks` / `WanAnimate2DistilledBlocks`. The original-layout repository
  (`Wan-AI/Wan2.2-Animate-2-14B`: `.pth` encoders, no configuration files) is not read; its transformer
  header is the one `testWanAnimateMatchesTheReleasedShapes` holds. The stages:
  - umT5-XXL (`NFKMLXT5Encoder`) for the prompt, the negative prompt, and the fixed reference prompt
    `人物动作的参考视频`. The modular pipeline tokenizes the prompt as written, with no whitespace cleanup.
  - The image encoder, `NFKMLXWanAnimateImageEncoderNet`: transformers' `CLIPVisionModel` (open-clip's
    XLM-RoBERTa ViT-H/14 visual tower, exact gelu) read at `hidden_states[-2]`, so the last block and
    the post-norm never run. Its input is the −1…1 frame resized to 224 by torch's non-antialiased
    bicubic (a = −0.75), mapped to 0…1 and normalized by CLIP's mean and std. It encodes the reference
    image (every denoising pass) and the driving video's first padded frame (the extract pass).
    transformers 5 saves the class without its `vision_model.` level; the release, saved by 4.57,
    has it, and the loader reads the release's names.
  - The Wan 2.1 autoencoder (`NFKMLXWanVideoVAENet`), the encode its deterministic mean, standardized
    by multiplying by the reciprocal standard deviation and undone by dividing by it, as the reference
    writes it.
  - The schedule the release's `scheduler_config.json` names (`NFKMLXWanAnimateSchedule`):
    `DPMSolverMultistepScheduler` (dpmsolver++, flow, midpoint, shift 5) for the standard release and
    `FlowMatchEulerDiscreteScheduler` (static shift 5) for the distilled one. The standard config writes
    `lambda_min_clipped: -Infinity`, which only a JSON5 read accepts. The variant is read from the
    scheduler class.
  The segment loop (`NFKMLXWanAnimatePipeline`), per the reference's blocks:
  - Frame size: the reference image's size rounded to multiples of 16 sets the aspect ratio (Python's
    `round`, ties to even), the target area sets the size, both sides truncated to multiples of 16. The
    crop box uses the ROUNDED image size while the letterbox paste uses the TRUE one, so the two can
    differ by a row (1920 × 1080 into 640 × 800 gives a 528 × 944 frame and a 527-row crop).
  - Letterbox: PIL's 8-bit resample (`NFKMLXPILResample`), bicubic for the image and bilinear for the
    video, pasted on black at `((W − w) / 2, (H − h) / 2)`.
  - Frame rate: the nearest source frame to each target time, ties to even; the backend resamples from
    the movie's own rate.
  - Zigzag padding: the last `padding` real frames appended reversed, so every segment is
    `segmentLength` frames. A clip shorter than its own padding is refused: the reference's slice wraps
    below zero there and its segment shapes no longer agree.
  - Per segment: the driving slice encoded on its own (the causal autoencoder makes slicing one
    whole-video encode a different computation) under an all-ones mask; the previous segment's last
    `carriedFrames` decoded frames, zeros after, encoded under a mask of those frames (the bicubic
    resize the reference applies is to the same size, an identity); fresh noise, schedule, and cache;
    the extract pass; the guided loop; the decode of the latent frames after the reference frame, the
    first `carriedFrames` dropped after the first segment.
  - Guidance: the unconditional pass skips block 9 by index (`unconditionalSkippedBlock`), a constant
    in the reference's forward; a stack of fewer blocks skips none. Both passes of a step run block by
    block (`generate(…passes:…)`), so a streamed block is read once per step.
  Runtime: staged residency (encoders, then transformer + autoencoder); `.streamed` holds the first
  transformer blocks that fit and reads the rest from the release on every pass (bit-exact against held
  blocks); `quantizationBits` 2 to 8 packs every linear layer inside the blocks as each block loads, so
  the bfloat16 form is never held whole (the released transformer at 8 / 6 / 4 / 3 / 2 bits: 17.65 /
  13.61 / 9.58 / 7.56 / 5.54 GB, against 32.79 GB). A packed transformer does not stream. Precision
  `.checkpoint` keeps every stage's stored type (bfloat16, the autoencoder float32); `.float32` widens all
  of them. The model has no key-value cache across requests, no draft, and no streamed partial output:
  each segment's frames exist only once its loop ends, and the backend reports progress per step.
  **Measured on a test-dev release.** `Tools/wan-animate-testdev/make_testdev.py` builds every component
  through its own reference class from the release's own `config.json`, with only the size keys reduced
  (transformer 64 wide, 2 heads, 10 blocks; umT5 32 wide over the full 256,384-token vocabulary;
  CLIP 1,280 wide, which the transformer's projector reads, 3 layers; autoencoder base width 8), seeds the
  weights, and saves them in the release's stored types, so the directory carries the release's layout,
  file names, and keys. The release's tokenizer, scheduler, image processor, and index files are copied.
  `run_reference.py wan_animate_pipeline` runs the modular blocks over it at float32: a 70 × 50 reference
  image and a 17-frame 30 × 40 driving clip at 30 fps, resampled to 13 frames, letterboxed to 64 × 48,
  padded to 17, two 9-frame segments carrying one frame, three steps. Standard release (DPM-Solver++, guidance 3, the unconditional pass skipping block 9): prompt features 0.9999999999998531, reference prompt 0.9999999999999162, reference image CLIP 0.9999999999994768, driving frame CLIP 0.9999999999994958, reference latents 0.9999999999999623, clip 0.9999999999992298 (max |difference| 4.5e-6).
  Distilled release (Euler, no guidance): the same encoder figures and clip 0.9999999999992134 (max |difference| 4.9e-6). The token ids of all three prompts are exact, and the letterboxed image and video agree with PIL's to 1e-6. An 8-bit packed transformer holds the clip at 0.999999113826966 against float32. The released stages are held to their loaders by shape
  (`testTheReleaseMatchesItsLoaders`: transformer 1303, umT5 242, image encoder 519, autoencoder 194
  tensors, none missing, mismatched, or unconsumed). Customization is ruled out as for the transformer.
- Wan 2.2 Animate 2 LoRA fine-tuning (`NFKMLXWanAnimate.fineTune`, `NFKMLXWanAnimateTraining.swift`) — as
  DiffSynth-Studio trains it (1e8ae1d, `lora/Wan2.2-Animate-2-14B.sh`; Wan-Video publishes no training
  code). The trainer runs both passes in one forward (`forward_origin`): the reference pass over the
  driving video's latents at timestep 1, then the generation pass over the noised stack of the reference
  image's latent and the clip's latents, the whole stack in the loss. The port's training forward
  (`NFKMLXWanAnimateNet.trainingPrediction`) runs `forward_origin` as DiffSynth does, block by block: each
  block's reference pass, then its generation pass over the keys and values that pass just formed, so no
  block's keys and values outlive it. Its prediction equals `extractReference` into a cache followed by
  `generate` over it, exactly. LoRA on the blocks' `self_attn.q/k/v`
  (rank 32, alpha 32), DiffSynth's Wan sampler and Gaussian weight (`NFKMLXWanTrainingPreset.diffSynth`'s
  objective), AdamW 1e-4, weight decay 0.01, warm-up 5. The adapters save under the release's original
  names with PEFT's `.default` level and the scaling folded into the up weight, as DiffSynth's loader fuses
  at 1; `NFKMLXWanLoRA` reads them back. `NFKMLXWanAnimateGenerator.trainingExample` builds an example with
  the generator's own encoders, following `WanVideoUnit_Animate2VAEEmbedder`: the condition is the
  reference image's all-ones mask and latent, then a zero-mask slot over the latents of a zero clip; the
  driving latents carry an all-ones mask; the image features are the reference image's and the driving
  video's first frame's. The reference prompt defaults to the one generation reads; DiffSynth's dataset
  supplies its own (`animate2_prompt_ref`).
  DiffSynth's attention is `torch.compile`d flex attention, which runs on no CPU here and has no CPU
  backward, so the oracle evaluates it densely from DiffSynth's own mask and score functions
  (`_diffsynth_dense_flex_attention`: `softmax(score_mod(q·kᵀ·scale))` over the admitted keys, zeros where
  none is admitted). Measured (`run_reference.py wan_animate_lora_diffsynth`, wantrain-diffsynth env,
  DiffSynth's own tiny `WanAnimate2Transformer` under its LoRA, `FlowMatchSFTLoss` over
  `model_fn_wananimate`): prediction 1.0000000; loss 1.9610462 against 1.9610443; lowest of 12 adapter
  gradients 0.99999994. A QLoRA run over the Animate test-dev release at 8 bits trains from examples the
  generator builds.
  The recipe passes `--use_gradient_checkpointing`, under which DiffSynth checkpoints each block's
  `forward_origin`. `NFKMLXWanAnimateNet.recomputesBlocksInBackward`, on by default in `fineTune`, does the
  same: each block's pair of passes runs through `NFKMLXCustomFunction.recomputed` over the block's trainable
  arrays and both passes' hidden states, with the cotangent wait and the parameter-gradient ordering
  (`mlx-runtime-gotchas.md`, "A recomputed block's backward needs two orderings"). The loss and every
  adapter gradient are bit-identical to the plain backward's, the plain backward held equal to itself first
  (`testRecomputingBlocksInTheBackwardKeepsTheLossAndGradients`). One LoRA step at the release's geometry in
  its stored bfloat16 (`testALoRAStepAtTheReleasedGeometryHoldsOneBlockAtATime`, rank 32, a 13-frame
  640 × 352 clip: 4,400 generation tokens, 3,520 reference tokens, 512 text and 257 image tokens a pass), MLX
  peak, 2026-11-22:

  | Blocks | Plain | Recomputed |
  |---|---|---|
  | 1 | 15.501 GB | 15.905 GB |
  | 2 | 31.326 GB | 17.163 GB |

  A plain block adds 15.825 GB to the step and a recomputed one 1.258 GB, its bfloat16 weights included. The
  test runs the plain arm at 1 block only, since its 2-block arm reaches the standing resident limit; its bars
  are 18 GiB for 2 recomputed blocks and 2 GiB a block. The 40-block release's step at this clip is
  unmeasured.
- `NFKMLXWanFunGenerator` (`@objc`) — the six Wan 2.2 Fun releases (alibaba-pai, Apache-2.0), end to end from
  their original-layout directories, at reference parity against VideoX-Fun's own `Wan2_2FunControlPipeline` and
  `Wan2_2FunInpaintPipeline` (aigc-apps/VideoX-Fun at 98fffa9). `NFKMLXWanFunVariant` names them:
  - 5B (`Wan2.2-Fun-5B-Control` @b8bc1a65, `-InP` @d3e911d5, `-Control-Camera` @4dde4056): Wan 2.2 TI2V-5B's
    geometry (3072 wide, 30 blocks) and the Wan 2.2 autoencoder (48 latent channels, 16× spatially);
  - A14B (`Wan2.2-Fun-A14B-Control` @297be6d5, `-InP` @2d27b954, `-Control-Camera` @da1f119d): two 14B experts
    (`high_noise_model/`, `low_noise_model/`) over the Wan 2.1 autoencoder; the high-noise expert runs every step at
    or above timestep 900, the `boundary` of VideoX-Fun's `config/wan2.2/wan_civitai_i2v.yaml`.
  The release defaults are VideoX-Fun's examples' (`examples/wan2.2_fun/predict_*.py`), since the repositories
  state none: Euler at a flow shift of 5 (`NFKMLXFlowMatchConfiguration.wan`), 50 steps, guidance 6, and 121
  frames of 704 × 1280 at 24 fps (5B) or 81 of 480 × 832 at 16 fps (A14B).
  - **The original layout** (`NFKMLXWanOriginalRelease`). The transformer's `config.json` and safetensors carry
    the original Wan names, the autoencoder is `Wan2.2_VAE.pth` or `Wan2.1_VAE.pth`, and umT5 is
    `models_t5_umt5-xxl-enc-bf16.pth` with its tokenizer under `google/umt5-xxl/`. The family shares these three
    files byte for byte (LFS sha256, 2026-11-26): every Fun release, S2V, VACE-Fun, Wan-Dancer, and SkyReels V3. The
    transformer names map as diffusers' `convert_wan_transformer_to_diffusers` maps them (`norm3` is the affine
    cross-attention norm, `modulation` the scale-shift table); the autoencoder's as `convert_vae` (Wan 2.1, whose
    decoder groups the flat list `numResBlocks + 1` residual blocks then an upsampler) and `convert_vae_22`; umT5's
    original `T5Encoder` names as transformers' `UMT5EncoderModel` (`gate` is `wi_0`, `fc1` is `wi_1`). Neither
    `.pth` carries a configuration, so the autoencoder's widths and umT5's geometry are read from the tensors.
  - **The tokenizer** is `google/umt5-xxl/tokenizer.json`'s fast tokenizer: the SentencePiece model's Unigram
    pieces behind a normalizer that replaces every run of two or more spaces with one and applies no character
    map. The bare `spiece.model` keeps a double space as an extra `▁` piece. VideoX-Fun passes the prompt as
    written, with no other cleanup.
  - **The transformer** is `NFKMLXWanTransformerNet` with three Fun inputs and the original arithmetic
    (`formsSinusoidsInFloat64`: the timestep sinusoid and the rotary angles in float64, as `sinusoidal_embedding_1d`
    and `rope_params` form them). `ref_conv` patchifies a reference image's latent into a frame of tokens ahead of
    the clip: rotary frame 0, the clip's frames shifted by one, the timestep of the last clip token, dropped before
    the unpatchify. The camera adapter (`control_adapter`, VideoX-Fun's `SimpleAdapter`) pixel-unshuffles a
    Plücker embedding by 8 (A14B) or 16 (5B), patchifies it, runs one residual block, and adds it to the patch
    embedding. A 5B release reads a timestep per token (`NFKWanConditioning.tokenTimesteps`, held as distinct
    values and an index so a clip never holds a modulation per token).
  - **The conditions** (`NFKMLXWanFunConditions`, `NFKMLXWanFunPipeline.prepare`) follow the pipelines' code. The
    input channels are the noise, the control latents (a Control release; zeros without a control video), the
    mask latents, and the masked video's latents: 148 for 5B Control, 100 for 5B InP and Camera, 52 and 36 for
    A14B. The mask repeats pixel frame 0 four times and folds every four frames into a latent frame's channels,
    then resizes trilinearly with `align_corners=False`, the first latent frame apart (`resize_mask`). A 16×
    autoencoder's release also forms the latent mask (trilinear, `align_corners=True`): each token's timestep is
    the step's times the mask, subsampled by the patch, and where the first frame is wholly kept every step blends
    the masked latents back. VideoX-Fun's Inpaint pipeline takes that path where `transformer.config.dim == 3072`,
    and its Control pipeline where the autoencoder compresses 16×; the two agree on every release. A start and an
    end image are the inpainting video and mask `get_image_to_video_latent` builds. `NFKMLXWanCameraTrajectory`
    reads a CameraCtrl pose file and builds the Plücker embedding as `process_pose_file` and `ray_condition` do
    (poses relative to the first in float64, intrinsics rescaled from 1280 × 720, normalized rays, `o × d` then
    `d`).
  - **Measured on test-dev releases** (`Tools/wan-fun-testdev/make_testdev.py`: each release's own `config.json`
    with 48 wide, 2 heads, 2 blocks, ffn 96, text 32; umT5 at 32 wide over the released vocabulary; the Wan 2.2
    encoder at 16 wide and its released 256-wide decoder, the Wan 2.1 autoencoder at its released width; the 5B InP
    transformer keeps 3072 wide and 24 heads, the value VideoX-Fun's Inpaint pipeline reads). `run_reference.py
    wan_fun_pipeline` (videoxfun env) runs 9 frames, three steps, guidance 4, three runs per release: every frame
    generated (with a control video or a camera trajectory, and a reference image where the release reads one), a
    kept start and end image, and a kept rectangle on frames 2 onward (fractional token timesteps, no blend).
    Token ids are exact and the prompt features 0.9999999999999197. Final latents and clip, first run:

    | Release | Final latents | Clip |
    |---|---|---|
    | 5B Control | 0.9999999999998469 | 0.9999999999799478 |
    | 5B InP | 0.9999999999794134 | 0.9999999999484279 |
    | 5B Control-Camera | 0.9999999999999299 | 0.9999999999801937 |
    | A14B Control | 0.9999999999998088 | 0.9999999999964444 |
    | A14B InP | 0.9999999999998035 | 0.9999999999962969 |
    | A14B Control-Camera | 0.9999999999999574 | 0.9999999999967171 |

    The start-image and masked runs sit at 0.99999999997 or closer on every release; the 5B InP rows are the
    3072-wide transformer's. The condition channels read 0.9999999999972 or closer, the reference latent
    0.9999999999988, the camera embedding from VideoX-Fun's own `asset/Zoom_In.txt` 0.9999999999999966 (largest
    difference under 1e-5), and the A14B expert of every step matches. The generator through each residency gives
    the pipeline's figures; a staged load and a streamed block compute exactly as held ones; an 8-bit pack holds the
    latents above 0.999.
  - **Released transformers held by shape** (`testEveryFunReleaseMatchesItsLoader`, `IK_SHAPES_ROOT`
    `wan22-fun-*`): 5B Control 827, InP 825, Control-Camera 831 tensors; each A14B expert 1097, 1095, and 1101; none
    missing, unconsumed, or mismatched. No released-weight run: a 5B release is 24 GB and an A14B release about 70 GB.
  - Runtime: staged residency (umT5, then the autoencoder with the 5B transformer, or each A14B expert as a stage
    of its own, held only for its own steps); a streamed transformer reads the blocks the plan cannot hold from
    the release on every pass; `quantizationBits` 2 to 8 packs the blocks' linear layers as each block loads
    (`NFKMLXWanBlockedLoad` with the original names); precision `.checkpoint` (bfloat16 as stored) or `.float32`.
    umT5 widens to float32 where the plan affords it. The model has no key-value cache across requests, no draft,
    and no streamed partial output; the backend reports progress per step. Not built: VideoX-Fun's TeaCache,
    CFG-skip, and RIFLEx options, which change the sampling rather than reproduce it.
- Wan 2.2 Fun LoRA fine-tuning (`NFKMLXWanFunGenerator.fineTune`, `NFKMLXWanFunTraining.swift`) — the A14B
  releases as DiffSynth-Studio trains them (1e8ae1d, `lora/Wan2.2-Fun-A14B-Control.sh`, `-InP.sh`,
  `-Control-Camera.sh`), the 5B releases as VideoX-Fun does (`scripts/wan2.2_fun/train_control_lora.sh`), which
  DiffSynth-Studio does not train. `trainingExample(video:prompt:conditions:)` builds an example with the
  generator's own encoders and the sampler's own condition preparation; DiffSynth's Fun units
  (`WanVideoUnit_FunControl`, `_FunReference`, `_FunCameraControl`, `_ImageEmbedderVAE`) build the same channels
  for a frame-wise mask. The objective runs the transformer's original forward over the example's conditions
  (`NFKMLXWanObjective.prediction`).
  - DiffSynth (`NFKMLXWanFunGenerator.diffSynthPreset(expert:)`): `NFKMLXWanTrainingPreset.diffSynth` on the
    expert's side of 0.358 (timestep 900), rank 32 on `q,k,v,o,ffn.0,ffn.2`. Measured (`run_reference.py
    wan_fun_lora_diffsynth`, wantrain-diffsynth env, DiffSynth's own `WanModel` with `has_ref_conv` or
    `add_control_adapter` under its LoRA, `FlowMatchSFTLoss` over `model_fn_wan_video` with `y`,
    `reference_latents`, and `control_camera_latents_input`): Control prediction 0.9999999999999518, loss
    1.9878968 against 1.9878968; Camera prediction 0.9999999999998113, loss 3.8798108 against 3.8798115; every
    adapter gradient at cosine 1.0000000000 to ten places.
  - VideoX-Fun (`NFKMLXWanTrainingPreset.videoXFun(expert:)`): an index uniform over the scheduler's construction
    table (`NFKMLXWanSigmaSampler.videoXFun`, diffusers' `FlowMatchEulerDiscreteScheduler` at shift 5), an A14B
    expert's side split at the entry nearest timestep 900, no loss weight, every element whose error exceeds 50
    dropped (`custom_mse_loss`), rank 64 at alpha 32 on `q,k,v,ffn.0,ffn.2` (`.queryKeyValueAndFeedForward`),
    AdamW at 1e-4 with weight decay 0.03 and epsilon 1e-10, a constant rate, the gradient norm clipped to 0.05, and
    the kohya layout VideoX-Fun saves beside its PEFT file. A 5B example carries its latent mask
    (`NFKMLXWanTrainingExample.latentMask`): the trainer sets every frame after the first to 1, and where the first
    is wholly kept it blends the masked latents into the noised latent and those tokens read timestep 0. The
    step is inline in the training loop, so `run_reference.py wan_fun_lora_videoxfun` transcribes its lines over
    VideoX-Fun's own transformer under PEFT: the 5B case (3072 wide, the first frame kept, a 12-token prompt the
    forward pads to 512) prediction 0.9999999999969619, loss 21.845076 against 21.845083, lowest gradient
    0.999999998378; the A14B low-noise case prediction 0.9999999999999551, loss 2.0726695 against 2.0726695,
    lowest gradient 0.999999999951. The A14B case carries a 512-token prompt: under 12 tokens and 500 padding
    rows its cross-attention query and key gradients fall near 1e-8, where float32 cancellation leaves them
    at cosine 0.9999982 and they discriminate nothing.
  - `testAFunExpertTrainsSavesAndReloadsThroughItsFactory` and `testA5BReleaseTrainsUnderVideoXFunsRecipeAndReloads`
    train on test-dev examples the generator builds, save in each trainer's layout, merge back into a fresh network
    from the factory (within 1e-4 at float32), and load through `addAdapter(at:scale:expert:)`. A 14B expert trains
    over a packed base (`network(directoryURL:expert:quantizationBits:precision:)`, QLoRA).
- Wan 2.2 VACE-Fun A14B (`NFKMLXWanFunGenerator`, `NFKMLXWanFunVariant.vaceA14B`, `wan2.2-vace-fun-a14b`;
  `alibaba-pai/Wan2.2-VACE-Fun-A14B` @4438caab, Apache-2.0, 81.2 GB) — the Fun A14B layout with VACE blocks in each
  expert. Measured at reference parity against VideoX-Fun's own `Wan2_2VaceFunPipeline` (aigc-apps/VideoX-Fun at
  98fffa9). The defaults are `examples/wan2.2_vace_fun/predict_*.py`'s with `config/wan2.2/wan_civitai_t2v.yaml`:
  Euler at a flow shift of 12 (`NFKMLXFlowMatchConfiguration.wanVACEFun`), 50 steps, guidance 5, the high-noise
  expert at and above timestep 875, and 81 frames of 480 × 832 at 16 fps.
  - **The VACE blocks** (`NFKMLXWanConfiguration.vaceLayers`, `vace_blocks`, `vace_patch_embedding`) are Wan blocks
    that run ahead of the main loop over a 96-channel context (`forward_vace`). The context is patchified. The
    first control block adds its projection (`before_proj`) to the clip's tokens, and each later one reads the
    previous one's output. All of them share the main blocks' modulation, text, and rotary tables. Each output's
    `after_proj` is a hint, which the block `vace_layers` names (0, 5, …, 35) adds after itself at
    `vace_context_scale`. The released names map as the main blocks' do, under `vace_blocks.N`.
  - **The context** (`NFKMLXWanVACEConditions`, `NFKMLXWanFunPipeline.prepareVACE`) follows `vace_encode_frames`,
    `vace_encode_masks`, and `vace_latent`:
    - the input is the control video, or the video where the mask (binarized at 0.5) keeps it, in `-1...1`;
    - its part outside the mask and its part inside it encode separately into 32 channels;
    - each subject image encodes alone, with zeros for its inside part, as a latent frame ahead of the clip;
    - the mask folds each 8 × 8 pixel block into 64 channels, resampled to the latent frames by nearest-exact
      interpolation, with zeros for the subject frames;
    - the noise covers the subject frames too, and they drop before the decode.

    A start and an end image are the video and mask `get_image_to_video_latent` builds.
  - **Measured on a test-dev release** (`Tools/wan-fun-testdev/make_testdev.py` on the release's small files, both
    reduced blocks VACE-fed). `run_reference.py wan_vace_fun_pipeline` (videoxfun env) runs 9 frames of 32 × 48,
    three steps, and guidance 5, twice: a control video with one subject image, then a video kept outside a
    rectangle on frames 2 onward. Token ids are exact and the prompt features 0.9999999999999197. The control run
    reads its VACE context at 0.9999999999997381, the final latents at 0.9999999999998047, and the clip at
    0.9999999999966065; the masked run reads 0.9999999999986078, 0.9999999999998026, and 0.9999999999965555. Each
    step's expert matches. The generator through each residency gives the pipeline's latents. A staged load and a
    streamed block compute exactly as held ones; an 8-bit pack holds the latents above 0.999.
  - **Released transformers held by shape** (`testEveryFunReleaseMatchesItsLoader`, `wan22-vace-fun-a14b-high` and
    `-low`): 1331 tensors each, 236 of them the VACE blocks', none missing, unconsumed, or mismatched.
  - Runtime: the Fun A14B staging, streaming, and packing. A packed load packs the VACE blocks' linear layers with
    the main blocks'. A streamed transformer streams its main blocks; the eight VACE blocks (5.6 GB of an expert's
    34.7 GB at bfloat16) stay held, because the layer stream walks one block list.
- Wan 2.2 VACE-Fun LoRA fine-tuning (`NFKMLXWanFunGenerator.vaceDiffSynthPreset(expert:)`) — DiffSynth-Studio's
  recipe (1e8ae1d, `lora/Wan2.2-VACE-Fun-A14B.sh`): the Fun A14B recipe on the expert's side of 0.358, on the VACE
  blocks alone (`--lora_base_model vace`, `NFKMLXWanLoRATargets.vaceAttentionAndFeedForward`).
  `trainingExample(video:prompt:vace:)` builds the clip's latents, led by each subject image's latent
  (`WanVideoUnit_InputVideoEmbedder`), and the context the sampler prepares (`WanVideoUnit_VACE`). The loss reads
  every frame. Measured (`run_reference.py wan_vace_fun_lora_diffsynth`, wantrain-diffsynth env: DiffSynth's own
  `WanModel` and `VaceWanModel`, split from the test-dev's low-noise expert, the LoRA on the VACE model):
  prediction 0.9999999999999534, loss 3.1937010 against 3.1937015, and the lowest of the 20 adapter gradient
  cosines 0.999999999907. `testAVACEFunExpertTrainsSavesAndReloadsThroughItsFactory` saves in DiffSynth's layout, merges
  into a fresh expert from the factory (20 layers), and loads through `addAdapter(at:scale:expert:)`.
- `NFKMLXWanS2VGenerator` (`@objc`) — Wan 2.2 S2V-14B (`Wan-AI/Wan2.2-S2V-14B` @dab4e9c5, Apache-2.0), speech-to-video:
  a reference image of a person speaking or singing a speech track, optionally driven by a pose video. Measured at
  reference parity against Wan-Video/Wan2.2's own `WanS2V.generate` (1ea34ff). The release is the original layout
  (`NFKMLXWanOriginalRelease`): umT5, `Wan2.1_VAE.pth`, the transformer's `config.json` and four bfloat16 shards
  (32.6 GB), and `wav2vec2-large-xlsr-53-english/` (1.3 GB at float32). The defaults are the release
  configuration's and `generate.py`'s: segments of 80 frames at 16 fps, as many as the speech covers, each over
  the official flow UniPC at a shift of 3 for 40 steps, guidance 4.5, the S2V negative prompt, and a clip within
  1280 × 720 pixels whose shape follows the reference image (`get_size_less_than_area`, sides multiples of 64).
  - **The transformer** (`NFKMLXWanS2VNet`, `WanModel_S2V`) runs the Wan block over three token kinds, marked by
    a learned embedding (`trainable_cond_mask`). The clip's tokens carry the pose latents' patches added
    (`cond_encoder`). The reference image's tokens sit at rotary frame 30. The previous segment's last 19 latent
    frames are frame-packed (`FramePackMotioner`, buckets 1, 2, and 16 patchified at 1×, 2×, and 4×), and their
    rotary frames run negative through `np.linspace(...).astype(int)` spreads with the conjugate rotation. The clip
    modulates at the step's timestep and every other token at timestep 0 (`zero_timestep`, two modulation rows).
    After each of the 12 `audio_inject_layers`, each latent frame's clip tokens pass an AdaIN on that frame's global
    audio token (diffusers' `AdaLayerNorm` with `chunk_dim=1`: shift first, then scale, eps 1e-5) and attend over its
    five audio tokens. The audio encoder (`CausalAudioEncoder`) weighs wav2vec2's 25 hidden states by a SiLU of
    learned weights and runs causal replicate-padded convolutions (`MotionEncoder_tc`) over the 73 motion frames'
    copies of the first bucket ahead of the segment's.
  - **The audio front end** (`NFKMLXWanS2VPipeline.audio`) follows `AudioEncoder`: the waveform normalized per
    utterance, wav2vec2's hidden states as transformers 4.51 lists them for a stable-layer-norm encoder (the
    encoder's input, each layer's output but the last, and the final layer norm's output), linear interpolation
    from 50 to 30 fps (`align_corners=True`, the positions in float32 as torch's CPU kernel forms them), and
    `get_audio_embed_bucket_fps`, which reads the feature frame nearest each video frame (NumPy's round half to
    even) and zeros past the speech. The official `requirements.txt` pins transformers 4.49 to 4.51.3.
    transformers 5 lists the last layer's raw output as the last hidden state, so a 5.x oracle's audio features
    read 0.987 against the release's arithmetic.
  - **The prompt** passes the official tokenizer's cleanup (`NFKMLXWanPromptCleaning`, `whitespace_clean(basic_clean(text))`):
    the character-level fixes of ftfy 6.3's `fix_text`, HTML unescaped twice, and whitespace collapsed. ftfy's
    fullwidth-to-halfwidth fix turns the S2V negative prompt's `，` into `,`. Mojibake repair and all named HTML
    references but six are not applied.
  - **The sampler** (`NFKMLXWanS2VPipeline.generate`) follows `generate()`. The first segment's motion frames are
    blank and dropped: it decodes from the reference latent and drops its first three frames. Each later segment
    reads the latents of the last 73 decoded frames. `startsFromReference` (`--start_from_ref`) puts the reference
    in the last six motion frames and keeps the first segment's. A pose video's frames are padded with −1 to the
    segments' length, and each segment encodes with its first frame ahead (`load_pose_cond`); without one the pose
    latents are zeros. The unconditional pass reads the negative prompt and silent audio. The schedule is
    `FlowUniPCMultistepScheduler`'s (`NFKMLXUniPCScheduler.setOriginalTimesteps`): `linspace(float32(0.999), 0)`
    shifted, each timestep truncated, under the same predictor-corrector diffusers' solver runs.
  - **Measured on a test-dev release** (`Tools/wan-s2v-testdev/make_testdev.py`). It uses the release's own
    `config.json` at 48 wide, 2 heads, and 2 blocks, both injecting audio, ffn 96, text 32. Alongside: umT5 at 32
    wide over the released vocabulary, the Wan 2.1 autoencoder at its released width, and wav2vec2 at 32 wide over
    the released 24 layers with the release's `weight_g`/`weight_v` names. `run_reference.py wan_s2v_pipeline` (the
    `wans2v` env) runs the official `generate()` unchanged. Its setup is a 64 × 128 reference, 0.8 s of 16 kHz
    speech (two 8-frame segments), three steps, guidance 4.5, and shift 3. On the CPU at float32,
    `flash_attention` becomes a float substitute (`Tools/wan-s2v-testdev/wan_shims.py`). The run is plain, then
    repeated with a pose video three frames short and `init_first_frame`. Results:
    - token ids exact, including the cleaned negative prompt;
    - prompt features 0.9999999999999358 and negative features 0.9999999999999095;
    - audio input values 0.9999999999999958, audio features 0.9999999999997524, audio buckets 0.9999999999997777;
    - audio local tokens 0.9999999999998374 and global tokens 0.9999999999999027;
    - reference latent 0.9999999999983651, and pose latents 0.9999999999966728 or closer;
    - every transformer call 0.9999999999992459 or closer;
    - clip 0.9999999999707869, and the pose run's clip 0.9999999999442221.

    The released transformer is held by shape (`testTheReleasedTransformerMatchesItsLoader`, `IK_SHAPES_ROOT`
    `wan22-s2v-14b`): 1260 tensors, none missing, unconsumed, or mismatched. The generator gives the pipeline's clip
    through every residency. A staged load and a streamed block compute
    exactly as held ones, and an 8-bit pack holds the clip above 0.999. No released-weight run: the release is
    49 GB.
  - Runtime: staged residency (umT5, then wav2vec2, then the transformer with the autoencoder, which every segment
    reads in turn). A streamed transformer reads the blocks the plan cannot hold from the release on every pass,
    and `quantizationBits` 2 to 8 packs the blocks' linear layers as each loads. Precision is `.checkpoint`
    (bfloat16 as stored) or `.float32`, and umT5 widens to float32 where the plan affords it. The model has no
    key-value cache across requests, no draft, and no streamed partial output; the backend reports progress per
    step. Not built: the CosyVoice text-to-speech front end (`--enable_tts`), a separate model, and the
    sequence-parallel path. The backend reads 16 kHz speech and refuses other rates (the reference resamples with
    librosa).
- Wan 2.2 S2V LoRA fine-tuning (`NFKMLXWanS2VGenerator.fineTune`, `NFKMLXWanS2VTraining.swift`) — DiffSynth-Studio's
  recipe (1e8ae1d, `lora/Wan2.2-S2V-14B.sh`): `NFKMLXWanTrainingPreset.diffSynth` (an index uniform over the shift-5
  table, its Gaussian loss weight, AdamW at 1e-4), rank 32 at alpha 32 on `q,k,v,o,ffn.0,ffn.2`. PEFT's name match
  takes both attentions and the feed-forward of every block and the audio injector's four projections. The
  example (`trainingExample(video:prompt:audio:pose:)`) is what `WanVideoUnit_S2V` and
  `WanVideoUnit_ImageEmbedderFused` leave: the clip's latents with its first frame, the reference image, kept
  clean (`first_frame_latents`) and left out of the loss; the speech's first segment of buckets; the blank motion
  clip dropped; and a pose video's latents where given. The loss runs the transformer as `model_fn_wans2v` does.
  Measured (`run_reference.py wan_s2v_lora_diffsynth`, wantrain-diffsynth env, DiffSynth's own `WanS2VModel` over
  the test-dev weights under its LoRA, `FlowMatchSFTLoss` over `model_fn_wan_video`): prediction
  0.9999999999999570, loss 4.2902217 against 4.2902217, and the lowest adapter gradient cosine 0.999999999862 over
  the 28 adapters (2 blocks × 10, 2 audio injectors × 4).
  `testS2VTrainsSavesAndReloadsThroughItsFactory` trains on an example the generator builds, saves in DiffSynth's
  layout (the release's names, `.default`, the scale in the up projection), merges back into a fresh network from
  the factory (`NFKMLXWanLoRA.merge(adapterURL:into:)`, 28 layers), and loads through `addAdapter(at:scale:)`.
- `NFKMLXWanDancerGenerator` (`@objc`) — Wan-Dancer-14B (`Wan-AI/Wan-Dancer-14B` @85ce88dd, Apache-2.0),
  music-to-dance: a reference image of a dancer dancing to a music track. Measured at reference parity against
  Wan-Video/Wan-Dancer's own pipeline (e6c87a9, a DiffSynth-Studio fork). The release is the original layout beside
  two transformers: umT5, `Wan2.1_VAE.pth`, open-clip's `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`
  (float32, 4.8 GB, the vision tower read), and `global_model.safetensors` and `local_model.safetensors` (34.5 GB
  each at bfloat16) under one `config.json`. The defaults are the released scripts': 720 × 1280 clips of 149 frames,
  48 global and 24 local steps of DiffSynth-Studio's flow match at shift 5 (`sigma_min` 0, `extra_one_step`),
  guidance 5 with the unconditional pass skipping block 9 (`enable_skip_layer`), and the scripts' negative prompt.
  - **The two stages** follow `gen_video_global.py` and `gen_video_local.py`. The global stage plans one 149-frame
    clip over the whole track at `30 / round(T / 149)` frames a second, the reference image its first keyframe, its
    prompt ending `帧率是<fps:.4f>`, and decodes each latent frame alone (`enable_vae_decode_framewise`). The local
    stage renders the track in 149-frame segments at 30 frames a second: each segment's keyframes are the global
    frames `process_global_video_firstlastframe` names (`total / N` frames apart, the last segment's final frame,
    each segment's last frame the next one's first), its music the segment's clip as `get_music_clip_149f` cuts it
    (written and read as 16-bit PCM, `floor(x · 32768)` as libsndfile converts), its prompt ending `, 帧率是30fps。`,
    its seed `10 · segment + seed`. The segments concatenate, trimmed to the track's container duration (to the
    hundredth, as ffmpeg reports it) less 0.2 s. The scripts' MP4 round trip of the global video is not modeled.
  - **The transformer** (`NFKMLXWanDancerNet`, `WanModel` with `dit_model_type=1`) is the Wan 2.1 I2V-14B block in
    the fork's layout (its RMS norm rounds before the scale) over a context of the reference image's projected CLIP
    tokens (`img_emb_refimage`), the first keyframe's (`img_emb`), and the text. The image-to-video cross-attention
    splits that context at token 257, so the reference image feeds the image branch and the keyframe's tokens join
    the text. The global stage reads its own patch embedding and head (`enable_unimodel`) and spreads its rotary
    frames `30 / (fps + 1e-6)` apart by interpolating the complex128 rotary rows (`get_single_freqs`). The music
    encoder (`music_projection`, two pre-norm `MusicEncoderLayer`s whose query and key inputs are rotated before
    torch's `MultiheadAttention` projects them) reads the 35 features; its output is interpolated bilinearly to
    149 × 4800 and then to eight tokens a latent frame at the model width. After each of the eight injected blocks,
    each group of `h·w/8` tokens attends to one music token, so the injector adds that token's projected value.
  - **The sequence-parallel path.** The released scripts run eight-way Ulysses sequence parallelism, and
    `model_fn_wan_video` reads the world size only on that path. Each rank encodes its eighth of the music
    features (truncated to a multiple of eight), which changes the encoder's result; `musicEncoderSpans` (8)
    reproduces it. Every other split (the attention, the tokens, the music injection) computes what one process
    computes.
  - **The music front end** (`NFKMLXWanDancerMusic`) ports `get_music_base_feature` under librosa 0.10.2: the onset
    envelope, 20 MFCCs, CENS chroma over a 252-bin constant-Q transform at the estimated tuning, the onset peaks,
    and the dynamic-programming beats from a tempo prior read off a 22050 Hz load, all computed as though the
    15360 Hz samples were at 22050 Hz, as the reference computes them. librosa resamples with soxr; this resamples
    with SciPy's `resample_poly`. Music at 15360 Hz loads without a resample; the chroma differs by the octave
    decimation alone. Against librosa with the polyphase decimation: mel 1.0000000000001592, envelope
    0.9999999999999877, MFCC 0.9999999999999947, CQT 0.9999999999999410, chroma 1.0000000000000000, the tuning
    equal, and every onset and beat exact (40 and 19). Against librosa's own soxr path: chroma 0.9999923753114697,
    every other feature as above. Measured on the reference's own tracks, replacing soxr in the load as well costs
    the MFCCs 0.9988 to 0.9995 and up to 2 onsets and 2 beats.
  - **Measured on a test-dev release** (`Tools/wan-dancer-testdev/make_testdev.py`): the release's `config.json` at 48
    wide over 2 heads and 28 blocks (so every injector and block 9 exist), ffn 96, text 32; umT5 at 32 wide over the
    released vocabulary; the Wan 2.1 autoencoder at its released width; CLIP's vision tower at its released width
    over 3 blocks. `run_reference.py wan_dancer_pipeline` (the `wandancer` env, `dancer_oracles.pipeline`) runs the
    fork's `WanVideoPipeline.__call__` unchanged, one global and one local call at 3 steps, 17 frames, 256 × 448,
    on the sequence-parallel path in a world of one. Results, global then local:
    - prompt features 0.9999999999998900 and 0.9999999999998799;
    - keyframe CLIP 0.9999999999954577 and 0.9999999999953713, reference CLIP 0.9999999999954577 in both;
    - the keyframe condition 0.9999999999948832 and 0.9999999999954617;
    - music tokens 0.9999999999999976 and 0.9999999999999941;
    - the first step's two calls 0.9999999999998230 / 0.9999999999998480 and 0.9999999999997364 / 0.9999999999997790;
    - the final latents 0.9999999999996806 and 0.9999999999996153;
    - the decoded clip 0.9999999999687290 (frame by frame) and 0.9999999999652606 (in DiffSynth-Studio's tiles,
      clamped after blending).

    The local stage's planning is exact against the script's own functions (`run_reference.py wan_dancer_planning`):
    the keyframe masks of six track lengths and the music clips of three tracks. Both released transformers are held
    by shape (`testBothReleasedTransformersMatchTheLoader`, `IK_SHAPES_ROOT` `wan-dancer-14b-global` and `-local`):
    1424 tensors each, none missing, unconsumed, or mismatched. The generator gives the reference's latents through
    every residency (0.9999999999996153): a staged load and a streamed block compute exactly as held ones, and an
    8-bit pack of each block's feed-forward output (32-wide groups, the widest the test-dev's 48 width admits)
    holds 0.9999999999513278. No released-weight run: the release is 85 GB.
  - Runtime: staged residency (umT5, then each transformer with the autoencoder, the CLIP tower one copy both
    stages read). A streamed transformer reads the blocks the plan cannot hold from its release file on every pass,
    and `quantizationBits` 2 to 8 packs the blocks' linear layers as each loads. Precision is `.checkpoint` or
    `.float32`, and umT5 widens to float32 where the plan affords it. The music features reach the model at the
    bfloat16 the scripts hand them over at. No key-value cache, draft, or streamed partial output; the backend
    reports progress per request. The backend returns the video without the music.
- Wan-Dancer LoRA fine-tuning (`NFKMLXWanDancerGenerator.fineTune`, `NFKMLXWanDancerTraining.swift`) —
  DiffSynth-Studio's own Wan-Dancer recipe (1e8ae1d, `lora/Wan-Dancer-14B-local.sh` and `-global.sh`, its
  `wantodance` modules in `wan_video_dit.py` and `model_fn_wan_video`): `NFKMLXWanTrainingPreset.diffSynth` at rank 32
  and alpha 32 on `q,k,v,o,ffn.0,ffn.2`, which PEFT's name match takes to both attentions and the feed-forward of
  every block and the music injectors' four projections, each stage on its own transformer file. Wan-Video's fork
  publishes inference only. DiffSynth trains without sequence parallelism and in the eval mode `freeze_except`
  leaves the transformer in, so its music encoder reads the whole track in one span and drops nothing; the example
  (`trainingExample(video:prompt:clip:)`, the clip's latents with its keyframe condition, CLIP features, music, and
  frame rate) sets `musicEncoderSpans` to 1. Measured (`run_reference.py wan_dancer_lora_diffsynth`, the
  wantrain-diffsynth env, DiffSynth's own `WanModel` with its WanToDance modules over the test-dev weights under its
  LoRA, `FlowMatchSFTLoss` over its `model_fn_wan_video`): the local stage's prediction 0.9999999999997966, loss
  0.16400957 against 0.16400956, the lowest adapter gradient cosine 0.999999999999; the global stage's prediction
  0.9999999999998244, loss 1.8738124 against 1.8738124, the lowest gradient cosine 0.999999999998; 312 adapters each
  (28 blocks × 10, 8 injectors × 4).
  `testTheTransformerTrainsSavesAndReloadsThroughItsFactory` trains on an example the generator builds, saves in
  DiffSynth's layout, merges back into a fresh network (312 layers), and loads through `addAdapter(at:scale:global:)`.
- `NFKMLXVJEPA2` (`@objc`) — V-JEPA 2 (`VJEPA2Model` and `VJEPA2ForVideoClassification`, Meta, MIT), a
  self-supervised video ViT that maps a clip to a per-token feature sequence for retrieval or as a video
  encoder for a vision-language model; the classification releases add an attentive pooler and a linear
  classifier. A 3D tubelet patch embedding (a `Conv3d` with kernel and stride `tubelet x patch x patch` =
  2 x 16 x 16 over `[B, T, H, W, C]`, flattened temporal-major then row-major) feeds the pre-norm blocks
  and a final layer norm. There is no class token and no learned position table; position enters only
  through a 3D rotary embedding on the queries and keys. The head dimension splits into temporal, height,
  and width bands of `2 * floor(floor(head_dim / 3) / 2)` channels each (20 for ViT-L, the rest
  unrotated), and each band rotates by the token's frame, row, or column index. The rotation is the
  reference's exact arithmetic: it pairs adjacent channels for the rotate but tiles the cosine and sine
  tables as two concatenated halves, so channel `k` is scaled by frequency `k mod (band / 2)`. Reading it
  as a standard interleaved rotation diverges. The pretraining predictor is dropped at load. The module
  keys mirror the checkpoint's `encoder.*` (a classification release nests it under `vjepa2.`, which the
  loader strips), so the loader only transposes the 5-D convolution weight (`[out, in, kT, kH, kW]` to
  MLX's `[out, kT, kH, kW, in]`).
- **The attentive pooler** (`pooler`, classification releases): `num_pooler_layers` (3) pre-norm
  self-attention layers over the encoder's tokens, then one learned query token cross-attends the
  normalized tokens (the query is not normalized, and there is no output projection, as in the
  reference's `CrossAttention`, whose `proj` is commented out), then an MLP; the linear `classifier`
  reads the query. `NFKMLXVJEPA2Net.classLogits`; the backend returns every class ranked under
  `NFKOutputClassifications`, labeled from `id2label`.
- **Consumer surface.** The `@objc` directory factory `backend(directoryURL:)` reads the release's
  `config.json` for the geometry and `video_preprocessor_config.json` for the resize; the backend embeds
  a video (`NFKInputVideo`) or an image (`NFKInputImage`, treated as a still frame the way the reference
  duplicates a sub-tubelet clip) into a mean-pooled feature vector under `NFKOutputEmbedding`. The video
  processor resizes the shortest edge to the release's `shortest_edge` (292 for a 256 crop, 438 for 384),
  center-crops, rescales to `0...1`, and applies the ImageNet normalization. Not registered, since the
  model loads a whole release directory (like Florence-2 and TrOCR).
- **Customization: probe** (the reference's own recipe). Meta's `evals/video_classification_frozen`
  freezes the encoder (`no_grad`) and trains an `AttentiveClassifier`, which is exactly the
  classification releases' `pooler` plus `classifier`. `NFKMLXVJEPA2.network(directoryURL:labels:)`
  builds the network: a release without a pooler gets a fresh one initialized as the reference's
  `AttentivePooler` (normal std 0.02 for every linear weight and the query, zero biases, each residual
  branch's last projection divided by `√(2·(i + 1))`, the cross-attention layer's by the last layer's
  factor); a classifier whose class count differs starts fresh; everything else loads.
  `NFKMLXVJEPA2Trainable` is `.probe` (pooler and classifier) or `.classifier`. `NFKMLXVJEPA2Objective`
  is `torch.nn.CrossEntropyLoss()`; the reference sums it over each clip's segments and views, and one
  view per example is the single-term case. `fineTune` runs the reference's `torch.optim.AdamW` with
  weight decay on every probe parameter and its `WarmupCosineLRSchedule` (no warm-up, a cosine to zero;
  `NFKMLXLearningRateSchedule.warmupCosine`, which steps before each update as the reference does). The
  reference sweeps twenty heads over five rates (5e-3 to 1e-4) and four weight decays (0.01 to 0.8) and
  keeps the best on validation; the defaults are the first, 5e-3 and 0.01. Its bfloat16 autocast is
  `precision: .bfloat16`, off by default. Its gradient scaler is not reproduced under bfloat16.
  `NFKMLXVJEPA2Processor.clip(frames:configuration:)` is the data adapter, and
  `NFKMLXVJEPA2.save(_:toDirectoryURL:)` writes a directory (`model.safetensors`, `config.json` with
  `id2label`, `video_preprocessor_config.json`) that the `@objc` `backendWithDirectoryURL:` loads.
  Measured against the reference's own code (`run_reference.py vjepa2_probe`, `IK_PARITY_VJEPA2_PROBE`,
  the reference files under `<validation root>/vjepa2-ref`): the loss on identical logits
  2.8547907 against 2.8547912; the schedule equal at every step, with and without a warm-up; and every
  fresh tensor's standard deviation within 2% of the reference's `AttentiveClassifier(depth=4)` at width
  512 (the query token within its 512-draw sampling error). `NFKMLXVJEPA2TrainingTests` holds the tiny
  fine-tune (the loss falls, the probe moves, the encoder does not) and the save and factory reload.
- **Parity.** Oracle: `run_reference.py vjepa2` (the `llm` env, transformers-native `VJEPA2Model`, or
  `VJEPA2ForVideoClassification` with its pooled output and logits recorded). Every release (the eight
  on the Hugging Face API, 2026-09-23) is at reference parity against its own record, and
  `testEveryOtherReleaseIsAtParity` also builds each through `backend(directoryURL:)`: the embedding has
  the hidden width, and a classifier ranks every class, most confident first, under its `id2label` name.

  | Release | Patch embedding | Middle block | Final features | Pooler | Logits |
  |---|---|---|---|---|---|
  | `vjepa2-vitl-fpc64-256` | 0.9999998 | 1.0000001 | 1.0000004 | | |
  | `vjepa2-vith-fpc64-256` | 0.99999994 | 0.99999994 | 1.0000002 | | |
  | `vjepa2-vitg-fpc64-256` | 1.0 | 1.0000001 | 0.99999917 | | |
  | `vjepa2-vitg-fpc64-384` | 0.99999964 | 0.9999997 | 1.0000005 | | |
  | `vjepa2-vitl-fpc16-256-ssv2` | 0.9999998 | 1.0000001 | 1.0000004 | 1.0000001 | 1.0000001 |
  | `vjepa2-vitl-fpc32-256-diving48` | 0.9999998 | 1.0000001 | 1.0000004 | 0.99999976 | 1.0000001 |
  | `vjepa2-vitg-fpc64-384-ssv2` | 0.99999964 | 0.9999997 | 1.0000005 | 1.0 | 0.9999999 |
  | `vjepa2-vitg-fpc32-384-diving48` | 0.99999964 | 0.9999997 | 1.0000005 | 1.0000001 | 0.99999994 |

  The 384 releases resize the shortest edge to 438, read from their `video_preprocessor_config.json`.
  A classifier's oracle peaks at 5.8 GB (ViT-g at 384) and runs in under 20 seconds on the CPU.
- `NFKMLXCosmosTokenizer` (`@objc`) — the Cosmos Tokenizer (NVIDIA; weights under the NVIDIA Open Model
  License, code Apache-2.0), all ten `nvidia/Cosmos-0.1-Tokenizer-*` releases, one per
  `NFKMLXCosmosTokenizerVariant`: continuous image CI8x8 / CI16x16, discrete image DI8x8 / DI16x16,
  continuous causal video CV4x8x8 / CV8x8x8 / CV8x16x16, discrete causal video DV4x8x8 / DV8x8x8 / DV8x16x16.
  Every variant patches its input with a two-level Haar wavelet (4× per side; a clip's first frame is
  repeated four times so it fills a temporal patch, and the inverse drops the copies), runs a 128-channel
  encoder, projects to a 16-channel latent or a six-channel FSQ input, and mirrors the path back. The
  image networks are Stable-Diffusion-style (32-group norm, one self-attention in the middle). The video
  networks factorize every convolution into a spatial `1×3×3` and a temporal `3×1×1` kernel padded toward
  the past by repeating the first frame, normalize each frame on its own (a one-group norm), and attend
  within each frame then causally over each position's frames, so frame `t` never reads a later one. A
  resampling level adds a strided convolution to an average pool of the same window, then a pointwise
  convolution; time doubles `t → 2t − 1`. The discrete variants bound each scalar with `tanh`, round it to
  one of 8, 8, 8, 5, 5, 5 levels (a 64,000-entry implicit codebook, no weights), and number a token
  mixed-radix. Five things the releases decide and no document states:
  - The releases are built by `nvidia-cosmos/cosmos-predict1`'s `tokenizer` package. The older
    `NVIDIA/Cosmos-Tokenizer` repository creates every resampling convolution, where the releases omit
    the ones a level does not use (a spatial-only level has no temporal convolution). The
    releases' `config.json` files are empty, so `NFKMLXCosmosTokenizerConfiguration.variant(_:)` fixes the
    geometry from the name and the tensors: patch 4 everywhere, a 256-wide encoder output for the discrete
    image tokenizers, and two levels on both halves of CV4x8x8.
  - The releases store the Haar taps as a persistent bfloat16 buffer (0.70703125 for 1/√2). A float32
    oracle that loaded it carried a 4e-4 error from the first convolution on, identically on the CPU
    and the GPU. The port and the oracle compute the taps; neither loads the stored copy.
  - For DI8x8, DV4x8x8, DV8x8x8, and DV8x16x16 the single `autoencoder.jit` holds different weights from
    the `encoder.jit` / `decoder.jit` pair (every tensor, up to 16% of one decoder tensor's norm); the
    other six are bit-identical. The factories take `autoencoder.jit`, so those four carry a second
    record.
  - A discrete tokenizer's tokens flip wherever a scalar sits within the arithmetic's error of a
    rounding boundary, and the releases put many there (DI8x8: 34 of 1,536 within 1e-3). NVIDIA's own
    graph at its stored bfloat16 agrees with the float32 tokens on 84–91% of positions. Token parity
    is therefore stated as the latent before rounding plus every token clear of a boundary, and at
    float32 every token of every record matches.
  - The shipped TorchScript graphs carry bfloat16 constants and run only at that precision, and CPU
    torch has no bfloat16 `avg_pool3d`; the oracle registers one for the process to run them.
  The loader reads the release's TorchScript directly through `NFKMLXWeights.loadCheckpoint` (no
  converter), converts bfloat16 to float32, transposes 4-D and 5-D convolutions, skips the stored
  derived constants, and verifies every built shape against the checkpoint. Reference parity against
  NVIDIA's own modules on the released weights, float32, seam by seam, on every variant (a 128×128
  photograph, and a 9-frame pan across it for the video variants): wavelet patch 0.999999999999999
  (image) / 0.999999999999838 (video); encoder output 0.9999999999961 or closer; continuous latents
  0.99999999999951 or closer and reconstructions 0.99999999999911 or closer; discrete input to FSQ
  0.9999999999952 or closer, every token equal, reference tokens decoded 0.99999999999818 or closer; the
  four differing `autoencoder.jit` files 0.9999999999960 or closer at every seam with every token equal.
  NVIDIA's shipped bfloat16 graphs agree with the float32 rebuild at their own floor (continuous latents
  0.99987–0.99998, every reconstruction 0.99997 or closer, the four combined discrete graphs
  0.998–0.9996 on their own flipped tokens). `@objc` factories: `backendWithVariant:weightsURL:error:`, the
  download `backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:error:` (the release's
  `autoencoder.jit`) and its `completionHandler:` peer, `tokenizerWithVariant:weightsURL:error:` and
  `tokenizerWithVariant:encoderWeightsURL:decoderWeightsURL:error:`, and `register()` (all ten under
  `cosmos-tokenizer-<variant>`). The backend reconstructs an image under `NFKInputImage` and, for a video
  variant, a clip under `NFKInputVideo`, padding and windowing as the reference does (space to a multiple
  of 16, frames to `1 + 8k` by repeating the ends, 17-frame windows). The latent or token grid reaches
  Objective-C as `NFKMLXCosmosTokenizerCode` (`codeForImage:error:`, `codeForFrames:error:`,
  `framesForCode:error:`). Customization ships in full (`NFKMLXCosmosTokenizerTraining.swift`): the
  reference's post-training objective, `NFKMLXCosmosTokenizerObjective` (L1 plus 0.1 × the layer-weighted
  L1 of VGG-16 features at `relu1_2` … `relu5_3`, `NFKMLXVGG16Features` over `timm/vgg16.tv_in1k`; the
  optional Gram term the reference leaves off for post-training ships too), measured against NVIDIA's own
  `ColorLoss` and `PerceptualLoss` on identical tensors (image color 0.15285844 vs 0.15285844, perceptual
  1.052784 vs 1.0527844, Gram 1.3198887 vs 1.3198897; a clip within 1e-5); `NFKMLXCosmosTokenizerTrainable`
  `.everything` (the reference's) or `.decoder` (the encoder frozen, so every latent and token stays the
  release's, which a world model trained on the released latents needs); `fineTune` over `NFKMLXTrainer`
  with the reference's AdamW (1e-4, betas 0.5 / 0.999, weight decay 0.01, bias-corrected), its
  5,000-step linear warm-up (`WarmupLambdaLR`), and its global gradient-norm clip at 1
  (`GradClipCallback` in the `basic` callbacks). The reference's EMA of the trainable weights
  (`EMAModelTracker`, beta 0.9999, constant) is `weightAverageDecay`, off by default, since a run far
  shorter than the reference's ends near the weights it started from. The flow and consistency terms, which the reference disables for
  post-training, are not ported. Oracles: `run_reference.py cosmos_tokenizer` (a release directory named
  for its variant, or its `autoencoder.jit`) and `cosmos_tokenizer_loss` (the VGG-16 file), under the
  `llm` env with the `cosmos_predict1` sources pinned in the manifest.
