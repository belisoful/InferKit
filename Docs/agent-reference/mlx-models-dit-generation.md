<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX models: DiT image and video generation

LTX-Video, Z-Image, SANA, Wan, SD3, FLUX, their samplers, text encoders, and ControlNets.

- `NFKMLXLTXVideoVAE` (`@objc`) — the LTX-Video VAE (`AutoencoderKLLTXVideo`), the toolkit's first piece of
  video generation (the DiT and a T5 text encoder are the remaining stages) and its first 3D model. A
  **causal 3D autoencoder**: a video compresses to a spatiotemporal latent and back. The encoder is causal
  in time — each temporal convolution left-pads by repeating the first frame `kernel-1` times, so a frame
  depends only on the past (`NFKLTXCausalConv3d`, a wrapper over MLX's `Conv3d` keeping the reference's
  `.conv` key); the decoder is non-causal (symmetric pad). Downsampling is a stride-2 causal conv;
  upsampling is a conv that widens the channels ×8 then a 3D pixel shuffle (`NFKLTXUpsampler`:
  interleave the extra channels into the frame/height/width axes, drop the first frame). Resnet blocks
  normalize with a parameter-free channel RMS norm (so norm1/norm2/norm_out carry no weights; only the
  shortcut's `norm3` LayerNorm does). Encoder: **patchify** (fold each `patchT×patch×patch` block into the
  channel axis, the reference's `channel, temporal, width, height` order) → conv_in → down blocks → mid →
  RMSNorm+SiLU → conv_out (latent+1 channels; the extra channel is the posterior's shared log-variance,
  dropped for the deterministic latent). Decoder mirrors it and unpatchifies. The whole thing works in
  NDHWC (`[B, T, H, W, C]`) where the reference is NCTHW, so every patchify/pixel-shuffle reshape is the
  reference's permute re-derived for channels-last; the 5-D Conv3d weights transpose `[out,in,kT,kH,kW]` →
  `[out,kT,kH,kW,in]` at load, and the causal-conv `.conv` naming makes the keys match with no remap.
  `NFKMLXLTXVideoVAE.encode`/`decode` (Swift, over `MLXArray`) and `vae(configuration:weightsURL:)` are the
  surface; the released diffusers safetensors loads directly (converter `Tools/ltx-vae-to-safetensors` is a
  passthrough). Reference parity against diffusers' `AutoencoderKLLTXVideo` on the first numeric run
  (`run_reference.py ltx_vae`, the `ltx` oracle env — diffusers ≥ 0.32, its own venv): every encoder seam
  exact (conv_in / first down block / mid ≥ 0.99999999999), latent cosine 0.99999999999, decode cosine
  0.99999999996. A weight-free test also asserts the encoder's temporal causality (two clips sharing their
  first frames but not the last produce the same first latent frame).
- `NFKMLXLTXTransformer` (`@objc`) — the LTX-Video DiT (`LTXVideoTransformer3DModel`), the denoising
  transformer of the video-generation pipeline (the stage after the VAE). A 2B sequence transformer over
  the VAE's flattened latent tokens: `proj_in` (128→2048), 28 `NFKLTXBlock`s, `norm_out`+adaLN, `proj_out`
  (2048→128). Each block is adaptive-layer-norm self-attention with 3D rotary + cross-attention to the
  text embedding + a gelu-approximate feed-forward, the six modulation parameters coming from the
  timestep through the block's own `scale_shift_table` (PixArt-α style). Attention (`NFKLTXAttention`)
  applies an across-heads RMS norm to the query and key (over the full 2048 width, before the head split),
  the 3D rotary to both (self-attention only), then SDPA; cross-attention reads the projected text and does
  not rotate. The **3D rotary** (`NFKLTXRotary`) is computed over the (frame, height, width) latent grid: a
  per-axis log-spaced frequency ramp times the scaled coordinate, cos/sin repeat-interleaved by 2, the
  leading `dim % 6` channels left unrotated. Timestep conditioning is `AdaLayerNormSingle` (a 256-wide
  sinusoidal embedding through an MLP, then SiLU + linear to `6·inner`); the text is a PixArt caption
  projection (4096→2048). The feed-forward's `net` is a `[Module]` array (`net.0.proj`, an activation
  marker, `net.2`) so the diffusers Sequential keys match. The module keys mirror the reference exactly
  (all 715 tensors), so the sharded release loads through `NFKMLXReleaseWeights.arrays` with a pass-through
  remap and no transpose (every weight ≤ 2-D). For parity the text embedding is supplied directly (the
  caption projection is inside the DiT), so the transformer is validated in isolation like the SD UNet — no
  T5 needed. Reference parity against diffusers on the first numeric run (`run_reference.py
  ltx_transformer`, the `ltx` oracle env, recorded random latent/text/timestep): every seam exact (rope
  cos/sin ≥ 0.9999999, proj_in and first block ≥ 0.99999999999) and the full 28-layer velocity cosine
  0.99999999999. The 2B weights are ~7.7 GB sharded.
- `NFKMLXT5Encoder` (`@objc`) — the T5 v1.1 text encoder (`T5EncoderModel`), the text conditioning for the
  LTX pipeline and a reusable building block (the same family conditions Wan / PixArt / SD3 / Flux). A
  stack of pre-normalized blocks with two T5-isms: the attention is unscaled and adds a bucketed
  **relative-position bias** (`NFKT5Attention.computeBias`, the Mesh-TensorFlow bidirectional bucketing,
  computed once from block 0's table and shared across all layers, passed as the SDPA additive mask), and
  the norm is **T5LayerNorm** — an RMS norm with a weight and no mean subtraction. The feed-forward is
  gated (`wo(gelu(wi_0(x)) · wi_1(x))`, tanh-approx GELU). Module keys mirror the reference
  (`shared`, `encoder.block.N.layer.0.SelfAttention.{q,k,v,o}`, `.layer.0.layer_norm`,
  `.layer.1.DenseReluDense.{wi_0,wi_1,wo}`, `.layer.1.layer_norm`, `encoder.final_layer_norm`), so the
  sharded release loads through `NFKMLXReleaseWeights.arrays` with no remap and no transpose (all 2-D).
  `NFKMLXT5Configuration.xxl` is T5-XXL (d_model 4096, 24 layers, 64 heads, d_ff 10240). Reference
  parity against transformers on the first numeric run (`run_reference.py ltx_t5`, the `ltx` oracle env):
  embedding seam exact, first block 0.9999999999996, full text embedding cosine 0.99999999998. ~19 GB fp32
  sharded — the memory crux of the LTX pipeline, which stages the encoders sequentially.
- `NFKMLXFlowMatchScheduler` — the rectified-flow sampler (`FlowMatchEulerDiscreteScheduler`), the sampler
  LTX / Flux / SD3 / Wan / Z-Image use, a value type with no parameters. The schedule is a sigma ramp from
  1 to 0 with **dynamic resolution-dependent shifting** (a per-sequence-length `mu = base_shift + slope·
  (seq − base_seq)` warps the ramp: `σ ← exp(mu)/(exp(mu) + 1/σ − 1)`) and a **terminal stretch** so the
  last non-zero sigma lands on `shiftTerminal` (0.1). A step is one Euler update `x + (σ_next − σ)·v`.
  **Verified against diffusers** (schedule exact: sigmas to 1e-4, terminal sigma 0.1) under `swift test`
  (pure Float math, no MLX eval).
- `NFKMLXLTXPipeline` — the LTX-Video text-to-video pipeline glue, chaining the three parity-verified
  stages: T5 encode → the DiT denoised over the flow schedule with classifier-free guidance → the VAE
  decode. With the DiT's patch size of 1 the latent packing to/from the token sequence is a single reshape
  of the VAE's NDHWC latent. The stages are held together for a run but a caller manages residency (the
  19 GB T5, the 7.7 GB DiT, and the VAE do not all fit resident on 32 GB, so they load and free in turn,
  the Music 3 pattern). Validated by a weight-free glue test on matching tiny configurations (the packing,
  the guided loop, unpacking, and decode produce a correct-shaped clip) plus the four stages' own parity —
  a sampled clip cannot be compared bitwise, as with Music 3. The VAE + DiT + T5 + flow are the complete
  LTX text-to-video path.
- `NFKMLXZImageTransformerNet` — the Z-Image S3-DiT (`ZImageTransformer2DModel`, Alibaba Tongyi), the
  denoising transformer of a 6B text-to-image model and the third DiT family beside the SD UNet and LTX.
  **Single-stream**: the image latent tokens and the caption tokens are concatenated and every layer's
  self-attention runs over the join, rather than a separate cross-attention branch. Three stages: a
  `noise_refiner` (2 modulated blocks over the image tokens alone), a `context_refiner` (2 UN-modulated
  blocks over the caption tokens alone), then 30 unified `layers` over the concatenation. The block is a
  Sandwich: `attention_norm1` before and `attention_norm2` after the attention, the same for the FFN,
  with a 4-chunk adaptive modulation (scale/gate for each of attention and FFN, gates `tanh`'d, scales
  `1 +`). SwiGLU FFN (hidden `dim/3·8` = 10240), per-head RMS q/k norm, and a **3-axis complex rotary**
  (`view_as_complex`, axes `[32,48,48]` summing to headDim 128, θ 256) over the (frame, height, width)
  grid. The caption is a Qwen3-4B embedding (`cap_feat_dim` 2560) projected in through
  `RMSNorm → Linear`; base text-to-image feeds only the latent + caption (the SigLIP visual-semantic
  tokens are the edit variant's input). Sequence lengths pad to a multiple of 32 with learned
  `x_pad_token`/`cap_pad_token` at (0,0,0) positions. Module keys mirror the reference exactly (73
  at the tiny config, `all_x_embedder.2-1`/`all_final_layer.2-1` ModuleDict keys included), so a release
  loads with no remap and no transpose (every weight ≤ 2-D). For parity the caption features are
  supplied directly (no Qwen3), so the DiT is validated in isolation, as the LTX DiT is. Reference
  parity against diffusers on the first numeric run (`run_reference.py z_image`, tiny random config,
  the `ltx` oracle env — diffusers 0.36 carries ZImageTransformer2DModel): the t_embedder seam exact and
  the full velocity cosine **0.9999999999999653**, with the pad-token path exercised (sequence lengths
  not a multiple of 32).
  The Flux VAE Z-Image encodes into is the shared `NFKMLXSDAutoencoder`: its `vae/config.json` is a
  diffusers `AutoencoderKL` (16 latent channels, `[128,256,512,512]`, `mid_block_add_attention`) that
  differs from Stable Diffusion's only in dropping the quant convolutions (`use_quant_conv: false`,
  `use_post_quant_conv: false`) and in a centering `shift_factor` 0.1159 / `scaling_factor` 0.3611. So
  `NFKMLXSDVAEConfiguration` gained `useQuantConv` (the two 1×1 convs are now optional `Conv2d?` — when
  absent the encoder's `conv_out` is the moments directly and decode reads the latent directly) and
  `shiftFactor`, and `.flux` is the preset. Reference parity against diffusers' AutoencoderKL at a
  tiny `use_quant_conv=False` config (`run_reference.py flux_vae`, `ltx` env): encoded-mean cosine
  0.9999998591898961, decode cosine 0.9999999948059418.
  `NFKMLXZImagePipeline` chains it end to end (S3-DiT denoised over the flow schedule with
  classifier-free guidance → Flux VAE decode). The caller supplies the caption embedding (the Qwen3-4B
  hidden states), as the SD pipeline takes a text context — Z-Image's text step is the shipped Qwen3
  decoder (already at reference parity), read for its penultimate hidden state (`hidden_states[-2]`, which
  is `NFKMLXLanguageNet.layerStates(tokens)[count − 2]` — the existing per-layer seam) after the Qwen chat
  template, run and freed separately. So no new port is needed for the text step; Qwen3 + the seam cover
  it. `NFKMLXFlowMatchConfiguration.zImage`
  is its schedule (a smaller resolution shift, no terminal stretch, `sigma_min` 0); the DiT timestep is
  `1 − σ` and the flow velocity is negated before the Euler step, both the reference's conventions.
  Validated by a weight-free tiny-config glue test (the guided loop, the timestep/latent conventions,
  the centered-latent decode) plus the DiT/VAE/flow parities — a sampled image is not bitwise-comparable,
  as with LTX and Music 3. The Qwen3-4B DiT + Flux VAE + flow are the complete Z-Image text-to-image path.
  `generate(image:strength:…)` is the image-to-image (edit) path — the Flux VAE encodes the source, the
  latent is noised to `strength` through the scheduler's flow `addNoise`, and the denoise runs from the
  matching step, diffusers' own `pipeline_z_image_img2img`. The paper's SigLIP-conditioned editing is a
  separate original-repo model not present in the diffusers Z-Image (its transformer takes only the image
  latent and the caption; neither diffusers pipeline references SigLIP), so there is no reference to port
  it against — the img2img path here is diffusers' actual edit capability.
- `NFKMLXSANATransformerNet` — the SANA linear-attention DiT (`SanaTransformer2DModel`, NVIDIA), the
  fourth DiT family. Two things set it apart: the self-attention is **linear** — ReLU feature maps,
  `O = ((V·1̂) @ ReLU(K)) @ ReLU(Q)` normalized by the ones row, O(N) rather than O(N²), which is what
  lets it run at high resolution — and the feed-forward is a **GLUMBConv** (an inverted-bottleneck gated
  Depthwise convolution over the 2-D token grid: `conv_inverted` 1×1 to `2·hidden`, SiLU, `conv_depth`
  3×3 depthwise, gate-split `x · silu(gate)`, `conv_point` 1×1, no bias), not a pointwise MLP. Each
  block also cross-attends to the Gemma text embedding through ordinary softmax attention (heads 20,
  head_dim 112). Conditioning is PixArt-α `AdaLayerNormSingle`: one shared timestep embedding plus a
  per-block learned `scale_shift_table` [6, inner], expanded to the six modulation parameters; the output
  norm reads the pre-linear embedded timestep against a top-level `scale_shift_table` [2, inner]. Patch
  embed is a stride-`patch` `Conv2d` (no positional embedding). The caption enters through
  `PixArtAlphaTextProjection` (Linear → GELU-tanh → Linear) and an RMS `caption_norm`. Module keys mirror
  the reference (54 at tiny); the convolution weights load transposed to MLX's NHWC. Reference parity
  against diffusers on the first numeric run (`run_reference.py sana`, tiny random config, `ltx` env):
  embedded-timestep seam exact, full velocity cosine **0.9999999999999959**.
  `NFKMLXDCAutoencoderNet` is SANA's Deep-Compression Autoencoder (`AutoencoderDC`), which compresses
  an image 32× spatially (against a Stable Diffusion VAE's 8×) — that is what keeps SANA's latent-token
  count low enough to run at high resolution. It is deterministic (`encode` returns one latent, not a
  Gaussian), built from two block families: `ResBlock`s at the shallow stages and `EfficientViTBlock`s
  (a multiscale ReLU linear attention — grouped multiscale-kernel q/k/v projections, then the same
  `O = ((V·1̂) @ ReLU(K)) @ ReLU(Q)` linear attention the DiT uses — plus a GLUMBConv with an RMS norm and
  a residual) at the deep stages. Down/up sampling is pixel-unshuffle / pixel-shuffle with a
  channel-averaging (`DCDownBlock`) or channel-repeating (`DCUpBlock`) shortcut; the released SANA uses a
  stride-2 Conv downsample and a nearest-interpolate upsample. The `pixelUnshuffle` / `NFKMLXPixelShuffle`
  helpers are shared with Real-ESRGAN / BiSeNet. The reference's `<blocks>.<i>.<j>` `nn.Sequential`
  indices map onto the module's `<i>.block.<j>` (a stage is an `NFKDCStage` holding a `[Module]`); the
  convolution weights load transposed to NHWC. Reference parity against diffusers' AutoencoderDC on
  the first numeric run (`run_reference.py dc_ae`, tiny random config, `ltx` env): latent cosine
  0.9999999999999344, decode cosine 0.9999999999999848. Also validated on the actual released SANA
  weights end to end (`NFKMLXDCAutoencoderNet.loadWeights` reads the diffusers checkpoint, remapping the
  `<blocks>.<i>.<j>` Sequential indices to `<i>.block.<j>` and transposing the convs): on the real 1.2 GB
  `Sana_600M` VAE over a 256×256 image at the full 32× compression, latent cosine 0.9999999999918, decode
  cosine 0.9999999998 (`run_reference.py dc_ae_real`, `IK_VAL_DCAE`) — the released config, the real
  checkpoint, and the loader confirmed, not just the tiny-config architecture.
  `NFKMLXSANAPipeline` chains it end to end (linear-attention DiT denoised over the flow schedule with
  classifier-free guidance → DC-AE decode). The caller supplies the caption embedding (the Gemma text
  encoder's last hidden state), as the SD pipeline takes a text context. The sampler is SANA's released
  `DPMSolverMultistepScheduler` (`NFKMLXDPMSolverScheduler`, at reference parity — see the scheduler
  entry below), not a stand-in. Validated by a weight-free tiny-config glue test plus the DiT/VAE
  parities. SANA's text encoder is `NFKMLXGemma2Net` (Gemma-2, ported here — see the Gemma-2 entry):
  the caller runs it for the caption features. The SANA text-to-image path is complete.
- `NFKMLXWanTransformerNet` — the Wan text-to-video DiT (`WanTransformer3DModel`, Alibaba Wan), the fifth
  DiT family. A 3-D sequence transformer over a `Conv3d`-patchified video latent (patch `(1,2,2)`), with
  the same 3-axis interleaved rotary as Z-Image (`NFKZImageRope` reused, θ 10000, axes `t = headDim −
  2·h`, `h = w = 2·(headDim/6)`) over the (frame, height, width) grid. Each block runs self-attention
  (with rotary), cross-attention to the text embedding, and a gelu-approximate feed-forward
  (`ffn.net.0.proj`/`net.2`), under PixArt-α adaptive norms — a shared `time_proj` [6·inner] plus a
  per-block `scale_shift_table` [1,6,inner]; the cross-attention norm (`norm2`) is an affine LayerNorm
  applied UN-modulated, where `norm1`/`norm3` are non-affine and modulated. The q/k norm is
  `rms_norm_across_heads` — an RMS norm over the whole inner width before the head split, where
  Z-Image norms per head. The condition embedder is `time_embedder` (TimestepEmbedding) → `time_proj`,
  plus `text_embedder` (`PixArtAlphaTextProjection`). This is the text-to-video path (no
  image-conditioning branch, no `added_kv`). Module keys mirror the reference (69 at tiny); the 5-D
  `Conv3d` weight loads transposed to NDHWC. Reference parity against diffusers on the first numeric
  run (`run_reference.py wan`, tiny random config, `ltx` env): full velocity cosine
  **0.9999999999999767**.
  `NFKMLXWanVideoVAENet` is the Wan 3D causal VAE (`AutoencoderKLWan`, the Wan 2.2 residual path),
  the last stage of the Wan pipeline and the hardest port of this batch. It compresses a video 4× in
  time and 16× in space. Unlike the LTX VAE (a clean full-clip causal forward), it runs a stateful
  streaming loop: the encoder consumes frames in chunks (1, then 4 at a time) and the decoder emits one
  latent frame at a time, threading a per-convolution feature cache (`feat_cache`) that supplies each
  causal convolution's temporal context across chunk boundaries — the temporal up/downsampling happens
  Only on that cache path, so a one-shot forward would skip it. `NFKWanCache` holds the per-conv slots
  (persisting across chunks, the index reset per chunk) with the reference's `Rep` / `frames` / `empty`
  states; `wanCausal` runs the caching dance (borrow the previous chunk's last frame when a chunk has
  fewer than two). `NFKWanCausalConv3d` holds its `weight`/`bias` directly (the reference is an
  `nn.Conv3d` subclass) via the functional `conv3d`, zero-padding the time axis on the left only. The
  residual down/up blocks carry the cacheless `AvgDown3D` / `DupUp3D` reshape shortcuts; the temporal
  resample `time_conv` doubles the frame count by splitting its doubled channel and interleaving it as a
  new frame sub-axis. Reference parity against diffusers' AutoencoderKLWan at a tiny residual config
  on a 5-frame clip (`run_reference.py wan_vae`, `ltx` env): encoder moments cosine 0.9999999999999997,
  decode cosine 0.999999999999892. The one load-bearing trap was the patchify channel order: the
  reference packs the `patch²` spatial block into the channel as `(C, pw, ph)`, not `(C, ph, pw)` — the
  swapped order left the encoder at 0.998 and the decode at 0.26 (the per-stage seams were exact through
  the last up-block, which pinned the fault to the final unpatchify and, symmetrically, the conv_in
  patchify). The RMS `gamma` loads flattened from its `[C,1,1,1]` layout; the 5-D/4-D conv weights
  transpose to NDHWC/NHWC. The non-residual Wan 2.1 path is implemented too (`isResidual` config,
  `.wan21` / `.tiny21`): Wan 2.1 replaces the residual down/up blocks (AvgDown3D / DupUp3D shortcuts) with
  a flat down-block list and a halving upsampler (`NFKWanUpBlock`, whose `WanResample` defaults
  `upsample_out_dim` to `dim/2`, so an inner decoder stage's input is halved), drops the patchify, and
  uses 16 latent channels. The encoder/decoder branch on `isResidual` and hold the blocks as `[Module]`
  with a type-dispatched forward. Reference parity against diffusers' AutoencoderKLWan at a tiny
  non-residual config (`run_reference.py wan_vae_21`): latent cosine 0.9999999999999978, decode
  0.9999999999999261, with the 2.2 residual path still at parity.
  `NFKMLXWanPipeline` chains it end to end (DiT denoised over the flow schedule with classifier-free
  guidance → the 3D VAE decode, over the `[C,F,H,W]`↔`[1,F,H,W,C]` bridge and the release's per-channel
  latent mean/std). The caller supplies the umT5 text embedding (a T5-family encoder). The sampler is
  Wan's released `UniPCMultistepScheduler` (`NFKMLXUniPCScheduler`, at reference parity — see the
  scheduler entry below), not a stand-in. Validated by a weight-free tiny-config glue test plus the
  DiT/VAE parities. Wan's text encoder is umT5, now verified: `NFKMLXT5Encoder` gained a
  `perLayerBias` configuration (`.umt5XXL` / `.tinyUMT5`) — umT5 (`UMT5EncoderModel`) differs from plain
  T5 only in giving every layer its own relative-position bias (plain T5 shares block 0's across the
  stack); everything else (T5LayerNorm, the gated FFN, the unscaled attention) is the same code.
  Reference parity against transformers' UMT5EncoderModel at a tiny configuration (`run_reference.py
  umt5`, the `llm` env): text embedding cosine 0.9999999999999984, with the plain-T5 shared-bias path
  still at parity. The Wan text-to-video path is complete.
- `NFKMLXDPMSolverScheduler` / `NFKMLXUniPCScheduler` — the released multistep samplers SANA and Wan use,
  ported in their flow-prediction configurations. Both are value types with no weights, so both are
  verified exactly against diffusers with no downloads (`run_reference.py dpm_solver` / `unipc`, `ltx`
  env): a fixed velocity sequence is run through the reference and this port, and the whole sample
  trajectory is compared step by step (worst |difference| 1.7e-6 and 1.4e-6), with the sigma schedule
  exact. **`NFKMLXDPMSolverScheduler`** is DPM-Solver++ (`algorithm_type: dpmsolver++`, `solver_order: 2`,
  `solver_type: midpoint`, `final_sigmas_type: zero`): each step converts the flow velocity to a data
  prediction `x0` and takes a first- or second-order multistep update, the coefficients computed as
  `Float` scalars (so the `log 0` at the terminal zero sigma resolves to a clean `x0`) and applied to the
  `MLXArray`. **`NFKMLXUniPCScheduler`** is UniPC (`solver_order: 2`, `solver_type: bh2`, `predict_x0`):
  a predictor-corrector — from step 1 on it corrects the previous sample before predicting the next — with
  the order-2 corrector's 2×2 `B(h)` linear system solved in closed form. Both take their flow sigma ramp
  from the release's `flow_shift` (SANA 3.0, Wan 5.0) and truncate the flow timesteps to integers as
  diffusers does. The `NFKMLXFlowMatchScheduler.sana`/`.wan` presets remain for a caller who wants the
  plain rectified-flow sampler, but the pipelines now run the released multistep samplers.
- `NFKMLXSD3TransformerNet` — the Stable Diffusion 3 MMDiT (`SD3Transformer2DModel`, Stability AI), the
  sixth DiT family and the flagship of the SD3.x line. Dual-STREAM: the image latent tokens and the text
  tokens each carry their own query/key/value projections, feed-forward, and adaptive-norm modulation
  (`norm1`/`norm1_context`, `ff`/`ff_context`), while attention runs jointly over the concatenation
  (`JointAttnProcessor2_0`: the image adds `add_*_proj` text keys and values, concatenates `[image,
  text]` on the sequence, and splits back). This is a different design from Z-Image's single-stream,
  which shares weights per layer. The last block runs `context_pre_only` — the text stream contributes
  keys and values but drops its own output and feed-forward, since nothing downstream reads the text.
  Conditioning is `time_text_embed` (a sinusoidal timestep MLP plus the pooled-text projection, summed);
  the text sequence enters through `context_embedder` (4096 → inner). The patch embed is a stride-2
  convolution plus a sincos positional table precomputed at `pos_embed_max_size²` and CENTER-CROPPED to
  the latent grid (the table is a persistent buffer, loaded from the checkpoint and cropped). The final
  `AdaLayerNormContinuous` + `proj_out` unpatchify via `nhwpqc->nchpwq`. SD3.5 adds two things over
  SD3.0: RMS query/key normalization (`qk_norm`), and — on SD3.5-**medium** (MMDiT-X), not the large —
  Dual attention (`SD35AdaLayerNormZeroX` giving nine modulation chunks, a second image-only `attn2`
  gated in beside the joint one). Reference parity against diffusers at a tiny random configuration
  that exercises the dual attention, the RMS q/k norm, the `context_pre_only` last block, and the
  cropped positional table (patch seam 0.9999999999999942, velocity cosine 0.9999999999999865). Two
  facts were load-bearing: `SD35AdaLayerNormZeroX` appends `(shift_msa2, scale_msa2, gate_msa2)` after
  the six (indices 6/7/8, not 8/6/7 — a moderate error localized to the dual layer); and the timestep /
  pooled-text embedders name their linears `linear_1` / `linear_2` (real submodules), not a
  `nn.Sequential`. Module keys mirror the release exactly, so `loadWeights` transposes only the 4-D
  patch-embed convolution to NHWC. `configuration(fromHuggingFace:)` reads `transformer/config.json`;
  presets `.sd3Medium` (2B, 24 layers, no qk-norm), `.sd35Medium` (2.5B, 24 layers, RMS qk-norm, dual
  attention 0…12, `pos_embed_max_size` 384), `.sd35Large` (8B, 38 layers × 38 heads, RMS qk-norm, no
  dual attention). Held to the released headers by shape: SD3.5-large 1227 tensors, SD3.5-medium 909
  (the dual-attention path), each 0 missing, 0 mismatched, 0 unaccounted. Oracle `run_sd3`, `IK_PARITY_SD3`.
- `NFKMLXFluxTransformerNet` — the FLUX.1 transformer (`FluxTransformer2DModel`, Black Forest Labs), the
  seventh DiT family. Two block kinds. The double-stream blocks (`transformer_blocks`) are MMDiT
  joint-attention blocks like SD3's, but concatenate `[text, image]` (text first, the opposite of SD3)
  and carry RMS q/k norm and an axial rotary. The single-stream blocks (`single_transformer_blocks`)
  concatenate the two streams and run a parallel attention-and-MLP over the join under one adaptive-norm
  gate (`AdaLayerNormZeroSingle`, three chunks): `proj_out([attention ‖ act(proj_mlp)])`, gated into the
  residual, then split back. Position is a 3-axis rotary (`FluxPosEmbed`, `get_1d_rotary_pos_embed` with
  `repeat_interleave_real`, adjacent-pair rotation) over the token ids — text ids all zero, image ids
  the `(0, row, col)` grid — whose axes must sum to the head dimension. Conditioning is
  `time_text_embed`: the timestep, an optional guidance embedding (the guidance-distilled `[dev]`
  carries one, `[schnell]` does not), and the pooled CLIP-L projection, summed; the timestep and
  guidance are scaled by 1000 inside the forward. FLUX operates on a packed latent (`x_embedder`:
  64 → inner), and `proj_out` returns the packed velocity (the pipeline unpacks). FLUX conditions on the
  T5-XXL sequence and the CLIP-L pooled embedding (no CLIP-G, no CLIP sequence). Reference parity
  against diffusers at a tiny random configuration covering both block kinds, the guidance embedding,
  the axial rotary, and the `[text, image]` order (velocity cosine 0.9999999999998679, first Swift run).
  Reuses the SD3 shared helpers (`NFKSD3AdaLinear`, `NFKSD3FeedForward`, `NFKSD3MLP`,
  `sd3AffineFreeLayerNorm`, `sd3TimestepEmbedding`). Every weight is at most 2-D, so `loadWeights` needs
  no transpose. `configuration(fromHuggingFace:)`; presets `.dev` / `.schnell` (both 19 double + 38
  single blocks, 24 heads × 128). Held to the released headers by shape: FLUX.1 [schnell] 1156
  tensors, FLUX.1 [dev] 1160 (with the guidance embedder), each 0 missing / mismatched / unaccounted.
  Oracle `run_flux`, `IK_PARITY_FLUX`.
- `NFKMLXSD3Pipeline` / `NFKMLXFluxPipeline` — the SD3 and FLUX text-to-image pipeline glue, chaining
  the transformer (denoised over the rectified-flow schedule) and the autoencoder. The caller supplies
  the joint text embedding and the pooled projection (SD3: T5 + the two CLIP sequences on the sequence
  axis, the two CLIP pooled on the channel axis; FLUX: T5 sequence + CLIP-L pooled), as the SD pipeline
  takes a text context. The autoencoder is the shared `NFKMLXSDAutoencoder`: SD3's VAE keeps the
  quant convolutions (`scaling` 1.5305, `shift` 0.0609), FLUX's drops them (`scaling` 0.3611, `shift`
  0.1159, the Z-Image `.flux` VAE). SD3 runs classifier-free guidance; FLUX takes the guidance embedding
  (`[dev]`) or none (`[schnell]`) with no CFG. FLUX packs the latent — each 2×2 spatial block folds
  into the channel axis (`pack` / `unpack`, round-tripped in a test), and `imageIds(height:width:)` is
  the `(0, row, col)` grid. Both step `sample + (σ_next − σ)·velocity` with no negation (Z-Image negated
  its convention). `NFKMLXFlowMatchScheduler` gained `.sd3` (static shift 3.0), `.flux` (dynamic shift),
  and `.fluxSchnell` (static shift 1.0). Validated by weight-free glue tests on matching tiny
  configurations; a sampled image is not bitwise-comparable, as with the other DiT pipelines.
- `NFKMLXSD3ControlNetNet` / `NFKMLXSD3ControlNetPipeline` — the Stable Diffusion 3 ControlNet
  (`SD3ControlNetModel`, Stability AI / InstantX), a partial copy of the MMDiT that steers a generation
  with a spatial control image (Canny, depth, pose, blur, tile). It runs the first N joint blocks over
  the noisy latent plus a control latent and emits one zero-initialized residual per block; the base
  `NFKMLXSD3TransformerNet` adds them into its own non-`context_pre_only` blocks, strided over the
  residual list by the reference's `interval_control` (`blockControlnetHiddenStates`, threaded back into
  the base forward, nil for plain text-to-image so the base is byte-identical). The control latent is
  VAE-encoded and patch-embedded through a zero-initialized `pos_embed_input` that carries no positional
  table, then added to the noisy latent before the blocks. Two released shapes, both ported and
  configurable: the InstantX SD3-medium / SD3.5 ControlNets carry a `context_embedder` and reuse the
  full dual-stream `NFKSD3JointBlock` (all `context_pre_only=false`); Stability's official SD3.5-large 8B
  ControlNets (Blur, Canny, Depth) drop the position embedding and the context embedder and run
  single-stream `NFKSD3SingleBlock`s (`AdaLayerNormZero`, no qk-norm, no text stream) over image tokens
  the base transformer's `pos_embed` supplies (`usePosEmbed` / `useContextEmbedder`; `extraConditioningChannels`
  widens the control patch embed). Reference parity against diffusers at tiny random configurations
  (`run_reference.py sd3_controlnet` / `sd3_controlnet_single`, `IK_PARITY_SD3_CONTROLNET` /
  `_SINGLE`, the `ltx` oracle env): the dual-stream per-block residuals 0.9999999999999958 /
  0.9999999999999937 and the base transformer with them injected (a four-block base, two residuals, so
  the striding is exercised) 0.9999999999999859; the single-stream residuals 0.9999999999999984 /
  0.9999999999999982. `configuration(fromHuggingFace:)` reads `config.json` (a `joint_attention_dim`
  present selects the dual-stream shape). The pipeline runs the ControlNet once per CFG branch (its
  residuals depend on the text stream) and offsets each base pass by its own residuals. Only the 4-D
  patch-embed convolutions transpose to NHWC.
- `NFKMLXFluxControlNetNet` / `NFKMLXFluxControlNetPipeline` — the FLUX.1 ControlNet
  (`FluxControlNetModel`, Black Forest Labs / InstantX / Shakker-Labs), a partial copy of the FLUX
  transformer. It runs a few double-stream and single-stream blocks over the packed noisy latent plus a
  packed control latent (added through the zero-initialized `controlnet_x_embedder`) and emits a
  zero-initialized residual per block — a `controlnet_blocks` list for the double blocks and a
  `controlnet_single_blocks` list for the single. The base `NFKMLXFluxTransformerNet` adds each into its
  own two block stacks (`controlnetBlockSamples` / `controlnetSingleBlockSamples`), strided by the
  reference's **`ceil` interval** (distinct from SD3's non-ceil rule), nil by default so the base is
  byte-identical. The **union** variant (`numMode`) prepends a learned control-type embedding to the
  text sequence and one txt-id row, so one ControlNet serves several control types. The
  `input_hint_block` shape (`NFKFluxControlNetHintEmbedding`, the reference `ControlNetConditioningEmbedding`:
  a `conv_in`, three stride-2 downsampling stages over a `(16, 16, 16, 16)` channel pyramid, a zero-init
  `conv_out`, SiLU between every convolution) is built: instead of a packed VAE control latent it takes a
  Full-resolution control image `[B, C, H·8, W·8]`, downsamples it 8× to the packed grid, and flattens to
  the control tokens; `conditioningEmbeddingChannels` selects it, and the pipeline feeds the raw NCHW
  image rather than a VAE latent when it is present. Reference parity against diffusers end to end
  (`run_reference.py flux_controlnet` / `flux_controlnet_hint`, `IK_PARITY_FLUX_CONTROLNET` / `_HINT`,
  the `ltx` oracle env): the double-block residuals 0.9999999999999967, the single-block residuals
  0.9999999999999962 / 0.9999999999999942, and the base transformer with both injected (a three-block
  base, two residuals of each kind) 0.9999999999999251; the hint variant's residuals ≥ 0.9999999999999913
  and injected velocity 0.9999999999999771. FLUX has no CFG (the guidance embedding), so the ControlNet
  runs once per step. Only the `input_hint_block`'s 4-D convolutions transpose to NHWC.
