<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX models: DiT image and video generation

LTX-Video, Z-Image, SANA, Wan, SD3, FLUX, their samplers, text encoders, and ControlNets.

At bf16 the FLUX, FLUX.2, Qwen-Image and Wan transformers place their roundings as diffusers does,
held on the tiny configurations of their float32 parity tests (`IK_DIT_DTYPE=bfloat16`, the method in
`mlx-parity-checklist.md`). FLUX and FLUX.2 are bit-exact; Qwen-Image reads 0.10 of the reference's
own bf16-versus-float32 distance, and Wan 0.24, all of it from the last float32 bit of MLX's
`exp`/`sin`/`cos` in the timestep sinusoid. Three things had been wrong. FLUX's rotary tables and every
sinusoidal timestep projection were float32 arrays that promoted the transformer's activations to
float32 against its bf16 weights; diffusers rotates in float32 and rounds once, casts the timestep and
guidance to the latents' type before the factor of 1000, and casts each projection to its embedder's
type. The attention is torch's default CPU kernel at bf16, its flash kernel (`flashAttention`), not
MLX's fused one. And the tanh GELU and SiLU round once. Wan's diffusers block also keeps its
modulation, its norms and its gated residual sums in float32, rounding each step once, and its query
and key norms are torch's `nn.RMSNorm`, which rounds once. A fourth reading, Qwen-Image at 41 times
the floor, came from the oracle: `model.to(bfloat16)` had rounded the timestep frequencies, a
non-persistent buffer that `from_pretrained` leaves float32.

The pipelines run their latents and conditioning in the transformer's type, as diffusers does
(`NFKReferenceRounding.parameterType(of:)`). FLUX, FLUX.2 and Qwen-Image draw their latents in that
type, pass the timestep as diffusers does (the schedule's value in that type, divided by 1000 there),
and take each Euler step with the velocity scaled and rounded, the sum formed in float32 and rounded
back (`eulerStep`). Wan keeps its latents float32 and casts only the transformer's input and the
prompt embeddings, as its reference pipeline does; its UniPC and DPM-Solver steps scale the velocity
in float32 and round once. Before this, a bf16 transformer received float32 latents, and its first
projection promoted every activation to float32 against bf16 weights. A float32 load takes the same
arithmetic as before.

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
  Customization: untrainable here. Lightricks/LTX-Video at 4b2d053 publishes no autoencoder training
  code: its only gradient through the autoencoder is a smoke test that backpropagates an MSE on random
  input in evaluation mode, and Lightricks/LTX-Video-Trainer freezes the autoencoder.
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
- `NFKMLXLTX2TransformerNet` — the LTX-2 audio-video transformer (`LTX2VideoTransformer3DModel`,
  Lightricks). One transformer denoises a video latent and an AUDIO latent together, so a generated
  clip carries its own sound. The LTX-Video DiT above is the video-only ancestor; this is a different
  model, not a configuration of it. Each block runs SIX attentions: video self-attention, audio
  self-attention, video-over-text and audio-over-text cross attention, and the two cross-modal
  directions (`audio_to_video_attn`, where the video asks and the audio answers, and
  `video_to_audio_attn`). Both cross-modal attentions run at the AUDIO head geometry whichever way
  they point, so `audio_to_video_attn.to_q` is `[audioInner, inner]` and its `to_out.0` is
  `[inner, audioInner]`. Every attention takes an ACROSS-HEADS RMS norm — query and key are normalized
  over the whole projected width before the heads are split, not per head — and scales each head by
  `2 · sigmoid(to_gate_logits(query input))`, so a zero-initialized gate leaves the attention
  unchanged.

  Modulation is PixArt-alpha's adaptive-norm-single raised to ten heads. Six live on the model (video
  and audio timestep embeddings at nine parameters each, two cross-modal scale/shift heads at four,
  two cross-modal gates at one) and each block adds its own `scale_shift_table` on top, so a block's
  parameters are the per-layer DELTA of a vector computed once. Two traps sit in that arrangement: the
  block's parameter order is (shift, scale, gate) per set while the OUTPUT layer's two-entry table is
  (shift, scale) read off the EMBEDDED timestep rather than the expanded one, and the output's layer
  norm fixes its epsilon at 1e-6 instead of reading `norm_eps`. A third sits in the cross-modal pair:
  both directions read the SAME normalized streams, so the video's update must not be visible to the
  audio's.

  The rotary is the `split` kind — the channel axis of each head halves into a real and an imaginary
  block rather than interleaving adjacent pairs — and its frequencies are a LINEAR ramp in the
  exponent (`theta ^ linspace(0, 1, steps) · π / 2`), not the usual geometric ramp in the index. A
  token's position is the MIDPOINT of the pixel-space interval its patch covers, in seconds on the
  frame axis, shifted by the causal VAE's first-frame stride and clamped at zero; the axes are
  interleaved per token rather than blocked, and the leading channels pad with an identity rotation.
  Reference parity against diffusers at a tiny random configuration on the first numeric run, in BOTH
  released arrangements: LTX-2.3's video velocity 0.9999999999996182 and audio 0.9999999999999584,
  LTX-2.5's 0.9999999999996512 and 0.9999999999996835, with the rotary cosine exact and its sine
  0.9999999999999996. LTX-2.0's arrangement is measured too (0.999999999999749), because its caption
  projections and LTX-2.5's keyframe embedding are paths no other measured configuration exercises.
  Oracle `run_ltx2` (no checkpoint, the `qwenimage` environment), `IK_PARITY_LTX2`. Held to the
  released headers by shape on LTX-2.3 22B, 4186 tensors, 0 missing / mismatched / unaccounted.

  **LTX-2.5 versus LTX-2.3, and what the gate withholds.** Every LTX-2.5 repository is gated
  (`Lightricks/LTX-2.5`, `-Diffusers`, `-Pre-Trained`), so the structural check runs against LTX-2.3,
  the same class, whose diffusers conversion is ungated. LTX-2.5's three declared differences are
  switches, not shapes: the video feed-forward drops its bias, the text cross-attention's key/value
  modulation becomes timestep-independent (`use_prompt_adaln_single` false, which drops the two prompt
  adaptive-norm heads and lets a sampler cache the text key and value across denoising steps), and a
  learned absolute-position embedding marks generated-keyframe tokens. All three are measured — the
  oracle records both arrangements — so the arithmetic of every switch is verified even where which
  switch the release sets cannot be read. `.ltx25` declares them; the rest of its geometry is
  LTX-2.3's and is unverified until the gate opens. `configuration(fromHuggingFace:)` REFUSES BY NAME
  every arrangement it does not build, rather than ignoring the field: the interleaved rotary, the text
  cross-attention modulation turned off (that arrangement carries six modulation parameters instead of
  nine, and a module built for nine would read the wrong slices of a vector that still loaded), the
  attention gates turned off, a per-head `qk_norm`, affine normalizations, and any activation but the
  tanh-approximate GELU. Without the refusal the gate case would surface as a missing `to_gate_logits`
  at load rather than as the configuration that asked for it.

  Spatio-temporal guidance is not built. It is a sampling-time trick (replace an attention with its
  value projection on perturbed batch elements) with no weights behind it, and the reference's
  perturbed processor is identical to the plain one when nothing is perturbed. The audio and video
  VAEs, the Gemma-4 text front end with its connectors, the vocoder and the pipeline are the remaining
  stages. Customization is offline-only: no size of this model fits a consumer machine.

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
  Customization of the DC-AE: untrainable here. mit-han-lab/efficientvit trains the diffusion model over
  precomputed latents, and dc-ai-projects/DC-Gen's autoencoder trainer calls `forward_train`, which no
  published model implements. The ruling rests on that gap: a later DC-Gen commit that implements
  `forward_train` reopens it.
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
  the last stage of the Wan pipeline and the hardest port of this batch. It also serves Qwen-Image 2.1
  under `NFKMLXWanVAEConfiguration.imageOnly`, which collapses every temporal kernel, pad, and stride
  to one; the flag defaults off and Wan's own measured path is unchanged by it. It compresses a video 4× in
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
  Customization of the Wan VAE: untrainable here. Wan-Video/Wan2.1 (`wan/modules/vae.py`) and Wan2.2
  (`wan/modules/vae2_2.py`) define the network with no objective and no training script.
- `NFKMLXQwenImageNet` / `NFKMLXQwenImage` (`@objc`) — the Qwen-Image 2.1 DiT
  (`QwenImage21Transformer2DModel`, Qwen), the denoising transformer of a 7.1B text-to-image model.
  Single-stream like Z-Image: one sequence carries the caption and the image latents. Two things
  distinguish it. Attention is **block-causal** — `q >= kv or same image block` — so the caption is
  strictly causal while each image block (every condition image, and the target image) stays
  internally bidirectional; the port builds that as one dense additive mask, where the reference
  chooses between a flex `BlockMask` and a per-segment decomposition. And **`causal_condition`** makes
  the text and condition-image tokens modulate from `t = 0` rather than from the sampled timestep,
  which is what makes the prefix independent of the step; the port carries both timestep rows and
  selects per token with a gather. The condition images do not sit beside the caption, they sit INSIDE
  it at the slots the vision-language encoder reserved, each slot standing for a 2x2 group of latent
  tokens, so the caption is expanded four-fold at those positions and the latents are spliced in.
  The block holds no modulation of its own: one shared `modulation` projection feeds every block,
  which slices `[scale1, gate1, scale2, gate2]` out of it, gates through `tanh`, scales as `1 + scale`.
  A SwiGLU feed-forward (`proj` / `gate_layer` / `out`, ratio 3), affine-free layer norms, per-head RMS
  q/k norms, a **three-axis complex rotary** (axes `[16, 56, 56]` summing to headDim 128, θ 10000)
  whose image blocks are CENTERED on zero and whose table carries positions −1024…−1 after 0…8191 (the
  reference reaches the negative half by Python's negative indexing, so the port maps a signed position
  to its row). The caption projection normalizes with a **zero-centered RMS norm** whose stored weight
  is `scale - 1`, which an ordinary `RMSNorm` would read as a scale near zero. The timestep's sinusoid
  puts COSINE in the first half of the channels, and the timestep is cast to the model's dtype BEFORE
  the projection, which is worth a thousandth of a cosine on a bfloat16 release.
  Only `causal_condition` true is measured, which is what the release sets.
  Reference parity against diffusers (`run_reference.py qwenimage21`, tiny random config, one condition
  image and one target, two padded caption slots): rotary 0.9999999999999997, output
  0.9999999999995779. On the RELEASED 7.1B weights, both sides at the bfloat16 they ship in
  (`qwenimage21_real`): 0.9988178088700854, with the per-seam record showing where that number comes
  from — `img_in` 0.9999999999, `txt_in` 0.9999986, blocks 0/7/15/23/31 at 0.999944 → 0.999906, then
  `norm_out` 0.997864. The final adaptive norm subtracts each row's mean, and at bfloat16 the residual
  stream's mean carries most of its magnitude, so removing it amplifies the accumulated rounding by an
  order of magnitude; normalizing BOTH sides' own last-block output reproduces the same drop
  (0.999906 → 0.998060), which is a property of the precision rather than of either implementation.
  Structural against the released checkpoint: 297 tensors consumed, 0 missing, 0 mismatched, 0
  unaccounted. The release's keys are the module names but for `modulation.1.weight` (a `Sequential`
  whose first element is the activation) and `norm_out.linear.weight`.
  Its text encoder is `Qwen3VLForConditionalGeneration` at the 8B geometry,
  which `NFKMLXQwen3VL` reads from a release's `config.json` unchanged. Its sampler is
  `NFKMLXFlowMatchConfiguration.qwenImage21`: dynamic shifting over the 256…8192 sequence range (base
  0.5, max 0.9) and a terminal stretch onto 0.02, measured against the release's own scheduler config
  (`run_reference.py qwenimage21_scheduler`) at 4, 20, and 50 steps over two sequence lengths, worst
  sigma 1.8e-7 and worst timestep 1.8e-4. One release fact came out of that: the pipeline passes its
  OWN sigma ramp, `linspace(1, 1/steps, steps)`, rather than the scheduler's default ramp to
  `1/num_train_timesteps`, which is what `rampEndsAtStepFraction` carries. The FLUX presets carry it
  too; the SD3 presets do not (see the FLUX / SD3 entry below for the measurement that settled
  each). The weights are under the **Qwen Research License**
  (non-commercial), unlike the Apache 2.0 of the Qwen3-VL retrieval pair.
  The oracle runs in its own interpreter: diffusers 0.41.0.dev0 from git main (the released 0.31 in
  `sdvenv` carries none of these classes), torch 2.8, transformers 5.17, at
  `~/.inferkit-validation/qwenimagevenv/bin/python3.12`.

- `NFKMLXQwenImageVAE` (`@objc`) — Qwen-Image 2.1's autoencoder (`AutoencoderKLQwenImage21`), which is
  not a new architecture: it is the Wan 2.2 residual VAE `NFKMLXWanVideoVAENet` already runs,
  SPECIALIZED TO ONE FRAME. In the reference its causal 3-D convolution subclasses `nn.Conv2d` — it
  squeezes the temporal axis away, runs a 2-D convolution, unsqueezes, and RAISES if handed a feature
  cache. Three things follow, and together they are the whole port: the checkpoint's convolution
  weights are 4-D rather than 5-D, every temporal kernel is 1 (the resamplers' `time_conv` becomes a
  1x1 convolution at stride 1), and for a single frame the temporal branches never execute at all,
  because the first chunk leaves their cache slots empty and that is the branch that skips them. The
  shipped network gained one configuration field, `imageOnly`, which collapses every temporal kernel,
  pad, and stride to one; Wan's own path is unchanged and its parity numbers are unmoved. The
  geometry is 64 latent channels over four spatial halvings (16x), base 96 encoding and 144 decoding,
  no patchify, and FOUR image channels rather than three. Loading translates three layouts: a spatial
  `Conv2d` takes `[out, kh, kw, in]`, a causal convolution the same weight with a temporal axis
  inserted, and the RMS normalizations store their scale with trailing singleton axes that flatten.
  Reference parity against diffusers on the released weights, first numeric run (`run_reference.py
  qwenimage21_vae`): encoder mean 0.9999999999995628, the pipeline's normalized latent
  0.9999999999996052, decoded image 0.9999999999995695.
  Customization: untrainable here. Neither QwenLM/Qwen-Image at 6b5e1f5 nor the Qwen-Image and
  Qwen-Image 2.1 releases carry autoencoder training code, and diffusers' generic autoencoder trainer
  loads plain `AutoencoderKL` only.
- `NFKMLXQwenImagePipeline` — the Qwen-Image 2.1 text-to-image glue, chaining the vision-language text
  encoder, the block-causal DiT over the flow schedule, and the single-frame VAE. Two details of the
  glue belong to the model rather than to the sampler. The prompt is a RAW template string handed
  straight to the tokenizer (`<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>…`)
  rather than the chat template's rendering of it, because the two tokenize differently and the
  release expects this one; and the features the transformer reads are the text encoder's last layer
  BEFORE its final normalization, which the reference obtains by neutralizing that norm with a forward
  hook and which `NFKMLXLanguageNet.hiddenStates(fromEmbeddings:cache:multimodal:applyFinalNorm:)`
  now exposes. The parameter defaults to true, so every other consumer of the shared decoder is
  unchanged; the Pixtral fused pipeline re-measures at logits 1.0000002 with argmax exact at every
  position. The system turn's leading tokens are dropped from the features. The joint sequence's
  image mask is the caption's own mask with one slot appended per 2x2 group of target latents, and
  packing is a plain spatial flatten because 2.1 consumes latents unpatched.
  Measured in two places. The glue against diffusers' own `QwenImage21Pipeline` at tiny random models
  (`run_reference.py qwenimage21_pipeline`): final latents 0.9999999999998141 and the decoded image
  0.9999999999999997, and the reference was run both with and without the prefix KV cache it uses by
  default — the two agree to the last digit, which is what makes this port's single uncached path
  correct for both. The prompt encoding against the released 8B text encoder at its own bfloat16
  (`run_reference.py qwenimage21_text`, eager attention): the template tokenizes to the reference's ids
  exactly, the dropped system length matches, and the features agree at 0.9988784826733571 in
  aggregate with a per-token profile of 0.98462 at the first position and 0.99870 or better at all
  twenty-one others. That is the reference's own bf16 spread: its SDPA run agrees with its eager run
  at 0.99867 in aggregate, 0.97649 at the first position and 0.99873 at the second. The earlier
  0.99955 came from float32 M-RoPE tables that promoted every attention to float32, more precise
  than the reference's own bf16 run and unlike it.
  The seam records say what that first position is: it carries a MASSIVE ACTIVATION, a residual of
  norm about 9390 from layer 18 through layer 34, which the last two layers cancel down to 717. At
  bfloat16 the rounding of the large residual survives that cancellation as a few percent of the small
  result, and the token's own cosine tracks at 0.9999997 until layer 35 and then falls. Every other
  token of the twenty-two is of comparable magnitude and agrees to 0.9998 or better, so the aggregate
  reads one cancelling position rather than the encoder's agreement; the per-token floor is the
  measurement that says the arithmetic is right.
  Customization is OFFLINE-ONLY, on the same grounds as SDXL and FLUX. The transformer is 7.1 billion
  parameters, which is 14 GB at the bfloat16 it ships in before a single activation, and the reference
  recipe is a diffusion fine-tune whose backward pass holds the activations of thirty-two blocks over
  the full latent sequence. Adapters trained in Python merge into the released tensors and load through
  `NFKMLXQwenImage.loadWeights(into:fromDirectory:precision:)` like any other checkpoint, which is the
  route a consumer takes. Nothing about the port blocks a LoRA run; the working set does.

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
  Oracle `run_flux`, `IK_PARITY_FLUX`. The released FLUX.1 [schnell] transformer is also at numeric parity
  against diffusers on a tiny-spatial input (4 image tokens) at the bfloat16 it ships in: velocity
  0.9999831167429072 (`run_reference.py flux_real`, `IK_PARITY_FLUX_REAL` / `IK_VAL_FLUX_SCHNELL`, from the
  ungated `cocktailpeanut/xulf-s` mirror). The 12B transformer needs ~24 GB resident, so that test runs
  where the machine can hold it, not by default.
- `NFKMLXFlux2TransformerNet` — the FLUX.2 transformer (`Flux2Transformer2DModel`, Black Forest Labs).
  It keeps FLUX.1's two block kinds and changes four things. MODULATION LIVES ON THE MODEL: three
  heads (`double_stream_modulation_img`, `…_txt`, `single_stream_modulation`) are evaluated once from
  the timestep embedding and every block of that kind reads the same vector, so the checkpoint carries
  three modulation tensors where FLUX.1 carries one per block. The feed-forward is a SwiGLU whose gate
  is the first half of one fused `linear_in`. The single-stream block is a parallel block in the
  ViT-22B sense: `to_qkv_mlp_proj` produces query, key, value and both SwiGLU halves, and `to_out`
  takes the attention output concatenated with the gated MLP (FLUX.1 keeps `proj_mlp` and `proj_out`
  separate under a GELU). The rotary runs over FOUR axes at theta 2000, and the text ids carry the
  token index on the fourth axis rather than being all zero — the `NFKFluxRope` table already loops
  over the declared axes, so it is reused unchanged. Every linear is bias-free and there is no pooled
  text embedding: the conditioning is the timestep plus the guidance scale. `jointAttentionDim` is
  three times a language model's width because the release concatenates three of its layers per token
  (FLUX.2 [dev] Mistral-Small 3 at 5120, FLUX.2 [klein] Qwen3 at 2560). Reference parity against
  diffusers at a tiny random configuration on the first numeric run: velocity cosine
  0.9999999999999934, with the conditioning 0.9999999999999806, the shared double- and single-stream
  modulation 0.999999999999973 / 0.9999999999999812, and the first double block's two streams
  0.9999999999999974 / 0.9999999999999977. Oracle `run_flux2` (no checkpoint, the `qwenimage`
  environment — `Flux2Transformer2DModel` needs a diffusers newer than the `sd`/`ltx` ones carry),
  `IK_PARITY_FLUX2`. Presets `.dev` and `.klein4B`; `configuration(fromHuggingFace:)` reads a released
  `transformer/config.json`. Held to the released headers by shape on FLUX.2 [klein] 4B, 169 tensors,
  0 missing / mismatched / unaccounted — the ONE FLUX.2 release that is not gated, so the inventory is
  Black Forest Labs' own rather than a mirror's.

  **What the gate withholds, and what replaces it.** FLUX.2 [dev] is gated, and accepting a license is
  a user action. The published parameter total substitutes for the headers exactly: summing the
  declared `.dev` module gives 32,223,281,152, the release's own figure to the tensor, qk-norm vectors
  included, and the same sum over `.klein4B` gives 3,875,544,576, which the shape walk confirms
  independently. `testFlux2DevParameterTotalMatchesTheRelease` asserts both, and `.klein9B` beside
  them at 9,078,581,248.

  The 9B split shows what a parameter total cannot do. It pins 32 heads, the 12288-wide text sequence
  and the absent guidance embedding, and leaves the double/single split open, because a double block
  costs exactly two single blocks and seventeen splits reach the same total. `.klein9B` was therefore
  left undeclared until a 9B release was reachable. `FLUX.2-klein-base-9B` opened first, and its
  `transformer/config.json` states `num_layers` 8 and `num_single_layers` 24; its own `_name_or_path`
  is the klein-9b conversion directory, so the two releases share the geometry. 233 tensors consumed,
  0 missing, 0 mismatched, 0 unaccounted. The lesson generalizes: a total constrains a geometry, and
  only the release decides among the arrangements that satisfy it.

  **The autoencoder and the text front end ship.** `AutoencoderKLFlux2` is the ordinary diffusers
  `AutoencoderKL` at 32 latent channels, which `NFKMLXSDAutoencoder` already builds: the preset is
  `NFKMLXSDVAEConfiguration.flux2` (quantization convolutions kept, no scalar scale or shift). What is
  new is `NFKMLXFlux2LatentCodec`, which carries the two pieces that belong between the autoencoder and
  the transformer — a 2×2 latent patching into the transformer's 128 channels, and the `bn` BatchNorm
  whose RUNNING STATISTICS whiten the patched latent in place of the `scaling_factor`/`shift_factor`
  every earlier release carries. The patching order is the trap: the reference's
  `permute(0, 1, 3, 5, 2, 4)` puts a channel's four sub-pixel offsets adjacent, so a pixel-unshuffle
  grouped by spatial position gives the same shape with different contents, and this package works
  channels-last where the reference works channels-first. Measured against diffusers: latent
  0.9999999999999983, patching and whitening 0.9999999999999988, decode 0.9999999999998801, with the
  decoder walked stage by stage (`post_quant_conv`, `conv_in`, `mid_block`, `up_blocks.0`,
  `conv_norm_out`) so a divergence names a stage. Oracle `run_flux2_vae`, `IK_PARITY_FLUX2_VAE`.
  Customization: `NFKMLXFlux2LatentCodec` is untrainable. It holds only the BatchNorm running statistics
  (the reference builds `bn` with `affine=False`), so a gradient has nothing to move, and
  black-forest-labs/flux2 publishes no training code. The autoencoder itself shares
  `NFKMLXSDAutoencoder`'s row; diffusers' `train_autoencoderkl.py` (L2, 0.5 × LPIPS, KL at 1e-6, a
  PatchGAN built fresh and switched on at step 50,001, AdamW at 4.5e-6) is a generic route, not BFL's
  recipe.

  That walk found a defect in SHARED code, not in FLUX.2: `NFKSDResnetBlock` normalized at the UNet's
  eps 1e-5 while diffusers builds every autoencoder block with `resnet_eps=1e-6`. See the entry in
  `mlx-runtime-gotchas.md`; the epsilon is now the caller's, and the FLUX.1, SD 1.5 and upscaler rows
  in `Docs/model-parity.md` moved because they had been recording the bug.

  `NFKMLXFlux2TextEncoder` is the text front end. The conditioning is not a language model's output:
  the release reads THREE intermediate hidden states and concatenates them per token, which is why
  `jointAttentionDim` is three times the text model's width (FLUX.2 [klein] reads a Qwen3 at layers
  9, 18 and 27; FLUX.2 [dev] reads Mistral-Small 3 at layers 10, 20 and 30 of 40, which this package
  now runs as `NFKMLXLanguageConfiguration.mistralSmall3`). The layers are DERIVED rather than
  written down: `jointAttentionDim / hiddenSize` gives how many states are read, and the encoder
  spaces them evenly over the stack, so 15360 / 5120 = 3 selects 10, 20 and 30. The prompt pads on the RIGHT and the whole padded sequence is encoded, so the pad positions'
  own states reach the transformer and the attention mask is load-bearing: measured on the reference,
  masking leaves the real tokens identical (cosine 1.0) and moves the pad positions by 1.2e-2, taking
  the whole embedding to 0.9993. `NFKMLXLanguageNet.layerStates` therefore takes an optional
  `keyPadding`, nil-defaulted so every existing caller is unchanged. Conditioning cosine
  1.0000000000000013 against the reference's own `_get_qwen3_prompt_embeds`, every hidden state exact.
  Oracle `run_flux2_text`, `IK_PARITY_FLUX2_TEXT`.

  The [dev] front end is measured on its own record, because Mistral differs from Qwen3 in two ways
  the arithmetic sees: it does not normalize queries and keys per head, and its head width is stated
  rather than implied (`head_dim` 128 against a 5120 residual over 32 heads, which divides to 160).
  Conditioning cosine 1.0 against transformers' own `MistralForCausalLM`, every hidden state at or
  above 0.9999999999999976, with the mask-off control at 0.9995260. Oracle `run_flux2_text_mistral`,
  `IK_PARITY_FLUX2_TEXT_MISTRAL`. `sliding_window` is null in the release and the oracle sets it so;
  MistralConfig's own default would apply a sliding mask the release does not.

  **`NFKMLXFlux2` runs [klein] END TO END** from a release directory: `NFKMLXFlux2Pipeline` chains
  the scheduler, the transformer and the codec, and the facade adds the prompt path. The prompt runs
  through the release's own `chat_template.jinja` rather than a reimplementation, because at
  `enable_thinking=False` a Qwen3 template appends an EMPTY think block after the assistant header —
  18 ids for "a red fox in the snow" where a bare prompt is 6. Rendered text and token ids are exact
  on three prompts, measured against the ungated klein-4B tokenizer (`run_reference.py flux2_prompt
  --checkpoint <tokenizer dir>`). The schedule is `NFKMLXFlowMatchConfiguration.flux2`, whose
  EMPIRICAL shift depends on the step count as well as the sequence length and replaces the
  `base_shift`/`max_shift` the released scheduler config still carries; above 4300 tokens the
  200-step line is used alone and the step count drops out. The `@objc` surface is the facade:
  `flux2WithDirectoryURL:error:` then `imageForPrompt:negativePrompt:width:height:seed:error:`.

  **Measured on the released weights.** Every FLUX.2 figure above is a tiny random configuration,
  and a structural match proves only that 169 tensors load at the right shapes. klein 4B is 3.88B and
  ungated, so each stage is also measured on the shipped checkpoint:

  - Transformer (`run_reference.py flux2_real`, `flux2_real_f32`): velocity 0.99999999990117 at
    float32 and 0.99892858 at the released bfloat16. The bfloat16 figure is 60 times further from 1
    than FLUX.1 [schnell]'s 0.99998, on a shallower network, so it could not be waved through as
    noise. The reference answers it: its own bfloat16 output agrees with its own float32 output only
    to 0.99919. That is this architecture's precision floor on real weights, and the float32 run is
    what separates the floor from a defect. klein 4B is 15.5 GB at float32, which this machine holds;
    FLUX.1 at 12B never could, so its released figure is bfloat16 alone.
  - Autoencoder and codec (`flux2_vae_real`): latent 0.9999999999986, packed tokens
    0.9999999999986, decode 0.9999999999887, through `loadVAEWeights` and
    `NFKMLXFlux2LatentCodec.codec(fromReleaseDirectory:)`, so the released BatchNorm statistics are
    read the way the pipeline reads them.
  - Prompt to conditioning (`flux2_text_real`): 0.99999999995 and 0.9999999998 on two prompts, from
    the prompt STRING through `NFKMLXFlux2.encode(prompt:)`, against the reference pipeline's own
    `_get_qwen3_prompt_embeds`. Token ids are exact and the pad id is 151643. The release's encoder is
    a `Qwen3ForCausalLM` that is NOT byte-identical to `Qwen/Qwen3-4B` (two shards against three, no
    hash in common), so the package's Qwen3-4B measurement could not be borrowed for it.

  The base 9B release is measured the same way where it fits. Whole, at its released bfloat16, the
  transformer reads 0.99784 (`flux2_real` against `IK_VAL_FLUX2_KLEIN_BASE_9B`). Float32 is 36 GB and
  does not fit, so `run_reference.py flux2_real_truncated` cuts the release to its first two double and
  two single blocks and reads only those tensors. The cut keeps everything the 9B geometry adds (the
  4096 width, 32 heads, the 12288-wide text projection, the model's modulation and output heads) and
  drops only repeated depth. At float32 the cut reads 0.9999999999994. The same cut measures the
  precision gap directly: the reference's own bfloat16 against its float32 is 0.99999024, this port's
  bfloat16 against the reference's float32 is 0.99999931, closer than the reference's own, and the two
  bfloat16 implementations agree to 0.99999039, the size of the reference's own gap. So the whole-depth
  0.99784 is bfloat16 accumulation over 32 blocks. That the floor reaches 0.99784 at full depth is
  inferred, not measured: the float32 run that would measure it does not fit. The base 9B's VAE is
  byte-identical to klein 4B's.

  The base 9B's text encoder is a Qwen3-8B, 15.3 GB as released and 30.5 GB at float32. From the
  prompt string at the released bfloat16, whole, the conditioning reads 0.99998016 and 0.99997714
  (`flux2_text_real_bf16`, `IK_PARITY_FLUX2_TEXT_9B`). Its width is 12288, three hidden states of
  4096. `flux2_text_real_truncated` cuts the encoder to 28 layers for float32: the conditioning reads
  hidden states 9, 18 and 27 and nothing past them, and the cut is 28 rather than 27 because the last
  entry of the reference's hidden-state tuple is taken after the final norm. The installed
  transformers validates that `layer_types` has one entry a layer, so the cut shortens both
  (`_qwen3_cut_config`); changing `num_hidden_layers` alone raises. At float32 the cut reads
  0.9999999999965 and 0.9999999999971 (`IK_PARITY_FLUX2_TEXT_9B_TRUNCATED`). Since the cut computes
  the whole conditioning, that is the full measurement, not an inference from depth. The reference's
  own bfloat16 against its float32 is 0.99993 on both prompts, a larger gap than this port's
  bfloat16 shows.

  The release lives at `/Volumes/WindowsBoot/InferKit/validation/flux2-klein-4b` (transformer, VAE,
  text encoder, tokenizer; 15 GB) under `IK_VAL_FLUX2_KLEIN_4B`, `IK_VAL_FLUX2_KLEIN_4B_VAE` and
  `IK_VAL_FLUX2_KLEIN_4B_ROOT`.

  **Image-to-image and editing** ship with it. FLUX.2 conditions on a reference image by encoding it
  through the same codec, APPENDING its tokens to the generated image's sequence, and keeping only
  the generated tokens from what the transformer returns. What separates them is the TIME axis: the
  generated latent sits at `t = 0` and reference `N` at `t = 10 · (N + 1)`
  (`referenceImageIds(height:width:index:scale:)`). Without the offset a reference patch and a
  generated patch at the same row and column would carry the same rotary position. This is what the
  fourth rope axis is for; the first separates images and the fourth numbers text tokens.

  Objective-C reaches image-to-image through `imageForPrompt:negativePrompt:references:width:height:seed:error:`,
  whose references are an `NSArray` of `CGImage` or `MTLTexture`, passed from Objective-C as
  `(__bridge id)image`. `[CGImage]` cannot be the Swift type: it compiles, and the generated header
  declares `NSArray<CGImageRef>`, which is not Objective-C, because a Core Foundation type is not an
  object type. The header is where that shows, not the Swift build.

  **Inpainting** is `NFKMLXFlux2Pipeline.inpaint` and, at the facade, `inpaint(prompt:image:mask:…)` and
  `inpaintImage:mask:prompt:negativePrompt:strength:seed:error:`, against diffusers'
  `Flux2KleinInpaintPipeline`. The source image conditions the generation twice. Its whitened latent
  is the first reference, at time coordinate 10, as in editing. It is also the starting point: the
  loop starts `int(steps - steps * strength)` steps in, from the image noised to that sigma, and after
  each Euler step the kept region is overwritten with the image noised to the NEXT sigma, with the
  noise drawn once at the start; the last step blends with the clean image. diffusers'
  `scale_noise` picks that sigma by INDEX (`begin_index` before the loop, `step_index` after a step),
  not by the timestep value it is handed. The mask is binarized at one half at full resolution and
  resampled bilinearly (`align_corners=false`) to the packed grid, so a mask edge that falls between
  packing cells blends at 0.25 or 0.5; `NFKMLXResample.resizeBilinear` is PyTorch's resampling. The
  facade reads a CGImage mask as luminance with PIL's `L` weights.

  `strength` is a `Double`, and the reason is measured rather than argued. The start step is
  computed in double precision, and a single-precision strength widens to a different product: at
  the pipeline's own defaults of 50 steps and 0.8 the reference starts at step 10, where 0.8 as a
  `Float` (0.80000001) starts at step 9. Over step counts 4 to 50 and strengths 0.01 to 0.99, 78
  pairs start differently. (10, 0.7) is not one of them: `10 * 0.7` rounds to exactly 7.0.

  Parity against the reference's own `__call__` at 50 steps and 0.8 (`run_reference.py
  flux2_inpaint`, a tiny random transformer and autoencoder): distilled 0.9999999999999998 and guided
  1.0000000000000007 over 40 steps, the packed mask exact, fractional edge cells included.

  **The reference cache (FLUX.2 [klein] 9B KV)** is `NFKMLXFlux2TransformerNet.extractingReferences`,
  a cached `callAsFunction(…referenceCache:)`, `NFKMLXFlux2ReferenceCache`, and
  `NFKMLXFlux2Pipeline.generateCachingReferences`, against diffusers' `Flux2KVAttnProcessor`,
  `Flux2KVParallelSelfAttnProcessor` and `Flux2KleinKVPipeline`. On the first step the reference tokens
  LEAD the image stream, the reverse of the order ordinary conditioning appends them in. Three things
  change for them. They take the modulation of a fixed timestep, 0 by default, spliced in per position
  (`flux2BlendedModulation`, the reference's `_blend_mod_params`): in the image stream of a double block,
  and between the text and the generated tokens in the joined single stream. They attend only to one
  another while the text and generated tokens attend to everything. And their post-rotary keys and
  values are cached per layer and dropped from the output. Every later step runs the generated tokens
  alone with the cached keys and values spliced between the text and the image. The standard path is
  untouched: the new arguments default to `.none`, which makes the same attention call as before, and
  the tiny transformer's seams read the same figures to the digit.

  That makes it a different function from ordinary conditioning over the same weights, which is why
  `FLUX.2-klein-9b-kv` is a separately trained release whose tensors are the base 9B's to the name and
  shape. Nothing in its files says so: `model_index.json` names the plain `Flux2KleinPipeline`, so
  the facade cannot detect it and takes `cachesReferences` from the caller. The reference pipeline
  has no guidance and defaults to 4 steps.

  The parity oracle draws its random weights at scale 0.4 rather than the harness's usual 0.05, and
  that choice is load-bearing. At 0.05 the modulation and the attention pattern barely move the
  output: the reference cache and ordinary conditioning agree to 1e-12, and a port of the wrong
  mechanism would have passed. At 0.4 they sit at 0.9119, and the port reads 0.9999999999998865
  extracting, 0.9999999999999287 cached, the cached keys and values of the first double and single
  layer at or above 0.99999999999996, and the reference pipeline's 4-step image at 0.9999999999999968
  (`run_reference.py flux2_kv`, `IK_PARITY_FLUX2_KV`). The test asserts the separation as well as the
  agreement. The pipeline refuses a reference image under 64 pixels on a side.

  On the released `FLUX.2-klein-9b-kv` weights (`flux2_kv_real`, `IK_VAL_FLUX2_KLEIN_9B_KV`), at
  bfloat16, whole: extracting 0.99945543 and cached 0.99971902, where ordinary conditioning on the same
  tokens sits at 0.93298, so the released weights separate the mechanisms as the scale-0.4 oracle
  does. Float32 is 36 GB, so `flux2_kv_real_truncated` cuts the release to two double and two single
  blocks as the base 9B's cut does: extracting 0.9999999999994421, cached 0.9999999999993744. At that
  depth ordinary conditioning sits at 0.99984, which the float32 agreement still separates by nine
  orders of magnitude. The reference's own bfloat16 against its float32 on the cut is 0.99998892
  extracting and 0.99999071 cached; the whole-depth floor is not measured, for the same reason as the
  base 9B's.

  **Staged loading.** The text encoder runs once per image and the transformer on every step, so the
  two never need to be loaded together, and `NFKMLXFlux2` holds them as `NFKMLXResidency` says, the
  residency every staged model here shares (`mlx-companion.md`, "Staged models"). A resident release
  loads both at construction and keeps them. A staged one loads nothing up front; for each image it
  loads the encoder, encodes the prompt and, where the release guides, the negative prompt in the same
  load, releases the encoder, then loads the transformer. `.automatic` asks `NFKMLXFlux2.plan`, which
  adds a choice Music 3 does not have, the encoder's precision:

  - a resident placement where one is known to fit, preferring a float32 encoder and falling back to
    the stored precision, which is diffusers' own default of a bfloat16 pipeline;
  - otherwise a staged placement, with a float32 encoder where the encoder alone is known to fit it.

  On a 32 GB machine (25 GiB recommended, a 21.25 GiB budget) klein 4B is resident with a bfloat16
  encoder: at float32 the encoder and transformer need 26.4 GiB with the reserve, which the facade
  used to hold anyway. Klein 9B is staged, its encoder as stored, and its 17.07 GiB transformer stage
  lands 0.18 GiB inside the budget. A machine that reports no budget is staged with its encoder as
  stored. A staged and a resident facade over the same weights produce the same image to the bit
  (`testFlux2StagingReleasesEachStageAndChangesNothing`). On the released base 9B the facade plans
  itself staged with its encoder as stored and makes a 256x256 image in 2 guided steps in 134 s, both
  loads included (`testFlux2Klein9BRunsEndToEndThroughTheFacadeOnReleasedWeights`), on the 32 GB
  machine that cannot hold the encoder and transformer at once.

  **Guidance follows the reference's rule.** Both klein pipelines guide where
  `guidance_scale > 1 and not is_distilled`, against an empty negative prompt when none is given.
  `is_distilled` comes from the release's `model_index.json` (klein 4B: true; the base releases: absent,
  so false). The facade had guided only when a caller passed a negative prompt, which left the base
  releases unguided by default and let a negative prompt turn guidance ON for the distilled release,
  where the reference ignores it. `NFKMLXFlux2.isDistilled` is read at load, and the rule is
  `NFKMLXFlux2.guides(isDistilled:guidance:)`.

  **The other releases.** FLUX.2 [klein] BASE 4B is ungated and geometrically identical to klein 4B,
  so it reads the same `.klein4B` preset — 169 tensors, which is the first check that the declared
  geometry describes the architecture rather than having been fitted to one checkpoint. FLUX.2's
  SMALL DECODER is ungated and is the first ASYMMETRIC autoencoder here: `decoder_block_out_channels`
  [96, 192, 384, 384] against the encoder's [128, 256, 512, 512], which `NFKMLXSDVAEConfiguration`
  now carries as `decoderBlockChannels` (nil mirrors the encoder, as every earlier release does).
  Preset `.flux2SmallDecoder`, 248 tensors plus the 3 `bn` buffers.

  **What is NOT measured, and why.** FLUX.2 [dev]'s text front end is built and measured, and its
  transformer geometry is pinned by the parameter total. The 9B releases are measured on their weights:
  the base 9B's transformer and text encoder, and `klein-9b-kv`'s transformer, whose tensors are the
  base 9B's to the name and shape (an identical config, 233 tensors). `.klein9B` covers the distilled
  klein 9B through the base 9B, whose config names the klein-9b conversion as its `_name_or_path`.
  The distilled klein 9B (`92196c8e`, 34.7 GB) and FLUX.2 [dev] (`26afe3a7`, 112.9 GB, diffusers
  folders only) are on the backup share under `validation/flux2-klein-9b` and `validation/flux2-dev`,
  and neither has been run: [dev]'s transformer is 60 GB at bfloat16, larger than a 32 GB machine
  loads whole. Customization is offline-only on the same
  grounds as FLUX.1 and SDXL.

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
  The sigma ramp under each preset was verified against diffusers on 2026-09-22 (0.31 and
  0.41.0.dev0 read the same): `pipeline_flux.py` and `pipeline_flux_controlnet.py` pass
  `sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)` to `set_timesteps`, so
  `.flux` and `.fluxSchnell` carry `rampEndsAtStepFraction`; `pipeline_stable_diffusion_3.py` and
  `pipeline_stable_diffusion_3_controlnet.py` pass `sigmas=None`, so `.sd3` stays on the scheduler's
  own default ramp. The two ramps differ at every sigma after the first and most at
  the last: FLUX [dev] at 4 steps over 4096 tokens ends on σ 0.5128 (pipeline ramp) against 0.0032
  (default ramp), [schnell] on 0.25 against 0.001, and at 28 steps the worst per-step gap is 0.10.
  The shipped presets ran on the default ramp until then. The same measurement caught a second
  fact about the default ramp: diffusers' scheduler shifts its 1000-entry sigma table at
  construction when `use_dynamic_shifting` is off, so `sigma_min` is the SHIFTED `1/1000` (0.002994
  at shift 3.0, 0.004980 at 5.0), and `set_timesteps` shifts the ramp built down to it a second
  time. SD3's last sigma is therefore 0.008929 (timestep 8.93), where the port's `1/1000` ramp end
  gave 0.002994 (timestep 2.99); `setTimesteps` now ends a static-shift default ramp at the shifted
  value. A dynamic-shift schedule (LTX, Z-Image) was never affected, its `sigma_min` is the raw
  `1/1000`. `NFKMLXFlowMatchSchedulerTests` holds
  diffusers' sigmas for `.flux` (20 steps at 1024 and 4096 tokens), `.fluxSchnell` (4 steps), and
  `.sd3` (20 steps), computed from the released scheduler configs (read from ungated mirrors; the
  `black-forest-labs` and `stabilityai` repos are gated and the local token is not on their lists).
- `NFKMLXFlux` / `NFKMLXFluxTextEncoder` — the end-to-end FLUX.1 text-to-image path (a prompt string in,
  an image out), the shipping layer over the transformer, the sampler, and the autoencoder. `NFKMLXFlux`
  assembles the whole model from a diffusers release directory (`transformer/`, `vae/`, `text_encoder/`,
  `text_encoder_2/`, `tokenizer/`, `tokenizer_2/`) and `image(forPrompt:width:height:seed:)` runs the text
  encoding, the flow-match sampler, and the decode; `@objc` reaches it through `fluxWithDirectoryURL:` and
  `imageForPrompt:width:height:seed:error:`. `NFKMLXFluxTextEncoder` is the text front end split out so it
  loads without the 24 GB transformer: CLIP-L (`NFKMLXSDTextEncoderNet` at the SD1.5 geometry) for the
  pooled projection — the end-of-text token's hidden state after the final norm, no projection, read from
  the padded `tokens(for:contextLength:)` sequence, not the bare-prompt `encode` — and T5-XXL
  (`NFKMLXT5Encoder`) for the sequence, padded to 256 and encoded whole with no attention mask, the way
  diffusers' `FluxPipeline` does. The T5 tokenization is the shared SentencePiece unigram
  (`NFKMLXSentencePieceSegmenter`); FLUX's `tokenizer_2` ships only `tokenizer.json`, so the model reads a
  `spiece.model` (the equivalent `google/t5-v1_1-xxl` one), which the parity below confirms is identical.
  Reference parity against transformers' `CLIPTextModel` and `T5EncoderModel` on the released weights
  (`run_reference.py flux_text`, `IK_PARITY_FLUX_TEXT` / `IK_VAL_FLUX_SCHNELL_FULL`): CLIP-L pooled
  0.999969542751324, T5-XXL sequence 0.9995037142323825. A full generation loads the 24 GB transformer
  beside the encoders, past the working set of a 32 GB machine, so the whole-pipeline image is left to a
  machine that can hold it; the two halves — the text encoding above and the released transformer velocity
  — are each measured.
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

## End-to-end generators (Qwen-Image, Z-Image, SD3, LTX-Video, Wan)

`NFKMLXQwenImageGenerator`, `NFKMLXLTXVideoGenerator` and `NFKMLXWanVideoGenerator` are the release-
directory facades over the three pipelines, each two stages under `NFKMLXResidency` (text encoder; then
transformer and autoencoder). Each pipeline's GLUE is measured against the reference pipeline at a tiny
random geometry, with the models measured separately: `run_reference.py ltx_pipeline` and
`wan_pipeline` (the `ltx` env, diffusers 0.36), keys `IK_PARITY_LTX_PIPELINE` / `IK_PARITY_WAN_PIPELINE`,
tokenized through `IK_VAL_T5_TOKENIZER` (a FLUX.1 release's `tokenizer_2/`). The released components are
held by shape (`IK_SHAPES_ROOT`: `wan21-t2v-1.3b-*`, `wan22-ti2v-5b-*`, `ltx-video-*`), 0 missing /
unconsumed / mismatched across all seven. Qwen-Image ran end to end on its released weights, staged on
32 GB (a red fox in snow at 256×256, 4 steps, 29 s).

What the glue had wrong or lacked before the oracle, all invisible to a shape test:

- LTX's pipeline passes `linspace(1, 1/steps, steps)` as `sigmas` (diffusers 0.36), where the older
  `.ltxVideo` preset mirrors the scheduler's own default ramp. `.ltxVideoPipeline` is the pipeline's;
  `.ltxVideo` stays as the scheduler's, which its own test measures.
- LTX masks the caption's padding in cross-attention as an additive `(1 − mask)·−10000`; the T5 encode
  itself is unmasked. The rotary scale is `(8/frame_rate, 32, 32)`, not ones. The decode denormalizes
  by the VAE's `latents_mean`/`latents_std` BUFFERS (in the checkpoint) over `scaling_factor`.
- Wan runs umT5 WITH the attention mask (`(1 − mask)·float32.min` on the position bias), cuts the
  features at the prompt length, and zero-pads to 512; the transformer takes no mask. The decode is
  `latent · std + mean` with the config's std (the reference writes `latent / (1/std)`), and the flow
  shift is per release (3.0 for 2.1 1.3B, 5.0 for 2.2 5B).
- Wan 2.2 TI2V's `expand_timesteps` gives every token the same timestep in text-to-video; the
  reference's two forms differ by 1.9e-6, so the scalar timestep serves both.
- `prompt_clean` runs `ftfy.fix_text`, which no oracle env carries and the port does not implement; it
  changes only mis-decoded text. The oracle stubs it to identity and says so.

Z-Image (`NFKMLXZImageGenerator`, `run_reference.py z_image_pipeline`, key `IK_PARITY_Z_IMAGE_PIPELINE`,
tokenizer `IK_VAL_QWEN3_4B`): glue exact to double-precision noise (final latents 0.9999999999973, image
0.9999999999999971; the guided and unguided records are 0.9958 apart, which the test asserts). What the
glue had wrong before the oracle:

- The `.zImage` preset used dynamic shifting (base 0.5, max 1.15). Both releases state
  `use_dynamic_shifting: false`, shift 3.0 (Turbo) and 6.0 (base), and the pipeline sets the
  scheduler's `sigma_min` to 0 before building the schedule, so the ramp ends at 0 and the last step
  moves nothing (`rampEndsAtZero`). The pipeline still computes `mu` and passes it; the static scheduler
  ignores it. Turbo's published "9 steps" is 8 transformer evaluations for this reason.
- Guidance applies only above a scale of 1 (`do_classifier_free_guidance`), with an empty negative prompt
  by default; the pipeline guided whenever a negative embedding was given. `cfg_normalization` and
  `cfg_truncation` default off and are not ported.
- The text step is `apply_chat_template(…, add_generation_prompt=True, enable_thinking=True)`, which adds
  nothing after `<|im_start|>assistant\n`; `hidden_states[-2]` is `layerStates` second to last (no final
  norm); the right-padded, masked batch is cut to each prompt's tokens, so the unpadded encode is equal.
- The text encoder ships as `Qwen3Model` in `model_index.json` but its shards carry `Qwen3ForCausalLM`'s
  `model.` keys and a tied head, so the ordinary language loader reads it.
- Z-Image-Turbo's transformer shards are float32 (24.6 GB). The generator loads them at bfloat16 through
  `NFKMLXReleaseWeights.arrays(inDirectory:converting:)`, which evaluates in ~256 MB groups; the planner
  weighs the stage with `NFKMLXStageWeights.bytes(inDirectory:holding:)`.

Stable Diffusion 3 (`NFKMLXSD3Generator`, `run_reference.py sd3_pipeline`, key `IK_PARITY_SD3_PIPELINE`,
tokenizers `IK_VAL_CLIP_TOKENIZER` (a FLUX.1 release's `tokenizer/`), `IK_VAL_CLIP_BANG_TOKENIZER` (the store's
`clip-bang-tokenizer/`, copied from SD 2.1's `tokenizer/`: the CLIP vocabulary padded with `!` as SD3's `tokenizer_2/`), `IK_VAL_T5_TOKENIZER`):
glue exact to double-precision noise. Notes:

- The joint sequence is `[77 + 256, 4096]`: CLIP-L's and bigG's penultimate states (`hidden_states[-2]`,
  no final norm) concatenated on channels, zero-padded to T5's width, then T5's unmasked sequence. The
  pooled projection is both towers' `text_embeds`, read at the first end marker (the max id, the
  reference's legacy `eos_token_id == 2` argmax).
- Guidance applies above a scale of 1 only, `uncond + g·(cond − uncond)`; the pipeline had guided
  whenever negatives were given. Skip-layer guidance defaults off and is not ported.
- The schedule is the scheduler's own ramp (the pipeline passes no `sigmas`), so a static shift is shifted
  twice at the end, which `.sd3` already mirrors; the reader builds the same configuration from
  `scheduler_config.json`.
- The oracle needed a text randomization scale of 0.5: at 0.1 the towers encode both prompts almost alike
  and the guided and unguided final latents agree to 1e-6, which no bar can tell apart. At 0.5 they are
  0.938 apart and the test asserts the separation.
- SD3's gated configs did not come back as JSON through `resolve` with this machine's token (2026-09-24),
  so the readers take every field from the release's own files rather than from presets.

The umT5 vocabulary is measured on its own: the release's 256k `spiece.model` through the port's
SentencePiece reader against the fast tokenizer the pipeline names (`run_reference.py umt5_tokenizer`,
`IK_VAL_UMT5_TOKENIZER` / `IK_PARITY_UMT5_TOKENIZER`), 9 of 9 prompts token-exact across Latin, CJK,
Cyrillic, accents, emoji, digits and runs of whitespace, padded and masked to 512. Unmeasured:
LTX-Video / Wan end to end on released weights (not on disk). The
0.9.1+ LTX autoencoder's decode timestep is refused, not ported. `framesForPrompt:` returns `[Any]`: an
`@objc` `[CGImage]` exports as `NSArray<CGImageRef>`, which clang rejects in the generated header.

