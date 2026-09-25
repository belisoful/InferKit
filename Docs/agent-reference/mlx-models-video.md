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
  anywhere between the frames, not only the midpoint. Its convolutions activate with a parameter-free
  leaky ReLU where HDv3 used a PReLU; that difference surfaced as exactly eight uncovered parameters,
  the coverage guard naming a structural mistake rather than a number going quietly wrong. `+register`
  under `rife-v4`; pads to a multiple of 64 and runs scales `[8, 4, 2, 1]`. Oracle: the architecture
  vendored by ComfyUI-Frame-Interpolation (`rife_arch.py`, `arch_ver="4.17"`), whose own `IFNet.py`
  ships inside the model zip rather than in the repository; only one ComfyUI device helper needs
  stubbing.
  Customization: trainable at `full`, with no recipe written yet. hzwer/Practical-RIFE links the v4
  training code from its README as Google Drive archives; v4.12 and v4.15 are kept under
  `~/.inferkit-validation/reference-sources/practical-rife-train`. The loss is a VGG19 perceptual term
  (torchvision's ImageNet weights) minus 0.1 × SSIM, plus 0.1 × the L1 of every scale's merge (0.05 in
  v4.15), 0.1 × a teacher term, and a flow-magnitude term. The teacher is the confidence-weighted blend
  of the student's own per-scale flows, and the released blocks already emit the confidence channel.
  AdamW with weight decay 1e-2 at a base rate of 1e-4, a 2,000-step linear warm-up then cosine to zero,
  batch 16. A recipe needs VGG19 features beside the VGG16 port.
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
  0.9999999999999837, reference pass 0.9999999999999781, generation pass 0.9999999999999792, chunked
  generation pass 0.9999999999999815 — and the released `wan_animate_2_bf16.safetensors` header holds
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
  keeps the best on validation; the defaults are the first, 5e-3 and 0.01. Its bfloat16 autocast and
  gradient scaler are a mixed-precision device the float32 recipe does not need.
  `NFKMLXVJEPA2Processor.clip(frames:configuration:)` is the data adapter, and
  `NFKMLXVJEPA2.save(_:toDirectoryURL:)` writes a directory (`model.safetensors`, `config.json` with
  `id2label`, `video_preprocessor_config.json`) that the `@objc` `backendWithDirectoryURL:` loads.
  Measured against the reference's own code (`run_reference.py vjepa2_probe`, `IK_PARITY_VJEPA2_PROBE`,
  the reference files under `~/.inferkit-validation/vjepa2-ref`): the loss on identical logits
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
  with the reference's AdamW (1e-4, betas 0.5 / 0.999, weight decay 0.01, bias-corrected) and its
  5,000-step linear warm-up (`WarmupLambdaLR`). The flow and consistency terms, which the reference disables for
  post-training, are not ported. Oracles: `run_reference.py cosmos_tokenizer` (a release directory named
  for its variant, or its `autoencoder.jit`) and `cosmos_tokenizer_loss` (the VGG-16 file), under the
  `llm` env with the `cosmos_predict1` sources pinned in the manifest.
