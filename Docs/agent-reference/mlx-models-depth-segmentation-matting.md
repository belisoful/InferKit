<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX models: depth, segmentation, and matting

- `NFKMLXDepthAnything` (`@objc`) — a real single-forward depth model: the Depth Anything V2 DINOv2 ViT
  encoder (`pretrained.*`) + DPT head (`depth_head.*`) in `MLXNN`, run through `NFKMLXModuleBackend`
  (image → grayscale depth under `NFKOutputImage`). `+register` under `depth-anything-v2-small`;
  `NFKMLXDepthConfiguration` holds the ViT-Small dims (Base/Large change them). The encoder runs a fixed
  518×518 (so `pos_embed` matches without interpolation) and the map resizes back. `loadWeights(into:from:remap:)`
  loads a **safetensors** checkpoint; the DPT key layout is intricate, so `Tools/depth-anything-to-safetensors/convert.py`
  is self-validating (matches every key against the module's expected layout, reports mismatches).
  Reference parity across all three released sizes: Small 0.99992 (encoder seam 0.9999924), Base
  0.99995, Large 0.99984, on the min-max-normalized 8-bit depth map. Two DPT-head fixes found while
  porting Depth Anything 3 raised these from ~0.998: the two `resize_layers` are `ConvTransposed2d`,
  whose PyTorch weight is `[C_in, C_out, kH, kW]` and needs the transposed-conv axis order `(1,2,3,0)` →
  `[C_out, kH, kW, C_in]`, not a regular convolution's `(0,2,3,1)` (both are square, so the wrong order
  loaded silently and scrambled the kernel, a small effect on the normalized map); and the
  FeatureFusionBlock upsamples bilinear with align_corners=true (`NFKMLXResample.resizeBilinearAlignCorners`),
  not nearest, the dominant fix. The oracle drives the authors' own `depth_anything_v2` package
  (`IK_DEPTH_VARIANT` picks the encoder config); it drove `transformers` until that package dropped the
  `depth_anything` model type, and the parity test kept passing throughout because it compares against a
  stored record, not a live oracle.
- `NFKMLXDepthAnything3` (`@objc`) — Depth Anything 3 monocular depth and camera estimation (DA3-SMALL).
  The whole released model is built: the DINOv2 ViT backbone, both branches of the DualDPT head, the
  camera decoder, and the camera encoder. Every released tensor loads, on all three sizes (437 for Small
  and Base, 637 for Large). Reference parity against the authors' `depth_anything_3` package on the
  released weights: the four hooked backbone features ≥ 0.9999999999962167, the exp-depth map
  0.9999999999999011 (mean-removed 0.9999999999101674), the ray map 0.9999999999998513 and its
  confidence 0.9999999999995774, the four camera tokens ≥ 0.9999999999950343, the predicted pose
  encoding 0.9999999999992621, and the camera encoder's pose encoding 0.9999999999998606 and tokens
  0.9999999999999734. The backbone is a DINOv2 ViT variant rather than a reuse of the V2
  encoder: from block 4 it adds a 2-D rotary embedding, per-head query/key normalization, a learned
  camera token injected into the class-token slot, and alternating local/global attention whose
  cross-view "global" blocks collapse to query/key-normalized self-attention for a single image (their
  uniform rotary positions cancel in the score). Each hooked feature concatenates the preceding local
  block's output with the current global block's (`cat_token`), the final norm applied to the global
  half only, so the `DualDPT` head reads twice the embedding width (`dim_in` 768). The head adds a UV
  positional embedding and upsamples bilinear with align_corners, and the output convention is exp-depth
  (`exp(logits)`), not V2's relative disparity.
  The main branch of the head predicts depth. The aux branch runs its own four fusion blocks
  (`refinenet1_aux` … `refinenet4_aux`) over the same reassembled pyramid, then a per-level neck
  (`output_conv1_aux`: five 3×3 convolutions alternating between the feature width and half of it, with
  no activation between them) and a per-level head (`output_conv2_aux`: a convolution, a channel
  LayerNorm, a ReLU, and a 1×1 to seven channels). Inference reads the finest level only. The ray map is
  six Plücker channels plus a confidence channel, and it stays at the finest fusion resolution rather
  than being interpolated to the image size, which is the reference's own convention.
  The aux head carries a **shared module instance**: the reference builds its `ln_seq` list once and
  splices that same list into all four `output_conv2_aux` Sequentials, so a single `nn.LayerNorm`
  instance serves the four levels. PyTorch deduplicates a shared module in a state dict and saves it
  once, under `output_conv2_aux.0.2`. Reading that absence as "unlearned, left at the `nn.LayerNorm`
  init" is wrong and quiet: the ray head is level 3, and the identity affine scored its ray logits at
  0.9991 against the reference's 0.9999999999. The loader copies level 0's parameters onto the other
  three.
  The camera decoder (`cam_dec`) runs two hidden layers over the last hook's camera token, then heads for
  the translation (3), the rotation quaternion (4, scalar-last `xyzw`), and the field of view (2, ReLU).
  The camera token is the concatenated local and global halves taken before the final LayerNorm. The
  camera encoder (`cam_enc`) is the conditioning path in the other direction: a pose branch MLP
  (9 → dim/2 → dim), a token norm, a four-block trunk, and a trunk norm turn a known camera into the
  token the backbone reads in place of its learned one.
  The reference carries **two Block implementations with different LayerNorm epsilons**. The backbone's
  (`dinov2/layers/block.py`) defaults `ln_eps` to 1e-6; the camera encoder's (`utils/block.py`) takes a
  plain `nn.LayerNorm`, so torch's 1e-5. The epsilon is a parameter here because both are built.
  `loadWeights` reads the released safetensors directly
  (`model.backbone.pretrained.*`, `model.head.*`, `model.cam_dec.*`, `model.cam_enc.*`); the two
  `resize_layers` take the transposed-conv axis order `(1,2,3,0)` (the same load-bearing detail as the V2
  port). `depth()` applies the pipeline's ImageNet normalization. `+register` under
  `depth-anything-3-small`. `NFKMLXDepth3Estimator` (`@objc`) carries what a single-image backend
  cannot: `estimatorWithVariant:weightsURL:error:` and its
  `estimatorWithVariant:repo:weightsPath:revision:cacheDirectoryURL:error:` peer build it,
  `cameraForImage:error:` returns an `@objc` `NFKMLXDepth3Camera` (translation, row-major rotation, focal
  lengths in pixels at the image's own size, fields of view), and
  `cameraForImage:knownRotation:translation:focalLengthX:focalLengthY:error:` runs the camera-encoder
  conditioning path. `rays(for:)` returns the ray map and its confidence as `MLXArray`s, so it is
  Swift-only under the parity rule. The depth path is unchanged
  (`NFKMLXDepthAnything3.backend(variant:weightsURL:)`). The oracle is reproducible in-repo
  (`run_reference.py depth3`, the `da3` oracle env, `IK_REF_SRC` = the unpacked `depth_anything_3`
  wheel), recording the input, the four hooks, the four head stages, the fused map, the pre-exp logits,
  the depth, the aux pyramid, the ray map and its confidence, the four camera tokens, the pose encoding,
  and the camera encoder's input and tokens; the parity test compares every seam by cosine plus
  mean-removed correlation (raw cosine on the near-constant ~1.0 depth is misleading), and a coverage
  test asserts every released tensor is loaded. Weights: `depth-anything/DA3-SMALL` (Apache-2.0, ~80M).
  Base and Large are also at reference parity (`NFKMLXDepth3Configuration.base` / `.large`,
  `NFKMLXDepth3Variant`, registered as `depth-anything-3-base` / `-large`): Base is the same recipe at
  ViT-B (768 wide, 12 heads, DPT features 128); Large is ViT-L (1024 wide, 24 blocks, 16 heads) hooked at
  blocks 11/15/19/23 with the query/key norms starting at block 8 rather than 4, and DPT features 256.
  Base: hooks ≥ 0.999999999994421, depth 0.9999999999999064 (mean-removed 0.999999999990152), ray
  0.9999999999998437 with confidence 0.9999999999985674, pose encoding 0.9999999999995328, camera encoder
  0.9999999999999702. Large: hooks ≥ 0.999999999997466, depth 0.9999999999998904 (mean-removed
  0.9999999999835174), ray 0.999999999999276 with confidence 0.9999999999926193, pose encoding
  0.9999999999993067, camera encoder 0.9999999999999863. `DA3-LARGE` is CC-by-NC.
- `NFKMLXU2Net` (`@objc`) — a real single-forward background remover: the U²-Net nested-U saliency
  network (Residual U-blocks) in `MLXNN`, run through `NFKMLXMattingBackend` (plate → straight
  foreground + saliency alpha, matte under `NFKOutputMask`). `+register` adds full `u2net` and light
  `u2netp`. Stage/side/`outconv` names match the reference; the RSU-internal convs are `enc`/`dec`
  arrays, and `Tools/u2net-to-safetensors/convert.py` renames `rebnconvN` → `enc`/`dec` so the file
  loads directly. Forward + matting round-trip tested under xcodebuild with the light config.
  Reference parity against U²-Net's own network on both releases (the full network 0.9999992900356037
  with mean absolute difference 7.263675130994506e-05, `u2netp` 0.9999997057950919); the light model is
  a separate class in the reference rather than a configuration of the full one. The loader calls
  `train(false)` after applying the weights: MLXNN modules start in training mode, and a `BatchNorm`
  left there normalizes over the plate instead of reading its released running statistics, which cost
  three digits of parity before it was fixed.
- `NFKMLXISNet` (`@objc`) — the IS-Net dichotomous segmentation network (`ISNetDIS`, the DIS project),
  U²-Net's successor by the same authors, run through `NFKMLXMattingBackend` and registered as `isnet`.
  The Residual U-blocks are U²-Net's, so `NFKU2NetRSU` and `NFKMLXU2Net.remapReferenceKey` carry over
  unchanged and a released `.pth` loads directly. Three things differ: a stride-2 `conv_in` stem with no
  norm and no activation halves the plate before stage 1, the stages are wider (32/32/64/128/256/256 mid
  channels over 64/128/256/512/512/512 out), and the six side maps stay separate with no fusion
  `outconv`, so the prediction is the first side map. The reference resizes to 1024x1024, scales to
  `0...1`, normalizes with mean 0.5 and unit standard deviation, and min-max stretches the returned map.
  Reference parity against the DIS `isnet.py` on `isnet-general-use.pth`: stem 0.9999999999990618,
  stage 1 0.9999999999987647, stage 6 0.9999999999996586, and every side map ≥ 0.9999999999999148.
- `NFKMLXBiRefNet` (`@objc`) — high-resolution background removal (`ZhengPeng7/BiRefNet`, MIT), run
  through `NFKMLXMattingBackend` and registered as `birefnet`. Three parts: a **Swin-v1-L backbone**
  that reuses SwinIR's window attention, a neck that concatenates a downscaled second view
  (`mul_scl_ipt='cat'`) with a context stack, and a decoder whose `ASPPDeformable` blocks run a
  **modulated deformable convolution**. That convolution reduces to the same bilinear-gather
  (`takeAlong`) primitive RAFT, RIFE, RVM and RT-DETR's deformable attention already use, so it needed
  no DCNv2 Metal kernel. The plate is resized to 1024 and ImageNet-normalized. Two facts are
  load-bearing. The decoder's `gdt` attention gating is **active at inference**, not a training-only
  branch as its siblings `pred`/`label`/`ms` are. And the model must be put in evaluation mode after
  loading, or `BatchNorm` normalizes over the plate — the backbone's LayerNorm masks the error, so it
  reads as a plausible matte rather than a broken one. Reference parity on the released weights, every
  seam: neck x1 0.9999999999975541, x2 0.999999999997728, x3 0.9999999999869097, x4 context
  0.9999999999948102, x4 squeezed 0.9999999999989743; decoder p4 0.9999999999997091, p3
  0.9999999999994882, p2 0.9999999999996263, p1 0.9999999999995682, logit 0.9999999999998718; and the
  assembled encoder-to-decoder chain 0.9999999999996281. The lite and other released variants need the
  hardcoded channel widths generalized and are left out.
- `NFKMLXSAM` (`@objc`) — real promptable segmentation (Segment Anything): a ViT image encoder, a prompt
  encoder (point → sparse tokens via a random-Fourier positional encoding), and a two-way-transformer
  mask decoder with a hypernetwork mask head, in `MLXNN`. Run through `NFKMLXMattingBackend` (plate +
  point under `NFKSAMPointKey` → mask alpha + matte). `+register` under `sam`. The ViT encoder uses real
  windowed attention (`windowSize`, `globalAttnIndexes`) with decomposed relative-position embeddings
  (`rel_pos_h`/`rel_pos_w`, added via `take` gather + batched matmul). `remapReferenceKey` maps the
  reference's nested MLP, positional neck/upscaling Sequentials, and `transformer` submodule; scope the
  `.mlp.lin` rule to the encoder, or it eats the decoder's. `NFKMLXSAMVariant`
  (`.compact`/`.vitB`/`.vitL`/`.vitH`) selects the geometry on both the local and the download
  factories; a released checkpoint fits only its own size. Reference parity against the official
  `segment-anything` predictor on ViT-B (encoder cosine 0.9999986, selected-mask cosine 0.99993, binary
  agreement 99.7%); the decisive fix was `skip_first_layer_pe` in the two-way transformer's first layer.
  ViT-L and ViT-H are also at parity (`NFKMLXSAMConfiguration.vitL`: 1024 wide, 24 blocks, 16 heads,
  global attention at 5/11/17/23; `.vitH`: 1280 wide, 32 blocks, global at 7/15/23/31): encoder
  0.99999799 / 0.99999669, binary mask agreement 1.0 on both. SAM 2 is `NFKMLXSAM2` below.
  Segment + round-trip tested under xcodebuild.
  `NFKMLXSAM2` ports SAM 2's Hiera image encoder, at reference parity against facebookresearch's own
  sources (finest FPN level 0.9999999999986, second 0.9999999999918, vision features 0.9999999999939),
  every parameter covered on the first triage run. Hiera is hierarchical where SAM's ViT is flat: four
  stages that halve the resolution and double the width, attention inside local windows except at
  designated global blocks, and stage transitions that max-pool the queries so one block changes both
  size and width (the shortcut takes the same projection and pooling so the residual still lines up). A
  transition block keeps the previous stage's window (the reference reads `window_spec[cur_stage - 1]`
  before advancing the stage), which is what makes block 10 use window 14 rather than 7; reversing that
  order makes the pooled windows the wrong size and the reassembly stops tiling. The position grid is
  resampled bicubically (`NFKMLXBicubic`, PyTorch's `a = -0.75` Keys kernel with half-pixel centers and
  border clamping) and a tiled window grid is added. The FPN neck projects each captured stage to 256
  and fuses top-down on the deeper levels only, dropping the coarsest (`scalp`).
  The prompt encoder and mask decoder are ported too, at reference parity (sparse prompt
  0.99999999999998, mask logits 0.9999999999950, object score matching to five figures), every
  parameter covered on the first triage. The decoder is SAM's two-way transformer (attention, block, and
  MLP shared with `NFKMLXSAM`) plus three SAM 2 additions: an object-score token leading the sequence
  with its own head, and high-resolution features from the FPN's two finer levels added during
  upscaling. Their `conv_s0`/`conv_s1` projections are the decoder's parameters but the reference
  applies them in its base model before calling it, so this port applies them internally and takes the
  levels as they come off the neck. Two traps: the query positional term is the original token embedding
  at every layer and again at the final attention (the running queries drift the masks without breaking
  anything visibly), and the reference shifts a click by half a pixel to the pixel's centre before
  normalizing.
  The video memory path is ported too, which completes the checkpoint: `NFKMLXSAM2MemoryEncoderNet`
  folds a frame's features and its predicted mask into a 64-channel memory, and
  `NFKMLXSAM2MemoryAttentionNet` conditions the next frame on that memory. Both are at reference parity
  (encoded memory cosine 0.9999999999999, attention output 0.9999999999997), every parameter covered on
  the first triage run. The encoder's mask downsampler is four stride-2 stages
  (1 → 4 → 16 → 64 → 256 channels), so it takes the mask at the full 1024 frame resolution, not at the
  decoder's low-resolution output; the tracker upsamples before encoding, and the total stride of 16
  lands it on the 64×64 feature grid. Its fuser blocks are ConvNeXt (7×7 depthwise, channel LayerNorm, a
  4× pointwise MLP, and a learned per-channel `gamma` scale). The attention applies axial rotary
  embeddings (`NFKMLXAxialRotary`): adjacent channel pairs are rotated, the first half of the pairs by
  an x-frequency and the second half by a y-frequency, matching the reference's `view_as_complex`
  layout. Its positional-encoding switches are asymmetric and each one matters: self-attention and
  cross-attention queries take no positional term, cross-attention keys do, and the input takes
  `0.1 × position` once. The trap is the reference's `batch_first=True`, which describes what its layers
  want: `MemoryAttention` takes its inputs sequence-first and transposes them itself, so handing it
  batch-first tensors makes the tokens the batch and every token attends only to itself (that scored
  0.86 and raised nothing). Checkpoint
  `dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt` (161 MB, 468 tensors, 39M
  parameters), of which 122 are the video path (`memory_attention`, `memory_encoder`); an image-only
  port needs the other ~250: `image_encoder` (154 trunk + 8 neck), `sam_prompt_encoder` (10), and
  `sam_mask_decoder` (~118). The `sam2` package cannot be installed here (it requires Python ≥ 3.10; this
  environment is 3.9), and the installed `transformers` is 4.33.3, which has no SAM 2, so the oracle
  vendors `backbones/{hieradet,image_encoder,utils}.py`, `position_encoding.py`, `sam2_utils.py`,
  `memory_attention.py`, `memory_encoder.py`, and `utils/misc.py`, all of which parse under 3.9, with
  `iopath` and `sam2.utils.misc` stubbed. The tiny config (embed 96, one head, stages `(1, 2, 7, 2)`,
  global attention at blocks 5/7/9, background window 7×7; neck `d_model` 256 over channels
  `[768, 384, 192, 96]`, top-down levels `[2, 3]`, nearest interpolation, `scalp` 1) loads strictly, and
  a 1024×1024 forward returns `vision_features [1, 256, 64, 64]` with FPN levels at 256/128/64.
  All four released Hiera sizes are at parity: small (`.small`: stages `[1, 2, 11, 2]`, windows
  `[8, 4, 14, 7]`, global attention at 7/10/13; level0 0.99999999999956, level1 0.99999999999791), large
  (level0 0.9999999999994, level1 0.9999999999986; 48 blocks against tiny's 12, weighted toward the
  third stage, a coarser window there, and global attention much later, its config overriding every axis
  the Hiera constructor defaults), and base_plus (level0 0.9999999999958, level1 0.9999999999895), which
  sets only the width and head count and takes the rest from the defaults, so it is the size that proves
  the defaults rather than the overrides. The oracle's `sam2` package is now in the manifest; it cannot
  be pip-installed here and had been vendored by hand, leaving that oracle unreproducible.
  **The tracker is ported, which is what completes the checkpoint.** `NFKMLXSAM2TrackerNet` holds the
  five networks under their released prefixes and the dozen parameters `SAM2Base` owns — the
  `no_mem_embed` that stands in for an empty memory, the `maskmem_tpos_enc` that tells one remembered
  frame from another, `obj_ptr_proj` and the `no_obj_ptr` an absent object falls back to, and the
  `mask_downsample` of the mask-as-output path — so one released file loads whole and strictly, and
  `track(image:frameIndex:points:session:)` follows a clicked object across a clip through
  `NFKMLXSAM2TrackerSession`. Every released size loads with nothing left over: 468 tensors for 2.0
  tiny, 516 small, 612 base_plus, 900 large, and 471 for 2.1 tiny. The prompt encoder gained the two
  box-corner embeddings and `mask_downscaling`, which the earlier port dropped, so a box or a mask is
  now a prompt the model can take.
  **SAM 2.1 changes no geometry**: `facebook/sam2.1-hiera-{tiny,small,base-plus,large}` is Apache-2.0
  and ungated, and its two flags each bring one parameter — `no_obj_embed_spatial`, added to a memory
  whose frame the model reads as empty, and `obj_ptr_tpos_proj`, which projects a sine encoding of how
  many frames back a pointer came from. `NFKMLXSAM2Release` selects 2.0 or 2.1, because a checkpoint
  carries exactly the parameters its own release declares and the difference is invisible to shapes.
  Reference parity against transformers' `Sam2VideoModel` on the released sam2.1-hiera-tiny, over a
  three-frame clip driven by one click: mask logits 0.99999999999806 / 0.99999999945 / 0.99999999985,
  the stored memory 0.999999956 / 0.999999992 / 0.999999988, and the object pointer
  0.99999999999778 / 0.99999998951 / 0.99999999523. The first frame is the encoder, prompt encoder,
  and decoder alone; the frames after add the memory encoder, memory attention, and the pointers.
  Three traps, each caught by the per-frame comparison rather than predicted:
  - `maskmem_tpos_enc` is `[frames, 1, 1, width]` and 4-D, so a loader that transposes every 4-D
    tensor as a convolution weight turns it into `[frames, 1, width, 1]`. Nothing fails at load; the
    first frame is exact and the second one cannot broadcast.
  - `get_1d_sine_pe` is not the 2-D grid encoding with one axis. It holds each frequency for two
    channels (`temperature ** (2·⌊i/2⌋/half)`) and puts the sines first, where the grid encoding
    interleaves them.
  - The object pointers' temporal span is `min(clip length, 16) - 1`, not the configured 16. A short
    clip normalizes its distances over its own length, so a tracker that uses the bound reads every
    pointer as nearer than it is. That is what `NFKMLXSAM2TrackerSession.frameCount` carries.
  **Customization is a HEAD RETARGET and it ships** (`NFKMLXSAM2Training.swift`): the prompt encoder
  and mask decoder train against a consumer's own masks with the Hiera trunk and the whole memory
  path frozen, which is about 5M parameters of the tiny release's 39M. `network(weightsURL:…)`
  builds, `fineTune(_:examples:trainable:…)` trains, `NFKMLXWeights.save` writes a file the `@objc`
  factory loads. Nothing is dropped on load the way a retargeted classifier is, because SAM 2's masks
  are class-agnostic: a consumer's subject needs a trained head, not a new one.
  The objective is the reference's `MultiStepMultiMasksAndIous` at the weights its own fine-tuning
  configuration sets (mask 20, dice 1, IoU 1, class 1, every IoU supervised, IoU by L1), and it
  matches term by term — mask 0.67431545 against 0.6743155, dice 0.6716747 against 0.6716747, IoU
  0.36395273 against 0.36395273, class 0.40081817 against 0.40081823. Three of its details are the
  kind a paper does not state: the focal and dice terms backpropagate ONLY through the multimask slot
  whose combined loss is lowest, the IoU head is supervised against the IoU its own mask achieves
  rather than a label, and every mask term is multiplied by whether the target holds an object, so an
  empty frame trains the object score and nothing else. The slot choice needs `stopGradient` on the
  indices, which PyTorch gets for free from `argmin` returning integers and MLX does not.
  What does NOT ship is the video objective, which backpropagates through the memory bank across a
  clip; a device holds the conditioning frame's path, which is what this trains.
  The default optimizer is the configuration's too: `torch.optim.AdamW`, bias-corrected, at the base
  learning rate 5e-6, with weight decay 0.1 on every parameter but those named `*bias*` and those of a
  `torch.nn.LayerNorm`. The mask embedder's two norms and the decoder's upscaling norm are `LayerNorm2d`
  there, so their weights decay. Under `.everything` the image encoder trains at 3e-6, and each trunk
  parameter is scaled by Hiera's `get_layer_id` under a decay of 0.9 (`pos_embed` exempt). The
  gradient clip is 0.1, and the rate follows fvcore's cosine to a tenth over the run.
  The round-trip test earned its place immediately: the prompt encoder's remap stripped `.weight`
  from every name under it, which is right for the four embeddings the reference stores as
  `nn.Embedding` and wrong for the mask embedder's convolutions, so a fine-tuned checkpoint could not
  reload. Nothing else would have caught it, because a released file never carries those names.
  Registered as `sam2` through `NFKMLXMattingBackend`, with the full `@objc` factory set taking both
  the variant and the release. A single plate with a click goes through the tracker's first frame with
  a fresh session. `no_mem_pos_enc` is loaded and unused: the releases set
  `directly_add_no_mem_embed`, under which the reference never reads it either.
  **SAM 3's two encoders are ported and at released-weight parity; its detector is not.** SAM 3 names
  its targets in words rather than pointing at them, so a prompt is text and the model segments every
  instance that text names. `facebook/sam3` and `facebook/sam3.1` stay gated behind Meta's SAM
  License, which this project accepts, and both answer 200 for its token. The two ship a
  BYTE-IDENTICAL `config.json`; 3.1's difference is its checkpoint. The release is one 3.44 GB
  float32 file of 1797 tensors, which this machine holds, so the stages are measured on the released
  weights rather than at a tiny configuration.
  `NFKMLXSAM3VisionNet` is the vision encoder (538 tensors): a 32-layer ViT 1024 wide over 14-pixel
  patches at 1008, and an FPN neck that reads its ONE output map at four scales
  (4, 2, 1, 0.5 → 288, 144, 72, 36), each projected to 256. Three things separate it from the SAM 1
  ViT it succeeds. Position is 2-D rotary over adjacent channel pairs, not a learned relative bias.
  The learned absolute grid is TILED to the input rather than interpolated, so a 24×24 pretraining
  grid covers a 72×72 input by repeating and cropping. And the layer normalization that usually ends
  a ViT runs BEFORE the block stack.
  The windowing is what makes the rotary table per layer rather than per model: a windowed layer
  attends inside a 24×24 window and rotates over a 24×24 grid at unit scale, while a global layer
  (7, 15, 23, 31) attends over the whole map and rotates over it at `window / grid`, so a position
  means the same distance in both. A port that shares one table across the stack is correct for one
  kind of layer and wrong for the other. The reference builds the global layers' table from the
  CONFIGURED image size, not the input's, which is why `NFKMLXSAM3Configuration.imageSize` has to
  match the plate.
  `NFKMLXSAM3TextNet` is the prompt side (391 tensors): a 24-layer causal CLIP text tower 1024 wide
  over a 49408-token vocabulary with a 32-position context, the exact gelu rather than CLIP's usual
  quick approximation, and epsilon 1e-5. Two projections leave it, and only one is read — the tower's
  own 512-wide `text_projection` is CLIP's contrastive head, which SAM 3 never calls, and the
  detector's 1024 → 256 projection reaches EVERY token rather than the pooled end-of-text one.
  Reference parity against transformers' `Sam3VisionModel` and `CLIPTextModelWithProjection` on the
  released weights, both on the first numeric run: the ViT 0.9999999999787388 and the FPN levels
  0.999999999991736 / 0.9999999999829633 / 0.9999999999852291 / 0.9999999999862851; the text tower
  0.9999999999997959 and the projected prompt 0.9999999999990974. The vision parity runs at 504
  pixels rather than the released 1008, which is what exercises both edges the full size hides: the
  position grid is tiled and cropped because 36 patches is not 24, and 36 is not a whole number of
  24-wide windows, so a windowed layer pads and unpads.
  `NFKMLXSAM3DetectorNet` is the detector (445 tensors), and `NFKMLXSAM3ImageModel` chains all three
  so a worded prompt goes in and every instance it names comes out. Four stages run in order. The
  DETR encoder fuses ONE vision level — the coarsest the detector is given, 72×72 at the released
  size — with the prompt over six layers of self-attention and cross-attention. The decoder runs 200
  learned queries and a presence token over that, refining one box per query at every layer under a
  relative-position bias built from the current boxes, so a query attends around where it currently
  points. The scoring head dots each query against the mean-pooled prompt. The mask decoder lifts the
  encoder's output back up the pyramid and dots the queries against it, one mask per query.
  Every attention in SAM 3 is DENSE: there is no deformable sampling anywhere, which is what makes
  the detector ordinary matrix arithmetic. Two things about it are easy to miss and both change
  numbers. Its `hidden_act` is RELU where the two encoders use a gelu. And its normalizations are
  constructed WITHOUT an epsilon, so they take PyTorch's 1e-5 default while the configuration's
  `layer_norm_eps` of 1e-6 goes unread — reading it would put the port a thousandfold off the weights
  it loads. A third is structural: `num_upsampling_stages` is 3 and the pixel decoder runs 2, because
  the stages it climbs are one fewer than the levels it is given, so one convolution and one group
  normalization in every release go unused.
  Detector parity against `Sam3Model` on the released weights, driven from the reference's own FPN
  levels and projected prompt: boxes 0.9999999999942162, per-query logits 0.9999999999956046, the
  presence logit exact, masks 0.9999999999608558, the semantic map 0.99999999999742. Chained end to
  end from the plate and the token ids: masks 0.9999999962068635, boxes 0.999999999821788.
  **Customization is a HEAD RETARGET and it ships** (`NFKMLXSAM3Training.swift`): the detector trains
  against a consumer's own boxes with the ViT and the text tower frozen, about 25M parameters against
  the release's 850M. Because the encoders are separate modules their output is computed ONCE per
  image and reused at every step (`encode(image:tokens:valid:using:)`), which is what makes the run
  cheap rather than merely smaller. `NFKMLXSAM3Trainable` offers the whole detector or the decoder
  alone with the DETR encoder frozen too. A FULL fine-tune is offline-only: the 3.44 GB model with
  gradients and optimizer state does not fit 32 GiB.
  The objective is the reference's own at the settings its `odinw_text_only_train.yaml` sets for
  exactly this case — a `BinaryHungarianMatcherV2` (class 2, box 5, GIoU 2, focal alpha 0.25, gamma
  2), then `Boxes` (L1 5, GIoU 2) and `IABCEMdetr` (classification 20, presence 20, positive weight
  5). That configuration sets `enable_segmentation: False`, so there is no mask term to ship and none
  is claimed. It matches term by term — box 0.31639794 against 0.3163979, GIoU 0.86075145 against
  0.86075145, classification 1.7220247 against 1.7220246, presence 0.41047144 against 0.4104715 —
  and the assignment is identical to `scipy`'s `linear_sum_assignment`, which is what
  `NFKMLXHungarian` had to reproduce (MLX has no assignment solver, and a greedy pick is not the
  same answer).
  The default optimizer follows the same configuration: `torch.optim.AdamW`, bias-corrected, weight
  decay 0.1 on every parameter but biases and `torch.nn.LayerNorm` weights (the detector's other norms
  are `GroupNorm`s, which decay there too), at 8e-5, the configuration's transformer rate. The gradient
  clip is 0.1, and the rate follows the reference's inverse square root (timescale 20) with a 20-step
  warm-up and a 20-step cool-down. The backbones' own rates do not apply: the recipe trains from
  precomputed features.
  Two details of `IABCEMdetr` only its code states. A matched query is NOT regressed toward 1: its
  target is `p^alpha · IoU^(1-alpha)` clamped at 0.01, so a query is asked to be exactly as confident
  as its box is good. And an image whose prompt names nothing present contributes no classification
  loss at all, only the presence term, which is how the model is taught to say no.
  **What remains is box prompts and video**, 423 of the release's 1797 tensors: the geometry encoder
  (94), which encodes a box by projection, ROI-pooled features, and the position encoding of its
  centre, and the video tracker with its own neck (329). A text prompt never reaches the geometry
  encoder, which is why the image path is complete without it. The oracle is `wananimatevenv`
  (transformers ≥ 5.16, which carries `Sam3Model`, `Sam3VideoModel`, and `Sam3TrackerModel`); every
  stage loads on its own out of the release, so one can be measured without instantiating the rest.

- `NFKMLXRVM` (`@objc`) — real video matting (Robust Video Matting): the reference `MattingNetwork` —
  a torchvision **MobileNetV3-Large** encoder (inverted residuals with squeeze-and-excitation,
  hardswish, **BatchNorm epsilon 1e-3**, the last stage dilated), the reference LR-ASPP, and a
  recurrent decoder whose ConvGRU runs on half of each stage's channels (one fused gate
  convolution emits reset then update), threading four hidden states across frames. Run through
  `NFKMLXMattingBackend` (single frame) or `NFKMLXRVMNet.forward` (video, state threaded; a
  `downsampleRatio` below one runs the network on a reduced frame and lifts the result through the
  deep guided filter refiner — the reference's high-resolution recipe). The foreground head
  predicts a residual added to the source, and the alpha is a clamp, not a sigmoid. `+register` under
  `robust-video-matting`. `remapReferenceKey` maps the positional `backbone.features.N.block.M`
  Sequentials (what position `M` holds depends on each block's expand/SE shape) and every other
  positional Sequential — module keys are semantic because MLX's `update(parameters:)` parses a
  numeric key as an array index (see `mlx-runtime-gotchas.md`). Reference parity against PeterL1n's
  own MattingNetwork on the released `rvm_mobilenetv3` (alpha cosine 0.9999999999999786, foreground
  0.9999999999996709, guided-filter pass at ratio 0.5: 0.9999999999998509); the parity plate must
  produce a **non-degenerate alpha** — the network returns an all-zero matte on a synthetic ellipse,
  and a cosine of zero vectors measures nothing (`run_reference.py --image`). A photograph of a real
  subject is what satisfies that; the shipped plate is an animal rather than a person, and its
  reference alpha reaches full opacity over about 8% of the frame, so the recorded numbers are a real
  measurement. An earlier note said the plate must contain a person, which is stricter than the
  requirement. Forward, recurrent-state carry,
  guided-filter shape, remap, and round-trip tested. The fine-tune question is answered by
  measurement: `testAFineTuneMovesTheSqueezeExciteAndHardswishBlocks` trains the tiny configuration —
  which carries every block form — and asserts the loss falls and the squeeze-excitation parameters
  move, because a decreasing loss alone could ride on the decoder while the backbone stays frozen.
  The ResNet-50 release is at parity too (`.resNet50`, `NFKMLXRVMVariant`, registered as
  `robust-video-matting-resnet50`): the shared `NFKMLXResNetBackbone` with its last stage dilated,
  tapped after the stem's ReLU, stage 1, and stage 2 (`taps(_:)`), LR-ASPP 2048 → 256, decoder
  `[128, 64, 32, 16]`, and no ImageNet normalization on this encoder — the reference applies it to
  MobileNetV3 alone. `encode(_:)` dispatches on the backbone; the loader remaps `backbone.` through the
  ResNet's own `remapReferenceKey` and drops `num_batches_tracked`. Alpha 0.99999999995280,
  foreground 0.99999999999967, guided-filter refine at 0.5: 0.99999999997671.
- `NFKMLXMODNet` (`@objc`) — real trimap-free portrait matting: the reference three-branch MODNet
  (ZHKKKe) in `MLXNN` over one **MobileNetV2** encoder — a low-resolution branch (squeeze-excitation
  on the deepest feature, then two 5×5 stages) deciding what the subject is, a high-resolution branch
  recovering boundary detail, and a fusion branch producing the matte. Run through
  `NFKMLXMattingBackend` (portrait → straight foreground + alpha). `+register` under `modnet`;
  factory sets `train(false)`. Its distinctive layer is **`IBNorm`**: the first half of a layer's
  channels are batch-normalized and the rest instance-normalized **without affine terms**, then
  concatenated — so the checkpoint carries parameters for only half the width. The input normalizes
  to `-1...1` (the demo's `Normalize(0.5, 0.5)`), and the strides need sides that are multiples of 32,
  so `matte(_:)` resizes for the network and resizes the alpha back. `remapReferenceKey` strips the
  `module.` prefix and the per-branch prefixes, translates the backbone's `features.N` and each
  inverted residual's `conv.M` (whose slots shift when the expansion is absent — the
  expansion-1 block's depthwise pair sits at 0/1, not 3/4), and unwraps every `Conv2dIBNormRelu`'s
  `layers` Sequential. The checkpoint stores the backbone **twice**, once per branch holding a
  reference to it; the copies are identical, so the loader keeps the first. Reference parity
  against MODNet's own network on the released photographic checkpoint (alpha cosine
  0.9999999999994, mean |difference| 1.4e-8), every parameter covered on the first triage run.
  Weights: `python3 -m gdown 1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz` (26 MB; the HF ONNX exports remain
  unreadable by this loader).
- `NFKMLXSegFormer` (`@objc`) — real semantic segmentation: the SegFormer MiT transformer encoder
  (efficient self-attention with spatially reduced keys/values + Mix-FFN depthwise conv, so no
  positional embedding) and an all-MLP decode head in `MLXNN`, run through `NFKMLXModuleBackend`. The
  argmax label map is emitted as a grayscale image under `NFKOutputImage`; recover the class index as
  `round(gray·(classCount−1))`. `+register` under `segformer-b0`. Reference parity against
  transformers' own `nvidia/segformer-b0-finetuned-ade-512-512` (logit cosine 0.99999992, label
  agreement 99.99%). `remapReferenceKey` regroups the reference's flat
  `segformer.encoder.block.<stage>.<index>` and its separate `patch_embeddings.N`/`layer_norm.N` lists
  onto per-stage names, and concatenates the reference's separate `key`/`value` into this port's one
  fused `kv` — a two-into-one a 1:1 key map cannot express. Forward, label-map, and round-trip tested.
  **Customization is a HEAD RETARGET and it ships** (`NFKMLXSegFormerTraining.swift`):
  `NFKMLXSegFormer.network(weightsURL:classCount:)` builds the net at the consumer's class count and
  drops a checkpoint classifier of another size, `NFKMLXSegFormerTrainable` picks `.decodeHead` or
  `.everything`, and `fineTune` runs the reference's recipe. `NFKMLXSegFormerObjective` is mmseg's
  cross entropy with the logits upsampled to the label resolution, measured against the reference
  (`run_reference.py segformer_loss`, `testSegFormerTrainingLossMatchesTheReference`). The reference
  optimizer is mmcv's grouped AdamW, and the schedule is `poly` decay after a 1,500-step warm-up.
  `NFKMLXSegFormerTrainingTests.testAFineTunedCheckpointRoundTrips` reloads the result.
- `NFKMLXDeepLab` (`@objc`) — real semantic segmentation (DeepLabV3): `NFKMLXResNetBackbone` with its
  last two stages dilated (so features reach the head at stride 8) and an Atrous Spatial Pyramid Pooling
  head (1×1 + three dilated 3×3 branches + global image pooling, fused, then a 3×3 convolution before
  the classifier) in `MLXNN`, run through `NFKMLXModuleBackend`. Emits a grayscale class-label map under
  `NFKOutputImage` (same convention as `NFKMLXSegFormer`); the logits upsample before the argmax, and
  the input takes ImageNet normalization. `+register` under `deeplabv3`; factory sets `train(false)`.
  Reference parity against torchvision (logit cosine 0.9999999999999, label agreement 1.0).
  `remapReferenceKey` maps the reference's positional `classifier.N` Sequential onto the module's names.
  Complements `NFKMLXSegFormer` (CNN vs transformer segmentation).
- `NFKMLXResNetBackbone` (`NFKMLXResNet.swift`) — the shared bottleneck residual backbone (ResNet-50 and
  up) in the reference layout, including the stride-to-dilation substitution DeepLab depends on
  (`replaceStrideWithDilation`; the reference gives a stage's first block the previous stage's dilation).
  `remapReferenceKey` names the projection shortcut the reference keeps in a `Sequential`
  (`downsample.0/1` → `downsample_conv`/`downsample_bn`). Pose's ResNet-50 reuses this.
  Its stem pools through `NFKMLXResample.maxPooled` (see `mlx-runtime-gotchas.md`).
- `NFKMLXBiSeNet` (`@objc`) — real real-time semantic segmentation: the reference **BiSeNetV1**
  (CoinCheung) in `MLXNN` — a shallow Spatial Path (three strided convolutions and a 1×1 projection)
  preserving the detail a deep path discards, a Context Path over **ResNet-18** with Attention
  Refinement at strides 16 and 32 plus a globally pooled branch, and a Feature Fusion Module that adds
  a channel-gated copy of the concatenated result. Run through `NFKMLXModuleBackend`; emits a
  grayscale class-label map under `NFKOutputImage` (same convention as `NFKMLXSegFormer`/
  `NFKMLXDeepLab`). `+register` under `bisenet`; factory sets `train(false)`. The context path
  upsamples with **nearest** (`nn.Upsample(scale_factor: 2)` defaults to it) while the output head is
  bilinear ×8 — two different resamplings in one network. Inputs take ImageNet normalization, and the
  Context Path's stride-32-onto-16 addition only lines up when both sides are multiples of 32, so
  `segment` resizes for the network and resizes the **logits** (never the labels — interpolating class
  indices invents classes) back. The two auxiliary heads are built because the checkpoint carries
  them, though inference never reads them. `remapReferenceKey` names the ResNet block's positional
  projection shortcut (`downsample.0/1`). Reference parity against BiSeNetV1's own network on the
  released Cityscapes checkpoint (logit cosine 0.9999999999989, **label agreement 1.0**), every
  parameter covered on the first triage run. Weights: the `CoinCheung/BiSeNet` GitHub release
  (`model_final_v1_city_new.pth`) — they were never actually unavailable.
- `NFKMLXBiSeNetV2` (`@objc`) — the second BiSeNet, a separate architecture rather than a variant, at
  reference parity on the released Cityscapes checkpoint (logit cosine 0.9999999999992, **label
  agreement 1.0**), every parameter covered on the first triage run. It replaces V1's ResNet context
  path with a purpose-built pair: a wide shallow **detail** branch and a narrow deep **semantic**
  branch of Gather-and-Expansion layers (each a residual block at stride one; at stride two it
  downsamples through two depthwise stages and carries a depthwise-then-pointwise shortcut), joined by
  a **bilateral aggregation** layer where each branch gates the other at its own scale. `+register`
  under `bisenet-v2`; emits the same grayscale label map as the other segmenters, aligning sides to a
  multiple of 32 and resizing the logits back.
  Its released checkpoint predates the repository's current head: it emits `classes × upFactor²`
  channels and **pixel-shuffles** to full resolution, where master now emits `classes` and
  interpolates. Everything before the heads is unchanged, so the oracle substitutes the older head
  rather than pinning a historical commit — the substitution stays visible. A factor-eight shuffle is
  not three ×2 shuffles (the channel interleaving differs), so `NFKBiSeNetPixelShuffle` implements
  PyTorch's `c·r² + i·r + j` order directly and a test asserts it. The four `aux*` heads supervise
  training only and are neither built nor loaded — they hold the largest tensors in the file.
