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
  **SAM 3 and SAM 3.1 are deliberately skipped**, not merely unported: their license terms are not
  acceptable for this project at this time. Revisit if that changes.
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
