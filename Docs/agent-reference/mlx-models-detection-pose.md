<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX models: object detection, face detection, and pose

- `NFKMLXRetinaFace` (`@objc`) — real face detection with five-point landmarks, and the detector the
  CodeFormer reference pipeline runs through facexlib. The released **mobile0.25** model: a MobileNetV1
  backbone at quarter width (a plain stem then depthwise-separable blocks), a three-level FPN fusing
  top-down with **nearest** resampling, three SSH context modules (a 3×3 branch beside 5×5 and 7×7
  receptive fields built from stacked 3×3s, concatenated then activated together), and per-level class
  / box / landmark heads over two anchors a cell. Run through `NFKMLXDetectionBackend`
  (`NFKInputImage` → `NSArray<NFKDetection *>` under `NFKOutputDetections`, boxes normalized 0…1,
  origin top-left) or `detector(weightsURL:)` for the landmarks, which is what alignment needs.
  `+register` under `retinaface-mobile025`. The input is BGR 0…255 minus `[104, 117, 123]`, because
  the reference reads its frames through OpenCV. Feeding RGB is a quiet defect, not a loud one:
  measured on the validation portrait, the face is still found and the confidence is unchanged to three
  decimals (0.9971 against 0.9975 — the wrong order scores marginally higher), while the box moves to
  IoU 0.962 and the landmarks shift by up to 5.7 px, which is enough to move the aligned crop and
  therefore the restoration. Nothing in the values reveals the order, so
  `testTheChannelOrderIsLoadBearing` pins it by measuring that displacement; asserting on confidence
  would have passed with the swap in place. The scale is detectable where the order is not, so
  `prepared` asserts its input is `0...1` rather than `0...255`. Neither is reachable from the public
  API — `faces(in:)`, `detector(weightsURL:)`, and `backend(...)` all take a `CGImage` and convert
  internally — so this is a maintainer hazard rather than a consumer one. Anchors are
  generated per level from `minSizes` / `steps` and decoded with the reference's variances
  (`0.1`/`0.2`): a centre is the anchor's centre plus a variance-scaled offset of the anchor's size,
  and a size is the anchor's size times the exponential of its offset. The checkpoint's ImageNet
  classifier (`body.fc`, `body.avg`) and every `num_batches_tracked` counter are dropped rather than
  loaded — they are not parameters of the detector. Reference parity against facexlib's own
  RetinaFace over the whole pre-suppression tensor (box cosine 0.9999999999980, class
  0.9999999999999969, landmark 0.9999999999980, and an exact anchor grid), and end to end through
  decoding and suppression against its `detect_faces` (same face count, **box IoU 1.0**, landmarks
  within a pixel). Weights: `github.com/xinntao/facexlib/releases` `detection_mobilenet0.25_Final.pth`,
  1.7 MB — negligible beside CodeFormer's own checkpoint, which is why it is the recommended detector.
- `NFKMLXYOLO` (`@objc`) — real object detection: the reference **YOLOv8** (ultralytics) in `MLXNN` —
  a CSPDarknet backbone of `Conv` (convolution + **BatchNorm epsilon 1e-3** + SiLU) and `C2f` stages
  ending in SPPF (three chained 5×5 stride-1 max pools through `NFKMLXResample.maxPooled`), a PAN-FPN
  neck fusing strides 8/16/32 both ways, and a decoupled head with **distribution-focal box
  regression**: each box side is a softmax over 16 bins whose expectation (the `dfl` convolution,
  fixed to 0…15, loaded from the checkpoint) is a distance from the cell's anchor point. v8 has no
  objectness — confidence is the best class probability. Box decode and greedy per-class NMS in
  Swift. The suppression thresholds are deliberately the reference's 0.25 / 0.7, not the 0.45 this
  module first shipped with — a ratified behavior change, so do not "restore" the older value.
  `NFKMLXYOLOBackend` reads `NFKInputImage` → `NSArray<NFKDetection *>` under `NFKOutputDetections`;
  boxes are normalized 0…1, origin top-left.
  The `+backendWith…labels:` factory attaches class names. `+register` under `yolo`.
  `remapReferenceKey` maps the reference's `model.N` module list onto named stages and the head
  branches' positional Sequential (`cv2.i.0/1/2` → `conv1`/`conv2`/`out`). Reference parity
  against ultralytics' own YOLOv8n on the released `yolov8n.pt` over the full pre-suppression tensor
  (box cosine 0.9999999999999638, class cosine 0.9999999999944721, same top class at the same
  anchor), and against ultralytics' `predict` end to end on a 16:9 frame through the public backend
  (9/9 detections, same classes, worst box IoU 0.9999984). A frame is fitted the reference's way:
  scaled by the smaller ratio, padded with gray 114 to a multiple of 32 (`auto` mode, so a wide frame
  runs at 640×384 rather than wasting a third of the input), and the decoded boxes have that padding
  and scale undone before they are normalized against the caller's own frame. Forward, decode, NMS,
  letterbox, remap, and round-trip tested. YOLOv8s is at parity too (box cosine
  0.99999999999997, class 0.9999999999997): it has the **same depth** as the nano model at twice the
  width, because the releases scale by two independent multiples — reading only one of them right
  still loads and is still wrong. **YOLOv8m** is the first size where both multiples change — wider
  stages and deeper C2f repeats `[2, 4, 4, 2]` — and it matches too (box 0.99999999999996, class
  0.999999999998). `NFKMLXYOLOVariant` (`.nano`/`.small`/`.medium`/`.large`/`.extraLarge`) selects the size. l and x
  are at parity too (box 0.99999999999993 / 0.99999999999994): both run the full depth multiple, so
  their C2f stages repeat `[3, 6, 6, 3]`, and x is wider again. The records must be made at
  `--size 640`, where the reference's letterboxing is an identity — generating one at another size
  produces a different anchor count and looks like a model failure.
- `NFKMLXYOLOGenerations` (`@objc`) — YOLOv9, YOLOv10, YOLO11, YOLOv12 and YOLO26, the generations
  after the shipped v8, as one graph interpreter rather than five ports. The reference states each
  release as a YAML list of `(from, repeats, module, args)` rows that `parse_model` scales by the
  size letter; `NFKYOLONode` carries those rows and `NFKMLXYOLOGenerationNet` builds them into one
  `[Module]` array, so a checkpoint's `model.<N>.…` keys land on array index N and only the head needs
  a remap. Twenty-six released checkpoints load and run: v9 t/s/m/c/e, v10 n/s/m/b/l/x, and 11 / 12 /
  26 at n/s/m/l/x. The new blocks are v9's GELAN set (`RepNCSPELAN4`, `ELAN1`, `RepCSP`, `RepConv`,
  `SPPELAN`, `AConv`, `ADown`), v10's `SCDown` / `PSA` / `C2fCIB` / `CIB` / `RepVGGDW`, YOLO11's `C3k2`
  / `C3k` / `C2PSA`, and YOLOv12's `A2C2f` area attention. Four facts are load-bearing, and three of
  them are visible only in the released checkpoints rather than in the current reference source.
  **YOLOv12's positional-encoding convolution carries a bias** beside its batch norm, which no other
  generation's does. **YOLO26's `SPPF` differs from every earlier one**: its narrowing convolution has
  no activation and the fused result adds the input, so the shipped `NFKYOLOSPPF` covers v9 through v12
  and `NFKYOLOSPPFShortcut` covers 26. **The end-to-end heads select over anchor-and-CLASS pairs**: the
  reference's `postprocess` flattens the scores before its top-k, so one anchor is reported twice when
  two classes clear the threshold, and taking the best class per anchor drops the second — which is
  exactly what the YOLOv10 consumer comparison caught. **YOLOv9e keeps its programmable-gradient branch
  at inference** (`CBLinear` splits a stage into per-level taps, `CBFuse` sums the matching tap back
  into the main path), so a graph row holds a list of tensors rather than one. `NFKYOLOGraphs` states
  which stages each YOLOv10 size runs as `C2fCIB` and which of those take the large-kernel branch,
  because the released configs differ stage by stage rather than by a rule. Reference parity against
  ultralytics' own model on every one of the twenty-six released checkpoints, over the full
  pre-suppression tensor (box cosines 0.99999945 to 1.0, class cosines 0.99993 to 0.9999993), plus the
  consumer path on a non-square frame for YOLO11, YOLOv10 and YOLO26 (same detection counts, same
  classes, worst box IoU 0.9993941187858582). The licence is ultralytics' AGPL-3.0, the same as the
  shipped v8; `NFKMLXRTDetr` and `NFKMLXRFDetr` remain the licence-clean detectors.
- `NFKMLXRTDetr` (`@objc`) — real object detection, the license-clean (Apache-2.0) alternative to the
  AGPL YOLO: RT-DETR (`RTDetrForObjectDetection`, PekingU/lyuwenyu) in `MLXNN` — a **ResNet-D**
  backbone (deep 3-conv stem, avgpool-in-shortcut bottleneck), a **hybrid encoder** (an AIFI transformer
  on the deepest feature plus a CSP-RepVGG FPN/PAN), **query selection** over generated anchors, and a
  **deformable-attention decoder** with iterative box refinement. Run through `NFKMLXRTDetrBackend`
  (`NFKInputImage` → `NSArray<NFKDetection *>` under `NFKOutputDetections`, boxes normalized 0…1, origin
  top-left). `+register` under `rtdetr`. DETR-family, so there is no non-max suppression — the
  one-to-one training makes the queries distinct — and the image processor **squashes** to 640×640 (no
  aspect-preserving pad), so a normalized box maps to the original frame unchanged.
  The decoder's cross-attention is multi-scale deformable attention, not DCNv2 deformable
  Convolution: a `grid_sample` bilinear gather at learned offset locations (`loc·size − 0.5`,
  `align_corners=false`, zero padding), which MLX expresses with `takeAlong` the way RAFT/RIFE do their
  warps — so unlike BiRefNet (which needs a deformable *conv* MLX has no op for), RT-DETR is portable.
  Anchors are generated per forward (`anchor_image_size=None`) as `logit(centre/size)` with a validity
  mask; query selection takes the top `num_queries` by the best class score; the decoder runs
  `sigmoid(reference)`, a per-layer `query_pos_head` MLP position, deformable cross-attention over the
  flattened encoder tokens, and `sigmoid(corner + inverse_sigmoid(reference))` box refinement, with
  `class_embed`/`bbox_embed` **cloned per layer** (`with_box_refine`). BatchNorm runs in eval (the
  reference freezes the backbone BNs). Reference parity against transformers' own
  RTDetrForObjectDetection, seam by seam at a tiny config: backbone 0.99999999999998, the PAN encoder
  0.9999999999999988, query-selection scores (`enc_class`) 0.9999999999999925 and boxes (`enc_coord`)
  0.9999999999999999, and the deformable-attention decoder exact over the reference's
  selection (logits 0.9999999999999958, boxes 0.9999999999999966). Also at parity on the released
  `PekingU/rtdetr_r50vd` weights end to end (`run_reference.py rtdetr_real`, `IK_PARITY_RTDETR_REAL` +
  `IK_VAL_RTDETR`): logits cosine 0.9999999999887, boxes 0.9999999999642 over the reference selection —
  the r50vd ResNet-50-vd geometry, the actual checkpoint, and the loader (the **stage-1 stride-1
  shortcut** re-indexed to `.0.`, the `model.` prefix stripped, the tied top-level `class_embed`/
  `bbox_embed` and every `num_batches_tracked` dropped) exercised, which the tiny config does not cover.
  Two facts are load-bearing, both found by the parity run. The oracle's shared `_randomized`
  randomizes all floating state including the BatchNorm `running_var` buffer, which can go negative, and
  `rsqrt(var + eps)` is then NaN in both the reference and the port; `run_rtdetr` therefore randomizes
  only the trainable parameters, leaving the BN buffers physical (mean 0, var 1). And the end-to-end
  top-k selection is float-tie-sensitive: `torch.topk` and MLX's `argSort` break a sub-ulp score tie
  differently, so one or two of the selected queries can swap (end-to-end boxes 0.981 where the decoder
  over the reference selection is exact) — the same near-tie class as the GGUF/Whisper greedy flips, so
  the decoder parity is measured over the reference's own selection and the end-to-end boxes are asserted
  at a tolerance that reflects the tie rather than a modeling error. `NFKMLXRTDetrConfiguration`
  (`.tiny`/`.r50vd`/`.r18vd`/`.r34vd`/`.r101vd`) selects the geometry; the backbone/encoder/decoder use
  `[Module]` arrays so the module keys mirror the checkpoint's nested `nn.Sequential` layout, with only
  the shortcut re-index in the remap. All four released sizes are at parity (`NFKMLXRTDetrVariant`,
  registered as `rtdetr-r18vd` / `-r34vd` / `-r101vd` beside `rtdetr`): r18vd and r34vd run ResNet
  **basic** blocks (`NFKRTDetrBasicLayer`, two ConvNorms and the avgpool-in-shortcut at stride 2),
  narrower stage widths, and three or four decoder layers; r101vd is the bottleneck backbone at depths
  `[3, 4, 23, 3]` with a 384-wide encoder and 2048 FFN. Over the reference's selection: logits
  0.99999999999881 / 0.99999999999911 / 0.99999999997880, boxes 0.99999999999285 / 0.99999999999693 /
  0.99999999979751.
  RT-DETRv2 (`RTDetrV2ForObjectDetection`, PekingU/lyuwenyu, Apache-2.0) runs through this same port. v2
  changes the deformable decoder's sampling and nothing else: the class inventory of transformers' v1 and
  v2 modeling files is identical but for the attention class. `NFKMLXRTDetrConfiguration` carries the two
  knobs that difference needs. `decoderOffsetScale` scales the learned sampling offsets, which RT-DETR
  fixes at 0.5 and v2 reads from `decoder_offset_scale`. `decoderMethod` selects the sampling through the
  `@objc` `NFKMLXRTDetrSamplingMethod`: `.bilinear` is v2's `decoder_method: "default"` and RT-DETR's only
  mode, and `.discrete` is v2's nearest-cell alternative, which scales the normalized location by the
  level's size, adds half a cell, truncates toward zero, and clamps each axis independently. The discrete
  path has no blend and no zero padding, so a location outside the map reads the nearest edge cell where
  the bilinear path reads nothing. v2's `n_points_scale` is `1 / n_points` when every level samples the
  same number of points, which every released configuration does, so it is the division RT-DETR already
  performs. Every released v2 configuration repeats its RT-DETR namesake's geometry and states the RT-DETR
  sampling settings (`decoder_method: default`, `decoder_offset_scale: 0.5`), so the four v2 presets
  (`.v2R18VD` / `.v2R34VD` / `.v2R50VD` / `.v2R101VD`) each return their v1 counterpart and the releases
  differ only in their trained weights. A consumer's own v2 configuration that sets either knob runs
  through the same code. `NFKMLXRTDetrVariant` gained the four cases, registered as `rtdetr-v2-r18vd` /
  `-r34vd` / `-r50vd` / `-r101vd`, and every existing `@objc` factory takes them. At reference parity on a
  tiny configuration that deliberately sets what no release does (`decoder_method: "discrete"`, an offset
  scale of 0.35), the only setting that exercises the discrete path: query-selection scores (`enc_class`)
  0.999999999999993 and boxes (`enc_coord`) 0.9999999999999999, with the decoder over the reference's own
  selection at logits 0.9999999999999964 and boxes 0.9999999999999982. All four released checkpoints are
  at parity too, over the reference's selection (r18vd / r34vd / r50vd / r101vd):
  logits 0.999999999996304 / 0.9999999999910995 / 0.9999999999202086 / 0.9999999999796655 and boxes
  0.9999999999901055 / 0.999999999885731 / 0.9999999990654127 / 0.9999999997898816. Oracle:
  `run_reference.py rtdetr_v2` for the tiny configuration and `rtdetr_v2_real` for a released checkpoint
  directory, both under the `llm` env, with the one real mode serving all four sizes.
- `NFKMLXRFDetr` (`@objc`) — real object detection under Apache-2.0 (RF-DETR base, Roboflow), a two-stage
  Group-DETR detector ported from transformers' `RfDetrForObjectDetection`, at reference parity on
  both a tiny random config and the released weights (`testRFDetrMatchesTheReference` /
  `…OnReleasedWeights`). A **windowed DINOv2** backbone (each block partitions the patch grid into
  `num_windows²` local windows with a replicated CLS; a global-attention block — the out-index layers
  2/5/8/11 — unpartitions to one sequence per image before attending and re-partitions after; selected
  stages are layernormed, the CLS dropped, unpartitioned, and reshaped to feature maps), a C2f /
  RepVGG scale projector (concat the stage maps → a C2FLayer → a channels-first LayerNorm, which in NHWC
  is a plain last-axis LayerNorm), **two-stage query selection** (an `enc_output` Linear + LayerNorm, the
  `enc_out_class`/`bbox` heads over every token, top-k by the class max), **mixed queries** (a learned
  `reference_point_embed` refined by the top-k coords in direct normalized box space — cxcy = Δxy·wh + xy,
  wh = exp(Δwh)·wh, not the sigmoid space RT-DETR uses — plus a learned `query_feat`), and an **LW-DETR
  deformable decoder** (self-attention with the query position added to q and k and the value without
  position; deformable cross-attention over one feature level; no iterative box refine — the reference
  points are constant and one box refinement runs at the end in the head). Group-DETR collapses to one
  group at inference.
  Two seam bugs the parity ladder caught. The global-attention block re-partitions using the
  Unpartitioned shape — the reference reassigns `hidden_states` before reading `.shape`, so it takes
  `[B/windows², windows²·seq, C]`, not the original windowed shape (a wrong shape crashes the reshape).
  And the released backbone interpolates its 518-trained pos_embed to 560 (37→40 patches) with `bicubic,
  align_corners=false, ANTIALIAS=true`, which is not a no-op when upsampling: antialias uses the PIL cubic
  coefficient **a = -0.5** (torch's non-antialias bicubic, and `NFKMLXBicubic`, use -0.75) with per-output
  weight normalization — `antialiasResampleMatrix` builds the `[out, in]` operator (measured to reproduce
  torch to 2e-6). The width/height-swapped window reshape, the direct box space, the channels-first
  LayerNorm eps (conv norms 1e-5, the projector norm 1e-6), and the DETR sinusoidal position embedding
  were all correct as first written.
  The released file loads directly on device. It is prefix-free with the original Roboflow naming
  (`backbone.0.encoder.encoder.*`, `transformer.*`, `refpoint_embed`, a fused `self_attn.in_proj_*`);
  `remapReferenceKey` converts it to the module names — the exact map derived by tensor-identity matching
  the raw checkpoint (487 keys) against the converted state_dict (499), since transformers exposes no
  mapping dict — and `loadWeights` splits the fused `in_proj` (packed `[q; k; v]`) into q/k/v_proj. The
  base geometry was read from the release: `num_labels` 91, `decoder_n_points` 2 (not 4), resolution
  560. `+register` under `rf-detr`; `detect()` applies the image processor's ImageNet normalization
  (mean/std, 560 resize; the PIL-bilinear resize is a documented approximation). The four later
  releases are at parity too (`laterRelease(resolution:decoderLayers:)`, `NFKMLXRFDetrVariant`,
  registered as `rf-detr-nano` / `-small` / `-medium` / `-large`): a patch-16 DINOv2 at the release's own
  resolution (384 / 512 / 576 / 704) with two windows a side and out-indices 3/6/9/12, over 2 / 3 / 4 / 4
  decoder layers. Over the reference's selection, logits 0.99999999998936 / 0.99999999997724 /
  0.99999999994142 / 0.99999999988494 and boxes ≥ 0.99999999221948. The tiny parity loads the
  oracle's converted weights and the released parity loads the raw file through `loadWeights`, so both the
  network and the on-device naming conversion are measured. Oracle: `run_reference.py rf_detr` /
  `rf_detr_real` under the `rfdetr` env (transformers 5.16.1).
- `NFKMLXPose` (`@objc`) — real top-down pose estimation (SimpleBaseline): `NFKMLXResNetBackbone` as
  ResNet-50 and a transposed-convolution head produce one heatmap per joint in `MLXNN`; the argmax of
  each heatmap is a joint location, refined a quarter cell toward its larger neighbor as the reference
  decode does. `NFKMLXPoseBackend` reads `NFKInputImage` → `NSArray<NFKKeypoint *>` (a new core value
  type) under the new core key `NFKOutputPose`; positions are normalized 0…1, origin top-left. The
  `+backendWith…jointNames:` factory attaches joint names. `+register` under `pose-simplebaseline`.
  A person crop is taller than it is wide, so the trained geometry is 256×192 (`inputHeight`/`inputWidth`)
  and the input takes ImageNet normalization. Factory sets `train(false)` for BatchNorm running stats;
  the converter swaps the deconv ConvT axes. Reference parity against microsoft's own
  SimpleBaseline (heatmap cosine 0.9999999999961, peak agreement 1.0), on the mmpose ResNet-50 COCO
  release — whose keys are the reference's under a `backbone.`/`head.` prefix, so a strict load of the
  reference doubles as proof the two architectures are one. `remapReferenceKey` maps that prefix and the
  head's positional `deconv_layers.{0,3,6}`/`{1,4,7}` Sequential.
- `NFKMLXVitPose` (`@objc`) — modern top-down pose estimation (`VitPoseForPoseEstimation`,
  ViTAE-Transformer / usyd-community, Apache-2.0), a plain ViT backbone under a small decoding head,
  beside the SimpleBaseline port. The backbone is a patch embedding, 12 pre-norm transformer blocks
  (separate query/key/value projections with bias, an exact error-function GELU), a final LayerNorm, and
  the token sequence reshaped to a feature map. Its `layer_norm_eps` is 1e-12, not a ViT's usual 1e-6.
  Two facts about that backbone are load-bearing. The patch-embedding convolution **pads by two**: at the
  trained geometry (256×192 at patch 16) the padding leaves the patch count alone, 16×12 either way, so a
  plain non-overlapping patchify loads cleanly and samples every window two pixels early, which scored the
  backbone at 0.998 against the reference's 0.9999999999. The position table carries a **class-token row**
  the sequence does not: the reference adds the patch rows and that row to every patch
  (`pos[:, 1:] + pos[:, :1]`), so the class position acts as a constant bias instead of a token.
  Both decoders are built, selected by the release's `use_simple_decoder`: the simple one is a ReLU, a
  bilinear upsample by `scale_factor`, and one 3×3 convolution; the classic one is two
  transposed-convolution blocks (no bias, each followed by a BatchNorm and a ReLU) and a 1×1 convolution,
  which is the head SimpleBaseline uses.
  Decoding is what most distinguishes ViTPose from SimpleBaseline, and it is ported in full.
  `NFKVitPoseDecoding` implements **DARK** (`post_dark_unbiased_data_processing`): SimpleBaseline nudges
  the peak a quarter of a cell toward its larger neighbor, while ViTPose blurs the heatmap, takes its
  logarithm, and refines the peak by one Newton step against the local derivative and Hessian, which
  places a keypoint between cells instead of on a quarter grid (Zhang et al., *Distribution-Aware
  Coordinate Representation for Human Pose Estimation*, and Huang et al., *The Devil is in the Details*,
  both CVPR 2020).
  `VitPoseImageProcessor.keypoints_from_heatmaps` defaults its `kernel` to **11**, so the Gaussian radius
  is 5, while `post_dark_unbiased_data_processing`'s own signature defaults to 3; reading the callee's
  signature instead of the caller's gives a three-tap blur and a decode that is close but wrong. scipy's
  default `reflect` mode is symmetric (it repeats the edge sample) where its `mirror` mode does not, and
  MLX pads with a constant or the edge value only, so the border is built by gathering reflected indices.
  And the refinement is faithful, so a flat neighborhood can move a peak past its own heatmap: clamping
  the shift to one cell disagreed with the reference by 1.04 cells on the classic-decoder release, so the
  decode does not clamp and `NFKMLXVitPoseNet.estimate` clamps the final normalized position, because
  `NFKKeypoint` promises one.
  `vitpose-plus-*` releases route their feed-forward through per-dataset experts (`num_experts` above
  one, `part_features` splitting the hidden width) and need a dataset index at inference.
  `NFKMLXVitPoseConfiguration.configuration(fromHuggingFace:)` refuses them instead of loading them into
  a dense stack.
  `NFKMLXVitPoseBackend` reads `NFKInputImage` → `NSArray<NFKKeypoint *>` under `NFKOutputPose`. The
  `@objc` enums are `NFKMLXVitPoseVariant` (`.baseSimple`, `.base`) and `NFKMLXVitPoseDecoder`
  (`.simple`, `.classic`); the factories are `+backendWithWeightsURL:jointNames:error:`,
  `+backendWithVariant:weightsURL:jointNames:error:`, `+backendWithDirectoryURL:jointNames:error:` (which
  reads the release's own `config.json`), the two download peers, and the two asynchronous peers.
  `+register` under `vitpose-base-simple` and `vitpose-base`.
  At reference parity on both released checkpoints. `usyd-community/vitpose-base-simple`: backbone
  feature map 0.9999999999995017, heatmaps 0.9999999999966225, every integer peak exact, the DARK
  refinement within 8.80751758813858e-05 of a cell. `usyd-community/vitpose-base` (the classic decoder):
  feature map 0.9999999999931072, heatmaps 0.9999999999953517, every integer peak exact, refinement
  within 4.622340202331543e-05 of a cell. Oracle: `run_reference.py vitpose --checkpoint <release
  directory>` under the `llm` env, one mode serving both decoders. It resizes and normalizes the image
  itself instead of using the release's image processor, because that processor warps a person's box
  through an affine transform and a whole-image caller has no box (the Swift backend resizes for the same
  reason), and it records the prepared pixels, the backbone feature map, the heatmaps, the integer peaks,
  the DARK-refined keypoints, and the scores.
