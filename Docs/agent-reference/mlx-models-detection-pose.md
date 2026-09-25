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
  **Customization ships: a class retarget and ultralytics' full fine-tune** (`NFKMLXYOLOTraining.swift`,
  ultralytics 8.4.120). `NFKMLXYOLO.network(variant:classCount:weightsURL:)` builds the now-public
  `NFKMLXYOLONet` for the consumer's class count, starts the head at `Detect.bias_init`'s priors (box
  outputs at 2, class outputs at `log(5 / classes / (640 / stride)²)`), and transfers every tensor
  shaped alike (`intersect_dicts`): at another class count the class branches, whose hidden width is
  `max(ch₀, min(classes, 100))`, stay fresh, and any other uncovered parameter throws.
  `NFKMLXYOLOObjective` is `v8DetectionLoss` with `TaskAlignedAssigner` on the host (the reference's
  `no_grad` assignment; the candidates inside the box, a box narrower than the first stride grown to
  the second, the top ten by `score^0.5 · CIoU^6`, a doubly claimed anchor to its best-overlapping
  box, targets scaled by normalized alignment) and the decode, CIoU, DFL, and BCE in MLX, gained 7.5,
  0.5, 1.5 and scaled by the batch. `fineTune(_:examples:trainable:…)` trains every weight by default
  (`.head` freezes the backbone and neck; `.dfl` is always fixed) with `optimizer=auto`'s choice for a
  short run (AdamW at `round(0.002 · 5 / (4 + classes), 6)`, decay `0.0005 · batch · accumulate / 64`
  on convolution weights only), `NFKMLXLearningRateSchedule.ultralytics` (the warm-up over
  `round(min(3, epochs − 1) · batches)` updates, then the per-epoch linear fall to 0.01), clipping at
  10, BatchNorm momentum 0.03, and `ModelEMA`'s average left in the network at the end, which is what
  the reference saves. Measured against ultralytics' own code: the loss (`run_reference.py yolo_loss`)
  97.40741 vs 97.407425 with the same 29 anchors assigned; the setup (`yolo_training_setup`, the
  trainer's `build_optimizer`, `_setup_scheduler`, `_get_warmup_iterations`, and `ModelEMA`) with
  groups 63 / 57 / 63, rate 0.001429, the schedule exact over 20 updates, and the average exact.
  `backend(variant:weightsURL:labels:)` reads the class count from the checkpoint. Not ported: the
  64-image nominal batch the reference accumulates to, and its mosaic and jitter augmentation.
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
  **Customization ships as YOLOv8's** (`NFKMLXYOLOGenerationsTraining.swift`):
  `NFKMLXYOLOGenerations.network(release:classCount:weightsURL:)` and `fineTune` with the same
  optimizer, schedule, average, bias priors (both branches), and retarget. v9, 11, and 12 train under
  `v8DetectionLoss`; v10 and YOLO26 under `NFKMLXYOLOEndToEndObjective`, `E2ELoss`: the one-to-many
  branch at ten candidates and the one-to-one branch at seven cut to one, weighted 0.8 / 0.2 falling
  linearly by epoch to 0.1 / 0.9, the one-to-one branch reading features with their gradient stopped
  as the reference detaches them. YOLO26's `reg_max` 1 head replaces DFL with an L1 on the side
  distances normalized by the input size. Measured (`run_reference.py yolo_e2e_loss`): YOLOv10n
  107.25517 vs 107.25522 and YOLO26n 111.36084 vs 111.36083 at epoch 0, and both again after the
  first epoch's weight update, each branch's terms within 1e-6 relative. The attention blocks (the PSA
  attention of v10, 11, and YOLO26, and v12's area attention) had reshaped with a batch of one, which
  inference never exceeds; they follow the input's batch now, and every release still matches.
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
  **Customization ships: a class retarget and each release's own fine-tune** (`NFKMLXRTDetrTraining.swift`,
  lyuwenyu/RT-DETR at 29320b6, `rtdetr_pytorch` and `rtdetrv2_pytorch`). The references disagree, and the
  recipe follows the original repository:
  - **The final layer's set.** transformers' `RTDetrForObjectDetection` takes `logits` before splitting off
    the denoising queries, so its model loss matches the final layer over denoising and matching queries
    together. The original splits them first (`rtdetr_decoder.py`). transformers' `RTDetrLoss` is the
    original's criterion, so the oracles run it on the original's sets and never compare the model's own
    `loss`.
  - **The box gradient.** In training, the original reports each later layer's boxes refined from the
    previous layer's undetached boxes, so a layer's box loss reaches the previous layer's box head, and
    each layer reads the previous boxes detached. transformers reports the detached form: the same
    values, different gradients. `NFKRTDetrDecoder` takes `training:` for the original's path.
  - **The recipe is per release.** The eight configuration files differ. r18vd and r34vd train their
    stem and their backbone's batch normalizations (`freeze_at: -1`, `freeze_norm: False`); r50vd and
    r101vd freeze both (`FrozenBatchNorm2d`). Each file writes its own `optimizer.params`. The config
    loader replaces a list rather than merging it, so v1 r101vd keeps one group, the backbone at 1e-6,
    and exempts nothing from decay. v2 r18vd's one group exempts normalizations at the base rate, so its
    backbone trains at 1e-4. RT-DETRv2 names its projections `conv`/`norm` and `proj`/`norm`, where v1
    indexes them, so the same pattern exempts a normalization in v2 and decays it in v1.
    `NFKMLXRTDetr.referenceRecipe(for:)` carries each file's freezing, groups, and warm-up (v2's 2,000-update
    `LinearWarmup`); `originalParameterName(_:version:)` maps the port's transformers names onto the names
    those patterns read. v2 trains under AMP; the recipe runs float32.
  `NFKMLXRTDetrObjective` is the criterion: the Hungarian match on focal class, L1, and GIoU costs (2, 5, 2),
  then varifocal (the matched query's target is its IoU), L1, and `1 − GIoU` (1, 5, 2). It sums over the
  final layer, each earlier layer and the encoder's top-k proposals (matched afresh), and every layer's
  denoising queries (matched to the box they were noised from, over the box count times the groups).
  `NFKMLXRTDetrDenoisingGroup` builds the contrastive-denoising queries as the reference does, with its
  random draws injectable. The padding class's embedding row takes no gradient, as `padding_idx` gives.
  `NFKMLXRTDetrNet.trainingOutputs` runs a batch: the encoder half shared with inference (`encode`), top-k
  per image, the selected features and starting boxes detached, the denoising queries ahead of the matching
  ones under the self-attention mask. `denoising_class_embed` exists only when
  `NFKMLXRTDetrConfiguration.denoisingQueries` is above zero, so inference and the tiny records are
  unchanged. `NFKMLXRTDetr.network(variant:classCount:weightsURL:)` starts the heads at `_reset_parameters`
  and transfers every tensor shaped alike (`load_tuning_state`): the class heads and the denoising
  embedding stay fresh at another class count, and any other uncovered parameter throws.
  `fineTune(_:variant:examples:trainable:…)` runs the release's recipe through `NFKMLXFineTune.run`, with
  clipping at 0.1 and `NFKMLXModelWeightAverage` (the original's `ModelEMA`). The backend reads a
  checkpoint's class count from `enc_score_head`, and the loader reads a saved network's own names.
  Measured: the loss 12/12 terms against `RTDetrLoss` to float32 rounding (`rtdetr_loss`); the training
  forward against transformers in training mode with the reference's own draws replayed, every layer's
  logits and boxes within 1.8e-7 and all 15 loss terms within 3e-7 (`rtdetr_training_forward`); and each
  release's element count under every (rate, decay) equal to the original's `YAMLConfig` for all eight
  releases (`rtdetr_training_setup`). Two process-killing traps surfaced here, both in
  `mlx-runtime-gotchas.md`: a tensor addressed to the nil optional embedding, and a gather differentiated
  through its indices.
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

- `NFKMLXRFDetrSegmentationNet` (`NFKMLXRFDetrSegmentation.swift`) — **RF-DETR instance segmentation**
  (`RfDetrForInstanceSegmentation`, the seven released `Roboflow/rf-detr-seg-*` sizes, Apache-2.0). The
  detector underneath is the one already at parity; the mask head is the only new network. It resamples
  the PROJECTOR's output (the reference's `backbone_features`, the tensor whose flatten becomes the
  encoder's source) to a quarter of the input, then walks one ConvNeXt-style block per decoder layer —
  depthwise 3×3, channels-last LayerNorm, pointwise linear, gelu, residual — and after each block
  projects that layer's queries through a normalized feed-forward and a linear, multiplies them against
  the block's 1×1-projected output, and adds a scalar bias. One set of masks per decoder layer; the
  reference's prediction is the LAST. Reference parity on the released seg-nano on the first numeric
  run: masks 0.9999999999502163 / 0.9999999999478614 / 0.999999999908166 / 0.9999999998673923 by layer,
  logits 0.9999999999925121, boxes 0.9999999998186787 (`run_reference.py rf_detr_seg`,
  `IK_PARITY_RF_DETR_SEG`). **Three details were load-bearing.** The activation is the EXACT
  error-function gelu, which is what the config's `gelu` names; the tanh approximation is a different
  function (`gelu_pytorch_tanh`) and costs real digits. The head's query feed-forward is a PyTorch
  `Sequential` whose index 1 is the activation, so its second linear arrives as `layers.2` where the
  module's array holds it at `layers.1`, which is the one remap the head needs; everything else in
  `segmentation_head.` passes through unchanged. And the decoder had to give up every layer's
  normalized output rather than only the last, which it already computed and discarded
  (`NFKRFDetrDecoder.states`). The seg configurations differ from the detector's in geometry alone —
  patch 12, `mask_downsample_ratio` 4, a 1024-wide head feed-forward, and per size the resolution,
  window count, decoder depth and query count. Reached from Objective-C through `NFKMLXRFDetrSegmentation`,
  whose factory set honors the `NFKMLXRFDetrSegmentationVariant` for all seven sizes and registers each
  under its own name; `NFKMLXRFDetrSegmentationBackend` emits the instances under `NFKOutputDetections`
  and their per-pixel maximum under `NFKOutputMask`, with the PER-INSTANCE masks on
  `segment(_:labels:)`, because the mask key carries one image.
- `NFKMLXTableTransformer` (`@objc`) — table-structure recognition (`TableTransformerForObjectDetection`,
  microsoft/table-transformer-structure-recognition, MIT), a vanilla DETR ported into `MLXNN`. A timm
  ResNet-18 backbone with frozen batch norm (a 7x7 stride-2 stem, a 3x3 max pool, four stages of two
  basic blocks each, the last stage 512 channels at stride 32), a 1x1 convolution to `dModel` 256, a
  normalized 2D sine position embedding, a six-layer transformer encoder, a six-layer decoder over 125
  learned object queries, and the class / box heads (`class_labels_classifier` a `Linear` to the six
  structure classes plus a no-object class, `bbox_predictor` a three-layer perceptron whose sigmoid is
  the box). Run through `NFKMLXTableTransformerBackend` (`NFKInputImage` → `NSArray<NFKDetection *>`
  under `NFKOutputDetections`, boxes normalized 0…1, origin top-left, labeled from the release's
  `id2label`). The six classes are `table`, `table column`, `table row`, `table column header`, `table
  projected row header`, and `table spanning cell`.
  One fact is load-bearing and is the only difference from post-norm DETR. Table Transformer is
  **pre-norm**: a layer norm precedes each sub-block (`self_attn_layer_norm` before the attention,
  `encoder_attn_layer_norm` before the cross-attention, `final_layer_norm` before the feed-forward), and
  a final `layernorm` follows each of the encoder and decoder stacks. Reading the DETR layer as post-norm
  loads cleanly and scores near 1 on the backbone while the encoder and decoder diverge. The other DETR
  facts were correct as first written: the attention adds the position embedding to the queries and keys
  and projects the values without it (the encoder's spatial sine embedding for its self-attention, and
  for the decoder the object queries on the self-attention and the spatial embedding on the
  cross-attention's keys); the frozen batch norm is a per-channel affine at epsilon 1e-5; the sine
  embedding is normalized (`normalize=True`, the row channels before the column channels, a shared
  frequency's sine and cosine interleaved); and DETR is one-to-one, so there is no non-max suppression.
  The image processor resizes the shortest edge to 800 (the longest capped at 1000, aspect preserved),
  then applies the ImageNet normalization; the resize is a uniform scale, so a normalized box maps to the
  original frame unchanged. Post-processing softmaxes the class logits, drops the trailing no-object
  class, takes the best remaining class per query, and converts the center-format box to corners.
  Reference parity against transformers' own `TableTransformerForObjectDetection` on the released weights,
  on the first numeric run: the ResNet-18 feature map 1.0000001, the encoder output 0.9999998, the
  decoder output 1.0, the class logits 0.9999999, and the predicted boxes 0.99999994; the backend
  recognizes a clean grid end to end as a table with its rows and columns. `NFKMLXTableTransformer` reads
  the geometry and the class labels from the release's `config.json`, so the same code serves the
  detection checkpoint (two classes) and the structure-recognition checkpoint (six). Reached from
  Objective-C through `backend(directoryURL:)` and its asynchronous peer; the model loads a whole release
  directory, so it takes a directory factory rather than the file-based registry, like Florence-2 and
  TrOCR.
- **Customization: head retarget, or the reference's own run.** microsoft/table-transformer trains
  with its vendored DETR: `SetCriterion` behind a `HungarianMatcher` (class 1, L1 5, GIoU 2 in the cost;
  `loss_ce` × 1, `loss_bbox` × 5, `loss_giou` × 2; the no-object class weighted 0.4), scored on the last
  decoder layer, because both released configurations set `aux_loss` false; AdamW with weight decay 1e-4
  on every parameter, 5e-5 and 1e-5 for the backbone; gradient norm clipped at 0.1; `StepLR` 0.9 per
  epoch. DETR trains only the backbone's last three stages, and every batch norm is frozen.
  `NFKMLXTableTransformerObjective` is that criterion, its matching on the shared `NFKMLXHungarian`
  solver and its boxes through the SAM 3 recipe's generalized IoU; a target is an `[N, 5]` array of
  `[class, cx, cy, w, h]`. `NFKMLXTableTransformerTrainable` is `.heads`, `.transformer`, or
  `.everything` (the reference). `NFKMLXTableTransformer.network(directoryURL:labels:)` retargets to
  a new class set (a fresh classifier; everything else loads), `fineTune` runs the reference optimizer
  and schedule (`stepsPerEpoch` sets the epoch), and `save(_:toDirectoryURL:release:)` writes the weights
  and a `config.json` carrying the new `id2label`, which `backendWithDirectoryURL:` loads. Measured
  (`run_reference.py table_transformer_loss`, `IK_PARITY_TABLE_TRANSFORMER_<RELEASE>_LOSS`, the
  reference's own `detr/models/matcher.py` and `SetCriterion` run on each release's outputs, pinned in
  the manifest): the matching equals the reference's on both releases; on the detection release
  `loss_ce` 5.9600515 against 5.96005, `loss_bbox` 0.55771196 against 0.557712, `loss_giou` 0.9057374
  exactly, the total 10.560086 against 10.560085; on the v1.1 structure release `loss_ce` 1.3405254
  against 1.3405252, `loss_bbox` and `loss_giou` exact, the total 3.6116304 against 3.6116302; the port's
  own forward reaches 10.560083 and 3.6116328. transformers' `labels=` loss equals the reference on both.
  `NFKMLXTableTransformerTrainingTests` also holds each policy's frozen set and a retargeted release's
  save and factory reload.
- **Every release** (2026-09-23; the five on the Hugging Face API) is at reference parity:
  - `table-transformer-detection`: 15 queries, two classes (`table`, `table rotated`).
  - `table-transformer-structure-recognition`: the v1.0 structure release.
  - `-structure-recognition-v1.1-all` / `-fin` / `-pub`: the v1.0 geometry.
  The v1.1 releases set `use_timm_backbone` false, so their backbone is transformers' `ResNetBackbone`
  under other names. `NFKMLXTableTransformerNet.timmBackboneKey` maps them onto the timm layout:
  - `embedder.embedder.{convolution,normalization}` → `conv1` / `bn1`
  - `encoder.stages.S.layers.B.layer.{0,1}` → `layer{S+1}.B.{conv,bn}{1,2}`
  - `shortcut` → `downsample`
  The input size comes from each release's `preprocessor_config.json` (`NFKMLXTableTransformerSizing`):
  - structure v1.0: shortest edge 800, longest capped at 1000.
  - detection: 800 capped at 800.
  - v1.1: `longest_edge` 800 alone. transformers 4.57's `DetrImageProcessor.resize` rejects that size,
    so the oracle bounds both edges with the processor's own `max_height` / `max_width` path, which
    truncates. The port follows it: 1234×567 becomes 800×367, not 368.
  Cosines, all seams:
  - detection: backbone 1.0, encoder 0.99999994, decoder 1.0, logits 0.99999976, boxes 1.0.
  - v1.1-all: backbone 1.0, encoder 1.0, decoder 1.0, logits 0.9999998, boxes 0.99999994.
  - v1.1-fin: backbone 1.0, encoder 1.0, decoder 0.99999994, logits 1.0000001, boxes 1.0000001.
  - v1.1-pub: backbone 0.9999999, encoder 0.9999998, decoder 1.0000001, logits 0.99999994, boxes
    1.0000001.
  Records: `run_reference.py table_transformer` per release directory, under `~/.inferkit-validation`
  (`IK_{VAL,PARITY}_TABLE_TRANSFORMER_{DETECTION,V11_ALL,V11_FIN,V11_PUB}`). Oracle: `run_reference.py table_transformer`
  under the `llm` oracle env (transformers, needs Pillow), which records the preprocessed pixels, the
  backbone feature map, the encoder and decoder outputs, the boxes, and the post-processed detections.
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
