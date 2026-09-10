<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX models: image restoration and enhancement

Upscaling, denoising, inpainting, stylization, low-light, colorization, face restoration.

- `NFKMLXRealESRGAN` (`@objc`) — a real single-forward upscaler: the Real-ESRGAN generator (RRDBNet)
  in `MLXNN` (`Conv2d` + `leakyRelu`, norm-free), run through `NFKMLXModuleBackend` for ×4 upscaling.
  `+register` puts it in the registry under `real-esrgan-x4`, so ObjC/MetalForge builds it by name (and
  downloads weights via `NFKMLXHub`). The module structure and parameter names mirror the reference
  PyTorch `RRDBNet`, so `loadWeights(into:from:)` loads a safetensors checkpoint (`loadArrays` →
  `update(parameters:)`), transposing 4-D conv weights from PyTorch `[out,in,kH,kW]` to MLX
  `[out,kH,kW,in]`. A `.pth` release converts to safetensors first with
  `Tools/realesrgan-to-safetensors/convert.py`; the weight-load path is proven offline by a test that
  saves the net's params in PyTorch layout, reloads through the transpose, and confirms the forward
  matches. Adds the `MLXNN` product of `mlx-swift` to the target. Reference parity against the general
  ×4 release (cosine 0.9999947) and the anime release, a six-block generator where the general one has
  twenty-three (0.9999956).
- `NFKMLXNAFNet` (`@objc`) — a real single-forward restoration network (denoise / deblur): a U-shaped
  stack of NAFBlocks (SimpleGate + Simplified Channel Attention, channel LayerNorm = last-axis in NHWC,
  PixelShuffle up) in `MLXNN`, run through `NFKMLXModuleBackend` (image → restored image at input size,
  padding to `2^levels` and cropping back). `+register` under `nafnet`; `NFKMLXNAFNetConfiguration` sets
  width and block counts (default SIDD width 32). Block names (`conv1`…`conv5`, `norm1/2`, `beta`,
  `gamma`) match the reference; `Tools/nafnet-to-safetensors/convert.py` renames `middle_blks`/`ups.N.0`/
  `sca.1` so a real checkpoint loads directly. Reference parity against megvii-research's own NAFNet on
  the released SIDD width-32 denoiser (cosine 0.9999972, mean |difference| 0.00098 through the backend's
  8-bit bridge), every parameter covered on the first triage run. The weights are not Drive-only:
  `huggingface.co/nyanko7/nafnet-models` mirrors all five releases. The GoPro deblurrer is also at
  parity (0.9999973), a block-layout change rather than a width one: it puts twenty-eight of its blocks
  in the last encoder stage and one in the middle, where SIDD spreads `[2, 2, 4, 8]` with twelve, so a
  checkpoint fits only the geometry it was trained as. The REDS release is that same distribution at
  twice the width (0.9999971), which separates a wrong width from a wrong block layout. The two width-64
  releases are also at parity: `siddWidth64` (SIDD's distribution at width 64, 0.99999986) and
  `goProWidth64` (the REDS geometry trained on GoPro, 0.9999999966). The GoPro width-64 record must be
  made from the photographic plate: on the synthetic plate the reference itself diverges (output range
  −111 to 110), so a record made from it reproduces exactly while measuring nothing (the first run read
  0.527 against that record). `NFKMLXNAFNetVariant` (`.sidd`/`.goPro`/`.reds`/`.siddWidth64`/`.goProWidth64`)
  selects the geometry from Objective-C.
- `NFKMLXLaMa` (`@objc`) — a real single-forward inpainter: the LaMa FFC-ResNet generator in `MLXNN`
  (each Fast Fourier Convolution runs a spatial branch and an FFT spectral branch via `MLXFFT`
  `rfft2`/`irfft2`, orthogonally normalized), run through an `NFKMLXMattingBackend` (plate under
  `NFKInputImage`, mask under `NFKInputMask`). `+register` under `lama-inpaint`. The configuration
  defaults are big-lama's own `config.yaml`: 64 base channels, three downsampling stages, 18 residual
  blocks, ratio 0.75 through the trunk and at the last downsample only, sigmoid output.
  `remapReferenceKey` translates the checkpoint's flat `model.N` Sequential (the parameter-free entries
  — reflection pads, the tuple concatenation, the activations — still consume an index, so the
  upsampling triples start at 24), plus the spectral branch's `conv1.0`/`conv1.1` narrowing. The
  upsampling transposed convolutions carry `outputPadding` 1 (without it they land a pixel short of
  doubling, which an output resize can only paper over) and load with `transposed(1, 2, 3, 0)`, not the
  forward convolutions' axis order. Reference parity against advimman's own FFCResNetGenerator on the
  released big-lama (inpainted cosine 0.9999999999997, mean |difference| 5.9e-8). The convolutions
  reflection-pad, as the reference's `padding_mode='reflect'` does; edge padding instead scores 0.99857
  / mean 0.00503, so the approximation this model shipped with was a real defect, measured both ways.
- `NFKMLXStyleTransfer` (`@objc`) — a real single-forward stylizer: Johnson et al.'s `TransformerNet`
  (three downsampling convs → five residual blocks → two nearest-upsample convs → output conv, each
  instance-normalized) in `MLXNN`, run through `NFKMLXModuleBackend` (image → stylized image at input
  size). `+register` under `fast-style-transfer`; the style is baked into the weights (one checkpoint =
  one style). Reference parity against pytorch/examples (cosine 0.9999926). Reaching it required real
  reflection padding (`NFKMLXResample.reflectPadded`, a mirror gather — MLX pads with a constant or the
  edge value only): with edge padding the mean pixel error was 0.049, and only a quarter of that sat at
  the border, because the instance norms are global and carry a border approximation into every pixel.
  Names match the reference, so `Tools/style-transfer-to-safetensors/convert.py` only drops the
  deprecated InstanceNorm running-stats keys.
- `NFKMLXCodeFormer` (`@objc`) — real face restoration: the reference CodeFormer (sczhou) in `MLXNN` —
  a VQGAN encoder and generator built as the reference's flat heterogeneous `blocks` list (residual
  blocks with a 1×1 skip projection where the width changes, single-head spatial attention at
  resolution 16, asymmetric-pad stride-2 downsamples, nearest ×2 upsamples; GroupNorm at epsilon
  1e-6), a codebook under `quantize.embedding`, and a Transformer code-predictor whose queries and
  keys carry the position embedding while the values do not (the reference's fused
  `in_proj_weight`/`out_proj` layout is kept). The quantized features **always** take the degraded
  latent's per-channel statistics (`adaptive_instance_normalization`): the reference's signature
  defaults `adain=False`, but its released `inference_codeformer.py` passes `True`, and matching the
  shipping behavior rather than the signature default is a ratified decision. The
  **controllable feature transformation** (`NFKCFFuseBlock`, one per connect resolution 32/64/128/256)
  modulates the generator with a learned scale and shift weighted by the fidelity `w` — 0 is full
  generative quality, 1 keeps the degraded input's detail. Run through `NFKMLXModuleBackend` (aligned
  face → restored face at the model resolution). `photoBackendWithFidelity:weightsURL:` takes a whole
  Photograph: it detects every face, aligns each to the reference's five-point 512 template, restores
  it, and composites the result back through the inverse transform with a feathered edge. Detection and
  alignment are `NFKMLXFaceAlignment`, built on **Vision** — no weights, no download, no third-party
  code, which is the rule the core applies to its own backends. It is not facexlib's RetinaFace, so a
  crop here is not byte-identical to the reference pipeline's and a restored photograph differs slightly
  from it; what the model does to a crop is unchanged, and that is what the parity record measures. The
  alignment is a **similarity** transform (uniform scale, rotation, translation, no shear), solved in
  closed form as one complex multiply over the centered point sets — a full affine would stretch the
  face onto the template exactly and hand the model a distorted subject. Vision reports each feature as
  a contour rather than a point, so an eye is its centroid, the nose is the lowest point of its contour,
  and the mouth corners are the outer lip's extremes in x. An image with no detectable face passes
  through unchanged. The detector is selectable and defaults to RetinaFace, the reference
  pipeline's own, so the crop is the crop facexlib produces; `photoBackendWithFidelity:weightsURL:
  detectorWeightsURL:` takes its 1.7 MB checkpoint. `NFKMLXVisionFaceDetector` is the alternative when
  a download-free path matters more than matching the reference. They disagree measurably — on the validation portrait, box IoU
  0.65 and a worst landmark disagreement of 15.7 px over a 960×1200 frame — so the choice changes the
  restoration, and `NFKMLXFaceAlignmentTests` records that number rather than describing it.
  A landmark assertion cannot validate the crop — the transform maps landmarks
  onto the template by construction, so it stays true however the drawing lands. Only the crop's
  Content can: `testTheAlignedCropContainsTheFace` detects a face inside the crop and checks it fills
  and centers it. That is what caught a real defect here — CoreGraphics orients an image for a y-up
  space, so drawing inside the flipped context produced a crop mirrored about the image's centre (the
  subject's chest instead of the face) while every number stayed in tolerance. Both the crop and the
  paste-back therefore carry a second, per-draw flip. `IK_VAL_FACE` is the portrait the detection tests
  read: a NASA Apollo XI photograph, a US government work and public domain, fetched by
  `Tools/validation-assets/fetch.py` as an `input` asset (no conversion step). `+register` under
  `codeformer`. `w` is the Objective-C knob (`+backendWithFidelity:weightsURL:error:` and its two
  download peers, clamped to 0…1) — the same role the variant enums play for the models that have
  them, since one backend restores at one fidelity. `remapReferenceKey` translates the fuse dictionary's resolution keys and the
  positional Sequentials (`scale.0`, `idx_pred_layer.0/1`) — the coders' `blocks.N` indices land on
  real arrays and pass through. Reference parity against CodeFormer's own architecture on the
  released `codeformer.pth` at the real inference settings (w 0.5, AdaIN on): code logits cosine
  0.9999999999985, **code agreement 1.0**, restored face 0.9999999999987, and the public backend path
  0.9999968 (8-bit CGImage quantization). Forward, fidelity effect, geometry, remap, and round-trip
  tested.
- `NFKMLXZeroDCE` (`@objc`) — a real single-forward low-light enhancer: the Zero-DCE DCE-Net (seven
  3×3 convs with U-style skip concatenations → 24 curve-parameter channels) in `MLXNN`, run through
  `NFKMLXModuleBackend` (dark image → brightened image). Enhancement applies `x = x + r·(x²−x)` eight
  times. `+register` under `zero-dce`. Names match the reference (`e_conv1`…`e_conv7`), so
  `Tools/zero-dce-to-safetensors` only extracts. Forward + round-trip tested.
- `NFKMLXSwinIR` (`@objc`) — real transformer super-resolution: SwinIR (shallow-feature conv → residual
  Swin Transformer blocks → pixel-shuffle upsampler) in `MLXNN`, with real window attention — window
  partition/reverse, cyclic shift with the standard attention mask, and a relative-position bias table
  gathered by a precomputed index. Run through `NFKMLXModuleBackend`; the input side must be a multiple
  of the window size. `+register` under `swinir-x4`. Reference parity against JingyunLiang's own
  `network_swinir.py` on the released `001_classicalSR_DIV2K_s48w8_SwinIR-M_x4` (cosine 0.99986, mean
  pixel |difference| 0.0037). The input is **RGB-mean centered** — the reference subtracts
  `(0.4488, 0.4371, 0.4040)`, scales by `img_range`, and restores it at the end; leaving that out was
  the fifth missing input normalization in this sweep.
  Non-power-of-two scaling is implemented and at parity on the released x3 (cosine 0.99987, mean
  0.0036). The reference `Upsample` reaches a power-of-two scale with repeated ×2 pixel-shuffle stages
  and a scale of three with one ×3 stage, because a factor-3 shuffle is not a composition of factor-2
  ones — so an x3 checkpoint packs `9·C` channels into a single stage where x4 packs `4·C` into each of
  two, and fits only its own geometry. `NFKMLXSwinIRVariant` selects the release, and `NFKMLXSwinIR.makeNet` throws
  `unsupportedConfiguration` for a scale the reference builds no upsampler for, rather than silently
  truncating `log2`. x8 is the same network with a third ×2 stage (0.99990). The lightweight
  release is not the classical network at a smaller size: it reconstructs through the reference's
  `pixelshuffledirect` — one convolution to `3·scale²` channels and a single shuffle, with neither the
  convolution before the upsampler nor the one after it — so `convBeforeUpsample` and `convLast` are
  absent rather than unused, and its checkpoint carries a single `upsample.0` (0.99991, mean 0.00097). The shuffle itself is
  the shared `NFKMLXPixelShuffle`, which BiSeNet, RIFE, VideoSR, and SwinIR all use. Forward, window
  helpers, and round-trip tested.
  Every released SwinIR now loads (`NFKMLXSwinIRVariant`: `.classicalX2`, `.lightweightSRX3`,
  `.lightweightSRX4`, `.realWorldX4Medium`, `.realWorldX4Large` beside the four above), each at float
  parity ≥ 0.9999999999993 with a mean pixel difference under 4e-7. The two real-world GAN releases
  reconstruct through the reference's **`nearest+conv`** upsampler (`NFKMLXSwinIRUpsampler.nearestConv`:
  a 64-wide tail, two nearest-×2 + convolution + leaky 0.2 stages, `conv_hr`, `conv_last`) instead of the
  pixel shuffle, and the large one is 240 wide over nine six-block groups with the **`3conv` residual
  connection** (`NFKMLXSwinIRResidualConnection.threeConv`: a 3×3 → 1×1 → 3×3 squeeze at a quarter width
  with leaky 0.2 between, after every RSTB and after the body; the remap moves `conv_after_body.N` and
  `layers.K.conv.N` onto `conv_after_body_3conv` / `layers.K.conv3`). Reaching them found a real
  defect in the classical tail: `conv_before_upsample` activates with a leaky ReLU at 0.01, not the
  plain ReLU this port had shipped with — on the released classical ×4 through the backend's 8-bit
  bridge the mean pixel difference fell from 0.0037 to 0.00136 (×3 0.0036 → 0.00119, ×8 0.0035 →
  0.00095); the lightweight release has no such tail and is unchanged.
- `NFKMLXColorizer` (`@objc`) — real colorization (Zhang et al. ECCV-16): eight VGG-style conv blocks
  (BatchNorm block ends; blocks 5–6 dilation 2) over the L channel predict a distribution over 313
  quantized ab bins; the annealed mean is the checkpoint's own `model_out` 1×1 conv (renamed
  `out_ab`), so no separate cluster file. `NFKLabColor` implements sRGB ↔ CIELAB (D65) in MLX ops,
  tested against CIE reference values (white L*=100, mid-gray L*=53.39) plus a full-gamut round-trip.
  Predicted ab recombines with the original full-resolution L, preserving luminance exactly. The
  factory sets `train(false)` so BatchNorm uses the checkpoint's running statistics.
  `Tools/colorizer-to-safetensors/convert.py` performs the complete `nn.Sequential` rename and the
  ConvTranspose axis swap (`[in,out,kH,kW]` → `[out,in,kH,kW]`), so the release loads directly —
  no remap is left to Swift. `+register` under `colorizer-eccv16`. Reference parity against the
  released eccv16 (ab cosine 0.9999999998, colorized sRGB cosine 0.9999971). The L and ab resampling is
  bilinear, as the reference's `nn.Upsample` is: nearest neighbour scored the ab prediction at 0.96, so
  this was a real defect. `abPrediction` exposes the network's output before the lightness
  goes back, so a parity failure says network or Lab conversion.
  `NFKMLXSiggraphColorizer` is the second released colorizer — a separate network, not a
  configuration of this one, and at reference parity against richzhang's own `siggraph17.py`
  (ab cosine 0.9999999999996, colorized 0.9999999999). It is 16 blocks (`model1…model10` plus `model8up`/`model9up`/`model10up` and
  the `model{3,2,1}short{8,9,10}` U-Net shortcuts), a **four-channel** input (L, an ab hint, and a
  hint mask), a `model_out` regression head emitting ab directly, and a 529-class `model_class`
  auxiliary head that only supervises training — the loader drops it. Blocks five and six **dilate**
  rather than downsample, and the downsampling elsewhere is a stride-2 **subsample**, not pooling.
  `remapReferenceKey` counts convolution slots per block because the encoder blocks open with a
  convolution while the decoder blocks open with a ReLU, so the same slot number means different
  layers. `+register` under `colorizer-siggraph17`; weights at
  `colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth`, converted with
  `Tools/colorizer-to-safetensors --passthrough` (the eccv16 rename does not apply). With an empty
  hint it colorizes automatically; `predictAB(lightness:hint:mask:)` takes user strokes. Forward, Lab math, bin softmax, and round-trip tested.
