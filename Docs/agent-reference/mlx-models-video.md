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
