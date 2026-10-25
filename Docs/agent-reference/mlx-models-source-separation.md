<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX models: source separation and speech denoising

- `NFKMLXDemucs` / `NFKMLXDemucsBackend` (`@objc`) — real music stem separation: the time-domain Demucs
  U-Net in `MLXNN` (strided Conv1d + GLU encoder, transposed-conv decoder with skips, via `NFKMLXDemucsBackend`
  audio → four stems, each an `NFKAudioAsset` under its name "drums"/"bass"/"other"/"vocals"). `+register`
  under `demucs`. 1-D transposed conv is a `ConvTransposed2d` with a singleton width. Reference parity
  against the released Demucs v2 (per-stem-channel cosine 0.9999999995). `NFKMLXDemucsConfiguration`
  carries everything the two released families differ in, so one network serves both: the music model is
  stereo, six blocks deep, mixes decoder channels over a `context` of 3, runs a **bidirectional**
  bottleneck (`NFKDemucsBLSTM`: forward and reversed passes concatenated, then a linear projection), and
  resamples ×2 through `NFKDemucsFractionalResample` (the polyphase `julius.resample_frac`); the speech
  denoiser is mono, five deep, context 1, causal, and resamples ×4 through the half-sample-shift
  `NFKDemucsResample`. `validLength` follows the reference exactly (ceiling division, plus `context - 1`
  per encoder stage), skips are added with `center_trim`, and `centersOutput` selects the music model's
  centered result trim over the denoiser's head crop. `NFKMLXDemucs.loadWeights` is the shared reference
  loader for both. Parity, round-trip, and per-stem stereo WAV tested.
  Demucs v4 (htdemucs) is `NFKMLXHTDemucs`, a separate architecture — `NFKMLXDemucsNet` is the v2
  time-domain U-Net and no v4 checkpoint fits it.
  **Customization ships** at `full` (`NFKMLXDemucsTraining.swift`), measured against the facebookresearch/demucs
  `v2` branch at 64ed2cb (`run_reference.py demucs_training`) on the released `demucs-e07c671f`.
  - `NFKMLXDemucs.fineTune` trains every parameter on stems `[N, 4, 2, L]`. Each batch is augmented, summed into
    the mix, and run through the unpadded forward (`NFKMLXDemucsNet.estimate`). Its output is shorter than the
    mix by what the decoder's context consumes, and `train.py` holds it to the stems trimmed to its middle by
    `nn.L1Loss`.
  - `NFKMLXDemucsObjective` takes the solver's per-stem weights. `NFKMLXDemucsAugmentation.demucsV2` ports
    `augment.py`'s crop, sign and channel flips, gain, and remix in the v2 order. The reference optimizer is
    `__main__.py`'s Adam at 3e-4, with the LSTMs' folded bias at twice the rate.
  - Measured on two clips of `valid_length(8192)` samples, each figure against float64: the estimate within
    2.36e-6 (reference float32 2.19e-6), the loss 0.10613971 vs 0.10613976, and the global gradient norm
    0.32932547 vs 0.32932603.
  - Eight gradients across the encoder, the bidirectional LSTM, and the decoder land within 8.45e-6 of float64,
    each within 1.05× the reference float32's own distance. The output transposed convolution's is 8.19e-7,
    against the reference's 1.61e-6. The loss after one Adam step is 0.08232483 vs 0.08232479.
  - The augmentation chain, replayed from the reference's draws, is exact.
  - The reference loader's tempo and pitch change runs the external `soundstretch` program and is not ported.
  - `network(weightsURL:)` and `backend(weightsURL:)` read the checkpoint's width, depth, bottleneck layers,
    stem count, and channels from the file, so a checkpoint trained at another size loads.
- `NFKMLXHTDemucs` / `NFKMLXHTDemucsBackend` (`@objc`) — Demucs v4 (Hybrid Transformer Demucs), at
  reference parity on the released `htdemucs` checkpoint (separated stems cosine
  0.9999999999997, mean |difference| 5.6e-8), every parameter covered on the first triage run and
  every stage seam exact on the first numeric run. Two branches run **in parallel**: a spectrogram
  branch over a complex-as-channels STFT (`nFFT` 4096, hop 1024, `torch.stft(normalized:)`, so the
  real and imaginary parts of each audio channel are two feature channels) and a waveform branch over
  the samples. Each is a four-stage U-Net of `HEncLayer`/`HDecLayer` — a strided convolution, a
  dilated `DConv` residual branch (compress ×4, GroupNorm, GELU, expand, GLU, a learned `LayerScale`),
  and a gated rewrite. They never merge by injection: every `tencoder` here is non-empty, so the
  only path between the branches is the **cross-transformer** at the bottleneck — five layers per
  branch alternating self-attention and cross-attention, pre-norm with two `LayerScale` factors and a
  `MyGroupNorm` over the whole sequence, reached through 1×1 channel samplers that widen 384 to 512
  and narrow back. The reconstructions are added. Adds the `MLXFast` product for
  `scaledDotProductAttention`: the bottleneck runs thousands of tokens, where an explicit score matrix
  would be hundreds of megabytes.
  Three details are load-bearing. The spectrogram branch's tokens are **frame-major**
  (`b c fr t -> b (t fr) c`) while the channel sampler flattens the same grid **frequency-major** —
  one grid, two orders. The two positional encodings follow different conventions: the 2-D grid
  alternates sine and cosine with the width in the low channels and the height in the high, and the
  1-D sequence puts all cosines first. `ScaledEmbedding` stores its weight divided by 10 and
  multiplies it back in the forward, so the frequency embedding's effective factor is 2.0, not 0.2.
  `NFKHTDemucsSpectrum` is the transform pair; framing and overlap-add run over Swift buffers because
  MLX has no scatter-add, and a round-trip test asserts the inverse. `separate` pads a short clip to
  the release's 7.8-second training segment and trims back, as the reference does; the parity record
  runs with that off, which is a padding policy rather than a shape. Oracle: `demucs` 4.0.1 is
  installed and `demucs.states.load_model` builds the network from the checkpoint directly, so
  nothing is vendored — but demucs predates torch 2.6, so `torch.load` must be patched to
  `weights_only=False` first. Checkpoint
  `dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th` (81 MB, 533 tensors, 42M
  parameters).
  The other two releases are at parity too. `htdemucs_6s` (`.htdemucs6s`, `NFKMLXHTDemucsVariant.sixStem`,
  registered as `htdemucs-6s`) predicts six stems, guitar and piano after the four
  (`NFKMLXHTDemucsConfiguration.stemNames`), and sets **`bottom_channels` to 0**: it carries no
  `channel_upsampler`/`downsampler` pair at all and runs the cross-transformer at the deepest encoder's
  own width (384, `transformerWidth`) with a 1536-wide feed-forward — so the samplers are optional
  modules, built only when the width is set, and a strict load of the 6s file is what pins that.
  `htdemucs_ft` is a **bag**: four checkpoints of the base geometry, each fine-tuned for one stem,
  combined by per-source weights as the reference's `BagOfModels` does (`NFKMLXHTDemucsBag`, one-hot
  weights so each stem comes from its own model; `backendWithFineTunedWeightsURLs:error:`;
  `run_reference.py htdemucs_bag` drives the same four files). Separated stems 0.99999999999975 (6s)
  and 0.99999999999975 (ft).
  **Customization ships** at `full` (`NFKMLXHTDemucsTraining.swift`), measured against demucs 4.0.1's `solver.py`
  (`run_reference.py htdemucs_training`) on `htdemucs_ft`'s drums model with its dropout at zero.
  - `NFKMLXHTDemucs.fineTune` trains every parameter through both branches. `NFKMLXHTDemucsNet.estimate(_:)`
    separates a batch at the clips' own length, each clip normalized by its own statistics.
  - The STFT pair runs in MLX as radix-4 stages with exactly rounded twiddles, at pocketfft's 1.2e-7 relative
    error. MLX's direct 4,096-point GPU transform carries 6.9e-7, and the gradients through the spectrogram
    branch carried that difference: the first self-attention's input projection stood at 2.5 times the reference
    float32's distance from float64.
  - The GELUs compute torch's error function and return torch's closed-form gradient
    (`NFKReferenceRounding.torchTrainableGELU`). MLX's GPU error function reads high from 0.93 to 2
    (`mlx-runtime-gotchas.md`). Behind MLXNN's GELU, on a random network saved at float16, the bias ahead of
    the first dilated branch's norm stood at 4.9 times the reference float32's distance from float64, and 21
    of 533 gradients past twice it. With these GELUs, 3 are.
  - `NFKMLXHTDemucsTrainingRecipe` holds the two recipes the releases' `training_args` record. `htdemucs` is
    Adam at 3e-4 with every augmentation. `htdemucsFineTune` is AdamW at 1e-4 with 0.05 decay on the
    cross-transformer, the gradient clipped at 5, and the crop and flips alone, with
    `NFKMLXDemucsObjective.stem(_:of:)` for one stem.
  - The releases store their tensors at float16. The loader copies them into float32 parameters, as
    `load_state_dict` does. A float16 parameter returns its gradient at float16, which left the attention
    projections' small gradients 29% off.
  - Measured on two one-second clips, each figure against float64: the estimate within 4.77e-7 (reference
    float32 4.80e-7), the drums and equally weighted losses equal to every printed digit, and the global
    gradient norm 0.061118927 vs 0.061118916.
  - Ten gradients across both branches, the transformer, and the spectrogram branch's last decoder land within
    6.57e-6 of float64. The widest is the first self-attention's input projection, against the reference
    float32's 5.98e-6. Each lands within 1.62 times the reference float32's distance, the time branch's first
    convolution the nearest to that bound. The loss after one step of the fine-tune's AdamW is 0.09660634, float64's to every
    printed digit.
  - The augmentation chain, replayed from the reference's draws, is exact.
  - The transformer drops activations at the releases' 0.02 in training. A new network starts at the
    reference's values: Xavier-uniform attention projections, layer scales of 1e-4 and 1e-3, and residual
    branches compressed by 8.
  - Not ported: the loader's tempo and pitch change (the `soundstretch` program), the batch EMAs that choose a
    checkpoint against a validation set, and `htdemucs_6s`'s D-Adaptation optimizer, which the public solver
    does not build.
- `NFKMLXConvTasNet` (`@objc`) — real time-domain speech separation: a 1-D convolutional encoder, a
  masking temporal convolutional network (depthwise-separable dilated Conv1d blocks with global layer
  normalization `NFKTasNetGlobalNorm` and PReLU), and a shared transposed-conv decoder in `MLXNN`.
  `NFKMLXConvTasNetBackend` reads `NFKInputAudio` → one `NFKAudioAsset` per speaker ("speaker-1",
  "speaker-2", …). `+register` under `conv-tasnet`. Reference parity against `asteroid`'s own
  ConvTasNet on `JorisCos/ConvTasNet_Libri2Mix_sepclean_16k` (per-speaker cosine 0.9999999995).
  `remapReferenceKey` unwraps asteroid's `filterbank` (`_filters`, no bias) and its positional
  `shared_block` Sequential. Every PReLU carries one slope by default, which is the reference's own
  shape — asteroid builds them as `nn.PReLU()`, whose `num_parameters` defaults to 1, and all 49
  slope tensors in the released checkpoint are `[1]`. An earlier note called per-channel slopes a
  sweep item as though the release used them; it does not, and making them the default would diverge
  from it. `perChannelPReLU` offers them anyway, for a fine-tune that wants the capacity: a shared
  slope applied to every channel is the same function, so `loadWeights` **widens** a released `[1]`
  slope to `[C]` against the widths the module reports, the model computes exactly what it computed
  before, and training moves the slopes apart from there. Tested by saving a shared-slope model,
  loading it into a per-channel one, and asserting the separation is unchanged. Forward, separation,
  and round-trip tested.
  **Customization is a FULL fine-tune and it ships** (`NFKMLXConvTasNetTraining.swift`): asteroid
  v0.5.2's `egs/librimix/ConvTasNet`, the recipe the release's model card names.
  `NFKMLXConvTasNet.network(weightsURL:)` builds the net at the checkpoint's own geometry, and
  `fineTune(_:examples:…)` trains every weight on a mixture and each speaker's signal.
  `NFKMLXConvTasNetObjective` is `PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")`: the zero-mean
  negative SI-SDR matrix and the assignment with the lowest mean. Measured against asteroid's own
  modules (`run_reference.py convtasnet_loss`, `testConvTasNetTrainingLossMatchesTheReference`):
  −11.85322 vs −11.853218, every pairwise entry within 8e-6 of values near 50. The optimizer is
  `torch.optim.Adam` at 1e-3 with gradient clipping at 5; the reference's halve-on-plateau schedule and
  early stopping need a validation set, so the recipe holds the rate. The network has no dropout and no
  batch normalization, so training computes the inference function. `backend(weightsURL:)` (`@objc
  backendWithWeightsURL:error:`) reads the filters, kernel, speaker count, bottleneck, and hidden widths
  from the checkpoint through `configuration(matching:)`. It had built `.base` (kernel 16) for every
  file, and the loader applies weights without a shape check, so the kernel-32 16 kHz release loaded
  into the wrong stride and separated wrongly without an error.
- `NFKMLXDenoiser` (`@objc`) — real speech noise suppression (Défossez et al.): the same Demucs
  time-domain U-Net as `NFKMLXDemucs` configured with `stems == 1`, so it reuses `NFKMLXDemucsNet` and
  `NFKMLXDemucs.loadWeights` (DRY). `NFKMLXDenoiserBackend` reads `NFKInputAudio` → one cleaned
  `NFKAudioAsset` under `NFKOutputAudio`. `+register` under `denoiser`. Reference parity against
  facebookresearch/denoiser dns48 (cosine 0.99999999999992), which also guards the shared network
  against a change made for the music model breaking the speech one. Single-output and round-trip tested.
  **Customization ships** at `full` (`NFKMLXDenoiserTraining.swift`), measured against
  facebookresearch/denoiser at 8afd7c1 (`run_reference.py denoiser_training`) on the released dns48.
  - `NFKMLXDenoiser.fineTune` trains every parameter on noisy and clean batches `[N, L]`.
  - `NFKMLXDenoiserObjective` offers `solver.py`'s distances: L1 (the DNS models' loss), L2, and Huber.
    It adds `stft_loss.py`'s multi-resolution STFT term when it is on: off in `.dns`, 0.1 per term in
    `.valentini`. The STFT is a pair of DFT matrices, so a gradient reaches the waveform.
  - `NFKMLXDenoiserAugmentation` ports `augment.py`'s remix, band mask, shift, and reverb in the solver's
    order. They are off by default, and `.dns` and `.valentini` are the two launch scripts.
  - The reference optimizer is `train.py`'s `torch.optim.Adam` at 3e-4, β 0.9 and 0.999, with no clip.
    The bottleneck LSTMs fold the reference's two biases into one, which steps at twice the rate.
  - Measured on a batch of three: the estimate within 1.1e-6. L1 0.035111126 vs 0.03511115, and the STFT
    terms 0.3397696 vs 0.3397504 and 3.196455 vs 3.1964548.
  - The L1 gradients are within 1.2e-5 relative for the first encoder weight, 4.6e-7 for the output
    transposed convolution, and 8.1e-6 for the LSTM bias. The loss after one Adam step is 0.01620323 vs
    0.01620335.
  - The four augmentations, replayed from the reference's recorded draws, are within 1.8e-7.
  - The reference needs two shims on torch 2.8 and numpy 2. `torch.stft` takes `return_complex=True`,
    viewed as real. `LowPassFilters` keeps float32 filters, numpy 1's result.
  - Training needed two network changes, both training-only so inference is unchanged.
    - The half-sample resampler's 112-tap sinc filter runs through
      `NFKMLXGradientSafeConvolution.convolve`.
    - `NFKDemucsConvT1d` runs the 1-D transposed operator, because MLX's GPU weight gradient of the 2-D
      form is wrong past 8,192 positions (`mlx-runtime-gotchas.md`).
  - `NFKMLXDenoiser.network(weightsURL:)` reads the base width from the checkpoint, so dns64 and a
    trained file load. `backend(weightsURL:)` now goes through it; it previously built width 48 whatever
    the file held.
