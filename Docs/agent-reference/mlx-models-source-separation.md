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
- `NFKMLXHTDemucs` / `NFKMLXHTDemucsBackend` (`@objc`) — Demucs v4 (Hybrid Transformer Demucs), at
  reference parity on the released `htdemucs` checkpoint (separated stems cosine
  0.9999999999996, mean |difference| 7.7e-8), every parameter covered on the first triage run and
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
  `run_reference.py htdemucs_bag` drives the same four files). Separated stems 0.99999999999949 (6s)
  and 0.99999999999958 (ft).
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
- `NFKMLXDenoiser` (`@objc`) — real speech noise suppression (Défossez et al.): the same Demucs
  time-domain U-Net as `NFKMLXDemucs` configured with `stems == 1`, so it reuses `NFKMLXDemucsNet` and
  `NFKMLXDemucs.loadWeights` (DRY). `NFKMLXDenoiserBackend` reads `NFKInputAudio` → one cleaned
  `NFKAudioAsset` under `NFKOutputAudio`. `+register` under `denoiser`. Reference parity against
  facebookresearch/denoiser dns48 (cosine 0.99999999999992), which also guards the shared network
  against a change made for the music model breaking the speech one. Single-output and round-trip tested.
