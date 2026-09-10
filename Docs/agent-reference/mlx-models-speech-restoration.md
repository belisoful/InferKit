<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX models: speech restoration and bandwidth extension

- The speech-restoration family shares two front-end primitives (`NFKMLXAudioSTFT.swift`):
  `NFKMLXComplexSTFT` reproduces `torch.stft` / `torch.istft` (`center=true`, `pad_mode="reflect"`,
  `normalized=false`) and returns either magnitude+phase (`transform`/`inverse`) or real+imag
  (`transformComplex`/`inverseComplex`) over one window-squared overlap-add; the window is a value the
  caller supplies (sqrt-Hann or Hann). `NFKMLXERB` is the Glasberg-Moore ERB filterbank. A second shared
  file `NFKMLXRecurrent.swift` carries `NFKMLXGRUCell` / `NFKMLXBiGRU` / `NFKMLXRecurrentFold` — the
  PyTorch `nn.GRU` weight fold (bidirectional → forward/backward cells; `b = bias_ih + [bias_hh[:2H], 0]`,
  `bhn = bias_hh[2H:3H]`). The GRU fix lives here: MLX's GRU drops the n-gate hidden bias `b_hn` at
  step 0 where PyTorch keeps it (`n₁ = tanh(W_in x + b_in + r₁·b_hn)`), so the cell adds `bhn` at every
  step; found in GTCRN, it improved MP-SENet too.
- `NFKMLXMPSENet` / `NFKMLXMPSENetFactory` (`@objc(NFKMLXMPSENet_Factory)`) — MP-SENet
  (`yxlu-0102/MP-SENet`, MIT), the first speech-restoration port: a time-frequency transformer that
  denoises the compressed magnitude and phase in parallel. A DenseEncoder, four **TS-transformer** blocks
  (`norm1 → MHSA → norm2 → FFN → norm3`, the FFN a bidirectional GRU over the shared `NFKMLXBiGRU`), and
  parallel mask / phase decoders producing a complex ratio mask. Reference parity against the
  reference `MPNet` on the released `g_best_dns` (waveform cosine > 0.999, every seam exact). The
  released core is the TS-TRANSFORMER, not the conformer the repo's `conformer.py` describes (it is
  unused). The bug was the `batch_first` trap: `nn.MultiheadAttention` / `nn.GRU` default
  `batch_first=False`, so the block runs over axis 0 of `[B·F, T, C]` — one `transposed(1, 0, 2)` around
  the block body fixed a 0.358 parity to > 0.999. `loadWeights` uses `NFKMLXRecurrentFold.fold` + a
  Sequential-index remap. Config: fftSize 400, hop 100, denseChannel 64, 4 blocks, 4 heads, compress
  0.3. The oracle (`run_reference.py mpsenet`) runs from the cloned source on `/usr/bin/python3` (3.9)
  via `IK_MPSENET_SRC`.
- `NFKMLXGTCRN` / `NFKMLXGTCRNFactory` — GTCRN (`Xiaobin-Rong/gtcrn`, MIT), a **~48.2K-parameter**
  real-time speech enhancer, the second restoration port. An ERB band merge/split (bias-free `Linear`s
  over the shared `NFKMLXComplexSTFT`), an SFE unfold, a grouped-convolution encoder/decoder, a **dual-path
  grouped RNN** (DPGRNN, intra/inter over the shared `NFKMLXGRUCell`), and a complex ratio mask.
  Reference parity on the released `model_trained_on_dns3` (waveform cosine > 0.999, every seam —
  encoder / skips / DPGRNN / decoder — exact). **Five bugs, each caught by seam isolation:** `erb_fc` /
  `ierb_fc` are `bias=False`; the grouped-deconv weight transpose is group-aware
  (`[in, out/g, kH, kW]` → `[out, kH, kW, in/g]`); the shared GRU dropped `b_hn` at step 0 (fixed in
  `NFKMLXRecurrent`); the deconv front-pads time for both conv and deconv with `ConvTranspose` padding
  `(2·dilation, 1)` and no crop; and MLX's grouped `ConvTranspose` does not match PyTorch, so a grouped
  deconv runs each group as its own `groups=1` transpose. STFT is sqrt-Hann. The oracle
  (`run_reference.py gtcrn`) runs from the cloned source via `IK_GTCRN_SRC` (torch only, 3.9).
- `NFKMLXSGMSE` (`@objc`) / `NFKMLXNCSNppNet` — SGMSE+ (`sp-uhh/sgmse`, MIT), score-based generative
  speech **dereverberation** / enhancement, the third restoration port and the first generative one. A
  forward OUVE variance-exploding SDE walks a clean complex spectrogram toward the observation; inference
  runs the reverse SDE (a predictor-corrector sampler) from `x_T = y + noise` to `x_0`, scored by an
  NCSN++ network, then inverts the STFT. `NFKMLXNCSNppNet` is ported as the reference's flat `all_modules`
  list (a `[Module]` array, keys match with no remap) walked by an index counter that mirrors the
  reference forward, plus a separate `output_layer`. The one new op is `NFKSGMSEFIR.upfirdn2d` — a
  depthwise FIR resample (kernel `[1,3,3,1]`) the DAC/SNAC resamplers resemble. The complex spectrogram
  packs into four real channels (`[xt.re, xt.im, y.re, y.im]`); the sampler is the OUVE SDE
  (`NFKMLXOUVEScheduler`, a value type with the closed-form mean / std / diffusion) driven by a
  reverse-diffusion predictor plus an annealed-Langevin corrector; the front end is a sqrt-Hann or Hann
  STFT with the `|X|^a · factor` amplitude compression and the time axis padded to a multiple of 64.
  Inference reads the EMA weights: `Tools/sgmse-to-safetensors` lets `torch_ema` apply them
  (`model.eval()` → `ema.copy_to(dnn)`) then dumps `dnn.state_dict()`. At reference parity on the
  released EMA weights — net-seam cosine 1.000000000000 on both released backbone variants: the classic
  `ncsnpp` (attention + progressive input/output skip, `sp-uhh/speech-enhancement-sgmse`) and `ncsnpp_48k`
  (no attention, `progressive='none'`, plain Hann, and the output projection applied before the sigma
  division; the ReverbFX release). The port is config-driven for both (`progressiveOutputSkip` /
  `windowPower` / `attentionResolutions`), and the oracle (`run_reference.py sgmse`, source via
  `IK_SGMSE_SRC`) records the net geometry from the DNN's own attributes so the Swift parity test builds a
  matching config. `backbone='ncsnpp_v2'` (a different two-arg forward) is out of scope. Two traps:
  `torch >= 2.6` defaults `weights_only=True` and refuses the pickled data module, so the converter and
  oracle patch `torch.load`; and the net geometry must match the front-end freq bins or the flat
  `all_modules` walk desyncs (the config carries both). A sampled clip is not bitwise-comparable (random
  stream), so the deterministic net seam is the numeric ground and the e2e asserts signal.
- `NFKMLXStoRM` (`@objc`) / `NFKMLXStoRMNet` — StoRM (`sp-uhh/storm`, MIT), a few-STEP stochastic-
  regeneration follow-on on SGMSE+. A discriminative predictor produces an initial denoised estimate,
  then the score network regenerates from it: the reverse SDE is re-centered on the denoised estimate and
  the score conditions on `[noisy, denoised]`, so the diffusion repairs only residual artifacts in far
  fewer steps (default corrector `none`). Both networks are `NFKMLXNCSNppNet`, which was generalized for
  the two roles — `inputChannels` (2 for the predictor, 6 for the `condition='both'` score), `conditional`
  (the predictor runs `discriminative=True` → no Gaussian-Fourier time embedding, the Dense weights load
  but are not applied), and `scaleBySigma` (off for the predictor) — and SGMSE+ stayed at parity as the
  `inputChannels=4`/conditional/scaled case. The sampler (`NFKSGMSESampler`) gained an `observation` (the
  SDE center = `y_denoised`) separate from the `conditioning` channels the score net reads, and a
  `useCorrector` flag. Keys mirror the reference `StochasticRegenerationModel` (`denoiser_net.*` /
  `score_net.*`), so the converted EMA safetensors loads with no remap. At reference parity at a tiny
  random configuration (denoiser seam and score seam cosine 1.000000000000): the released combined
  checkpoints are GDrive-only, and the NCSN++ backbone is already at released-weight parity via SGMSE+, so
  the tiny-random oracle (`run_reference.py storm`, built from the backbone registry, saving both nets'
  weights into the record under `w::…`) validates the new two-net architecture exactly. The StoRM clone
  omits `upfirdn2d_native.py` and its op imports the fused CUDA extension, so the oracle injects an inline
  native `upfirdn2d` + a leaky-ReLU shim into `sys.modules`. Registered under `storm`.
- `NFKMLXMossFormer2SENet` / `NFKMLXMossFormer2Factory` (`@objc(NFKMLXMossFormer2_Factory)`) — MossFormer2
  SE 48K (modelscope/ClearerVoice-Studio, Apache-2.0), full-band speech enhancement, at reference
  parity on the released `last_best_checkpoint.pt`, measured on the M1 at float32: the Kaldi fbank+Δ
  **1.0000000**, the encoder and FLASH block 0 **0.99999994**, FLASH block last and the 961-bin mask
  **1.0000000**, and the enhanced waveform **0.9999998**. The shared MossFormer2 backbone is a mask-predicting
  net over a Kaldi-fbank front end: a `GroupNorm(1)` input norm, a `Conv1d` bottleneck, a scaled sinusoidal
  positional embedding, 24 `MossformerBlock_GFSMN` layers, and a gated output to a real 961-bin mask
  (final ReLU). Each block interleaves `FLASH_ShareA_FFConvM` (gated single-head attention: quadratic
  ReLU-squared local attention within 256-groups + a linear global path, `to_hidden`/`to_qk` as `FFConvM`
  norm→Linear→SiLU→depthwise-`ConvModule`, an `OffsetScale(heads=4)`, adjacent-pair rotary over the first
  32 of the 128 qk dims, gate `(att_u·v)·sigmoid(att_v·u)`) and a `Gated_FSMN_Block` (a `UniDeepFsmn`
  depthwise `Conv2d[39,1]` memory, the Chatterbox-S3 FSMN family). Norms are the `ScaleNorm`/`CLayerNorm`/
  `LayerNorm(1e-6)` zoo. The front end is `NFKMLXKaldiFbank` — `torchaudio.compliance.kaldi.fbank`
  reproduced (DC-removal, pre-emphasis 0.97, Povey/hamming, pow2-padded FFT, kaldi-mel, log) + `compute_deltas`
  ×2 → 180-dim, with `dither` forced to 0 (it is random noise; parity is impossible with it on). The
  mask multiplies a hamming/`center=false` STFT (phase kept) and inverts. Two facts are load-bearing, both
  found on the M1: **`num_spks=2`** (the wrapper builds the MaskNet with the default, so `conv1d_out`
  widens to `dModel·2` and the net returns speaker 0 — the port slices the first `dModel` channels, exact
  because the 1×1 gated output and decoder are per-position), and the FLASH linear-attention path divides
  by the original sequence length `n = x.shape[-2]`, not the padded length (only wrong when the frame
  count is not a multiple of the group size; it dragged FLASH block 0 to 0.9975, localized by the block
  seams). `remapReferenceKey` strips the `mossformer.` wrapper prefix, drops the pos-enc/rotary buffers,
  and translates the `FFConvM.mdl`/`ConvModule`/`Gated_FSMN_Block.conv1`/output-gate Sequential indices.
  The oracle is `run_reference.py mossformer2_se` (dither=0; records the 180-dim feature, STFT, encoder,
  first/last block, mask, waveform), run against the source files (`IK_MOSSFORMER2_SE_SRC`) on
  `~/.inferkit-validation/llmvenv` (needs `rotary_embedding_torch` + `torchinfo`); the parity test feeds
  the recorded feature to isolate the backbone from the fbank. The SR sibling is `NFKMLXMossFormer2SRNet`
  below. Registered under `mossformer2-se`.
- `NFKMLXDeepFilterNet` / `NFKMLXDeepFilterNetBackend` (`@objc(NFKMLXDeepFilterNet_Factory)`) —
  **DeepFilterNet3** (Rikorose/DeepFilterNet, dual **MIT/Apache-2.0**), a ~2.3M-parameter real-time
  48 kHz speech denoiser, the cheap counterpart to `NFKMLXDenoiser`. The `DfNet` is a clean torch
  `nn.Module`: an **encoder** (an ERB convolution pathway `erb_conv0..3` + a DF convolution pathway
  `df_conv0/1`, summed — `enc_concat=False` — into a `SqueezedGRU_S` embedding), an **ERB decoder** (a
  second `SqueezedGRU_S` and a U-net of depthwise-1x1 skips + (transposed) convolutions → a 32-band
  sigmoid ERB mask), and a **DF decoder** (a `SqueezedGRU_S` + a grouped-linear skip → the deep-filter
  coefficients `[B, 5, T, 96, 2]`). The enhanced spectrum is the ERB mask applied to the full spectrum,
  with the lowest 96 bins replaced by a 5-tap causal complex deep filter (`MF.DF`, a per-frame
  complex MAC over a `(dfOrder-1-lookahead, lookahead)`-padded window). The STFT / ERB / normalization
  DSP is Rust `libdf` in the reference, reproduced in `NFKMLXDeepFilterNetDSP` (MLX + Swift) and each
  step validated numerically against a libdf recording: **analysis** left-pads by `nFFT-hop`, windows
  with the **VORBIS window** `sin(π/2·sin²(π(n+0.5)/N))`, rffts, and scales **`1/N`**; the **ERB
  feature** is `10·log10(|spec|²·erb_fb + 1e-10)` then a per-band EMA mean-normalization `(x−s)/40`
  (α = 0.99, `s` init `linspace(-60,-90,32)` = `MEAN_NORM_INIT`); **unit_norm** on the lowest 96 bins is
  `x/√s`, `s` init `linspace(0.001,0.0001,96)` = `UNIT_NORM_INIT`; **synthesis** is `irfft(spec·N)` with
  window-squared overlap-add. Reference parity on the released DeepFilterNet3 weights, seam by seam
  and end to end against the `deepfilternet` pip package: every net seam exact (encoder `e0..e3` /
  `emb` / `c0`, the ERB mask `m`, the deep-filter coefficients, `spec_e` all cosine 1.0000000), the DSP
  features exact (spec / feat_erb / feat_spec 1.0000000), and the enhanced waveform end to end
  **0.9999999**. `+register` under `deepfilternet3`.
  Seven facts are load-bearing, most found by the seam ladder. The `Conv2dNormAct` layout is
  derived from the shapes, not a per-layer flag: `groups = gcd(in, out)`, a pointwise 1x1 follows
  only when `groups > 1` And the kernel is not 1x1, and a causal time pad precedes the conv only when the
  time kernel exceeds 1 — so `erb_conv0` (`gcd(1,64)=1`) is a plain conv while `conv3p` (`gcd(64,64)=64`,
  1x1) is a **depthwise 1x1**. `SqueezedGRU_S`'s `linear_out` carries a trailing **ReLU** (missing it
  dropped `emb` to 0.68), and its `df_gru` variant has `output_size=None` → no `linear_out` (an
  Identity) while running its grouped linears at **8** groups where `emb_gru` runs 16. The DF skip fed
  to the decoders is `c0` (the `df_conv0` output), not `c1`. `df_fc_a` and `pad_spec` exist in the
  checkpoint/module but the reference `forward` never applies them (loaded, unused — there is **no alpha
  blend**). The ERB decoder's `convt2`/`convt1` are grouped depthwise `ConvTranspose2d` (the shared
  GTCRN per-group workaround + the `[in,out/g,kH,kW]→[out,kH,kW,in/g]` transpose, with `output_padding`).
  **The one oracle trap:** `df_state.synthesis(as_complex(spec_e).numpy())` writes IN place through the
  view, corrupting a `spec_e` recorded afterward — snapshot it before synthesis. The oracle is
  `run_reference.py deepfilternet --checkpoint <DeepFilterNet3 dir>` (the `dfnvenv`: `deepfilternet` +
  `deepfilterlib` + torch, Python 3.9; `init_df` downloads the model, and `IK_DEEPFILTERNET_WEIGHTS_OUT`
  dumps the state dict as the weights). No offline converter: the pip package ships the weights and the
  DSP oracle.
- `NFKMLXVoiceRestore` / `NFKMLXBigVGAN` / `NFKMLXVoiceRestoreBackend` (`@objc(NFKMLXVoiceRestore_Factory)`)
  — **VoiceRestore** (skirdey/voicerestore, **MIT**), a ~301M-parameter flow-matching (CFM) universal
  speech restorer, text-free, that fixes noise, reverberation, clipping, and band-limiting together in
  one model. **E2-TTS-derived, not F5** (deps pin `x-transformers==1.34.0`,
  `gateloop-transformer==0.2.5`). The pipeline (24 kHz): degraded audio → a BigVGAN log-mel → a CFM ODE
  `dx/dt = v_θ(x_t, t | degraded_mel)` from noise to the restored mel → BigVGAN → waveform. At
  reference parity on the released weights, seam by seam and end to end.
  **The velocity net** `NFKMLXVoiceRestore` is the vendored E2-TTS transformer: the condition is
  **Additive and frame-aligned** (`x = proj_in(x_t) + cond_proj(degraded_mel)`, not F5's concat), the
  sequence carries 32 learned register tokens prepended (unpacked after the blocks) plus an absolute
  positional embedding on the mel frames, and each of the 20 blocks is a **residual
  `SimpleGateLoopLayer`** → adaptive-RMSNorm attention (adaLN-zero gated) → adaptive-RMSNorm GEGLU
  feed-forward (adaLN-zero gated), with U-net concat skips over the second half. At reference parity
  against the `x-transformers 1.34.0` / `gateloop 0.2.5` reference (`run_reference.py voicerestore`):
  every seam exact (`x_in` / each block's gateloop, attention, feed-forward / the final velocity all
  cosine 0.99999994–1.0). The **`SimpleGateLoopLayer`** (the one research-grade piece) is a data-dependent
  gated linear recurrence: an RMSNorm, a bias-free `Linear(dim, 3·dim)` → q/kv/a, a **sigmoid** forget
  gate, a per-channel first-order recurrence `h_t = a_t·h_{t-1} + kv_t`, and the output `q_t·h_t` — a
  sequential scan, which is exact in MLX (no associative scan needed). The **x-transformers `Attention`**
  carries `gate_value_heads` (a per-head sigmoid value gate) and `softclamp_logits` (`tanh(logit/50)·50`);
  **`AdaptiveRMSNorm`** is `F.normalize(x)·√dim·(1 + γ)` and **`AdaLNZero`** is a `-2`-biased **sigmoid**
  gate; rotary is adjacent-pair over `dim_head`.
  **The vocoder** `NFKMLXBigVGAN` is BigVGAN v2 (`nvidia/bigvgan_v2_24khz_100band_256x`, MIT), a
  HiFi-GAN-style generator with two BigVGAN additions: **SnakeBeta** periodic activations
  (`x + (1/(exp(β)+1e-9))·sin²(exp(α)·x)`) and an **anti-aliased `Activation1d`** — a fixed kaiser-sinc
  up/down FIR (cutoff 0.25, half-width 0.3, kernel 12) around each activation, which this port
  **recomputes** (the Bessel-I0 Kaiser window in Swift) rather than loading. All convolutions are
  weight-normed (`g·v/‖v‖`, the shared `NFKMLXMusic3.fusedWeightNorm`); the `ups` are ConvTranspose1d.
  At reference parity against the released generator end to end (mel → waveform cosine 0.9999997,
  `run_reference.py bigvgan`). The mel front end reproduces `meldataset.mel_spectrogram` (n_fft 1024,
  hop 256, win 1024, fmin 0, fmax 12000, Hann, center=False with a reflect pad of `(n_fft-hop)/2`,
  magnitude `sqrt(·+1e-9)`, `log(clamp(·, 1e-5))`, the shared Slaney filterbank). **The sampler**
  reproduces `VoiceRestore.sample` (`torchdiffeq` fixed-step **midpoint**, `times = linspace(0, 1, steps)`,
  default 32 steps, classifier-free guidance 0.5). End to end at reference parity with the sampler
  seeded from the reference's `y0` (`run_reference.py voicerestore_e2e`): the mel 1.0, the restored mel
  0.9999999, the restored waveform ~1.0. `+register` under `voicerestore` (a directory holding the
  transformer checkpoint + `bigvgan_generator.pt`).
  **The costliest debugging was an oracle bug, not a port bug:** the transformer was correct as first
  written; the seam hooks fired during both the conditioned velocity pass and the CFG null pass, and the
  null pass overwrote every recorded seam — so conditioned inputs were being compared against null-pass
  seams. Removing the hooks before the null pass turned every seam green at once. The oracle runs in a
  dedicated `vrvenv` (Python 3.9, torch 2.2.2, the pinned x-transformers / gateloop / torchdiffeq, plus
  jaxtyping). Weights: `jadechoghari/VoiceRestore/pytorch_model.bin` (the transformer, keyed
  `transformer.*` / `proj_in` / `cond_proj` / `to_pred`, loaded through the native torch reader; the
  `abs_pos_emb` is 2000 rows) and `nvidia/bigvgan_v2_24khz_100band_256x/bigvgan_generator.pt`. No offline
  converter.
- `NFKMLXResembleEnhance` / `NFKMLXResembleEnhanceBackend` (`@objc(NFKMLXResembleEnhance_Factory)`) —
  **Resemble Enhance** (resemble-ai, **MIT**), a five-network general speech restorer (noise +
  reverberation + clipping + band-limiting together), the eighth restoration-vein port and the largest of
  the family. All five networks plus the mel front end are at reference parity on the released
  enhancer_stage2 weights, seam by seam and end to end (`run_reference.py reenhance_*`, the
  `reenhancevenv` oracle env — Python 3.12; the torch source is the only oracle, no community MLX port
  existed): mel 0.9999998, IRMAE encode 0.9999985 / decode 1.0000002, CFM velocity 1.0000001 / sample
  1.0000001, UnivNet 0.9999365, denoiser 0.9999996, and the full `enhance()` waveform 0.9999971. The
  `enhance()` path (`resemble_enhance/enhancer/{enhancer,inference}.py`, defaults nfe 32 / lambd 0.5 /
  tau 0.5): peak-normalize → mel → the mix mel and the denoised mel blended by `lambd` → the LCFM stage
  samples a latent from the encoded-prior-plus-noise (`tau`) → decode to the vocoder input → the UnivNet
  vocoder.
  - **Mel front end** (`NFKMLXResembleMel`): `resemble_enhance.melspec.MelSpectrogram` — preemphasis
    0.97, a torchaudio magnitude mel (Slaney scale + Slaney normalization, reusing `NFKMLXMel.melFilters`,
    a zero-centered STFT `pad_mode="constant"`, n_fft 2048 / hop 420 / 128 mels), `amp_to_db`
    (`clamp(1e-4).log10()·20`), and a headroom normalization `(s + 80) / 95`. `to_mel` drops the last
    frame. A global scalar `Normalizer` (`(x − mean) / std`, `std = sqrt(var + 1e-9)`) loaded from the
    checkpoint follows it.
  - **IRMAE** (`NFKMLXResembleIRMAE`): an implicit-rank-minimizing autoencoder. The encoder (a 1024-wide
    conv, four dilated GroupNorm/GELU ResBlocks, four bias-free 1×1 rank-minimizing convs, a Tanh)
    compresses the 128-mel to a 64-channel latent; the decoder mirrors it to the 160-channel
    (`num_mels + vocoder_extra_dim` 32) vocoder input. The training-only `head` and `estimator` are not
    built. `GroupNorm(pytorchCompatible: true)`.
  - **CFM** (`NFKMLXResembleCFM`): the flow-matching stage. The velocity net is a **WaveNet**
    (`NFKMLXResembleWN`, a DiffWave-style stack of 30 gated dilated-conv layers, dilation cycle 5, an
    InstanceNorm on the local condition, a `SinusodialTimeEmbedding`), not a transformer. The sampler is
    the reference's **exponential-decay midpoint ODE**: `ts = h(linspace(0,1,n+1))` with
    `h(t) = (a^t − 1)/(a − 1)`, `a` solving `h(1/4) = 0.5` (Newton, matching scipy `fsolve`); nfe 32 →
    16 midpoint steps. Distinct from VoiceRestore's plain-linspace midpoint.
  - **UnivNet** (`NFKMLXResembleUnivNet`): a GAN vocoder over **location-variable convolutions**. A noise
    input through `conv_pre` (reflect-padded), four `LVCBlock`s (each: an upsampling transposed conv at
    stride 7/5/4/3 = 420 = hop, an anti-aliased-SnakeBeta AMP block reusing the BigVGAN kaiser-sinc FIR
    family, then four dilated conv stages whose kernels a `KernelPredictor` generates per cond segment and
    applies through a GAU gate), then `conv_post` (LeakyReLU → conv → Tanh). The LVC (dilation 1) is a
    per-segment im2col matmul (MLX has no unfold). The noise is non-deterministic; parity feeds a recorded
    `z`.
  - **Denoiser** (`NFKMLXResembleDenoiser`): the stage-1 STFT-mask model — a complex STFT (reusing
    `NFKMLXComplexSTFT`, n_fft 1680 / hop 420), a 2-D (frequency × time) UNet predicting a magnitude mask
    and a phase residual, and the inverse STFT.
  Three facts are load-bearing, all found by seam localization. The KernelPredictor's LeakyReLU is
  slope **0.2** (the LVCBlock overrides the KernelPredictor's own 0.1 default) — with 0.1 the vocoder
  scores ~0.84. The LVCBlock `convt_pre` is `Sequential(LeakyReLU, ConvTranspose)`, so the activation runs
  Before the transposed conv. And the oracle must load the released `hparams.yaml`, because the enhancer
  default `lcfm_z_scale` is 5 but the release is **6** — a difference cosine cannot see (it changes only
  the encoded-prior/noise blend magnitude) and that surfaces only in the end-to-end path. The DeepSpeed
  shard nests everything under `module`, which the native reader does not unwrap, so the loader strips
  that prefix; it uses old `weight_g`/`weight_v` weight-norm (`fusedWeightNorm` handles it). `+register`
  under `resemble-enhance`; `@objc backendWithDirectoryURL:error:` (the `enhancer_stage2` dir holding
  `ds/G/default/mp_rank_00_model_states.pt`). Weights: `ResembleAI/resemble-enhance` (MIT). No offline
  converter (the native torch reader loads the shard). The oracle imports the released `resemble_enhance`
  leaf modules directly to avoid deepspeed (which the top-level `enhancer.py` pulls in).
- `NFKMLXMetricGANPlus` / `NFKMLXMetricGANPlusBackend` (`@objc`) — **MetricGAN+**
  (`speechbrain/metricgan-plus-voicebank`, Apache-2.0), the smallest member of the restoration family
  and the roadmap's "plumbing smoke test" for it: a magnitude-mask enhancer whose generator is a
  two-layer bidirectional LSTM (257 → 200 per direction) over `log1p(|X|)` frames, then Linear 400→300,
  LeakyReLU(0.3), Linear 300→257, and a per-bin learnable sigmoid `1.2 · sigmoid(slope · x)`. The
  enhanced magnitude is `expm1(mask · features)` under the noisy phase, inverted, then peak-normalized
  (`x / (max|x| + 1e-14)`). At reference parity on the released weights on the first numeric run
  against speechbrain's own `SpectralMaskEnhancement` (`run_reference.py metricgan`, the `llm` env plus
  the `speechbrain` package, `IK_PARITY_METRICGAN` + `IK_VAL_METRICGAN`): features 1.0, mask 1.0000001,
  enhanced waveform 1.0000001 (float32 cosines). Two front-end facts are load-bearing. speechbrain's
  STFT pads with zeros (`pad_mode="constant"`), where `torch.stft` and the shared `NFKMLXComplexSTFT`
  reflect; the shared transform gained `zeroPadded` for it (Parakeet's front end had the same fact,
  measured there at 0.974 the other way). And speechbrain's `resynthesize` calls `istft` with
  `sig_length = the input length`, which keeps the last frame's tail past the symmetric center trim
  (48000 samples where the plain inverse returns 47872), so the shared inverse gained `torch.istft`'s
  `length`. The window is a 512-sample periodic Hamming at hop 256. The release is a plain state dict
  the native torch reader opens; the two LSTM layers fold through the shared PyTorch→MLX
  `Wx`/`Wh`/`bias` treatment under `blstm.N.forward` / `.reverse` (the gate order matches, so the
  matrices transfer as they are), and `Learnable_sigmoid.slope` loads by name. `+register` under
  `metricgan-plus`; `backendWithWeightsURL:` and the repo / async peers; the gallery example runs it.
- `NFKMLXCMGAN` / `NFKMLXCMGANNet` / `NFKMLXCMGANBackend` (`@objc`) — **CMGAN** (`ruizhecao96/CMGAN`,
  MIT), a 1.83M-parameter conformer-based metric GAN whose generator `TSCNet` denoises a
  power-compressed (`mag^0.3`) complex spectrogram: a dense encoder (a 1×1 convolution, a four-layer
  dilated dense net, a `(1,3)` stride-`(1,2)` convolution halving the frequency axis) over
  `[magnitude, real, imaginary]`, four **two-stage conformer blocks** (a lucidrains conformer over time
  with each frequency a sequence, then one over frequency with each frame a sequence, each residual:
  half-weighted macaron feed-forwards, attention with Shaw's relative position embedding — a learned
  `[1025, 16]` table indexed by the clamped query-key distance, dotted with the query — a GLU →
  depthwise-31 → BatchNorm → Swish convolution module, and a post-norm), then a magnitude-mask decoder
  (the dense net, the MP-SENet sub-pixel frequency upsample, a `(1,2)` convolution to one channel, a
  per-bin `prelu_out` initialized at -0.25) and a complex-residual decoder. The output is
  `mask · mag` under the noisy phase plus the residual, decompressed `^(1/0.3)`. The front end is
  `evaluation.enhance_one_track`: the clip scaled to unit RMS (`c`), padded to a multiple of the hop by
  repeating its first samples, a 400-point periodic-Hamming STFT at hop 100 (center, reflect), and the
  output divided by `c` and trimmed. Only the generator runs; the discriminator is a training device.
  The MP-SENet blocks are reused directly (`NFKMPSEDenseConv`, `NFKMPSESubpixelUp`) — MP-SENet
  descends from CMGAN — and the reference's `nn.Sequential`s are held as `[Module]` arrays so the
  numeric keys match with no remap; only the dense nets' flat `conv{i}` / `norm{i}` / `prelu{i}`
  attributes and the `TSCB_{i}` blocks map onto arrays. At reference parity on the released weights on
  the first numeric run against the repository's own `TSCNet` (`run_reference.py cmgan`, the `llm` env,
  `IK_CMGAN_SRC` = the cloned `src/`, `IK_PARITY_CMGAN` + `IK_VAL_CMGAN`, the repository's own noisy
  VCTK-DEMAND clip `p232_052`): compressed spectrum 0.99999994, encoder 1.0, TSCB 1–4 1.0 / 1.0 /
  0.9999998 / 1.0, mask 1.0, complex residual 1.0, final real / imaginary 0.99999994 / 1.0, enhanced
  waveform 0.99999994. The released `ckpt` is a plain state dict the native torch reader opens.
  `+register` under `cmgan`.
- `NFKMLXFRCRN` / `NFKMLXFRCRNNet` / `NFKMLXFRCRNBackend` (`@objc`) — **FRCRN SE 16K**
  (modelscope/ClearerVoice-Studio, `alibabasglab/FRCRN_SE_16K`, Apache-2.0), a frequency-recurrent
  complex CRN: two complex UNets (`unet`, then `unet2` reading the first's raw output) over a
  convolutional STFT (a 640-point square-root periodic Hann at hop 320, **no centering**, the
  reference's `ConvSTFT` kernel, whose pseudo-inverse is the windowed irfft), the mask
  `tanh(unet2) + tanh(unet1)` applied as a complex product. Each UNet is seven complex encoders
  (`(5,2)` kernels over `(frequency, time)` at stride `(2,1)` padding `(0,1)`, so every stage halves
  the 321 bins to one and adds a frame; the decoders' `(·,2)` transposed convolutions remove it) with
  a **frequency-recurrent FSMN** before each encoder but the first (`ComplexUniDeepFsmn_L1`: each
  Frame is a sequence over the frequency axis, a causal 20-tap depthwise memory added to a
  Linear→ReLU→Linear projection, the whole residual; the complex form pairs `re`/`im` sub-nets as
  `re(x_re) − im(x_im)`, `re(x_im) + im(x_re)`), a **complex squeeze-excite** after each (`SELayer`:
  real and imaginary parts pooled and gated separately, the two gates combined as a complex product,
  then applied part by part, an elementwise scale rather than a complex multiply), a two-layer FSMN over
  Time at the one-bin bottleneck, and the encoders' excited outputs concatenated on channels into the
  decoders. Complex BatchNorm is two BatchNorms (eval), LeakyReLU at 0.01. The released checkpoint
  stores every stage twice, as flat `encoder{i}` / `fsmn_enc{i}` / `se_layer_enc{i}` attributes and
  as the `ModuleList`s `encoders.{i}` …; the module is keyed by the lists and the loader drops the flat
  copies (the MODNet backbone trap again). Three tensors exist but never run (`fsmn_enc0`,
  `fsmn_dec6`, `se_layer_dec5`) and are declared so the strict load holds. The consumer path
  reproduces `decode_one_audio_frcrn_se_16k`'s zero padding (to a 1 s window, to window + 0.75 s
  stride, or past that by `t − ⌊(t − window)/stride⌋·stride` off the stride grid): the FSMN memories
  and the squeeze-excites' global pools read the padded clip, so the padding changes every output
  sample and is part of the model's input; the output is trimmed back to the input length. At
  reference parity on the released weights on the first numeric run against ClearerVoice's own
  `DCCRN` (`run_reference.py frcrn`, the `llm` env, `IK_FRCRN_SRC` = the curled `models/frcrn_se/`
  sources, `IK_PARITY_FRCRN` + `IK_VAL_FRCRN`, the CMGAN noisy clip padded to 58368): conv-STFT
  spectrum 0.9999999, encoder 0 1.0, its squeeze-excite 1.0, the bottleneck FSMN 1.0, decoder 0
  0.99999994, the first UNet 1.0, mask 1.0, masked spectrum 1.0, enhanced waveform 1.0. The shared
  `NFKMLXComplexSTFT` gained `centered: false` for it. The FSMN's `[C, 1, order, 1]` depthwise memory
  loads as a 1-D `[C, order, 1]` convolution; the transposed convolutions through `(1, 2, 3, 0)`.
  `+register` under `frcrn`; weights `alibabasglab/FRCRN_SE_16K/last_best_checkpoint.pt` (161 MB).
- `NFKMLXMossFormer2SRNet` / `NFKMLXMossFormer2SRGenerator` / `NFKMLXMossFormer2SRFactory`
  (`@objc(NFKMLXMossFormer2SR_Factory)`) — **MossFormer2 SR 48K** (modelscope/ClearerVoice-Studio,
  `alibabasglab/MossFormer2_SR_48K`, Apache-2.0), speech super-resolution (bandwidth extension), the
  SR sibling the SE entry promised. Three stages plus a DSP post-process: the HiFi-GAN log-mel
  (`meldataset.mel_spectrogram` at 48 kHz, 1024/256, 80 bands to 8 kHz — the shared
  `NFKMLXVoiceRestoreMel`, now parameterized), the **mel-to-mel MossFormer2 backbone** (the shipped
  `NFKMLXMossFormer2SENet` under `NFKMLXMossFormer2Configuration.superResolution`: 80 in, 80 out,
  `num_spks` 1 — the reference's block / FSMN / conv-module sources are byte-identical to the SE
  ones, so nothing in the backbone is new; its final ReLU stays, so the restored log-mel is clipped
  at zero), a **Snake HiFi-GAN generator** (`ResBlock1` with per-channel Snake activations in place
  of every leaky ReLU, a Snake before each of the four transposed-convolution upsamples `[8, 8, 2, 2]`
  = the 256 hop, `snake_post`, `conv_post`, `tanh`; the DAC Snake `NFKMusic3Snake` is reused), and
  **`bandwidth_sub`**, the decode path's scipy post-process ported in double precision
  (`NFKMossBandwidthSubstitution`): the input's effective bandwidth is the first bin where a 256-point
  Hann STFT's cumulative energy (zero boundary padding, `scipy.signal.stft` defaults) reaches 0.9996,
  the input is kept below it through a fourth-order Butterworth low-pass and the generator's output
  added above it through the matching high-pass (both `scipy.signal.butter` — prototype poles,
  pre-warp, bilinear at fs 2, `zpk2tf` — under `filtfilt`'s odd extension of 15, `lfilter_zi`
  initial state, forward and backward passes), and the result crossfades from the input over the
  first 100 ms. At reference parity on the released weights on the first numeric run against
  ClearerVoice's own `Mossformer` + `Generator` + `bandwidth_sub` (`run_reference.py mossformer2_sr`,
  the `llm` env plus `pydub` for the `meldataset` import, `IK_MOSSFORMER2_SR_SRC`,
  `IK_PARITY_MOSSFORMER2_SR` + `IK_VAL_MOSSFORMER2_SR`; the CMGAN clean 16 kHz clip resampled to
  48 kHz by torchaudio and recorded, so both sides read one waveform): mel 1.0, backbone 1.0,
  generator 0.9999999999987, the detected cutoff 6937.5 Hz exactly, the substitution 1.0, and the
  whole path 0.9999999999992. The backbone checkpoint is `{"mossformer": {"mossformer.…"}}`: the
  container name is not one the native reader unwraps, so it lands as a doubled prefix the loader
  strips before the SE remap; the generator checkpoint (`{"generator": …}`) is weight-normed
  (`fusedWeightNorm`) with the Snake `alpha` moving from `[1, C, 1]` to `[1, 1, C]` through the same
  3-D transpose the convolutions take. `+register` under `mossformer2-sr`; `@objc
  backendWithDirectoryURL:error:` (the directory holding `last_best_checkpoint_m.pt` and
  `last_best_checkpoint_g.pt`, 220 MB each). The reference's long-clip sliding window (past 20 s) is
  not reproduced; a clip runs whole.
- `NFKMLXNUWave2` / `NFKMLXNUWave2Net` / `NFKMLXNUWave2Backend` (`@objc`) — **NU-Wave 2** (maum-ai,
  BSD-3), diffusion bandwidth extension and the family's first generative up-sampler: a
  WaveGrad-style noise predictor whose 15 residual blocks are **short-time Fourier convolutions**
  (`FFC`): the 64 channels split into a local half (3-tap convolutions) and a global half whose
  `SpectralTransform` takes every channel's normalized STFT (1024/256, periodic Hann, center reflect,
  `normalized=true` on both transforms), interleaves the real and imaginary parts on the channel axis
  (`2c`, `2c + 1`), modulates them per bin by **BSFT** (the input's bandwidth as a one-hot over the
  513 bins through a shared 3-tap convolution to a per-bin scale and shift), ReLU, a bias-free 1×1
  convolution across those 64 channels, and the inverse STFT; the halves cross-connect, a gated
  activation gathers its gate and filter from both, and a 1×1 projection splits the residual (`/√2`)
  and the skip. The noise level is `(logsnr_max − logsnr) / 40` through a 50000-scaled sinusoidal
  embedding and two SiLU projections, added per block. The sampler is `denoise_ddim` over the
  released eight-value logSNR schedule (`[-2.6, -0.8, 2.0, 6.4, 9.8, 12.9, 14.4, 17.2]`, the last
  step landing on `logsnr_max` 20): from standard-normal noise, `x̂ = (y − σ_t ε) / α_t`,
  `y_s = α_s x̂ + σ_s ε` with `α² = sigmoid(logSNR)`, then a clamp to `1 − ε_fp16`. At reference
  parity on the official checkpoint on the first numeric run against the repository's own
  `Diffusion` (`run_reference.py nuwave2`, the `llm` env plus `omegaconf`, `IK_NUWAVE2_SRC`,
  `IK_PARITY_NUWAVE2` + `IK_VAL_NUWAVE2`), from the reference's own seeded start noise: the diffusion
  embedding 1.0, the first block's residual and skip 1.0, the step-0 noise prediction 1.0, every one
  of the eight DDIM steps 1.0, the clamped output 1.0. The conditioning follows `inference.py`: the
  clip peak-normalized, upsampled to 48 kHz (the reference's scipy `resample_poly`; the consumer path
  uses the shared `NFKMLXAudioRate.matched`, a documented approximation — the parity reads the
  recorded upsampled clip), trimmed to a multiple of the hop, and the band the first
  `int((rate / 2) / 24000 · 513)` bins (171 for a 16 kHz source, the reference's own float
  arithmetic). `NFKNUWaveSpectrum` is the batched normalized STFT pair over `[N, L]`: reflect padding
  and framing by gathers, and an overlap-add that reshapes each frame into `fftSize / hop` chunks
  and sums the shifted chunk sequences (MLX has no scatter-add). **Three oracle facts.** The official
  checkpoint is a Lightning file whose pickled callbacks need a `pytorch_lightning` stub to unpickle
  (a stub module whose `__getattr__` must still raise on dunders, or `inspect` inside `torch.load`
  breaks); the state dict sits under `model.model.` and the STFT window buffers are dropped; and the
  repository predates torch 2.x, handing `istft` the real `(…, 2)` view — the oracle patches
  `torch.istft` to take the complex view of the same numbers. `NFKParameterSeed` fixes the diffusion
  start; a step count other than eight walks the logSNR range evenly, as the reference does.
  `+register` under `nuwave2`; weights: the README's Google Drive checkpoint (20.9 MB, manifest route
  `gdrive`), which the native torch reader opens.
- `NFKMLXApollo` / `NFKMLXApolloNet` / `NFKMLXApolloBackend` (`@objc`) — **Apollo** (JusperLee,
  **CC-by-SA-4.0** code and weights, `JusperLee/Apollo/pytorch_model.bin`, 66 MB), music restoration
  of lossy-codec artifacts (MP3 at 24–128 kbps → lossless), the last of the audio fillers and the one
  music-leaning model. An 80-band split of a 20 ms STFT at 44.1 kHz (882/441, periodic Hann, center
  reflect, un-normalized): 79 bands of 5 bins and a 47-bin remainder over the 442 bins, each band's
  real and imaginary parts divided by the band's power (`sqrt(Σ|X|² + ε_fp32)`) and joined by the
  log power, an RMS norm and a 1×1 projection to 256 per band (`BN[i]`). Six **band-sequence layers**
  (`BSNet`): a Roformer across the 80 bands (every frame a sequence: an RMS-normed fused q/k/v 1×1
  projection whose 768 channels are head-major with q, k, v inside each of 8 heads, adjacent-pair rotary
  over a 100-position table, non-causal fused attention, a bias-free output projection with a residual,
  and a gated MLP — `silu` over the whole `8·dim` projection, then `silu(gate) · z` over its halves, so
  the gate is silu'd twice, reproduced as written), then an **ICB along time** (every band a sequence:
  three `ConvActNorm1d` blocks of a depthwise 7-tap convolution, an RMS norm, a 1×1 expansion ×4, SiLU,
  a 1×1 projection, residual). A head per band (an RMS norm, a 1×1 to `4·width`, a GLU to the band's
  real and imaginary bins, the real bins first) and the inverse STFT at the input's length. Every RMS
  norm is over the channels at eps 1e-5 (MLXNN's `RMSNorm`). The reference's `nn.Sequential` indices
  in `BN[i]` / `output[i]` map onto `norm` / `conv`; the rotary tables are recomputed (held as Swift
  arrays, off the parameters). At reference parity on the released weights on the first numeric
  run against the repository's own `Apollo` (`run_reference.py apollo`, the `llm` env, `IK_APOLLO_SRC`
  = the curled `look2hear` package, `IK_PARITY_APOLLO` + `IK_VAL_APOLLO`, channel 0 of the
  repository's own `asserts/input_wav.wav` for two seconds): band features 0.9999988, band 0's
  bottleneck 0.9999999, the first band-sequence layer 1.0, the last 1.0, band 0's head 1.0, and the
  restored waveform 0.9999973. The batched STFT pair (`NFKNUWaveSpectrum`) gained `normalized: false`
  and `torch.istft`'s `length` for it. The network runs each channel on its own; the backend runs the
  mono clip the WAV reader yields, at 44.1 kHz. The `inference.py` chunked overlap-add for long files
  is not reproduced; a clip runs whole. `+register` under `apollo`.
