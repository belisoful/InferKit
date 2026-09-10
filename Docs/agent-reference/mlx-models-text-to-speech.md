<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX models: text-to-speech

- `NFKMLXChatterbox` / `NFKMLXChatterboxTTS` (`@objc` factory) — **Chatterbox** (Resemble AI, MIT), zero-shot
  VOICE-CLONING text-to-speech, the third TTS beside FastSpeech2 and Kokoro and the first that takes a
  reference voice. Five networks, ported stage by stage with a parity record gating each
  (`run_reference.py chatterbox_voice` / `chatterbox_t3` / `chatterbox_s3gen`, the `chatterbox` oracle env,
  `IK_VAL_CHATTERBOX` = the release directory), all on the released weights and the validation clip:
  - **VoiceEncoder** (`NFKMLXChatterboxVoiceEncoderNet`, `ve.safetensors`): a 40-band unscaled power mel
    (no log; librosa Slaney to 8 kHz, every STFT frame kept) over the librosa-trimmed 16 kHz prompt
    (20 dB below the loudest 2048-sample frame, ported exactly), cut into 160-frame partials at step 77
    (`round((16000 / 1.3) / 160)`), a 3-layer LSTM → linear → ReLU → L2 per partial, mean → L2. Trim
    exact, mel 0.99999999999984, partials 0.999999999999, embedding 0.9999999999995.
  - **S3 speech tokenizer** (`NFKMLXS3TokenizerNet`, the `tokenizer.` subtree of `s3gen.safetensors`):
    Whisper's 128-band log-mel exactly (`NFKMLXMel.logMel` is reused), two stride-2 GELU convolutions
    (100 Hz → 25 Hz), six FSMN-attention blocks at 1280 over 20 heads (a depthwise 31-tap memory over
    the values added to the attention output; rotate-half rotary at 64, base 10000; the reference's
    `headDim^-0.25` on q and k is one `1/√headDim`), and an 8-channel base-3 FSQ (`round(tanh · 0.999) + 1`
    read as a base-3 number, 6561 codes). Mel 0.99999999998, states 0.9999999999, codes **87/87 exact** on
    both the six-second crop and the whole prompt.
  - **T3** (`NFKMLXT3Net`, `t3_cfg.safetensors`): the shipped dense decoder `NFKMLXLanguageNet` as a
    Llama 520M (30 layers, 16 heads × 64, no q/k norm, theta 500000) under **llama3 rope scaling**, which
    `NFKMLXRoPEScaling` now implements (factor 8, low/high frequency factors 1/4 over the 8192 window; at
    parity with transformers' `ROPE_INIT_FUNCTIONS` on two configurations, worst relative difference
    < 1e-5, added to the rope record). Conditioning is a 34-token prefix: `spkr_enc(speaker)`, a
    **Perceiver** (32 learned queries cross-attend to the prompt codes' `speech_emb + speech_pos_emb`, then
    self-attend once; 4 heads, one shared LayerNorm) and `emotion_adv_fc(exaggeration)`; then the text
    with learned positions and the start-of-speech token. Three reference quirks are reproduced, not
    fixed: `inference` feeds the start-of-speech embedding twice at speech position 0; the CFG
    unconditional row zeroes the text embedding but keeps the text positions; and generated code `i`
    takes speech position `i + 1`. The text tokenizer (`NFKMLXChatterboxTextTokenizer`) reads
    `tokenizer.json` directly — a plain character BPE with a `Whitespace` pre-tokenizer and literal added
    tokens, spaces replaced by `[SPACE]` first, plus `punc_norm` — token-exact against `tokenizers`.
    Sampling is the reference's chain (CFG 0.5 → repetition penalty 1.2 over every generated id → temperature
    0.8 → min-p 0.05 → top-p; `NFKMLXT3Sampler.processed`), pinned against transformers' own processors on
    the first step to 2e-5; temperature 0 is this port's greedy addition. Teacher-forced over the
    reference's own sampled sequence: cond 0.9999999999997, embeds 0.9999999999997, logits
    0.99999999999943 / 0.9999999999992 (both CFG rows), **argmax 79/79**. HF `hidden_states[-1]` Is the
    post-norm state (verified empirically on transformers 5.2, since T3 reads it as the head input).
  - **S3Gen token-to-mel** (`NFKMLXS3GenNet.flow`): the **CAMPPlus x-vector** over a ported Kaldi fbank
    (25/10 ms, 512-point, DC removed per frame, pre-emphasis 0.97 with replicate padding, the POVEY window
    `hann^0.85`, HTK mel 20 Hz–Nyquist, `log(max(x, ε))`, then the utterance mean removed) — an FCM 2-D
    head over the (frequency, time) plane whose stride hits frequency only, three CAM-dense TDNN blocks
    (12/24/16 layers, growth 32, a sigmoid gate from the utterance mean plus a 100-frame segment mean),
    statistics pooling (unbiased std), a 192-wide affine-free BatchNorm embedding; fbank 0.99999999995,
    x-vector 0.999999999999. The 24 kHz prompt mel (Matcha's: 1920/480, reflect pad 720, `sqrt(power +
    1e-9)`, log-clamp 1e-5) 0.9999999998. The **UpsampleConformerEncoder** (espnet `rel_pos` positions
    `T-1 … -(T-1)` with `√d` input scale, a 3-frame pre-lookahead convolution, six pre-norm rel-pos layers
    at epsilon 1e-12 with the appendix-B shift, nearest ×2 + left-padded 5-tap conv, four more layers, a
    final norm) 0.9999999999997; `mu` 0.9999999999998. The **CausalConditionalCFM**: ten Euler steps on
    `1 − cos(t·π/2)`, the estimator run as a batch of two (the unconditional row with zero mu, speaker, and
    prompt mel), `(1 + 0.7)·cond − 0.7·uncond`; the **ConditionalDecoder** (input `[x, mu, speaker, cond]` =
    320 channels; one down stage, twelve mid, one up, each a causal resnet block — left-padded 3-tap convs,
    LayerNorm over channels, Mish, the timestep MLP added between — plus four diffusers
    `BasicTransformerBlock`s with plain LayerNorms and a GELU FF; the released single level makes both
    resamples a causal 3-tap conv). First-step velocity 0.9999999999997, the solved mel 0.9999999999999.
  - **HiFT vocoder** (`NFKHiFTGeneratorNet`, `mel2wav.`): ConvRNNF0Predictor (five ELU convs, `|linear|`)
    0.99999999999997; the NSF harmonic source (nine harmonics, `2π·(cumsum(f0·k/sr) mod 1)`, voiced above
    **10 Hz**, `tanh(linear)`) **cosine 1.0** with the random phases and noise zeroed (the oracle zeroes
    them; the consumer path draws them); the generator — three transposed-conv upsamples (×8, ×5, ×3), the
    source's own 16-point STFT injected at each scale through strided convolutions, Snake residual blocks, a
    One-sample reflection pad before the last scale, the one bare `leaky_relu` at 0.01 before `conv_post`
    (the HiFi-GAN/Kokoro trap again), `exp` magnitude clipped at 100, `sin` phase, iSTFT at 16/4 through the
    shared `NFKKokoroSTFT` — waveform 0.99999999999, with the reference's 40 ms leading fade.
  End to end on the released weights: the validation clip as the voice, "The quick brown fox jumps
  over the lazy dog." → 3.38 s of 24 kHz speech that the package's own Parakeet (at parity) transcribes
  back exactly (`testChatterboxSynthesizesSpeechParakeetTranscribes`), and the release's built-in voice
  (`conds.pt`, read through the native torch reader: nested dicts flatten to `t3.` / `gen.` keys)
  speaks through the backend. `NFKMLXChatterboxTTS(directoryURL:)` loads all five; `conditionals(voice:
  sampleRate:)` is `prepare_conditionals` (resample to 24 kHz then to 16 kHz through
  `NFKMLXAudioRate.matched` — the reference's librosa/torchaudio resamplers are not bitwise, so the
  parity records take the recorded waveforms at each rate as inputs and the consumer path is a documented
  resampler approximation); `@objc chatterboxBackendWithDirectoryURL:voiceURL:error:` returns an
  `NFKMLXSpeechBackend` (24 kHz WAV under `NFKOutputAudio`; nil voice = `conds.pt`). **Two oracle
  traps**: `solve_euler` calls `estimator.forward` Directly, so a forward hook never fires (wrap the
  method); and the prompt mel's frame count can be odd (173) while the codes are trimmed to `mel // 2`
  (86), so the generated mel is `2n − 1` frames rather than `2n`. Weights: `ResembleAI/chatterbox`
  (`ve.safetensors` 7 MB, `t3_cfg.safetensors` 2.1 GB, `s3gen.safetensors` 1.1 GB, `tokenizer.json`,
  `conds.pt`). pyannote diarization stays blocked (gated `pyannote/segmentation-3.0`, no token here).
- `NFKMLXPhonemizer` (protocol) + two paths for the TTS text→phoneme front-end. `NFKMLXEspeakPhonemizer`
  (macOS only) shells out to a **system-installed** espeak-ng — InferKit does not bundle it (GPLv3);
  `Tools/espeak/install.sh` installs it and the phonemizer uses it only when present (`isInstalled`).
  `NFKMLXNeuralG2P` is the in-toolkit path: a compact encoder-decoder transformer (reusing `NFKWhisperBlock`)
  mapping graphemes → phonemes, no external dependency, permissively licensed. Both conform to
  `NFKMLXPhonemizer`; the neural model has a `loadWeights` + round-trip test (grapheme/phoneme vocabs are
  load-time artifacts). These are the front-end for a full TTS chain (phonemes → acoustic → vocoder).
- `NFKMLXTTS` + `NFKMLXAcousticNet` + `NFKMLXHiFiGAN` — the complete text-to-speech voice.
  `NFKMLXAcousticNet` (FastSpeech2-style: phoneme embedding → transformer encoder → duration predictor
  → length regulator (gather-expand by rounded durations) → decoder → mel projection, reusing
  `NFKWhisperBlock`). `NFKMLXHiFiGANNet` is the vocoder (mel → waveform: `conv_pre` → transposed-conv
  upsampling via `NFKDemucsConvT1d` + multi-receptive-field dilated `NFKHiFiResBlock`s → `conv_post`/tanh).
  `NFKMLXTTS` chains a `NFKMLXPhonemizer` + acoustic + vocoder and exposes `makeSpeechBackend()`
  (text → WAV) via `NFKMLXSpeechBackend`. Acoustic/vocoder load safetensors separately; each has a
  round-trip test, and the full text→audio chain is tested end to end.
  The vocoder runs real released weights at reference parity: jik876's UNIVERSAL_V1 generator
  (whose geometry is this port's default configuration), cosine 0.9999999999341 against the
  reference's own `models.py` on a deterministic mel — a vocoder is a pure function of its mel, so
  nothing about speech needs assuming. The release stores every convolution weight-normalized
  (`weight_g`/`weight_v`); `Tools/hifigan-to-safetensors/convert.py` fuses `g·v/‖v‖`, which is the
  reference's own `remove_weight_norm`. Reaching parity found a real defect: the reference's one bare
  `F.leaky_relu(x)` before `conv_post` runs at PyTorch's default slope 0.01 where every other
  activation is 0.1 — with 0.1 the released weights score 0.99954, measured both ways. The
  upsampling stages load through the Demucs ConvT treatment (`[in, out, k]`, name-gated).
  The trained acoustic model is ported too, and the voice is complete. `NFKMLXFastSpeech2Net`
  is the espnet FastSpeech2 conformer (through the transformers layout, whose implementation is the
  oracle): relative-position attention with the Transformer-XL shifting trick, macaron post-norm
  conformer layers (`normalize_before: false`), a GLU convolution module with BatchNorm, conv-FFN
  blocks, duration/pitch/energy variance adaptors (pitch and energy predicted per phoneme and
  embedded before the durations stretch to frames), the length regulator, and the residual postnet.
  Reference parity on the released LJSpeech weights on the first numeric run: encoder
  0.9999999999998, durations exact frame for frame, pitch/energy/mel all ≥ 0.9999999999. Module keys
  are the checkpoint's names.
  `NFKMLXVoice` chains it with the vocoder and the release's own 78-symbol ARPAbet vocabulary (the
  matching phoneme table), exposed through `makeSpeechBackend(phonemize:)`. The vocoder must be the
  Paired release (`espnet/fastspeech2_conformer_with_hifigan`, `vocoder.` prefix, weight norm
  already fused): espnet's acoustic model emits mels normalized by its training statistics, and the
  universal jik876 generator — same geometry, raw-log-mel convention — turns them into loud garbage.
  Measured, not assumed: the end-to-end test synthesizes "hello world" and has the package's own
  Whisper (real weights, at parity) transcribe it — with the universal vocoder Whisper hears
  "(indistinct)", with the paired one **" hello, world."** — which closes the loop TTS → audio → ASR
  entirely inside this package on released weights.
- `NFKMLXKokoro` / `NFKMLXKokoroNet` (`@objc`) — **Kokoro-82M**, a StyleTTS2 / iSTFTNet text-to-speech
  voice (hexgrad, Apache-2.0), a second TTS beside FastSpeech2 and the most popular on-device one. The
  pipeline: a **PL-BERT (Albert)** phoneme encoder (12 parameter-shared layers over a factorized 128-wide
  embedding), a `bert_encoder` projection, a **duration predictor** (a DurationEncoder of bidirectional
  LSTM + AdaLayerNorm blocks that re-concatenates the style vector each layer, then an LSTM and a
  duration head), the **alignment** by the rounded durations, an **F0/energy predictor** (a shared LSTM
  then AdaIN residual blocks with a depthwise-ConvTranspose upsample), a separate **TextEncoder** (CNN +
  LSTM), and an **iSTFTNet decoder** — AdaIN residual blocks over the alignment-expanded encoding, then a
  generator with a **harmonic sine source** (`SourceModuleHnNSF`), upsampling transposed convolutions,
  Snake-activated `AdaINResBlock1`s, a noise band from the source's STFT, and an **inverse-STFT** head.
  Reference parity on the released weights, seam by seam against the vendored `KModel` (`run_reference.py
  kokoro`, `IK_PARITY_KOKORO` + `IK_VAL_KOKORO`, the `llm` env): the PL-BERT 0.99999999999, the projection
  0.99999999999, the DurationEncoder 0.99999999999, durations 0.99999999999, F0/energy 0.99999999999, the
  TextEncoder 0.99999999999, the asr 0.99999999999, and the decoder's encode and full decode stack exact
  (0.9999999999995 / 0.9999999999991 — every AdaIN residual block, the F0/N stride-2 convs, and the
  depthwise-ConvTranspose upsample). The vocoder is float-precision-limited, not a modeling gap: the
  reference multiplies the accumulated sine phase by the upsample scale (≈18000 radians — the NSF sine
  oscillates at F0 over the whole clip) before `sin`, and a float32 argument of that size holds ~two
  fractional digits, so a torch/MLX rounding difference bounds the sine source at 0.99999 and the waveform
  cosine near 0.996 — the same class as the Music3 e2e, so the deterministic seams are the ground and the
  audio is compared loosely. The full consumer path (`loadVocab` + `loadVoice` + `synthesize(phonemes:voice:)`,
  per-Unicode-scalar phoneme mapping) reproduces the reference waveform at 0.997.
  Three facts are load-bearing, all found by the parity run. The released `.pth` stores each top-level
  module's state_dict under a `module.` DataParallel prefix (`bert.module.embeddings…`) which KModel strips
  at load and the loader strips too. The sine source's voiced threshold is **10, not 0** — SourceModuleHnNSF
  passes `voiced_threshod=10` to SineGen. And the generator's one bare `leaky_relu` before `conv_post` runs
  at the default slope 0.01 where every other activation is 0.1 (the same HiFi-GAN trap). All the
  bidirectional LSTMs load through the shared PyTorch→MLX fold (`weight_ih_l0`/`hh` → `Wx`/`Wh`, biases
  summed, forward/reverse), the weight-norm convs fuse `g·v/‖v‖`, and the `AdaINResBlock1` Snake `alpha`
  ParameterLists stack into one parameter. The misaki phonemizer is not required (it pulls spaCy, which
  will not build here): the oracle vendors just `KModel` + `istftnet` + `modules` (torch/transformers/scipy,
  no spaCy), and the backend takes a phoneme string under `NFKInputPrompt` directly — a caller brings the
  grapheme→phoneme front end (`NFKMLXNeuralG2P` or espeak). The `@objc` `NFKMLXKokoro.backend(directoryURL:voiceName:)`
  returns an `NFKMLXSpeechBackend` (24 kHz WAV under `NFKOutputAudio`); a voicepack `.pt` is a bare tensor
  the native reader will not interpret, so it converts to a single-`voice` safetensors offline (the
  `Tools/kokoro-voice-to-safetensors` treatment), which `loadVoice` reads. The weights are
  `hexgrad/Kokoro-82M` (327 MB `kokoro-v1_0.pth` + `config.json` + `voices/*.pt`).
