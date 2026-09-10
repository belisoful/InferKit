<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# MLX models: speech recognition, voice activity, audio tagging

- `NFKMLXWhisper` / `NFKMLXWhisperBackend` (`@objc`) — real on-device speech-to-text: the Whisper
  encoder-decoder transformer in `MLXNN` (log-mel via MLXFFT `rfft` → audio encoder → greedy text
  decoder). Audio → text backend: reads `NFKInputAudio` (an `NFKAudioAsset` WAV via `NFKMLXWaveFile.read`,
  or NSData), returns `NFKOutputText`. `+register` under `whisper-tiny`. Module names follow OpenAI
  Whisper (`encoder`/`decoder`, `blocks.N`, `attn`/`cross_attn`, `mlp.0`/`mlp.2`), so a converted
  checkpoint loads with the conv transpose (`loadWeights` handles 4-D and 3-D Conv1d). `NFKTokenizer`
  (optional) detokenizes; else token ids. `Tools/whisper-to-safetensors/convert.py` targets the OpenAI
  `.pt`. Reference parity against openai-whisper itself (log-mel cosine 0.9999999999940,
  first-step decoder logit cosine 0.9999999997432, and an exact greedy token match).
  The decoder's suppression rules are the reference's, measured rather than approximated:
  `suppressTokens` masks the curated non-speech set at every step, and `suppressesBlankStart` masks a
  space and an immediate `<|endoftext|>` at the first sampled position only, which is `SuppressBlank`.
  `NFKMLXWhisperSuppression.nonSpeechTokens(using:)` computes that set from the model's own tokenizer
  by the reference's rule — a symbol contributes its first token when it encodes to exactly one token,
  a musical symbol contributes its first token however many it encodes to — because the ids differ
  between the English-only and the multilingual vocabulary. `backendWithWeightsURL:tokenizer:` wires it.
  The record carries both decodings: `output` under the plain special/timestamp mask, which isolates
  the network, and `ruled_tokens` under the reference's own policy, which the port reproduces exactly.
  They differ on the synthetic clip (five tokens against three), which is why the policy needed
  measuring rather than describing. Timestamped decoding is implemented and at an exact token
  match against the reference (`transcribeWithTimestamps`, and `emitsTimestamps` on the backend,
  which adds `NSArray<NFKAudioSegment *>` under `NFKOutputSegments`). It is a different decode rather
  than a different reading of one: the times only exist when `<|notimestamps|>` is left out of the
  prompt and the timestamp range stays unmasked, so the model is asked a different question and may
  answer it with different words — which is why it is off by default. `ApplyTimestampRules` then
  orders the result: a timestamp is followed by text and text by a timestamp, so they come in pairs;
  a timestamp never precedes an earlier one, and the `+ 1` in the reference's bound is what forbids an
  empty segment; the opening position must be a timestamp no later than `max_initial_timestamp`
  (one second, 50 ids); and where the timestamps together hold more probability than any single word,
  a timestamp is taken even though no single one leads. On the synthetic clip the reference and the
  port both emit `<|0.00|>` "Thank you." `<|3.00|>` — and the tone in that clip does stop at 3.0
  seconds, so the span is a real measurement rather than a shape check. `timestampBegin` is the id of
  `<|0.00|>`, one past `<|notimestamps|>`, so large-v3's extra language token shifts both together.
  The mel question an early note left open is settled by measurement, not assumption: `melFilters` is
  the Slaney scale with Slaney area normalization — librosa `mel(htk=False, norm='slaney')`, which is
  what OpenAI's precomputed filters hold — and the parity record scores the port's log-mel at
  0.999999999994 against the reference's own. HF-format checkpoints load too: `loadWeights`
  detects the transformers naming (`model.encoder.layers.N.self_attn.q_proj`) and remaps it onto the
  OpenAI layout, asserted by renaming a real checkpoint into HF form and getting the identical
  transcription. `small`, `medium`, and `large-v3` are ported too, each at an exact token match
  against the reference (mel cosine 0.99999999999). Every size shares one encoder-decoder structure and
  differs in a width, a head count, and a depth — except large-v3, which also produces 128 mel bands
  instead of 80 and carries one more language token, shifting `<|transcribe|>` and `<|notimestamps|>`
  up by one. Both come from the model's own tokenizer rather than the smallest size's constants; the
  parity record carries the prompt the reference used, so a shifted id surfaces as a prompt mismatch
  rather than a mysterious token difference. `base`, `large` (large-v1 and large-v2 share one
  geometry: 1280 wide, 20 heads, 32 layers, 80 mels, vocabulary 51865), and `largeV3Turbo` (large-v3's
  128-mel encoder over a four-layer decoder) complete the released sizes, each at an exact greedy token
  match; `NFKMLXWhisperVariant` carries all of them. The `.en` releases share the multilingual geometry
  and differ only in their tokenizer files.
- `NFKMLXParakeet` / `NFKMLXParakeetNet` / `NFKMLXParakeetBackend` (`@objc`) — **Parakeet-TDT 0.6B v2**
  (NVIDIA NeMo, CC-by-4.0), a second on-device speech recognizer beside Whisper and the fast one: a
  **FastConformer** encoder — the NeMo mel front end (128 mels, 25 ms / 10 ms, a 512-point transform,
  `log(x + 2^-24)`, **per-feature normalization** by each band's unbiased standard deviation over time
  plus 1e-5), a **depthwise-striding 8× subsampler** (a full 3×3 stride-2 conv, then two depthwise
  3×3 stride-2 + pointwise 1×1 pairs over the `(time, mel)` plane, flattening `(channel, mel)` into a
  4096→1024 linear), and 24 relative-position (Transformer-XL) conformer layers (half-weighted
  macaron feed-forwards, rel-pos attention with per-layer `pos_bias_u`/`pos_bias_v` and the appendix-B
  shift, a GLU → depthwise-9 → BatchNorm → Swish convolution module, **no biases** on any projection or
  convolution) — and a **token-and-duration transducer**: a two-layer LSTM prediction network (640) over a
  blank-as-pad embedding, a joint (`enc` 1024→640 + `pred` 640→640, ReLU, a linear to 1024 tokens + blank
  + 5 duration classes), and greedy TDT decoding — at each encoder frame the joint scores the next token
  And how many frames to skip (`[0, 1, 2, 3, 4]`), a non-blank advances the LSTM state, and a zero
  duration keeps decoding the same frame (at most 10 symbols). `NFKMLXParakeetBackend` reads
  `NFKInputAudio` (resampled to 16 kHz) → the transcript under `NFKOutputText` plus one
  `NFKAudioSegment` per token under `NFKOutputSegments` (an encoder frame is 80 ms, so every token
  carries its onset). The release is an unpacked `.nemo` tar (`model_weights.ckpt` through the native
  torch reader; the SentencePiece **BPE** `*_tokenizer.vocab` piece table — recognition only decodes,
  so the pieces alone reproduce SentencePiece's `decode`: concatenate, `▁` → space, drop the leading
  space); `@objc backendWithDirectoryURL:error:`.
  Reference parity on the released weights, on the real validation clip, seam by seam against
  NeMo's own EncDecRNNTBPEModel (`run_reference.py parakeet --checkpoint <the .nemo>`,
  `IK_PARITY_PARAKEET` + `IK_VAL_PARAKEET`, the new `nemo` oracle env): features 0.9999999996, the
  subsampler 0.99999999989, the first layer 0.99999999975, the encoder 0.99999999999, the joint
  0.99999999999997, and the greedy TDT decode reproducing the reference's 19 tokens and their frame
  timestamps exactly — transcription "The quick brown fox jumps over the lazy dog." — through the
  public backend too (`testParakeetBackendTranscribesTheValidationClip`).
  Three facts are load-bearing, all found by the parity run. NeMo's valid frame count is
  `floor((samples + n_fft − n_fft) / hop)` — without the `+ 1` the transform's own frame count
  carries — so the transform's last frame lies past the valid length: it is zeroed (`pad_value`),
  excluded from the normalization statistics, and masked through the encoder; cropping to the valid
  length is exactly that through the stride-2 subsampler (with it, features went from 0.9975 to
  0.9999999996 and every downstream seam became exact). The blank / start state of the prediction net
  feeds the LSTM a zero vector in place of an embedding (`blank_as_pad`) and the LSTM still runs — its
  output and state are what the joint and the next step read; returning raw zeros scored the frame-0
  joint at −0.52 and dropped the first word. And NeMo's `joint()` log-softmaxes on the CPU when
  `log_softmax` is null — a constant shift over the whole 1030-vector that leaves both argmaxes alone,
  so the port keeps raw logits and the seam is compared in log-softmax space. The STFT pads with
  **zeros** (`pad_mode="constant"`), where the older MarbleNet VAD front end here reflects — measured
  both ways, reflect scores 0.974. The LSTM loads through the shared PyTorch→MLX fold (`weight_ih_l<n>`
  / `hh` → `Wx` / `Wh`, biases summed) under `dec_rnn.lstm.<n>`; the `nn.Sequential` indices of the
  subsampler (ReLU at 1, 4, 7) and the joint (ReLU 0, Dropout 1, Linear 2) are kept with marker modules
  so every other key matches with no remap; the 4-D and 3-D convolutions transpose to channels-last.
  Weights: `nvidia/parakeet-tdt-0.6b-v2` (2.47 GB `.nemo`; v3 and the CTC release are the same encoder).
- `NFKMLXVAD` (`@objc`) — real voice activity detection (MarbleNet): a mel front end feeding a stack of
  QuartzNet-style blocks — runs of time-channel-separable convolutions with an optional projected
  residual — and a two-class per-frame head; consecutive above-threshold frames merge into spans.
  `NFKMLXVADBackend` reads `NFKInputAudio` → `NSArray<NFKAudioSegment *>` (a new core value type) under
  the new core key `NFKOutputSegments`. `+register` under `vad-marblenet`; factory sets `train(false)`.
  Reference parity against NeMo (cosine 0.99999999999983). The front end (`NFKVADFrontEnd`) is the
  reference preprocessor — preemphasis, a centered 512-point transform, power spectrum through the mel
  filterbank, natural log with a `2⁻²⁴` guard, frames padded to a multiple of two — and it loads its
  window and filterbank from the checkpoint, which carries both; the defaults reproduce them for a
  randomly initialized net. Held in a plain box, not on the `Module`, so those constants stay out of
  `parameters()`. `remapReferenceKey` maps NeMo's flat positional `mconv` list (five entries per
  separable convolution, four per plain one) and its `res.0` shortcut onto the module's names.
  A clip arriving at another sample rate is resampled to 16 kHz through `NFKMLXAudioRate.matched`
  (the parity-proven `julius.resample_frac` port, with the ratio reduced by its greatest common
  divisor first — 44100 → 16000 would otherwise build 16000 polyphase kernels instead of 160). Frame
  times are computed at the model's rate, which is the caller's own seconds because resampling
  preserves duration.
- `NFKMLXSileroVAD` (`@objc`) — real voice activity detection (Silero VAD v6, snakers4), a second VAD
  architecture beside MarbleNet and the first of the per-modality roadmap adds. A learned-STFT
  front end (`Conv1d(1→258, k256, s128)`, no bias) → magnitude of `real[:129]`/`imag[129:]` → four
  `Conv1d+ReLU` (129→128→64→64→128, convs 2/3 stride-2) → a one-layer `LSTM(128→128)` → ReLU →
  `Conv1d(128→1, k1)` → sigmoid, scoring one speech probability per 512-sample chunk (32 ms). It streams:
  each chunk carries the previous chunk's last 64 samples as look-back (context roll; the first chunk
  zeros) and the LSTM state threads across chunks. The whole clip runs as one pass with the chunks on
  the LSTM's sequence axis, which reproduces the reference's chunk-by-chunk stream exactly (zero-init
  state, sequential). `NFKMLXSileroVADBackend` reads `NFKInputAudio` → `NSArray<NFKAudioSegment *>` under
  `NFKOutputSegments`; `+register` under `silero-vad`. v6 differs from v5 in the STFT padding alone:
  v5 pads the 576-sample (64 context + 512 chunk) input symmetrically by 128 and drops transform frame 0;
  v6 pads the right by 64 → 640 → four frames directly, no drop (`NFKMLXHTDemucs.reflectPadded(left:right:)`
  gives the right-only reflect). The LSTM reuses the Demucs bottleneck idiom — MLXNN's `LSTM`
  (`Wx`/`Wh`/fused `bias`, gate order `i,f,g,o`), PyTorch's `bias_ih`+`bias_hh` folded — which is what
  de-risked the port. Reference parity against the released snakers4 JIT (`silero_vad` 6.2.1) on the
  first numeric run: per-chunk cosine 0.9999999999998, max |difference| 6.9e-7, threshold agreement
  32/32. `remapReferenceKey` maps `_model.stft.forward_basis_buffer`→stft, `_model.encoder.{0..3}.reparam_conv`
  →conv1..4, `_model.decoder.rnn.weight_ih/hh`→`Wx`/`Wh`, `_model.decoder.decoder.2`→final; the released
  `.jit` also carries an 8 kHz `_model_8k.*` branch this port drops (`"_model."` is not a prefix of
  `"_model_8k."`, char 7 being `_` not `.`, and the loader skips `_model_8k` explicitly). The converter
  `Tools/silero-vad-to-safetensors` reads the `.jit` with `torch.jit.load` (torch alone, no `silero-vad`
  package) and keeps the 16 kHz `_model.*` in PyTorch layout; the native `.pth`/JIT reader reads the raw
  `.jit` too. The parity oracle (`run_reference.py silero_vad`, llm env, needs `silero-vad`+`torchaudio`)
  streams the JIT chunk by chunk. Resampled to 16 kHz through `NFKMLXAudioRate.matched`.
- `NFKMLXAudioTagger` (`@objc`) — real audio tagging (PANNs Cnn14): a log-mel spectrogram, normalized
  across its mel bands (`bn0`), feeds six VGG-style blocks (two 3×3 convolutions and an average pooling
  each), and the result pools over time — max plus mean — into an independent score per class; the top
  scores become tags. `NFKMLXAudioTaggerBackend` reads `NFKInputAudio` → `NSArray<NFKClassification *>`
  (a new core value type, most-confident first) under the new core key `NFKOutputClassifications`; the
  `+backendWith…labels:` factory attaches class names. `+register` under `audio-tagger-panns`.
  Reference parity against PANNs' own Cnn14 (mel cosine 0.99999999, embedding 0.99999994, tag
  0.99999988, same top class). The front end is 32 kHz / 1024-point / hop 320 over a 50 Hz–14 kHz
  filterbank, which the **checkpoint ships** (`logmel_extractor.melW`), so it loads rather than being
  recomputed — held in `NFKAudioTaggerFrontEnd`, off the `Module`, or it inflates `parameters()`.
  Its decibel scale is `10·log₁₀` floored at `1e-10`, not the natural log the other front ends use.
  The last block pools with a window of one, i.e. not at all; pooling it like the others cost
  embedding cosine 0.9942 and moved the top class. `remapReferenceKey` maps the reference's 1-based
  `conv_blockN` onto the module's array. A clip arriving at another sample rate is resampled to
  32 kHz through `NFKMLXAudioRate.matched`: the filterbank is built for one rate, so feeding another
  puts every frequency in the wrong mel bin — wrong tags, with nothing that looks like an error.
