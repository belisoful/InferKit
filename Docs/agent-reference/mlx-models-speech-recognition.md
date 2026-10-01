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
  **Customization is LoRA and it ships** (`NFKMLXWhisperTraining.swift`): `NFKMLXWhisper.network(weightsURL:configuration:)`
  builds the net, `spectrogram(for:sampleRate:configuration:)` pads a clip to the 30-second window, and
  `fineTune(_:examples:rank:alpha:…)` adapts the decoder's query and value projections with the encoder
  frozen, over `(mel, tokens)` pairs. `NFKMLXWhisperObjective` is teacher-forced next-token cross
  entropy, the loss transformers' `WhisperForConditionalGeneration` computes from `labels=`; its
  alignment is pinned on constructed logits, and no reference oracle records it. The reference
  optimizer is transformers' `Trainer` default, bias-corrected AdamW with no weight decay, at 1e-4.
  `NFKMLXLoRA.merge(into:)` writes one ordinary checkpoint the backend loads
  (`NFKMLXWhisperTrainingTests.testAnAdaptedRunMergesBackToAnOrdinaryCheckpoint`).
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
  **zeros** (`pad_mode="constant"`, NeMo 2.5 and later, as the MarbleNet VAD front end does). Reflect
  padding, measured against the same record, scores 0.974. The LSTM loads through the shared PyTorch→MLX fold (`weight_ih_l<n>`
  / `hh` → `Wx` / `Wh`, biases summed) under `dec_rnn.lstm.<n>`; the `nn.Sequential` indices of the
  subsampler (ReLU at 1, 4, 7) and the joint (ReLU 0, Dropout 1, Linear 2) are kept with marker modules
  so every other key matches with no remap; the 4-D and 3-D convolutions transpose to channels-last.
  Weights: `nvidia/parakeet-tdt-0.6b-v2` (2.47 GB `.nemo`; v3 and the CTC release are the same encoder).
- `NFKMLXGraniteSpeech` / `NFKMLXGraniteSpeechNet` / `NFKMLXGraniteSpeechBackend` (`@objc`) — **Granite
  Speech 3.3-2b** (`GraniteSpeechForConditionalGeneration`, IBM, Apache-2.0), the first speech language
  model here: a Conformer acoustic encoder, a BLIP-2 Q-former projector, and a dense Granite decoder that
  generates the transcription with the audio embeddings scattered into the prompt. Three components,
  measured seam by seam. The **Conformer CTC encoder** (16 layers, hidden 1024) reads the stacked log-mel
  features: an input projection, blocks of a half-weighted macaron feed-forward pair, **Shaw
  relative-position block-local attention** (each `context_size`-frame block attends within itself, a
  `rel_pos_emb` embedding adding a per-offset bias), a GLU → depthwise → **BatchNorm** → SiLU convolution
  module, and a post-norm, with a **mid-stack CTC skip** that adds `out_mid(softmax(out(h)))` at the
  half-way layer. The **BLIP-2 Q-former projector** (two layers) windows the encoder output into
  `window_size`-frame blocks and cross-attends `window_size / downsample_rate` learned query embeddings
  into each (self-attention, then cross-attention into the window, then a gelu feed-forward, each residual
  through a LayerNorm), then projects to the decoder width; a cross-attention block sits on every layer
  whose index is a multiple of `cross_attention_frequency` (the release's is 1). The **dense Granite
  decoder** (2048 wide, 40 layers, RoPE) is a Llama-family decoder with Granite's four scalar multipliers
  (embedding ×12, residual ×0.22, attention scale 1/64, logits ÷8) and a SwiGLU feed-forward; the audio
  features scatter into the token embeddings at the `<|audio|>` positions before the embedding multiplier
  scales the fused sequence. The released model ships a trained **audio LoRA adapter** (r 64, α 32, on the
  decoder's query and value projections) that Granite Speech enables for audio inputs; the loader folds it
  into those projections (`W += (α/r)·B·A`), so the loaded weights are the audio-active model.
  **Reference parity** against transformers' own `GraniteSpeechForConditionalGeneration` (transformers
  4.57.6, the llm oracle). Tiny config (`run_reference.py granite_speech`): the encoder 1.0000002, the
  projector 0.99999994, the fused logits 1.0000001. **Released granite-speech-3.3-2b** at float32 with the
  adapter folded (`run_reference.py granite_speech_real`): the checkpoint's 937 base tensors all match by
  name and shape (0 missing, 0 mismatched, 0 unaccounted; `num_batches_tracked` excluded), the encoder
  1.0000002, the projector 0.99999994, the logits 1.0000001, and the greedy continuation 8/8 tokens. The
  audio front-end (`NFKMLXGraniteSpeechFeatures`) matches the release's own torchaudio Mel spectrogram
  (n_fft 512, 400-sample Hann window, hop 160, 80 HTK mel bands, no area normalization, power 2), its
  `log10` with the `max(x, max − 8) / 4 + 1` normalization, and frame-pair stacking (2 × 80 → 160), at
  cosine 1.0000001 on a real clip. `NFKMLXGraniteSpeech.backend(directoryURL:)` (`@objc
  graniteSpeechBackendWithDirectoryURL:error:`) reads `NFKInputAudio` (resampled to 16 kHz) and an
  instruction under `NFKInputPrompt`, and transcribes the validation clip to "the quick brown fox jumps
  over the lazy dog." — exactly the reference processor's transcription — under `NFKOutputText`. Several
  facts are load-bearing. The whole projector uses **real nested submodules** (`qformer` → `encoder` →
  `layer`, each attention split into `attention` and `output`), because MLX splits a parameter key on `.`
  and a dotted `@ModuleInfo` key never routes (the SigLIP2 / Mamba lesson). Conv1d weights transpose from
  PyTorch `[out, in, kernel]` to MLX channels-last `[out, kernel, in]`, gated on the `.weight` suffix so
  the 3-D learned `query` parameter is left alone. The BatchNorm runs on its released running statistics
  (`train(false)` after load). The release ships `vocab.json` without `merges.txt`, so the shared
  release-tokenizer reader was generalized to extract the byte-level pair from `tokenizer.json` unless
  both files are present. **Customization is trainable at LoRA, with no recipe written yet.**
  transformers' `GraniteSpeechForConditionalGeneration` computes a `labels=` loss, and the 2B decoder
  is under the 4B float line. The shipped Whisper recipe already trains on `(mel, tokens)` pairs from a
  closure, which is the labeled audio-and-transcript input this model needs. The release's own audio
  LoRA adapter is folded in at load, so a recipe adapts on top of the merged weights. Weights: `ibm-granite/granite-speech-3.3-2b` (~6 GB;
  the 4-shard set the index names, plus `adapter_model.safetensors`). `IK_VAL_GRANITE_SPEECH`,
  `IK_PARITY_GRANITE_SPEECH_TINY` / `_REAL`, `IK_SHAPES_GRANITE_SPEECH` / `IK_CONFIG_GRANITE_SPEECH`,
  `IK_PARITY_GRANITE_SPEECH_FEATURES` / `IK_SPEECH_WAV`.
  **At bf16**, the backend's default, every Conformer piece and every probed decoder piece matches
  transformers' bf16 run on its own input, the worst at 0.05 of the reference's bf16-versus-float32
  distance, and the logits from the features sit 1.14 times that distance from float32
  (`testGraniteSpeechInBFloat16MatchesTheBFloat16Reference`). The Conformer's attention is torch's
  MATH backend at bf16 (`mathAttention`), its BatchNorm forms its scale and shift in float32
  (`NFKBatchNorm`), and the GLU and the norms round once. Two measurements shape the test. The
  reference applies the float32 adapter as its own bf16 branch, while the loader folds it in float32:
  that difference alone moves the logits by 1.2 times the floor, so the decoder is held to a reference
  with the adapter folded the same way (`IK_GRANITE_SPEECH_DTYPE=bfloat16-folded`, which also restores
  the rotary's float32 `inv_freq` that `model.to` rounds). And whole blocks amplify: a one-ulp
  difference in the Conformer's pointwise up-convolution spreads across its frame through the
  pointwise down-convolution, and the decoder amplifies about tenfold per layer from layer 1, so
  the isolated check runs on sub-pieces (`IK_PROBE_ENCODER=1`, `IK_PROBE_LAYERS`).
- `NFKMLXVoxtral` / `NFKMLXVoxtralNet` / `NFKMLXVoxtralBackend` (`@objc`) — **Voxtral-Mini 3B**
  (`VoxtralForConditionalGeneration`, Mistral, Apache-2.0), a second speech language model, built almost
  entirely from parts already at parity. Its audio encoder is the **Whisper large-v3 encoder reused
  verbatim** (`NFKWhisperEncoder`): the released `audio_tower` is that encoder in transformers naming, so
  the loader remaps it to the openai names the Whisper port loads (`layers.N.self_attn.{q,k,v,out}_proj`
  → `blocks.N.attn.{query,key,value,out}`, `self_attn_layer_norm` → `attn_ln`, `fc1`/`fc2` → `mlp.0`/`mlp.2`,
  `final_layer_norm` → `mlp_ln`, `layer_norm` → `ln_post`) and drops `embed_positions`, the fixed Whisper
  sinusoids the encoder computes. A **two-layer projector** groups every four encoder frames
  (`reshape(-1, 4·1280)`) and maps them to the decoder width (`linear_1` → gelu → `linear_2`). The
  decoder is a **Llama (Ministral 3B) decoder reused from `NFKMLXGraniteTextNet`** with Granite's four
  scalar multipliers set to identity (the dense Granite decoder is a superset of Llama); its `head_dim`
  128 exceeds the hidden size split across 32 heads. The audio embeddings scatter into the prompt at the
  `[AUDIO]` (24) positions. **Reference parity** against transformers' own `VoxtralForConditionalGeneration`.
  Tiny config (`run_reference.py voxtral`): the encoder 1.0000001, the projected audio embeddings
  0.99999994, the fused logits 0.99999994. **Released Voxtral-Mini-3B** at float32 (`run_reference.py
  voxtral_real`): the checkpoint's 761 base tensors all match by name and shape (0 missing, 0 mismatched,
  0 unaccounted; `embed_positions` is the computed sinusoid, not a parameter), the encoder 0.99999934, the
  audio embeddings 1.0000007, the logits 1.0000002, and the greedy continuation 8/8 tokens. A `.float32`
  load widens the encoder and the projector whole and, in the decoder, only the embedding table and the
  norms: the projection matrices keep the release's bfloat16 and promote exactly inside float32
  arithmetic, which holds the test inside a 32 GB machine's working set (the all-float32 load peaked at
  24.6 GB and swapped). One trap:
  the tiny oracle randomized `embed_positions`, but the port computes Whisper sinusoids, so the oracle
  sets it to the sinusoids the released model already carries (encoder 0.52 → 1.0 once fixed).
  `NFKMLXVoxtral.backend(directoryURL:)` (`@objc voxtralBackendWithDirectoryURL:error:`) reads
  `NFKInputAudio` (padded to a 30-second window, Whisper 128-band log-mel through `NFKMLXMel.logMel`) and
  a language code under `NFKInputPrompt` (default `en`), and transcribes the validation clip to "The
  quick brown fox jumps over the lazy dog." under `NFKOutputText`. The tokenizer is Mistral's **tekken**
  (`NFKMLXTekkenTokenizer`, a tiktoken byte-pair encoder read from `tekken.json`: base64 `token_bytes`
  with a merge rank, control tokens in `[0, numSpecial)`, regular tokens at `numSpecial + rank`, a
  tiktoken split regex, the byte-pair merge by lowest rank), token-exact against mistral-common. The
  transcription prompt is Voxtral's own: `<s> [INST] [BEGIN_AUDIO] [AUDIO]×N [/INST] lang:<code>
  [TRANSCRIBE]`. **Customization is trainable at LoRA, with no recipe written yet.** The 3B decoder
  over a frozen Whisper encoder is the shipped Whisper recipe with one more billion parameters, and
  transformers' `VoxtralForConditionalGeneration` computes a `labels=` loss. Weights: `mistralai/Voxtral-Mini-3B-2507` (~9 GB; the 2-shard set plus `tekken.json`,
  `preprocessor_config.json`). `IK_VAL_VOXTRAL`, `IK_PARITY_VOXTRAL_TINY` / `_REAL`, `IK_SHAPES_VOXTRAL` /
  `IK_CONFIG_VOXTRAL`.
  **At bf16**, the backend's default, every Whisper-encoder and decoder piece matches transformers'
  bf16 run on its own input at 0.027 of the floor or less, and the logits sit 0.97 of the floor from
  float32 (`testVoxtralInBFloat16MatchesTheBFloat16Reference`, `IK_VOXTRAL_DTYPE=bfloat16`). Before, the
  computed float32 sinusoids promoted the whole encoder to float32. transformers keeps
  `embed_positions` float32 under a bf16 load (`_keep_in_fp32_modules_strict`) and rounds
  `x + table` once, which the encoder now does. transformers' Whisper attention scales the query by
  `head_dim^-0.5` in float32; the openai layout splits the scale over queries and keys, and the shared
  encoder keeps that on its float32 path.
- `NFKMLXCanary` / `NFKMLXCanaryNet` / `NFKMLXCanaryBackend` (`@objc`) — **Canary-1B-v2** (NVIDIA NeMo
  `EncDecMultiTaskModel`, CC-BY-4.0), a multitask speech model that transcribes and translates, built
  from the FastConformer encoder already at parity and a new attention encoder-decoder. The acoustic
  front is the **biased FastConformer** — the same layer as Parakeet's, reused verbatim through
  `NFKParakeetEncoder` with `useBias` set (the released encoder carries `attention_bias` and
  `convolution_bias`, which the flag turns on for the attention, feed-forward, and convolution
  projections; the subsampling keeps its biases in either case) — 32 conformer layers over the shared
  mel front end. The head is a **Transformer decoder**: a token embedding, a fixed sinusoidal position
  table, an embedding LayerNorm, then pre-norm blocks of self-attention, cross-attention into the
  encoder frames, and a ReLU feed-forward, a final norm, and a tied output projection. It generates the
  transcription from a **task prompt of control tokens** — `<|startofcontext|><|startoftranscript|>
  <|emo:undefined|><|src|><|tgt|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>`, where a source language
  equal to the target transcribes and a different target translates. **Reference parity** against NeMo's
  own `EncDecMultiTaskModel` on the released weights and the validation clip (`run_reference.py canary`),
  seam by seam: the normalized mel features 0.999999999662, the subsampler 0.999999999715, the first
  conformer layer 0.999999999760, the encoder output 0.999999999913, and the decoder's first-step
  logits at the transcription prompt 0.9999999999998 (compared in the reference's log-softmax space —
  the tied-embedding classifier's raw logits carry a large shared component, as with Parakeet's joint).
  The checkpoint's 1475 base tensors all match by name and shape (0 missing, 0 mismatched, 0
  unaccounted; `num_batches_tracked` and the computed position table excluded), and the greedy decode
  reproduces the reference's 15 tokens and transcription "The quick brown fox jumps over the lazy dog."
  exactly, through the public backend too. Several facts are load-bearing. The FastConformer keys match
  `NFKParakeetEncoder` with no remap; the decoder renames NeMo's `transf_decoder._decoder.layers.N`
  (three sub-layers: self-attention `first_sub_layer`, cross-attention `second_sub_layer`, feed-forward
  `third_sub_layer`) onto the module, and `log_softmax.mlp.layer0` onto `proj_out`. The fixed position
  table is the sinusoid **scaled by `1/sqrt(d)`**, which the reference adds to the unscaled token
  embedding; the unscaled table left the decoder predicting the wrong distribution while the encoder
  stayed exact. The output projection is tied to the token embedding (shared storage), which the oracle
  clones so both tensors save independently. The tokenizer is the release's `tokenizer.json`, a
  **Metaspace BPE** (`NFKMLXCanaryTokenizer`: the `▁` word marker, control tokens dropped from decoded
  text), read for the task prompt and to decode the ids. `NFKMLXCanary.backend(directoryURL:)` (`@objc
  backendWithDirectoryURL:error:`) reads `NFKInputAudio` (resampled to 16 kHz) and a language code or a
  `src>tgt` pair under `NFKInputPrompt` (default `en`), and transcribes the validation clip to "The
  quick brown fox jumps over the lazy dog." **Customization is trainable at LoRA, with no recipe
  written yet.** NeMo publishes `transf_loss` with a prompt loss mask, and a 1B encoder-decoder trains
  on `(audio, transcript)` pairs the way the shipped Whisper recipe does. The release is the unpacked `.nemo` (`model_weights.ckpt`
  through the native torch reader, `tokenizer.json` beside it); `nvidia/canary-1b-v2` (~6 GB `.nemo`),
  the `nemo` oracle env. `IK_VAL_CANARY`, `IK_PARITY_CANARY`.
- `NFKMLXWav2Vec2` / `NFKMLXWav2Vec2Net` / `NFKMLXWav2Vec2Backend` (`@objc`) — **Wav2Vec2 and HuBERT**
  (transformers `Wav2Vec2Model` / `HubertModel` and their `…ForCTC` heads, Meta, Apache-2.0), the
  self-supervised speech encoders that read the raw 16 kHz waveform, in one configuration-driven network.
  A strided convolutional feature encoder (seven layers, 320 samples a frame) feeds a feature projection,
  a grouped convolutional position embedding (128 taps, 16 groups, weight-normalized, its trailing frame
  dropped), and a transformer encoder that is post-norm in the base releases and pre-norm ("stable layer
  norm", a final layer norm after the stack) in the large ones. The base releases group-normalize the
  first convolution alone; the large ones layer-normalize every convolution and carry biases. HuBERT is
  the same network under a `hubert.` prefix. A CTC release adds `lm_head` over a 32-character vocabulary
  (`Wav2Vec2CTCTokenizer`: the pad token is the blank, `|` the word delimiter). **At reference parity on
  every release measured, each against its own record** (`run_reference.py wav2vec2`, float32, eager
  attention, the validation clip through the release's own `Wav2Vec2FeatureExtractor`): the
  normalization within 4.8e-7, every seam at 0.99999999 or better, and the last hidden state at
  0.999999995267 (`wav2vec2-base-960h`), 0.999999999994 (`wav2vec2-base`), 0.999999999840
  (`wav2vec2-large-960h`), 0.999999999968 (`xls-r-300m`), 0.999999992094 (`xls-r-1b`), 0.999999999997
  (`hubert-base-ls960`), 0.999999998048 (`hubert-large-ll60k`), and 0.999999997454
  (`hubert-large-ls960-ft`). The CTC releases' logits read 0.9999999990 to 0.99999999999 with every
  greedy token equal, and the backend transcribes the clip as each release does: "THE QUICK BROWN FOX
  JUMPS OVER THE LAZY DOG" from base-960h, "THEQUICK …" from large-960h, "EQUICK …" from
  hubert-large-ls960-ft (the releases' own readings of the clip's start). Several facts are load-bearing.
  The position convolution keeps its weight-norm gain and direction as parameters (`weight_g`,
  `weight_v`), read from either spelling the releases use, because a fine-tune trains them rather than
  the product. `masked_spec_embed` is optional at load: base-960h does not ship it. The releases store
  `pytorch_model.bin` except base-960h and hubert-xlarge-ls960-ft, and the native torch reader loads
  them directly. **The GPU gradient of a convolution wider than 16 taps is wrong in mlx 0.32.2**
  (`mlx-runtime-gotchas.md`), so the position convolution sums eight 16-tap slices; its forward is the
  same, and without the slices the feature projection's first-step gradient read 0.997.
  **Customization ships at `full`**, the reference's `run_speech_recognition_ctc.py` (the copy shipped
  beside `facebook/wav2vec2-xls-r-2b`): `NFKMLXWav2Vec2.network(directoryURL:vocabulary:)` retargets the
  head to a consumer's characters (`NFKMLXWav2Vec2Tokenizer(characters:)`), `fineTune` trains everything
  but the frozen feature encoder under `NFKMLXWav2Vec2Objective` (`ctc_loss` over the float32
  log-softmax, reduction mean) with SpecAugment's time masks (`NFKMLXSpecAugment`, the reference's span
  rule), the Trainer's AdamW at 5e-5 with no weight decay, a linear decay, and clipping at 1.0, and
  `save(_:tokenizer:toDirectoryURL:)` writes a directory the `@objc` factory loads. The script's 8-bit
  optimizer states (bitsandbytes `Adam8bit`) are a CUDA memory saving and are not reproduced; the update
  is Adam's. Measured against transformers' own `Wav2Vec2ForCTC` (`run_reference.py wav2vec2_loss`): the
  loss within 1.2e-7 relative at both reductions, including a label set whose doubled letters block the
  recursion's skip, its gradient at 0.99999999995, and the impossible alignment and its
  `zero_infinity` form. Three recipe steps under the reference's recorded masks: every first-step
  gradient at 0.99999997 or better, every parameter's movement at 0.9999 or better, and the step losses
  within 6.2e-6, 2.0e-4, and 1.6e-4 of a float64 replay of the same steps, against the reference's own
  float32 drift of 1.5e-5, 4.7e-5, and 3.2e-4. Adam's first steps move near-zero gradients by nearly the
  full rate, so per-step drift is not monotone; the test holds each step to twice the reference's worst
  drift. `IK_VAL_WAV2VEC2_*`, `IK_VAL_HUBERT_*`, `IK_PARITY_*` for each, `IK_PARITY_WAV2VEC2_LOSS`.
- `NFKMLXWav2Vec2Bert` / `NFKMLXWav2Vec2BertNet` / `NFKMLXWav2Vec2BertBackend` (`@objc`) — **W2V-BERT 2.0**
  (transformers `Wav2Vec2BertModel`, Meta, MIT), the 600M-parameter multilingual encoder behind Seamless.
  It reads filterbanks, not the waveform: `SeamlessM4TFeatureExtractor`'s 80-bin Kaldi filterbank of the
  waveform scaled to 16-bit range, each bin normalized over the utterance (sample variance), padded to an
  even frame count with the release's padding value 1.0, and frames stacked in pairs (160 wide, 50 per
  second), with a mask marking a stacked frame whose second half is padding. The encoder is 24 Conformer
  layers: a half-step swish feed-forward, self-attention with a **clipped relative-key position table**
  (`q · E[clamp(j − i, −64, 8)]` added to the scores, both scaled by `1/√d`), a convolution block (a
  pointwise projection, a GLU, a causal depthwise convolution of 31 taps padded on the left only, a layer
  norm, a swish, a pointwise projection), a second half-step feed-forward, and a final layer norm per
  layer. Masked frames are zeroed before the encoder and before each depthwise convolution and excluded
  as attention keys. **At reference parity on the release** (`run_reference.py w2v_bert`, float32, the
  extractor's attention mask passed as the processor returns it): the filterbank within 1.9e-6 of the
  extractor's, the stacked features within 4.8e-7, the projection 0.99999999999987, layers 0, 12, and 23
  at 0.99999999999945, 0.99999999999645, and 0.99999999999744, and the output from this port's own
  features 0.99999999999633. Two facts are load-bearing. The filterbank is computed in double precision
  (`NFKKaldiFbank.logEnergiesInDoublePrecision`): the extractor's transform is NumPy's float64, and a
  near-silent frame sits within float32 round-off of the log floor, where a float32 transform read 2.1
  above it. The extractor pads to an even frame count, so a clip of 345 filterbank frames yields 173
  stacked frames, the last one masked.
  **Customization ships at `full`**, Hugging Face's W2V-BERT recipe ("Fine-Tune W2V2-Bert for low-resource
  ASR"): `NFKMLXWav2Vec2Bert.network(directoryURL:vocabulary:)` adds the output adapter (`add_adapter`: one
  layer that pools the residual and the attention input to half the frames through a stride-2 convolution
  and a GLU, then a plain self-attention and a relu feed-forward) and a CTC head over the consumer's
  characters, both initialized as `_init_weights` does; `fineTune` trains every parameter under the CTC
  loss over the adapter's real frames, with the Trainer's AdamW at 5e-5, no weight decay, a 500-step
  linear warm-up then a linear decay (`NFKMLXLearningRateSchedule.linearWithWarmup`), and clipping at 1.0;
  `save(_:tokenizer:toDirectoryURL:)` writes a directory the factory loads. The recipe's own settings leave
  `conformer_conv_dropout` and `final_dropout` at the release's 0.1, and it trains in fp16 over batches of
  16 with two accumulation steps; the port runs without dropout at float32, one utterance a step. Measured
  against transformers' own `Wav2Vec2BertForCTC` from the reference's adapter and head initialization
  (`run_reference.py w2v_bert_loss`, the dropouts zeroed so the replay is deterministic): the loss over 86
  pooled frames 5.2926073 vs 5.2926073 with its gradient at 0.99999999978, the adapted logits
  0.9999999999948, three steps' losses within 1.2e-6 relative (5.2926073 / 3.6675189 / 3.2128937 vs
  5.2926073 / 3.6675167 / 3.21289), every watched first-step gradient at 0.9999999987 or better and every
  watched movement at 0.9999975 or better, and the warm-up schedule equal to transformers' at every sampled
  step. `IK_VAL_W2V_BERT_2`, `IK_PARITY_W2V_BERT_2`, `IK_PARITY_W2V_BERT_LOSS`.
- `NFKMLXPhi4MM` (`@objc`) / `NFKMLXPhi4MMBackend` (`@objc`) — **Phi-4-multimodal** (Microsoft, MIT)
  transcribes as one of its modes: audio alone runs the speech LoRA on its Phi-4-mini decoder over a
  Conformer speech tower with a T5 relative bias. At reference parity on the released weights (encoder
  0.9999999999982836, fused logits 0.9999999999994174, transcription exact), with SpeechLib's
  filterbank and the reference's sample-rate handling reproduced at 16, 44.1, 48, 8, and 11.025 kHz.
  Several clips encode as one masked batch, and a clip past 40 s unfolds into the encoder's 500-frame
  windows, both as the reference computes them. The full entry, with the image and text modes, is in
  `mlx-models-vision-language.md`.
- `NFKMLXVAD` (`@objc`) — real voice activity detection (MarbleNet): a mel front end feeding a stack of
  QuartzNet-style blocks — runs of time-channel-separable convolutions with an optional projected
  residual — and a two-class per-frame head; consecutive above-threshold frames merge into spans.
  `NFKMLXVADBackend` reads `NFKInputAudio` → `NSArray<NFKAudioSegment *>` (a new core value type) under
  the new core key `NFKOutputSegments`. `+register` under `vad-marblenet`; factory sets `train(false)`.
  Reference parity against NeMo 3.0 (probability cosine 0.9999999999998, mel 0.9999999999997). The
  front end (`NFKVADFrontEnd`) is the reference preprocessor — preemphasis, a 512-point transform
  centered with zeros (`pad_mode="constant"`), power spectrum through the mel filterbank, natural log
  with a `2⁻²⁴` guard, the frames past `floor(samples / hop)` zeroed, then padded to a multiple of
  two — and the encoder masks every convolution's input past the valid length as it shrinks through the
  strides (`conv_mask: true`). Both edge rules are NeMo 2.5's (NVIDIA/NeMo #13827, July 2025). NeMo
  2.4 and earlier center with `torch.stft`'s default reflect padding and count `floor(samples / hop) + 1`
  valid frames. Neither rule is a preprocessor config key, so NeMo 3.0 applies the new pair to this
  release even though its `model_config.yaml` records `nemo_version: 2.1.0rc0`, the version it trained
  under. The parity record is `reference/vad-reference-nemo3.safetensors` (NeMo 3.0.0, torch 2.14.0),
  the file `IK_PARITY_VAD` names. The August 2026 record, renamed `reference/vad-reference-pre-nemo2.5.safetensors`, is the
  pre-2.5 pair: numpy reproduces it from its own waveform with reflect padding and 101 valid frames to
  9.5e-7, and reproduces the NeMo 3.0 record with zero padding and 100 valid frames to 9.5e-7. Between
  the two records, mel frames 0, 1, 99, and 100 differ by up to 2.0, 1.3, 0.12, and 11.9 (mel cosine
  0.9979). The only probability that moves is the last frame, index 50 (0.055 → 0.459, cosine 0.919),
  which falls past NeMo 3.0's valid length and scores zero padding. The test's 0.999 mel bound and
  0.02 per-frame bound each reject the old record. The front end loads its
  window and filterbank from the checkpoint, which carries both; the defaults reproduce them for a
  randomly initialized net. Held in a plain box, not on the `Module`, so those constants stay out of
  `parameters()`. `remapReferenceKey` maps NeMo's flat positional `mconv` list (five entries per
  separable convolution, four per plain one) and its `res.0` shortcut onto the module's names.
  A clip arriving at another sample rate is resampled to 16 kHz through `NFKMLXAudioRate.matched`
  (the parity-proven `julius.resample_frac` port, with the ratio reduced by its greatest common
  divisor first — 44100 → 16000 would otherwise build 16000 polyphase kernels instead of 160). Frame
  times are computed at the model's rate, which is the caller's own seconds because resampling
  preserves duration. `detect` builds spans from the frames inside the encoder's valid length alone
  (`logitsAndLength`, NeMo's `encoded_len`) and clamps each span's end to the clip's duration, through
  `NFKMLXSpeechSpans.merge`, which Silero VAD shares. `speechProbabilities` still returns every frame
  the forward emits, padding frames included, because the parity record holds all of them. The
  even-count padding adds one frame past the valid length whenever the valid mel count is even: a
  16000-sample clip yields 51 frames, 50 of them valid. On the parity clip the extra frame scores
  0.459 against the 0.5 threshold, so on another clip it can open a span that reaches past the audio.
  NeMo's own `binarization` postprocessing reads every frame it is given; its loss and metrics mask
  by the label count.
  **Customization is a FULL fine-tune and it ships** (`NFKMLXVADTraining.swift`), the recipe the
  release's own `model_config.yaml` names. `NFKMLXVAD.network(weightsURL:)` builds the now-public
  `NFKMLXVADNet`, `frameCount(samples:)` and `frameLabels(speech:frameCount:)` label its 20 ms frames,
  and `fineTune(_:examples:…)` trains every weight. `NFKMLXVADObjective` is NeMo's masked per-frame
  `CrossEntropyLoss`; the optimizer is SGD (momentum 0.9, rate 0.01, weight decay 0.001), and the
  schedule is `PolynomialHoldDecayAnnealing` (5% warm-up, 15% hold, power 2, floor 1e-8) as
  `NFKMLXLearningRateSchedule.nemoPolynomialHoldDecay`. While it trains the front end adds its 1e-5
  dither, `NFKMLXVADSpecAugment` applies `SpectrogramAugmentation`'s vectorized masks over the valid
  frames, and each block drops 10% after every activation. Measured against the release restored
  through NeMo (`run_reference.py vad_training`, `testVADTrainingMatchesTheRelease`): the loss on the
  release's logits 2.6228473 vs 2.6228476, on this port's logits for the same 2 s clip 2.6228466
  (logit cosine 0.99999999999993), and the schedule exact over 45 steps. The network defaults to
  evaluation mode in its initializer.
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
  streams the JIT chunk by chunk. Resampled to 16 kHz through `NFKMLXAudioRate.matched`. The last
  chunk is zero-padded to 512 samples, so a span that reaches it ends at the clip's duration, as the
  reference's `get_speech_timestamps` sets the last end to `audio_length_samples`.
  **Customization is a HEAD RETARGET of the decoder and it ships** (`NFKMLXSileroVADTraining.swift`),
  following snakers4's own `tuning/` at v6.2.1: `NFKMLXSileroVAD.network(weightsURL:)` builds the net,
  `NFKMLXSileroVADTrainable` picks `.decoder` (the STFT and encoder frozen, as `tune.py` freezes them)
  or `.everything`, and `chunkTargets(speech:sampleCount:noiseWeight:configuration:)` is the dataset's
  `get_ground_truth_annotated`: a chunk is speech when more than half its samples are, and the mask
  weighs a non-speech chunk by 0.5. `NFKMLXSileroVADObjective` is the masked mean of `BCELoss` with its
  −100 log floor. The decoder drops its LSTM output at 0.1 while training, as `VADDecoderRNNJIT` does.
  The reference optimizer is Adam at 5e-4 with one deliberate difference in form: the reference's
  `LSTMCell` keeps `bias_ih` and `bias_hh`, both receive the gradient the folded bias receives, and Adam
  moves each by the same step, so the folded bias steps at twice the rate to move exactly as the pair
  moves. Measured against the released JIT and `tune.py`'s own forward loop: targets and mask exact,
  probabilities within 9.2e-7, the loss 0.23334178 against 0.233342 and after two Adam steps 0.11447279
  against 0.11447296, every decoder update at cosine 0.999996 or better with norms within 6e-6, the
  folded bias's included (`testSileroVADTrainingMatchesTheReference`). The reference's augmentations and
  eight-second crop are the caller's data choices.
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
  **Customization is a HEAD RETARGET and it ships** (`NFKMLXAudioTaggerTraining.swift`), from PANNs at
  d2f4b8c: `NFKMLXAudioTagger.network(weightsURL:configuration:)` builds the net at the configuration's
  class count and leaves the classifier at its random initialization when the checkpoint's differs,
  and `NFKMLXAudioTaggerTrainable` picks `.classifier` (`finetune_template.py`'s frozen base) or
  `.everything`. `NFKMLXAudioTaggerObjective` is `losses.py`'s `clip_bce` on the classifier's sigmoid,
  and the reference optimizer is `main.py`'s Adam at 1e-3 with AMSGrad (`NFKMLXAMSGrad`). The template
  keeps the model in training mode, so Cnn14's dropouts (0.2 after each block, 0.5 on the pooled
  vector) and torchlibrosa's SpecAugment (`NFKMLXAudioTaggerSpecAugment`, two time stripes up to 64
  frames and two band stripes up to 8) run over the frozen base as well; the base's BatchNorms hold
  their statistics, where the template's frozen base updates them. The recipe retargets
  `fc_audioset`, which reads the embedding before its final dropout, where the template's
  `fc_transfer` reads it after. `NFKMLXAudioTagger.save(_:to:)` writes the filterbank beside the
  parameters (`NFKMLXWeights.save(_:extraArrays:to:)`), because a recomputed one differs from the
  stored one by up to 4.8e-8, 1e-3 relative at the filter edges. Measured against the released Cnn14
  with a fresh five-class layer: embedding cosine 0.99999994, losses within 2.8e-5 relative over three
  AMSGrad steps (0.72625655 against 0.72627217 first, 0.48410437 against 0.4841176 last), and the
  layer's update at cosine 0.9999999999999 with norm ratio 1.0000002
  (`testAudioTaggerTrainingMatchesTheReference`). The reference's default mixup needs a batch; a caller
  mixes its own examples for the same effect.
