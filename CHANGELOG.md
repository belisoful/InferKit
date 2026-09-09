# Changelog

All notable changes to InferKit are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below 1.0.0, a minor bump may change public API. SwiftPM treats a 0.x minor as
breaking, so `from: "0.1.0"` resolves 0.1.x only and a consumer opts into each minor deliberately.

## [Unreleased]

### Core (`InferKit`)

- Grammar-constrained sampling for the Core ML language backend: `NFKTokenVocabulary`,
  `NFKJSONConstraint` (well-formed JSON with a chosen root), and `NFKChoiceConstraint` (exactly one of a
  fixed set) mask the sampler's logit buffer through an `NFKTokenConstraintCursor`, so sampling and
  greedy decoding alike stay inside the grammar. `NFKCoreMLLanguageBackend` reads the new
  `NFKParameterOutputFormat` and `NFKParameterChoices` request parameters and returns JSON it was asked
  for parsed under `NFKOutputStructured`; the MLX language backend honors the same keys.
- `NFKByteLevelBPETokenizer` reads the `o200k` pre-tokenization (see the gpt-oss entry below).

### InferKitMLX (companion)

#### Released sizes

Every shipped family now covers every size its authors released, each with a configuration preset, a
case on the family's `@objc` variant enum, the full factory set, and a registered name. A size whose
checkpoint is a modest download is measured at reference parity; a size too large to run here is held to
the module by shape against its released safetensors headers (`Tools/validation-assets/shapes.py`,
`NFKMLXReleasedSizesTests`). The measured numbers are in `Docs/model-parity.md`.

- Whisper `base`, `large` (fits large-v1 and large-v2), and `largeV3Turbo` (large-v3's encoder over a
  4-layer decoder), each at an exact greedy token match.
- SAM `vitL` and `vitH` (encoder cosine 0.9999980 and 0.9999967, binary mask agreement 1.0); SAM 2
  Hiera `small` (FPN levels 0.9999999999996 / 0.999999999998), so all four Hiera sizes are at parity.
- Depth Anything 3 `base` and `large` (`NFKMLXDepth3Variant`, registered as `depth-anything-3-base` /
  `-large`; every hooked feature ≥ 0.99999999999, the depth mean-removed 0.99999999999 / 0.99999999998).
- NAFNet `siddWidth64` and `goProWidth64` (the REDS geometry trained on GoPro), 0.99999986 and
  0.9999999966.
- SwinIR `classicalX2`, `lightweightSRX3`, `lightweightSRX4`, and the two real-world ×4 releases,
  `realWorldX4Medium` (the reference's `nearest+conv` upsampler) and `realWorldX4Large` (240 wide, nine
  six-block groups, the `3conv` residual connection), every one ≥ 0.9999999999993.
- Robust Video Matting `resNet50` (`NFKMLXRVMVariant`, `robust-video-matting-resnet50`): the
  torchvision bottleneck backbone with its last stage dilated, tapped through
  `NFKMLXResNetBackbone.taps`, LR-ASPP 2048 → 256, and no ImageNet normalization on this encoder, as
  the reference; alpha 0.99999999995, foreground 0.9999999999997.
- RT-DETR `r18vd`, `r34vd`, `r101vd` (`NFKMLXRTDetrVariant`; the two smaller run basic blocks and
  fewer decoder layers, the 101 a 384-wide encoder), logits ≥ 0.99999999998 over the reference's
  selection.
- RF-DETR `nano`, `small`, `medium`, `large` (`NFKMLXRFDetrVariant`; the later releases run a patch-16
  DINOv2 at 384 / 512 / 576 / 704 with two windows), logits ≥ 0.99999999988.
- SNAC `music32kHz` and `music44kHz`: four codebooks at strides 8/4/2/1 and a 32-frame windowed
  attention at the bottleneck (`NFKSNACLocalAttention`, rotary over the window positions), which the
  24 kHz speech model omits; codes 60/60 exact, reconstruction ≥ 0.9999999999998.
- Hybrid Transformer Demucs `sixStem` (`htdemucs_6s`: guitar and piano after the four base stems; the
  release sets `bottom_channels` to 0, so it carries no channel samplers and runs its cross-transformer
  at the encoder's own width) and `NFKMLXHTDemucsBag` for the fine-tuned `htdemucs_ft` release (four
  checkpoints combined by per-source weights, `backendWithFineTunedWeightsURLs:error:`); both
  0.99999999999995.
- CLIP `vitB16`, `vitL14`, `vitL14At336` (`NFKMLXCLIPVariant`), and every SigLIP 2 release
  (`NFKMLXSigLIP2Variant`: base at patch 16 / 32, large, so400m at patch 14 / 16, giant-opt, each at its
  released resolutions; the giant-opt text tower projects 1152 → 1536), registered under their release
  names.
- Language presets `qwen3_14B`, `qwen3_32B`, `gemma2_9B`, `gemma2_27B`; `NFKMLXLanguage.configuration(fromJSON:)`
  reads a config already parsed, which is what a multimodal release's `text_config` needs.
  `NFKMLXQwen3VL` gained `visionNet(directoryURL:)`, `decoderConfiguration(directoryURL:)`, and
  `decoder(directoryURL:)`, reading the release's own `config.json` and sharded weights (the 30B-A3B's
  fused experts split into the module's gate and up projections). Structurally verified: Qwen3 14B /
  32B, Qwen3-Embedding 4B / 8B, Qwen3-VL 4B / 8B / 32B / 30B-A3B, Gemma 3 12B / 27B, Gemma 2 9B / 27B,
  SigLIP 2 ×14, Gemma 3n E4B, each with 0 missing, 0 mismatched, 0 unaccounted.
- Gemma 3n E4B measured at bf16 on both sides (logit cosine 0.99989, argmax 5/6); the E2B, exact at
  float32, reads the same 0.99989 at bf16, which is the rounding floor the E4B is held to.
- Fixed: SwinIR's classical reconstruction tail activated with a plain ReLU where the reference's
  `conv_before_upsample` uses a leaky ReLU at 0.01. On the released classical ×4 through the 8-bit
  backend bridge the mean pixel difference falls from 0.0037 to 0.00136 (×3 from 0.0036 to 0.00119, ×8
  from 0.0035 to 0.00095); the lightweight release has no such tail and is unchanged.

#### Gemma 3n

- `NFKMLXGemma3n` runs Gemma 3n tri-modally from a release directory, at reference parity on the
  released E2B weights for every stage. It is a distinct architecture rather than a Gemma 3 variant:
  the decoder (`NFKMLXGemma3nNet`) carries **AltUp**, which makes the residual stream four parallel
  copies with a learned per-token map predicting them before each block and correcting them after;
  **LAuReL**, a rank-64 detour beside the attention residual; **per-layer embeddings**, a second wide
  table giving every layer its own slice, gated into the inactive copies; and **activation sparsity**,
  which holds the first ten layers' feed-forward gates at zero below a per-token Gaussian cutoff.
  Attention runs at scale 1.0, the values carry an unweighted normalization, and the trailing
  `num_kv_shared_layers` compute no keys or values at all, reusing the last non-shared layer of their
  own kind. Measured: every one of the 30 hidden states 1.0000000000, logit cosine 0.9999999999943566,
  and the cached greedy continuation 11/11 token for token; a tiny record additionally pins every
  mechanism and both attention kinds.
- `NFKMLXGemma3nAudioNet` is the Universal Speech Model Conformer: two strided convolutions under a
  **cumulative group normalization**, then twelve blocks of feed-forward, chunked attention with a
  Transformer-XL relative-position shift, a causal depthwise convolution, and a second feed-forward,
  with the activation clamp applied at inference. Released-weight parity: front end
  0.9999999999999402, first block 0.9999999999998025, encoded 0.9999999999999257.
- `NFKMLXGemma3nVisionNet` is **MobileNetV5-300M**, a convolutional encoder the release reaches
  through `timm`: edge residuals, universal inverted residuals, and multi-query attention over the
  feature map, joined by a multi-scale fusion adapter. Padding is TensorFlow's `SAME`, asymmetric at
  stride two. Released-weight parity: stem 0.9999999999996717, every stage ≥ 0.9999999999571081,
  fused grid 0.9999999999981896.
- `NFKMLXGemma3nModel` splices the towers into the prompt in two stages — a hard per-modality
  embedding of the placeholder ids, then the soft tokens over them. End to end on the released E2B the
  prompt reproduces the release processor's 272 ids exactly, the argmax matches at **272/272** fused
  positions, the last-16 logit cosine is 0.9999999999785188, and the cached greedy continuation
  reproduces the reference's caption token for token.
- `NFKMLXGemma3nBackend` reads `NFKInputPrompt` / `NFKInputMessages` with `NFKInputImage` or
  `NFKInputAudio` beside them, honors temperature / top-p / max-tokens / seed, and streams through a
  submitted job. `NFKMLXGemma3nAudioFeatures` is the mel front end (HTK's mel scale with no area
  normalization, frames cut one sample longer than the window, the transform at twice its length),
  measured at 0.999999999999857. From Objective-C: `[NFKMLXGemma3n backendWithDirectoryURL:error:]`,
  `gemma3nWithDirectoryURL:error:`, `answerForImage:question:error:`, `answerForQuestion:error:`.
- `NFKMLXGemma3Language.configuration(fromHuggingFace:)` still refuses a `gemma3n` config, which now
  names the reader that takes it rather than only saying no.

#### Gemma 3

- `NFKMLXGemma3` runs the Gemma 3 line end to end from a release directory. The text decoder
  (`NFKMLXGemma3Net`: `(1 + w)` RMS norms, sandwich blocks, five sliding-window layers to one
  full-attention layer at a local and a global rotary base, the 4B's 8× linear rotary scaling on the
  full layers, per-head QK norm, GeGLU, a tied head) is at reference parity on the released 270M, 1B,
  and 4B against transformers' own Gemma 3 (every hidden state exact; logit cosine 0.99999999999368,
  0.99999999999821, and 0.99999999999735; the greedy continuation through the cache token for token). A
  tiny record pins the 4-position sliding window, the linear rotary scaling, both soft-caps, and the
  cached step-by-step decode against a teacher-forced pass (worst step 1 − cosine 6.7e-8).
- `NFKMLXGemma3Backend` generates through a hybrid key-value cache (`NFKMLXGemma3Cache`: unbounded for
  the full layers, bounded to the window for the sliding ones), streams each token through a submitted
  job, honors cancellation between tokens, renders `NFKInputMessages` through the release's own Jinja
  chat template (token-exact against `apply_chat_template`), and takes `NFKInputImage` beside the text
  on the multimodal 4B. `NFKMLXGemmaLanguage.backend(directoryURL:)` routes a `gemma3` / `gemma3_text`
  release here. Objective-C: `backendWithDirectoryURL:error:`, `gemma3WithDirectoryURL:error:`,
  `answerForImage:question:error:`, `answerForQuestion:error:`.
- The multimodal 4B: `NFKMLXGemma3VisionNet` (SigLIP so400m at 896×896, reusing the shared SigLIP
  encoder with row-major positions), `NFKMLXGemma3MultimodalProjector` (a 4×4 average pool of the 64×64
  patch grid to 256 soft tokens, a Gemma norm, one `[1152, 2560]` matrix), the processor's
  `\n\n<start_of_image>` + 256 soft tokens + `<end_of_image>\n\n` expansion done in the text before
  tokenizing (the newlines merge with their neighbours' into one token), and the reference's
  bidirectional attention among an image's tokens (`NFKMLXGemma3Masks`) — the full
  `Gemma3ForConditionalGeneration` chain at reference parity on the released weights (vision tower
  0.99999999875, soft tokens 0.99999999971, the fused decoder's argmax 275/275 with every layer
  ≥ 0.99999995, the greedy continuation token for token; see `Docs/model-parity.md`).
- The loader reads both checkpoint layouts: the released files' transformers 4.x naming
  (`language_model.model.`, `vision_tower.vision_model.`, `multi_modal_projector.`) and the 5.x
  `model.`-nested one.
- `NFKMLXGemmaTokenizer` matches a release's `added_tokens` (`<bos>`, `<start_of_turn>`,
  `<start_of_image>`, `<image_soft_token>`, …) as literals before the merge, as the reference does, and
  `decode(_:skipSpecial:)` leaves the markers out; `specialIds` lists them.
- `NFKMLXGemma3EncoderNet` (EmbeddingGemma's backbone) is now built on the decoder's blocks, and its
  bidirectional window is the reference's exclusive bound `span / 2 + 1` (257 for the released 512),
  where it had used the span itself — measured at a tiny configuration whose sequence exceeds the
  window (last hidden cosine 0.99999999999967); no EmbeddingGemma input had reached either number.

#### MetricGAN+

- `NFKMLXMetricGANPlus` runs MetricGAN+ (speechbrain, Apache-2.0), a two-layer bidirectional LSTM
  magnitude mask over `log1p(|X|)` frames with a per-bin learnable sigmoid, at reference parity on the
  released weights against speechbrain's own `SpectralMaskEnhancement` (features 1.0, mask 1.0000001,
  enhanced waveform 1.0000001). Registered under `metricgan-plus`.
- `NFKMLXComplexSTFT` gained `zeroPadded` (speechbrain's `pad_mode="constant"`, where `torch.stft`
  reflects) and a `length` on both inverses (`torch.istft`'s `length`, which keeps the last frame's
  tail past the center trim); the default behavior is unchanged.

#### CMGAN

- `NFKMLXCMGAN` runs CMGAN (ruizhecao96/CMGAN, MIT), a conformer metric GAN's generator: a dense
  encoder, four two-stage (time, frequency) conformer blocks with Shaw's relative position embedding,
  and magnitude-mask + complex-residual decoders over a `mag^0.3` compressed spectrogram, at reference
  parity on the released checkpoint (every seam ≥ 0.9999998, enhanced waveform 0.99999994). Registered
  under `cmgan`.

#### FRCRN

- `NFKMLXFRCRN` runs FRCRN SE 16K (ClearerVoice, Apache-2.0), a frequency-recurrent complex CRN: two
  complex UNets over a conv-STFT with frequency-recurrent FSMN memories, complex squeeze-excites, and
  a time FSMN bottleneck, the mask `tanh(unet2) + tanh(unet1)` as a complex product, at reference
  parity on the released weights (every seam ≥ 0.99999994, enhanced waveform 1.0). The backend
  reproduces the reference decode path's zero-padding grid. Registered under `frcrn`.
- `NFKMLXComplexSTFT` gained `centered: false` (a convolutional STFT with no padding); the default
  behavior is unchanged.

#### MossFormer2 SR

- `NFKMLXMossFormer2SRNet` runs MossFormer2 SR 48K (ClearerVoice, Apache-2.0), speech
  super-resolution: the HiFi-GAN log-mel, the shipped MossFormer2 backbone as a mel-to-mel restorer,
  a Snake HiFi-GAN generator, and the decode path's `bandwidth_sub` post-process (Butterworth
  low-pass / high-pass under `filtfilt`, ported in double precision), at reference parity on the
  released weights (mel 1.0, backbone 1.0, generator 0.9999999999987, substitution 1.0, end to end
  0.9999999999992). Registered under `mossformer2-sr`; `backendWithDirectoryURL:error:`.
- `NFKMLXMossFormer2Configuration.superResolution` configures the SE MaskNet as the SR backbone, and
  the BigVGAN mel front end takes its geometry as parameters.

#### NU-Wave 2

- `NFKMLXNUWave2` runs NU-Wave 2 (maum-ai, BSD-3), diffusion bandwidth extension to 48 kHz: a noise
  predictor of short-time Fourier convolutions with BSFT band modulation, sampled by the released
  eight-step logSNR DDIM from seeded noise (`NFKParameterSeed`), at reference parity on the official
  checkpoint from the reference's own start noise (every seam and every step 1.0). Registered under
  `nuwave2`.

#### Apollo

- `NFKMLXApollo` runs Apollo (JusperLee, CC-BY-SA-4.0 code and weights), music restoration of
  lossy-codec artifacts at 44.1 kHz: an 80-band split of a 20 ms STFT, six band-sequence layers (a
  rotary transformer across the bands, a depthwise convolutional block along time), and GLU band
  heads, at reference parity on the released weights (band features 0.9999988, every band-sequence
  layer 1.0, restored waveform 0.9999973). Registered under `apollo`.

#### Schema-constrained decoding

- `NFKMLXJSONSchemaConstraint` constrains generation to a JSON Schema, so the keys, the types, the
  enumerations, and the array bounds are guaranteed as well as the syntax. `NFKMLXJSONSchema` compiles
  the schema (`type` as a name or a list, `properties` / `required` / `additionalProperties`, `items` /
  `minItems` / `maxItems`, `enum` / `const`, `anyOf` / `oneOf`, `$ref` into `$defs` or `definitions`
  with recursion; an empty schema admits any value) into nodes the byte-level engine walks. An `anyOf`
  keeps every alternative alive until the bytes decide; keys are written in any order and an object
  closes only once its required keys are present. A keyword the grammar cannot enforce byte by byte is
  ignored when it only narrows content (`pattern`, `minimum`, `minLength`) and refused at compile time
  when it changes what is admissible (`allOf`, `not`, `if`).
- `NFKMLXGenerationOptions.jsonSchema` carries it, and `NFKMLXLanguageBackend` fills it from the
  core's `NFKParameterJSONSchema` request parameter — the key the remote backends read — so a
  structured-output request is engine-agnostic. A schema the grammar cannot enforce is an error rather
  than a silently unconstrained run.
- JSON that was asked for (a schema or `outputFormat`) now comes back parsed under `NFKOutputStructured`
  beside `NFKOutputText`, as the remote backends return it.

#### Qwen2-MoE

- The dense decoder reads `qwen2_moe` (Qwen1.5-MoE-A2.7B, Qwen2-57B-A14B): the routed feed-forward plus
  a shared expert every token runs, gated by `sigmoid(shared_expert_gate(x))` and summed with the routed
  output. Its releases leave the routing weights unnormalized, which the reader defaults for this type,
  and carry query/key/value biases (`qkv_bias`, true by default and absent from the released config).
  Reference parity against transformers' own `Qwen2MoeForCausalLM` at a tiny configuration: every hidden
  state exact, logit cosine 0.9999999999999813.
- Dense Qwen2 releases load: their config carries no `attention_bias`, and the reader now defaults the
  bias to true for `qwen2` as the architecture does, where the previous default refused their bias
  tensors at load.

#### gpt-oss

- The dense decoder reads `gpt_oss` (gpt-oss-20b, Apache-2.0): layers alternating a sliding window with
  full attention, a learned attention sink per head (the softmax is written out for it), biases on every
  attention projection and the router, fused interleaved experts with biases and the clamped SwiGLU,
  and YaRN whose correction band is left fractional (`NFKMLXRoPEScaling.truncatesCorrectionRange`).
  Reference parity against transformers' own `GptOssForCausalLM` at a tiny configuration: every hidden
  state exact, logit cosine 0.9999999999999721.
- The released MXFP4 experts stay packed: the `_blocks` bytes viewed as `uint32` are MLX's `mxfp4`
  words as stored, measured against MLX's own decode and against transformers' decode of the real
  released bytes (worst difference 0.0), and `gatherQuantizedMM` runs them in `mxfp4` mode. A release
  declaring `quant_method: mxfp4` loads at `.checkpoint` precision, so the 20B is 13.8 GB resident.
  `NFKLMQuantizedSwitchLinear` carries its mode; a saved checkpoint records `bits:groupSize[:mode]`.
- `NFKByteLevelBPETokenizer` gains the `o200k` pre-tokenization (digit runs of at most three, words
  split by their case pattern, case-insensitive contractions), and the language backend's release
  tokenizer reads a `tokenizer.json`-only release, choosing the pattern from its `Split` regex. Token-exact
  against the `tokenizers` library on gpt-oss's o200k_harmony.

#### Measurements

- Speculative decoding on a bandwidth-bound target (Qwen3-4B at bf16 drafted by the 0.6B) runs at 0.57×
  the plain wall clock with 0.435 acceptance, so it stays off by default; at bf16 the speculative and
  plain greedy runs can part at a near-tie (the target's own top two, margin 0.125), where at float32
  they are identical.

#### Key-value cache: keys grouped per channel

- `NFKMLXKeyValueCache.Quantization` gains `keyAxis`. `.sequence`, now the default, groups each key
  channel over `groupSize` consecutive positions (KIVI's axis) and holds the positions of the unfinished
  group in full precision; `.headDimension` is the previous per-position layout. Values stay per
  position. The window and rollback work over groups, a rollback that cuts into a packed group returns
  its surviving positions to the residual, and a saved prompt cache records the axis (an older file
  loads as the layout it was written with). From Objective-C, `cacheQuantizationPerChannelKeys`
  (default true) selects it.
- Measured on the released Qwen3-0.6B. Per position, 8-bit tracked the reference record on a short
  prompt (cosine 0.99956, continuation 16/16) but 4-bit collapsed (0.577 at group 64, 0.928 at group
  32), and the keys' 4-bit reconstruction error was 0.133 per position against 0.045 per channel while
  the values barely differed by axis. Over a 132-token prompt against the float cache, 8-bit per-channel
  keys reproduce it (last-logit cosine 0.99997, the 24-token greedy continuation exact) where 8-bit
  per-position read 0.994; 4-bit per-channel reads 0.994 (group 64) and 0.997 (group 32) where
  per-position read 0.68 and 0.97.

## [0.3.0] — 2026-09-06

### InferKitMLX (companion)

#### Text embeddings

- `NFKMLXQwen3Embedding` embeds text on device: the Qwen3-0.6B dense decoder read one layer earlier
  (its post-final-norm hidden states), pooled at the last token over an appended `<|endoftext|>` and
  L2-normalized, so a dot product between two embeddings is their cosine similarity. It reuses
  `NFKMLXLanguageNet` through its `hiddenStates(fromEmbeddings:)` seam, so nothing about the transformer
  is re-implemented. The package embedded images (CLIP) and had no path for text; this adds semantic
  search, retrieval, clustering, and reranking over a consumer's own corpus.
- The released checkpoint is the base model (`AutoModel`/`Qwen3Model`), so its keys carry no `model.`
  prefix and no `lm_head`; the embedder's loader prepends `model.` and drops the absent projection.
- `NFKMLXTextEmbeddingBackend` reads `NFKInputPrompt` (or a joined `NFKInputMessages`) and returns the
  vector under `NFKOutputEmbedding`. `NFKMLXQwen3Embedding.instruct(task:query:)` formats a retrieval
  query the way the model is trained to read it; a document is embedded as-is.
  `backendWithDirectoryURL:outputDimensions:error:` truncates each embedding to a smaller Matryoshka
  width before normalizing, and a Swift caller with token ids already reads them through
  `embedding(forTokens:)`.
- Reference parity against the model card's own transformers recipe on the released 0.6B weights: query
  embedding cosine 0.99999999999, document 0.99999999999, and the retrieval score reproduced to 1e-6
  end to end. A separate tokenizer-agreement test reproduces the reference's ids from the shared text,
  since the `qwen2` pre-tokenization is what makes them right.
- `NFKMLXEmbeddingGemma` is a second text embedder over a second architecture: EmbeddingGemma-300M, the
  bidirectional Gemma 3 encoder (`gemma3_text`, `use_bidirectional_attention`) mean-pooled over every
  token, run through a Dense bottleneck (768 → 3072 → 768, no bias), and L2-normalized. The backbone is
  Gemma 3, not the causal Gemma 4 (`gemma4_text`) the language model runs — `(1 + w)` normalization,
  sandwich norms, dual RoPE (local 10000 / global 1000000), QK-norm, GeGLU, no per-layer embeddings — so
  it is `NFKMLXGemma3EncoderNet`, its own implementation.
- `NFKMLXGemmaTokenizer` reads Gemma's byte-fallback BPE `tokenizer.json` directly (a metaspace
  normalizer, the whole string one pre-token, merges by rank), so the text path needs no offline
  conversion. Gemma's SentencePiece scores are merge ranks, so the unigram Viterbi `NFKUnigramTokenizer`
  runs is the wrong algorithm for it — measured, then implemented as BPE.
- The gated `google/embeddinggemma-300m` is mirrored ungated at `unsloth/embeddinggemma-300m`, as SD 2.1
  is. Reference parity against the sentence-transformers pipeline on the released 300M weights on the
  first numeric run: every one of the 24 layers exact, query and document embedding cosine
  0.99999999999, retrieval score to 1e-7, and the tokenizer reproducing the reference's ids token for
  token.
- `NFKMLXTextEmbeddingBackend` now serves both families through a small `NFKTextEmbedding` protocol and
  a tokenization closure, so a family with its own pooling, projection, and tokenizer (EmbeddingGemma's
  mean pooling, Dense head, and BPE) plugs into the same backend as the last-token pooled decoder.

#### Reranking

- `NFKMLXModernBERTReranker` is a cross-encoder reranker over `gte-reranker-modernbert-base`: a
  bidirectional ModernBERT encoder reads a `[CLS] query [SEP] document [SEP]` pair, mean-pools, and a
  prediction head plus a single-logit classifier gives a relevance score. An embedder scores a query and
  a document independently; a cross-encoder reads the pair together and is more accurate, which reorders
  an embedder's shortlist. `scores(query:documents:)` and `rankedIndices(query:documents:)` score and
  order candidates; ObjC reaches them through `scoresForQuery:documents:` and `rankedIndicesForQuery:documents:`.
- `NFKMLXModernBertRerankerNet` is the encoder: RoPE (global base 160000 every third layer, local base
  10000 with a 128-token sliding window elsewhere), a GeGLU feed-forward, LayerNorm without biases, no
  absolute position embeddings, and layer 0's `attn_norm` an identity (the embeddings are pre-normed).
- The tokenizer is GPT-2-family byte-level BPE, which the core `NFKByteLevelBPETokenizer` reads; the
  release ships only `tokenizer.json`, so its vocabulary and merges are extracted into the
  `vocab.json`/`merges.txt` the core reader takes.
- Reference parity against transformers' own `ModernBertForSequenceClassification` on the released
  weights: every one of the 22 layers exact by the per-layer isolation harness (over a pair long enough
  to engage the local sliding window), and both the relevant and irrelevant scores reproduced to within
  5e-3, with the tokenizer reproducing the reference's ids token for token. ModernBERT's
  `output_hidden_states` does not final-norm its last entry, which the harness matches.

#### Vision-language model

- `NFKMLXSmolVLM` is the package's first vision-language model, SmolVLM2-500M: an image and a question
  in, an answer out. Three parts — a SigLIP vision encoder (`NFKMLXSigLIPNet`), a pixel-shuffle
  connector (`NFKMLXSmolVLMConnector`, scale 4: 1024 patch tokens per tile fold to 64), and a Llama
  decoder (`NFKMLXLanguageNet`, loaded from the checkpoint's `text_model` subtree) — with the projected
  vision tokens spliced into the decoder's input embeddings at the image-token positions.
- The consumer path: a CoreGraphics image processor (`NFKMLXSmolVLMImageProcessor`) tiles an image the
  way SmolVLM does (longest edge to 2048, split into 512×512 sub-tiles plus a global thumbnail), a
  token-exact prompt builder expands the `<image>` structure, and `answerForImage:question:` greedily
  decodes an answer. The tokenizer is GPT-2-family byte-level BPE from the release's `tokenizer.json`.
- Reference parity against transformers' own SmolVLMForConditionalGeneration on the released 500M
  weights, staged by the isolation harness: the SigLIP embeddings, encoder, and connector each exact,
  the fused decoder logits predicting the reference's token at every one of the 1140 positions (last
  position cosine 0.9999999999), and the greedy continuation token for token. The prompt expansion is
  token-exact. Two bugs the harness located: SigLIP's position ids use a `1 - 1e-6` bucketize (a full
  32-patch row maps to `[0, 0, 1, …, 30]`, not `0 … 31`), and SmolVLM's `lm_head` is NOT tied to the
  embedding despite the geometry.

#### A second vision-language model's vision tower (Qwen3-VL)

- `NFKMLXQwen3VLVisionNet` is Qwen3-VL-2B's vision encoder, a second vision architecture beside SmolVLM's
  SigLIP: a 2D-rotary ViT whose patches are laid out in 2×2 merge blocks, a bilinearly interpolated
  position embedding (the learned 48×48 grid resampled to the image's grid), a merger that folds each
  block to the decoder width, and a three-layer "deepstack" of feature maps (from vision layers 5/11/17).
  The patch embedding is the reference's full-kernel convolution written as one linear projection.
- Reference parity against transformers' own Qwen3-VL vision model on the released 2B weights, on the
  first numeric run: the patch embedding, the interpolated position embedding, the merged output, and
  every one of the three deepstack features exact (cosine 0.9999999997+).
- The decoder is the Qwen3 dense stack the language model already runs. Qwen3-VL turned out to be much
  more than "reuses the Qwen3 decoder": the decoder adds interleaved M-RoPE (3D positions) and deepstack
  injection at its first layers, so the decoder integration is the remaining wiring.

#### Native GGUF reader

- `NFKMLXGGUF` reads a GGUF model natively — the container's header, typed key/value metadata, and tensor
  table — and dequantizes the block-quant formats a real model uses into `MLXArray`s, with no Python and
  no llama.cpp. The sequel to the native PyTorch checkpoint reader, and the same contract: pure Foundation
  below the MLX materialization (so parsing and dequantization run under `swift test`), and a type it does
  not implement is refused per-tensor rather than misread.
- Dequantizers: `F32`, `F16`, `Q4_0`, `Q5_0`, `Q8_0`, `Q4_K`, and `Q6_K` — the k-quants a `Q4_K_M` model
  is built from, plus the legacy and full-precision types. GGUF stores the fastest-varying dimension
  first, so a tensor's row-major shape is the reverse of its stored dimensions.
- Bit-exact against the `gguf` package on the released SmolLM2-135M-Instruct Q4_K_M model: for the first
  tensor of each type, the dequantized values match to worst |difference| 0.0 across 262144 values each.
  `NFKMLXGGUFTensorInfo` and the `metadataString`/`metadataInteger` accessors inspect a model before
  reading it; `array(forTensor:)` and `arrays()` dequantize.
- The GGUF reader is **wired into the language-model loader**, so a GGUF release generates text end to
  end (`NFKMLXLanguage.backend(ggufURL:)` / `@objc backendWithGGUFURL:error:`). The metadata becomes an
  `NFKMLXLanguageConfiguration`, the llama.cpp tensor names are remapped onto the decoder's keys, the
  weights are dequantized by the reader, and the embedded byte-level BPE tokenizer is rebuilt. Only the
  dense `llama`/`qwen2`/`qwen3` families are read. The query and key projections are un-permuted per
  head — llama.cpp stores them interleaved for its own rotary, and loading them raw runs mostly-right
  and subtly wrong (logit cosine 0.95). Reference parity against transformers loading the same GGUF
  (`run_reference.py gguf_lm`): logit cosine 0.9999999999971 with the same argmax at every prompt
  position, on the released SmolLM2-135M-Instruct Q4_K_M.

#### Chat-template renderer

- `NFKMLXChatTemplateRenderer` renders the Jinja `chat_template` an instruct release ships, so the
  language backend reproduces the model's trained input rather than approximating it. Rendering the
  template wrong silently changes the model's input — the same failure class as the `qwen2`
  pre-tokenization defect — so the faithful path is to render the release's own template.
- A compact Jinja interpreter for the subset chat templates use: text with `{{ … }}` output and
  `{% … %}` control (for / if / elif / else / set), the whitespace controls transformers compiles a
  template with (`trim_blocks`, `lstrip_blocks`, and the explicit `{%-`/`-%}` markers), and the
  expression language — attribute and index access, slicing (`messages[::-1]`), `namespace`, the `loop`
  variable, the operators and `is` tests, string methods (`startswith`/`split`/`strip`/…), and the
  `tojson`/`trim` filters. Pure Foundation, so it runs under `swift test`.
- Reference parity against transformers' own `apply_chat_template` over six shipped cases — the Qwen3
  template (namespaces, reversed slicing, `is` tests, the tool-call and tool-role branches), Llama-3
  (`bos_token` and `| trim` precedence), and Gemma (`%`, `!=` on booleans, `set role`, the
  `raise_exception` guards) — rendered byte-for-byte. Regenerate with
  `Tools/reference-parity/generate_chat_templates.py`.
- `NFKMLXChatTemplate.jinja(template:bosToken:eosToken:)` is the new generation option; from
  Objective-C, a `chatTemplate` request parameter carrying Jinja delimiters is rendered the same way.
  One documented divergence: `tojson` emits an object's keys in sorted order, where transformers emits
  insertion order (Foundation dictionaries do not preserve it), which affects a tool schema's key
  order, not a plain or multi-turn chat.

#### Native PyTorch checkpoint reader

- Every `weightsURL:` factory and every registry build accepts a raw PyTorch checkpoint (`.pth`,
  `.pt`, `.ckpt`, `.th`, HF `.bin`) wherever it accepted a converted safetensors, with no Python
  toolchain. The loader sniffs a file's leading bytes rather than its extension, so an HF torch `.bin`
  and a safetensors `.bin` are told apart by content. Both containers parse: the modern ZIP archive
  (zip64 and deflate) and the pre-1.6 five-pickle stream. The file stays memory-mapped, and the
  restricted pickle machine never executes a global; every construction it does not recognize becomes
  inert data that flattening drops.
- Training wrappers unwrap in the offline converters' own precedence (`state_dict`,
  `model_state_dict`, `params_ema`, `params`, `model`, `generator`, `state`) before the root is
  flattened, non-tensor sidecars drop, a tensor stored as a strided view (Whisper's transposed
  `Linear` weights) is gathered to row-major, and float64 narrows to float32.
- Three checkpoint shapes beyond a plain state dict load: a pickled live `nn.Module` tree (YOLO's
  `ultralytics` DetectionModel, walked through its `_parameters` / `_buffers` / `_modules` state), a
  TorchScript archive (CLIP, walked through its attribute-keyed scripted-module state), and a `.nemo`
  tar (VAD, unwrapped to the checkpoint inside). No class is constructed and no serialized `code/` is
  interpreted.
- `NFKMLXTorchCheckpoint` (`@objc`) is the inspection and conversion API over the same reader:
  `checkpointWithContentsOfURL:error:`, `tensorNames`, `infoForTensor:` (an `NFKMLXTorchTensorInfo`
  with `shape` and `NFKMLXTorchScalarType`), `dataForTensor:error:`, and `writeSafetensorsToURL:error:`,
  which converts on device to the file the model's `Tools` converter produces. Swift adds `arrays()`
  for feeding a model directly.
- Each model's loader carries its converter's renames and transforms, so a raw release loads end to
  end wherever a converted one does: U²-Net's `rebnconvN` index rename, the colorizer's `Sequential`
  table and transposed-convolution axis swap, HiFi-GAN's weight-norm fusion and `vocoder.` / `ups.N`
  renames, the NAFNet / RAFT / RIFE renames, LaMa's `generator.` and FastSpeech2's `model.` prefixes,
  and the pose head's transposed-convolution axis order. Conv-TasNet and the denoiser needed no change.
  `NFKMLXTorchParityTests` holds each raw load to its converted file, parameter for parameter.
- A big-endian save and a storage kind the reader has no counterpart for are refused with an error
  naming the offline converter; a file whose bytes do not parse fails as a malformed checkpoint rather
  than crashing.

#### Generation runtime (dense decoder)

- `NFKMLXGenerationOptions.cacheQuantization` stores the key-value cache affine-packed at 4 or 8 bits
  (`NFKMLXKeyValueCache.Quantization(bits:groupSize:)`; `groupSize` must divide the head dimension).
  It changes storage, not positions, so the window and mask accounting are untouched. Off by default
  because it is lossy; measured, 8-bit decoding tracks the full-precision logits at a cosine above 0.99.
- `NFKMLXGenerationOptions.prefillChunkSize` runs a long prompt through the cache in slices, bounding
  the prefill's attention peak by the chunk rather than the prompt. It is exact: each chunk attends
  through the cache to the keys a single pass would.
- `NFKMLXGenerationOptions.chatTemplate` (`NFKMLXChatTemplate.none` or `.chatML`) renders an
  `NFKInputMessages` list in the `<|im_start|>role … <|im_end|>` format an instruct release is trained
  on, using the release's own special tokens. Off by default; a base model wants the plain text, and a
  raw `NFKInputPrompt` is always used verbatim.
- `NFKMLXLanguage.backend(directoryURL:…)` refuses a release whose weights exceed the memory budget
  before materializing any of them. The stored bytes are compared against Metal's recommended working
  set, doubled for a `.float32` load of a 16-bit release, and the error names the shortfall and the
  remedies (`.checkpoint` precision, quantization, a smaller size). A machine with no readable budget
  does not gate the load.
- The dense loader validates every checkpoint shape against the built module, so a checkpoint whose
  shapes disagree fails at load instead of computing wrong numbers. The check is scoped to the dense
  decoder alone: shape adoption is load-bearing in the Gemma E4B and Conv-TasNet builders.

#### Long-form diffusion output

- `NFKMLXDiffusionBackend.windowedContinuation(totalWidth:height:channels:windowWidth:hop:scheduler:steps:seed:denoiseWindow:progress:)`
  denoises a latent longer than the model's native window by tiling one axis into overlapping windows
  and keeping continuity inside the sampler. Each window's overlap is held to the previous window's
  finished latent, re-noised to the current step's level through `scheduler.addNoise` (the same
  primitive the inpaint path uses, so any `NFKDiffusionScheduler` serves), locked after the loop, and
  the windows are stitched at the hop stride; the final window is pulled back to end exactly at the
  total. Window `i` draws its noise from `seed + i`; `progress` runs once per solver step and cancels
  by returning false. This is the `NFKMLXMusic3` mechanism lifted out for the shared-conditioning case
  (text-to-long-image, an extendable texture). Swift-only: it takes a closure over `MLXArray`.

#### Speech: a second recognizer and voice cloning

- `NFKMLXParakeet` runs Parakeet-TDT 0.6B v2 (NVIDIA NeMo, CC-BY-4.0): a FastConformer encoder with a
  depthwise-striding 8× subsampler and 24 relative-position conformer layers, and a token-and-duration
  transducer decoded greedily, so each step scores the next token AND how many frames to skip.
  `NFKMLXParakeetBackend` reads `NFKInputAudio` and returns the transcript under `NFKOutputText` plus one
  `NFKAudioSegment` per token under `NFKOutputSegments`. `backend(directoryURL:)` /
  `backendWithDirectoryURL:error:` read an unpacked `.nemo` through the native torch reader. At reference
  parity against NeMo's own model on the released weights and the validation clip: every encoder seam
  ≥ 0.99999999999, tokens and frame timestamps exact.
- `NFKMLXChatterbox` runs Chatterbox (Resemble AI, MIT), zero-shot voice cloning, ported stage by stage:
  the VoiceEncoder speaker embedding (`NFKMLXChatterboxVoiceEncoderNet`), the S3 speech tokenizer
  (`NFKMLXS3TokenizerNet`, FSMN attention over Whisper's log-mel, an 8-channel base-3 FSQ), T3
  (`NFKMLXT3Net`, the shipped dense decoder as a Llama 520M under llama3 rope scaling with a Perceiver
  prompt resampler, sampled with the reference's guidance, repetition-penalty, min-p, top-p chain), and
  S3Gen (`NFKMLXS3GenNet`: a CAMPPlus x-vector over a Kaldi filterbank, an upsampling conformer encoder,
  a causal conditional flow-matching U-Net, and the HiFT vocoder). `NFKMLXChatterboxTTS(directoryURL:)`
  loads the release; `conditionals(voice:sampleRate:)` prepares a voice from any WAV;
  `speechBackend(directoryURL:voiceURL:)` / `chatterboxBackendWithDirectoryURL:voiceURL:error:` return an
  `NFKMLXSpeechBackend` (24 kHz WAV under `NFKOutputAudio`; nil voice uses the built-in `conds.pt`).
  Every stage at reference parity on the released weights (speech codes 87/87 exact, teacher-forced T3
  argmax 79/79, waveform 0.99999999999); the validation sentence synthesized in the validation clip's
  voice transcribes back through `NFKMLXParakeet` exactly.
- `NFKMLXRoPEScaling` implements `llama3` beside `linear` and `yarn` (Chatterbox's T3 declares it), at
  parity with transformers' own initializer; `dynamic` and `longrope` stay refused.

#### Objective-C parity

- `NFKMLXGenerationParameterKey` carries the MLX generation options that have no core parameter key
  as request-parameter strings, read by `runInference` and overriding the backend's build-time
  options for that request: `contextWindow`, `cacheQuantizationBits` (4 or 8),
  `cacheQuantizationGroupSize` (default 64), `prefillChunkSize`, and `chatTemplate` (`"chatml"`).
- `NFKMLXLanguage.backendWithDirectoryURL:error:` builds the dense language model from a downloaded
  release directory (`config.json`, tokenizer, shards) at the default options.
- Every released size reaches every factory. Whisper: `backendWithVariant:weightsURL:tokenizer:timestamps:error:`,
  `backendWithVariant:weightsURL:error:`, `backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:error:`,
  and its `…completionHandler:` peer, over `NFKMLXWhisperVariant` (tiny / small / medium / large-v3),
  which was declared and reached by no factory. NAFNet and SwinIR gain
  `backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:error:` and the asynchronous peer;
  YOLO gains `backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:labels:error:` and its
  peer.
- RetinaFace: `NFKMLXRetinaFace.backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:`
  is the asynchronous download peer. `detectorWithWeightsURL:confidenceThreshold:suppressionThreshold:error:`
  returns an `NFKMLXRetinaFaceDetector`, now an `@objc` class, whose `facesInImage:error:` hands back
  `NFKFaceObservation` objects. `NFKFaceObservation` is now an `@objc` class carrying `boundingBox`,
  `confidence`, and the five landmarks by name (`leftEye`, `rightEye`, `nose`, `leftMouthCorner`,
  `rightMouthCorner`; the `landmarks` array stays Swift-only).
- `NFKMLXGPU.measuredMemoryBandwidth`, `measuredMemoryBandwidthWithMegabytes:repetitions:`, and
  `resetMeasuredBandwidth` expose the bandwidth probe `NFKMLXModelSizing` measures, beside the other
  machine readings.

#### Language models (continued)

- `NFKMLXDeepSeek` runs DeepSeek V4: Multi-head Latent Attention over a mixture of experts, with a
  learned per-head attention sink, a compressor and a sparse indexer, and Hyper-Connections that carry
  the residual as parallel copies. The arithmetic is at reference parity against transformers' own
  DeepSeek V4 at a tiny all-sliding configuration (every layer ≥ 0.9999995, logits 0.9999999999); the
  released Flash and Pro checkpoints are accounted for by shape, and `NFKMLXDeepSeekQuantization`
  decodes the release's fp8 and packed-fp4 storage exactly.
- `NFKMLXHybridLanguage` runs the hybrid Qwen3.5 / 3.6 / 3.8 decoder: three quarters of its layers
  replace attention with a gated delta-rule recurrence (a fixed-size state, cost linear in sequence
  length), every fourth layer a gated full attention. At reference parity on the released Qwen3.5-4B
  (all 33 hidden states exact, logit cosine 0.9999999999962); the 27B is accounted for by shape.

#### Text-to-image and video generation

- LTX-Video text-to-video, end to end: `NFKMLXLTXTransformer` (a 2B 3-D DiT), `NFKMLXLTXVideoVAE` (a
  causal 3-D autoencoder), `NFKMLXT5Encoder` (a T5-XXL v1.1 text encoder), and a rectified-flow
  sampler, assembled by `NFKMLXLTXPipeline`. Each stage is at reference parity against diffusers or
  transformers (DiT velocity 0.99999999999, VAE decode 0.99999999996, T5 embedding 0.99999999998).
- Z-Image text-to-image: `NFKMLXZImageTransformerNet`, a 6B single-stream S3-DiT over the shared
  `NFKMLXSDAutoencoder` in a new Flux configuration, with the shipped Qwen3 decoder as its text encoder
  (`NFKMLXZImagePipeline`). DiT velocity cosine 0.9999999999999 against diffusers.
- SANA text-to-image: `NFKMLXSANATransformerNet`, a linear-attention DiT over the Deep-Compression
  autoencoder `NFKMLXDCAutoencoderNet` (32× spatial), with Gemma-2 (`NFKMLXGemma2Net`) as its text
  encoder and the released DPM-Solver++ sampler (`NFKMLXSANAPipeline`). DiT velocity 0.9999999999999,
  DC-AE decode on the released Sana_600M VAE 0.9999999998.
- Wan text-to-video: `NFKMLXWanTransformerNet` (a 3-D DiT), `NFKMLXWanVideoVAENet` (a stateful causal
  3-D VAE, the 2.1 and 2.2 residual paths), the umT5 text encoder, and the released UniPC sampler
  (`NFKMLXWanPipeline`). DiT velocity 0.9999999999999.
- `NFKMLXTAESD` — the tiny Stable Diffusion autoencoder for an instant latent preview, at parity
  against madebyollin's taesd (latent and decode 0.9999999999996).
- `NFKMLXIPAdapterImageProjection` / `NFKMLXIPAdapterAttention` — IP-Adapter image conditioning,
  threaded into the shipped `NFKMLXSDUNet` cross-attention, at reference parity against diffusers.
- The released multistep samplers ship as value types, each verified exactly against diffusers:
  `NFKMLXDPMSolverScheduler` (DPM-Solver++) and `NFKMLXUniPCScheduler` (UniPC).

#### Speech synthesis, restoration, and enhancement

- `NFKMLXKokoro` runs Kokoro-82M (StyleTTS2 / iSTFTNet), a phoneme-to-waveform voice at reference
  parity seam by seam against the vendored `KModel`: the deterministic seams exact, the NSF vocoder
  float-precision-limited (waveform ~0.997). It takes a phoneme string, so a caller brings the
  grapheme-to-phoneme front end.
- Eight speech restoration and enhancement models, each at measured reference parity on the released
  weights and reached through an `NFKMLXSpeechBackend`: `NFKMLXMPSENet` (MP-SENet, magnitude and
  phase), `NFKMLXGTCRN` (GTCRN, ~48K parameters, real-time), `NFKMLXSGMSE` (SGMSE+, score-based
  generative dereverberation), `NFKMLXStoRM` (StoRM, few-step stochastic regeneration),
  `NFKMLXMossFormer2SENet` (MossFormer2 SE 48K), `NFKMLXDeepFilterNet` (DeepFilterNet3, real-time
  48 kHz denoising), `NFKMLXVoiceRestore` (a flow-matching universal restorer with a BigVGAN v2
  vocoder), and `NFKMLXResembleEnhance` (a five-network general restorer). Waveform cosines run from
  above 0.999 to 0.9999999.

#### Detection, matting, depth, vision, and audio codecs

- `NFKMLXRTDetr` (RT-DETR) and `NFKMLXRFDetr` (RF-DETR) — transformer object detectors under Apache
  licenses, each at reference parity on the released weights (RT-DETR r50vd logits 0.9999999999,
  RF-DETR base decoder 0.9999999999). Both return `NFKDetection`s with no non-max suppression.
- `NFKMLXBiRefNet` — high-resolution background removal (MIT), at reference parity on the released
  weights (every seam ~1e-12). Its `ASPPDeformable` decoder's deformable convolution reduces to the
  shared bilinear-gather primitive, so no DCNv2 Metal kernel is needed.
- `NFKMLXDepthAnything3` — Depth Anything 3 monocular depth (DA3-SMALL), at reference parity against
  the authors' package (every backbone and head stage ≥ 0.9999999999, the exp-depth map 0.9999999999).
- `NFKMLXSigLIP2` — SigLIP 2 image and text embeddings (base-patch16-224), at reference parity against
  transformers (image and text embedding 0.999999999999, sigmoid logits to 1e-5).
- `NFKMLXDAC` (Descript Audio Codec) and `NFKMLXSNAC` (multi-scale) — the first neural audio codecs,
  each at reference parity (codes exact, reconstruction 0.99999999999).
- `NFKMLXSileroVAD` — Silero VAD v6, a streaming STFT-plus-convolution-plus-LSTM voice-activity
  detector, at reference parity against the released JIT (per-chunk 0.99999999999, threshold agreement
  exact).

### Changed

- `NFKFaceObservation` and `NFKMLXRetinaFaceDetector` are classes rather than structs. Reads are
  source-compatible; a Swift consumer that relied on value semantics now holds a reference.
  `NFKFaceObservation` gains a public `init(boundingBox:landmarks:confidence:)`.
- A dense-decoder release that does not fit the working set now throws at
  `NFKMLXLanguage.backend(directoryURL:…)` instead of proceeding to a process kill. A release that
  loaded at `.float32` with little margin may now be refused; load it at `.checkpoint` precision.
- A dense-decoder checkpoint whose tensor shapes disagree with `config.json` now fails at load.

### Documentation

- `Docs/examples.md`: a "Loading a PyTorch checkpoint directly" section in Swift and Objective-C;
  generation-option examples (cache quantization, chunked prefill, chat template, the fit refusal);
  an Objective-C example configuring generation through `NFKMLXGenerationParameterKey`, building the
  language model from a directory, reaching Whisper's sizes, reading a face's landmarks, and the
  bandwidth probe. Each is mirrored by a compiled example test in `InferKitMLX/Examples` and
  `InferKitMLX/ObjCExamples`.
- The `WeightsAndConversion` DocC article gains "Reading a PyTorch checkpoint natively" and a
  "PyTorch checkpoints" topic group (`NFKMLXTorchCheckpoint`, `NFKMLXTorchTensorInfo`,
  `NFKMLXTorchScalarType`).
- `Docs/mlx-runtime-hazards.md`: duplicate keys crash `ModuleParameters.unflattened` with a stack
  overflow (found by loading the raw RAFT checkpoint, whose `norm3` / `downsample.1` rename collides
  deliberately). Build remapped parameters into a dictionary first.
- `Docs/companions.md`: the quantized Music 3 stack is 7.7 GiB, and `transformerBits: 6` is documented
  as the measured fallback (DiT velocity cosine 0.99844 against 0.99990 at 8-bit).
- `Tools/README.md` (new) records that nothing under `Tools/` ships in any distribution, that a
  consumer needs no Python, and why the converters and the Python reference-parity tooling remain.
  Every `*-to-safetensors/convert.py` docstring now says the converter is optional.
- `CLAUDE.md` / `AGENTS.md`: Objective-C parity is a written, required rule for new and changed code,
  with the fixed list of what legitimately stays Swift-only.

### Tooling

- `Tools/validation-assets/fetch.py` gains two acquisition routes for a raw checkpoint, `gdrive` (a
  Google Drive id, MODNet) and `extract` (a member of a zip at `url`, LaMa's Lightning `best.ckpt`),
  and stamps `IK_RAW_<KEY>` for every asset whose raw download is on disk; the manifest adds the raw
  NAFNet, RIFE, LaMa, MODNet, and YOLO entries. The raw-versus-converted parity tests read those keys
  and skip when absent.
- `NFKMLXZipArchiveTests`, `NFKMLXPickleTests`, and `NFKMLXTorchCheckpointTests` run under
  `swift test`: the parsing layers are pure Foundation below the MLX materialization.

#### Language-model runtime, continued

- `NFKMLXKeyValueCache.rollback(by:)` discards the newest positions by moving cursors; it returns
  false, changing nothing, where a window has already dropped what it would reach.
- `NFKMLXPromptCache` keeps a key-value cache and its token ids between generations and rolls back to
  the longest prefix a new prompt shares, so a conversation's next turn prefills only what it adds.
  `save(to:)` / `load(from:)` persist it, quantized rows included.
  `NFKMLXGenerationOptions.reusesPromptCache` (request key `NFKMLXGenerationParameterKey.reusesPromptCache`)
  makes the backend keep one; `NFKMLXLanguageBackend.resetPromptCache` and `promptCacheLength` reach it.
- Speculative decoding: `NFKMLXLanguageNet.generate(prompt:options:draft:promptCache:report:onToken:)`
  verifies a draft model's proposals in one cached pass and keeps the leading run that agrees, so the
  output is the model's own greedy run token for token (measured on Qwen3-1.7B drafted by 0.6B:
  identical, 73% of proposals accepted, no wall-clock gain at float32 on an M1 Max, where a step is
  launch-bound). `NFKMLXLanguage.backend(directoryURL:draftDirectoryURL:options:)`, the `@objc`
  `backendWithDirectoryURL:draftDirectoryURL:error:`, `NFKMLXGenerationOptions.draftTokens`, the
  request key `draftTokens`, and `NFKMLXSpeculativeReport`.
- Mixture of experts: the dense decoder reads Qwen3-MoE (`qwen3_moe`) and Mixtral (`mixtral`)
  releases through the same factories. `NFKMLXLanguageConfiguration` gains `expertCount`,
  `activeExpertCount`, `expertIntermediateSize`, and `normalizesExpertWeights`; the loader stacks the
  released per-expert tensors into one tensor per projection and runs the chosen experts through a
  gathered matrix multiplication; `NFKMLXQuantization` packs the stacked experts and the checkpoint
  contract carries them. Both families match `transformers`' arithmetic layer by layer at a tiny
  configuration, and every released tensor of Qwen3-30B-A3B is accounted for by shape. Configs that
  interleave dense layers, use a sliding window, or name another expert family are refused.
- Constrained decoding: `NFKMLXTokenConstraint` masks the logits before sampling. `NFKMLXJSONConstraint`
  admits only well-formed JSON (root `.container` / `.object` / `.array` / `.any`, whitespace runs
  capped) and ends the run when the document closes; `NFKMLXChoiceConstraint` admits exactly one of a
  fixed set of strings; `NFKMLXByteConstraint` is the base for any byte-level grammar with a hashable
  state, with the admissible mask cached per state. `NFKMLXVocabulary` reads every token's bytes from
  the tokenizer. `NFKMLXGenerationOptions.constraint`, `jsonOutput`, `jsonRoot`, and `choices`; the
  request keys `outputFormat` (`"json"`, `"json-object"`, `"json-array"`) and `choices`.

### Core (`InferKit`)

- `NFKTokenizer.bytesForTokenId:` returns the bytes one id contributes to decoded text (a fragment of
  a multi-byte character as that fragment, a special token as its literal), implemented by the
  byte-level BPE, unigram, and word-piece tokenizers.
- Remote model discovery. `NFKRemoteProvider.modelsWithAPIKey:error:` (and a completion-handler form)
  lists the models a provider currently serves as `NFKRemoteModel`s (identifier, display name, owner,
  creation time, and context length where the provider publishes them, plus the entry as received),
  so an app fills a picker from the server rather than from a constant. Every preset answers the
  same `data[].id` envelope, hosted and local alike; the credential travels in the provider's own
  header style, and Anthropic's pagination is followed to the end. `NFKRemoteModelCatalog` is the
  object under the convenience, with a timeout, a session, `isReachableWithError:`, and the
  overridable transport seam the other remote classes carry.
- `NFKRemoteProvider.baseURL` is the one address a preset carries; `endpointURL` and `modelsURL` are
  derived from it, and `URLForPath:` builds any other operation's URL. `providerWithBaseURL:` re-points
  a preset at another port or another machine, keeping its identity, protocol, and key requirement.
- `NFKRemoteTransport` holds the blocking request, the per-style credential headers, and the
  status-to-error mapping the chat, Anthropic, transcription, and catalog classes each carried a copy
  of. Each class keeps its own `sendRequest:response:error:` seam, delegating there by default.
- `kNFKError_RemoteUnreachable`: a remote endpoint produced no response at all (the host is down or
  refused the connection), with the URL-loading error under `NSUnderlyingErrorKey`. A local runner that
  is not running is now told apart from one that answered with an error or an empty list.
- `NFKRemoteEmbeddingBackend`, the embeddings counterpart of the chat client (`POST /embeddings`, which
  the hosted providers and every local runner serve): `NFKInputPrompt` (or joined `NFKInputMessages`)
  in, the vector under `NFKOutputEmbedding` out, the key the on-device embedders in InferKitMLX
  answer with. `embeddingsForTexts:error:` embeds a batch in one request, ordered by the provider's
  index rather than by arrival. `backendForProvider:apiKey:modelName:` derives the endpoint and
  answers nil for Anthropic, which serves none.
- Local-runner management. `NFKLocalModelRunner` is the shape a runner's native API is reached
  through — `isRunning`, `installedModelsWithError:`, `loadedModelsWithError:`,
  `detailsForModel:error:`, and the optional `versionWithError:`, `pullModel:`, and
  `deleteModel:error:` that change the machine — and `-[NFKRemoteProvider localRunner]` hands back
  the adapter for a preset. `NFKOllamaRunner` adopts the whole set over `/api/tags`, `/api/ps`,
  `/api/show`, `/api/version`, `/api/pull`, and `/api/delete`; the pull is streamed as an
  `NFKInferenceJob` with the layer fraction as progress, the runner's status line as `partialResult`,
  and cancellation. `NFKLMStudioRunner` adopts the reading set over `/api/v0/models`, whose entries
  carry the loaded state. llama.cpp and vLLM have nothing beyond the OpenAI surface, so their presets
  answer nil. Measured against Ollama 0.33: the installed list already carries each model's context
  length, quantization, and capabilities, and a failing pull answers HTTP 200 with the failure in an
  error line inside the stream, which the job reads rather than trusting the status.
- `NFKRemoteModel` gains `sizeBytes`, `quantization`, and `capabilities`, read where a runner reports
  them, and takes its identifier from `model` or `name` where a list carries no `id`.
  `NFKRemoteModelCatalog.modelWithIdentifier:error:` reads one model (`GET /models/{id}`), which
  answers 404 for a name the provider does not know.
- `NFKRemoteSpeechBackend` (text → speech, `POST /audio/speech`): `NFKInputPrompt` in, an
  `NFKAudioAsset` under `NFKOutputAudio` out, a WAV by default so it is interchangeable with what
  `NFKMLXSpeechBackend` writes; the voice is required and has no default. Verified by probe to be
  served by openai, groq, together, xai, mistral, and openrouter.
- `NFKRemoteImageBackend` (`POST /images/generations` and `/images/edits`): a prompt alone generates,
  an image under `NFKInputImage` edits it (a multipart body carrying the image as PNG), and a mask
  under `NFKInputMask` inpaints — the operation chosen from the request the way `NFKMLXBackend`
  chooses. `NFKParameterWidth`/`Height` become the service's size, seed and steps pass through, and an
  inline base64 reply or a URL the backend fetches both decode to a 32BGRA `CVPixelBuffer` under
  `NFKOutputImage`. Generations verified for openai, together, xai, and openrouter; edits for openai
  and xai. This is the synchronous shape; `NFKAsyncGenerationBackend` remains the job-style one.
- Vision through the chat backends. An image under `NFKInputImage` beside a prompt or a conversation
  rides on the last user turn: `NFKRemoteBackend` sends it as an inline `image_url` content part
  (base64 PNG), which the OpenAI-compatible vision models read, the local runners' included —
  measured against Ollama with `qwen3.5:27b` — and `NFKAnthropicBackend` as a base64 `image` block
  before the text. An image that is not one of the accepted representations is refused before any
  request as `kNFKError_InferenceMissingInput`.
- `NFKImageCoding`, the public codec under the above: PNG bytes or a data URL from a `CGImage`, a
  32BGRA/32RGBA `CVPixelBuffer`, or a BGRA8/RGBA8 `MTLTexture`, and a 32BGRA pixel buffer back from
  any format ImageIO reads. The core now links ImageIO.
- The chat backends stream. `submitInferenceJobForRequest:` on `NFKRemoteBackend` and
  `NFKAnthropicBackend` sends the request with streaming on and reads the reply as server-sent
  events: each text delta appends to the job's `partialResult`, a tool call assembles across its
  argument deltas, and the job finishes with the same result the blocking form returns.
  **Cancelling the job cancels the request**, where the generic wrapper let a cancelled remote call
  run to the end on the server. `NFKInferenceSubmit` reaches the streamed form. Measured against a
  live Ollama: the text arrives in more than one piece and the last partial is the final text.
- Tool calling and structured output over remote. `NFKParameterTools` (`{name, description,
  parameters}`) becomes OpenAI `function` tools or Anthropic's tool shape; what the model called comes
  back under `NFKOutputToolCalls` (`result.toolCalls`: `{id, name, arguments, argumentsJSON}`, the
  arguments parsed). `NFKParameterJSONSchema` becomes a `json_schema` response format, or on Anthropic
  a forced tool whose input is the reply, and the parsed reply comes back under `NFKOutputStructured`;
  a `response_format` folded in by name is parsed the same way. A tool already in a provider's wire
  shape passes through unwrapped. Measured against a live Ollama: `qwen3.5:27b` calls `get_weather`
  with `{"city": "Paris"}`.
- `NFKInputImages` carries further images beside `NFKInputImage`, attached in order.
- `NFKRemoteTransport` retries a blocking call on 429, 502, 503, or 504 after the provider's
  `Retry-After` or an exponential delay from half a second, `retryAttempts` (default 2) more times and
  never after a delay above `maximumRetryDelay` (default 8 s); a refused connection is not retried.
  `streamRequest:session:lineHandler:completionHandler:` is the line-delimited streaming primitive the
  chat backends and the Ollama pull share, delivering a failing status's body whole; `SSEDataForLine:`
  reads a server-sent-events data line.
- More directions in and out of the chat backends. `NFKInputAudio` beside the prompt is sent as an
  `input_audio` part (the Messages API refuses audio rather than dropping it); `NFKInputDocument` and
  `NFKInputDocuments` (PDFs, as an NSURL or NSData) as `file` parts or `document` blocks;
  `NFKInputVideo` is sampled into `NFKParameterVideoFrameCount` evenly spaced frames (default 8) for
  a vision model, on both backends — measured against a live Ollama, which names the colours of a
  sampled clip. `NFKParameterAudioOutput` (`{voice, format}`) asks an OpenAI-compatible model to
  speak its reply, returned as an `NFKAudioAsset` under `NFKOutputAudio` beside the text, its chunks
  assembled when streamed.
- `NFKVideoSampling`, the public piece under the clip input: `framesOfVideoAtURL:count:error:`
  returns evenly spaced `CGImage`s through AVFoundation, which the core now links. A sample is taken
  a millisecond into the frame its midpoint falls in rather than on a frame edge, and a decode
  session that comes up broken is recreated once — measured: right after a large model had
  occupied the GPU, the hardware decoder reported `kVTVideoDecoderMalfunctionErr` after a
  four-minute timeout, and the next session decoded correctly.
- `NFKRemoteTranscriptionBackend.emitsTimestamps` asks for the verbose reply and adds its segments
  under `NFKOutputSegments` as `NFKAudioSegment`s (confidence from the decoder's mean log
  probability), which is what the on-device Whisper backend emits; `translates` sends the audio to
  the sibling `/audio/translations` endpoint for an English transcript. `backendForProvider:apiKey:
  modelName:` derives the endpoint.
- `NFKRemoteVideoBackend`, the first shipped `NFKAsyncGenerationBackend`: OpenAI's videos API
  (verified by probe; no other preset serves one) as submit, poll, and download. A prompt, or a
  prompt with `NFKInputImage` as the reference frame (a multipart submit), `NFKParameterDurationSeconds`
  and the size; the service's percentage progress becomes the job's fraction, and the finished clip
  is downloaded as an `.mp4` `NFKVideoAsset` under `NFKOutputVideo`. The base gained two hooks:
  `submitRequestForRequest:` (a subclass building a multipart submit) and
  `failureReasonFromStatusResponse:` (the service's own message in the job's error).
- `NFKRemoteReranker`: query and documents → one relevance score each (`POST /rerank`; verified on
  together and openrouter), put back in the documents' order; `rankedIndicesForQuery:documents:error:`
  and `scoreForQuery:document:error:` beside it, the shape of the on-device `NFKMLXModernBERTReranker`.
- `NFKRemoteModerationBackend` (`POST /moderations`; verified on openai and mistral): text, and an
  image where the service reads one, → per-category `NFKClassification`s under
  `NFKOutputClassifications`, most confident first, and the verdict under `NFKOutputStructured`
  (`flagged`).

### Changed

- The remote backends report a connection-level failure as `kNFKError_RemoteUnreachable` in
  `NFKInferenceErrorDomain` where they previously passed the `NSURLErrorDomain` error through; that
  error is now the underlying one. A failing HTTP status carries the response body in its description
  for every remote backend, where only the Anthropic backend included it before.
- `NFKMLXLanguage.backend(directoryURL:)` now reads the release's special tokens and `eos_token` from
  `tokenizer_config.json`. A ChatML marker encodes to its single id where it previously encoded as
  plain text, and generation stops at the release's end-of-sequence token when a request names no
  stop tokens where it previously ran to `maxTokens`.
- `NFKMLXLanguageBackend` is exported to Objective-C under that name (`@objc(NFKMLXLanguageBackend)`).

### Tooling

- `Tools/validation-assets/shapes.py` fetches a release's `config.json` and every tensor's shape and
  dtype by HTTP range request over the safetensors headers, without downloading weights.
- `run_reference.py` gains the `qwen3_moe` and `mixtral` tiny-configuration modes, and parses under
  the LLM oracle environment's Python 3.9 again (a backslash inside an f-string expression, legal
  only from 3.12, had made every mode there unrunnable).

### Documentation

- `Docs/inference-guide.md`, deleted in an earlier commit, is restored and rewritten for the current
  backends, with a roadmap. `Docs/examples.md` gains prompt-cache, speculative-decoding,
  mixture-of-experts, and constrained-decoding sections in Swift and Objective-C, each mirrored by a
  compiled example, and Objective-C examples for the fit-before-load refusal and the context window,
  both run through the directory factory against a release written on the fly;
  `Docs/companions.md` lists the language backend.
- `AGENTS.md` is again a byte-identical copy of `CLAUDE.md`. The two had drifted since v0.2.0, with
  `AGENTS.md` taking abbreviated versions of the same edits; `CLAUDE.md` is the one to edit.


## [0.2.0] — 2026-08-29

### Core (`InferKit`)

- `NFKInferKit.version` — a class property returning the core's semantic-version string
  (`NFKInferKit.h`, in the umbrella).
- `NFKComputePlan` reports where Core ML plans to run each operation of a compiled model, without
  running it: `planForCompiledModelAtURL:…` (synchronous and completion-handler forms),
  `neuralEngineOperationCount` / `gpuOperationCount` / `cpuOperationCount` / `unknownOperationCount`,
  `neuralEngineFraction`, `runsEntirelyOnNeuralEngine`, `operatorNamesOffNeuralEngine`, and
  `describedPlacement`. `MLComputeUnits` is a request, and Core ML says nothing about moving an
  operation the Neural Engine cannot run; this is the answer. Needs macOS 14.4 / iOS 17.4 / tvOS 17.4
  (`isAvailable`); an older system fails with `kNFKError_InferenceUnsupported` rather than returning
  an empty plan.
- `NFKHardwareProfile` reports what the machine is and how much of it is left: `currentProfile` with
  `chipName`, `modelIdentifier`, `graphicsArchitecture`, `performanceCoreCount`,
  `efficiencyCoreCount`, `hasUnifiedMemory`, and three distinct ceilings (`physicalMemory`,
  `recommendedWorkingSetSize`, `maximumBufferLength`); `+availableMemory` is a live reading. Every
  reading degrades to zero or an empty string rather than throwing.
- `NFKCoreMLBackend.computeUnits` selects the compute units a model loads with (default
  `MLComputeUnitsAll`; set before `loadModelFromURL:error:`).
- `NFKInputLyrics` — the lyrics a music-generation backend sings, distinct from `NFKInputPrompt`.
- `NFKByteLevelBPETokenizer` takes a `pretokenization:` name (`"gpt2"`, the default, or `"qwen2"`),
  and `NFKTokenizer.tokenizerForManifest:` reads it from the manifest's `tokenizer.pretokenizer`
  key. An unknown name is refused rather than defaulted.

### InferKitMLX (companion)

- MiniMax Music 3: `NFKMLXMusic3` and `NFKMLXMusicBackend` (`@objc`, `backendWithDirectoryURL:error:`,
  registry name `minimax-music3`) generate stereo 44.1 kHz music from a description
  (`NFKInputPrompt`) and lyrics (`NFKInputLyrics`), honoring `NFKParameterDurationSeconds`, `Seed`,
  `Steps`, and `GuidanceScale`. The five networks are each at measured reference parity against
  diffusers' own implementation, and the prompt contract is token-exact against the release
  tokenizer. The weights are under the MiniMax-Music3 Community License, which is not permissive.
- Runtime MLX quantization. `NFKMLXMusic3.quantizeRelease(at:to:bits:transformerBits:groupSize:)`
  writes a quantized copy of the release in the release's own layout (4-bit language model including
  its untied input embedding, 8-bit DiT, the split measured), which brings the stack from 27 GB to
  7.7 GiB and lets it stay resident between runs. A quantized checkpoint records its bits and group
  size in metadata (`inferkit.quantization`), the language-model and music loaders rebuild the
  matching packed structure before applying it, and it loads at its stored types whatever precision
  the caller requests.
- `NFKMLXGenerationOptions.contextWindow` bounds the key-value cache (`NFKMLXKeyValueCache(layerCount:window:)`):
  the oldest positions are dropped, so memory stops growing with the conversation, while rotary
  offsets stay absolute. The cache holds its rows in a buffer that grows in blocks rather than
  concatenating per token. Off by default; for a model whose attention is not natively windowed it
  is exact while the conversation fits and an approximation past that.
- `NFKMLXModelSizing` / `NFKMLXModelFit` decide whether a language model fits before anything is
  allocated: `parameterCount(of:)` counts a dense decoder from its geometry, `keyValueBytesPerToken`
  counts the cache at the key-value head count, `fit(of:tokens:precision:budget:)` returns `fits` /
  `fitsWithinWindow(n)` / `tooLarge(shortfall:)`, and `options(for:requesting:…)` derives a
  `contextWindow` from the machine. `measuredMemoryBandwidth` times a large read rather than
  tabulating per chip; `decodeCeiling` and `achievedFraction` turn it into tokens per second.
  Swift-only (it takes `NFKMLXLanguageConfiguration`).
- `NFKMLXGPU` reports the machine beside MLX's counters: `physicalMemory`,
  `recommendedWorkingSetSize`, `reclaimableMemory`, `memoryPressure`, `deviceArchitecture`, and
  `applyStandingLimits` / `applyStandingLimitsWithCacheBytes:fractionOfRecommendedWorkingSet:`
  (a standing cache cap, default 256 MB, plus a soft memory limit). Swift adds the scoped async
  `withWiredLimit(_:_:)`; there is deliberately no persistent wired-limit setter.
- `NFKMLXRoPEScaling` applies a release's `rope_scaling` (`linear` and `yarn`) in the dense decoder
  and DeepSeek, at reference parity against transformers' own frequency functions. `dynamic`,
  `llama3`, and `longrope` are refused rather than approximated.
- `NFKMLXLanguage.configuration(fromHuggingFace:)` reads transformers-5.x-shaped configs: a
  `layer_types` list naming only `full_attention` is dense (a mixed stack is still rejected), and
  `rope_theta` is read from `rope_parameters`. A release directory in the diffusers spelling
  (`diffusion_pytorch_model.safetensors`, single or sharded) also resolves.
- `NFKDiffusionLatentPreview` gives a diffusion job's progress callback an image: a per-channel map
  from latent to RGB, reported as the job's `partialResult` each step through
  `NFKDiffusionConfiguration.latentPreview` and thinned by `previewEverySteps`. `.passthrough`,
  `.stableDiffusion`, and `.stableDiffusionXL` are the published factors (the SD map tracks a real
  decode at a mean-removed correlation of 0.93); `fitted(latentChannels:decode:sample:)` derives a
  map for any decoder by least squares.
- RAFT's correlation lookup runs as one fused Metal kernel per pyramid level on the GPU
  (`MLXFast.metalKernel`, compiled and cached by MLX on first use): 3.98 ms against 2753 ms at the
  model's eighth-resolution geometry. The gather path remains for the CPU stream and is what the
  kernel is held to, bit for bit.
- `NFKMLXTrainer.clipGradientNorm` sanitizes before it clips: non-finite entries are zeroed and the
  norm is taken relative to the largest magnitude, so a finite gradient set whose sum of squares
  overflows is scaled to the norm rather than to zero.
- `NFKMLXLoRA.apply` and `merge` refuse a quantized model with `loRANotApplicable`: a
  `QuantizedLinear` satisfies a `Linear` type test while holding packed integers, and merging a delta
  into a quantized base then requantizing rounds the training away.
- `NFKMLXReferenceModels.registerAll` registers `minimax-music3`.

### Changed

- The Qwen text path (`NFKMLXLanguage.backend(directoryURL:)`) tokenizes with the `qwen2`
  pre-tokenization pattern rather than the byte-level BPE GPT-2 default, which encodes the same prompt
  to different, valid-looking token ids. A consumer that cached Qwen-encoded text from an earlier
  build re-encodes it.
- Every asynchronous path in InferKitMLX runs at user-initiated quality of service (`Task.detached(priority: .userInitiated)`).
  An unprioritized task inherits the default, which Apple Silicon schedules on the efficiency cores; a
  decode loop there ran several times slower.

### Documentation

- `Docs/mlx-runtime-hazards.md` (new, linked from the README): where MLX, Metal, and Core ML return a
  wrong answer quietly, each entry with an executable probe in `NFKMLXRuntimeHazardTests`. Four
  hazards reported elsewhere against mlx-swift 0.31.6 were reduced and do not reproduce here.
- `Docs/companions.md`: the `NFKMLXMusicBackend` entry and a "Model weight licenses" section
  recording the MiniMax-Music3 Community License obligations.
- `Docs/examples.md`: the MLX language backend in Swift, bounding the cache, sizing a model to the
  machine, extended context, `NFKComputePlan`, `NFKHardwareProfile`, and the latent preview, each
  mirrored by a compiled example.
- DocC pages for `NFKMLXMusic3` and `NFKMLXMusicBackend`, and a "Text → music" topic group.

### Tooling

- `Tools/ane-placement/`: paired Core ML models differing in exactly one property, to measure what
  lands on the Neural Engine. The measurement isolated why a converted language model runs on the
  GPU: a single-token forward is not placed on the Neural Engine, and a multifunction package takes
  one placement decision for both functions.
- `Tools/reference-parity/run_reference.py` gains the `music_vocoder`, `music_depth`,
  `music_condition`, `music_dit`, `music_ar`, `music_tokenizer`, and `rope_scaling` modes; the
  validation manifest gains the `MUSIC3` entry and records the music oracle's own interpreter under
  `oracle_environments`. `fetch.py` lands an `input` asset under `inputs/` as-is.

## [0.1.0] — 2026-08-22

First public release.

### Core (`InferKit`)

- `NFKInferenceBackend`, `NFKInferenceRequest`, `NFKInferenceResult`, and the thread-safe
  `NFKInferenceJob` handle, with a shared input/parameter/output key vocabulary.
- Backends with no third-party dependency: the passthrough mock, in-process Core ML (image and tensor),
  an on-device Core ML language model, OpenAI-compatible chat and transcription clients, and a
  submit-poll-fetch base for generation services.
- `NFKDynamicBackend` — runtime discovery, so an optional engine activates only when its classes are
  linked, with no build dependency on it.
- `NFKRemoteProvider` — named presets for the services a consumer is likely to call: OpenAI, xAI Grok,
  Google Gemini, Groq, Mistral, DeepSeek, Together, OpenRouter, and the local servers Ollama, LM
  Studio, llama.cpp, and vLLM, all OpenAI-compatible. `NFKAnthropicBackend` covers Anthropic's Messages
  API, whose authentication, required fields, and response envelope differ.
- Value types for detections, pose keypoints, classifications, audio segments, video, and audio.
- Tokenizers as a class cluster: byte-level BPE, CLIP, WordPiece, and Unigram, built from tokenizer
  files.
- `NFKHFHub` — Hugging Face resolve, download, checksum, and cache, with bearer-token support for a
  gated repository through `accessToken` or `HF_TOKEN`.
- RGBA-interleaved ↔ planar tensor conversion and an `MLMultiArray` bridge.
- A DocC catalog, and examples compiled in both Objective-C and Swift so a documented snippet cannot
  drift from the API.

### InferKitMLX (companion)

- Around thirty-five models implemented in `MLXNN` and validated at measured reference parity against
  each model's own reference implementation, spanning image, video, audio, and text.
- Stable Diffusion text-to-image built entirely from this package's own components — CLIP text tower,
  UNet, DDIM sampler, autoencoder — at end-to-end parity for SD 1.5, SD 2.1, and SDXL-Turbo.
- On-device text generation: `NFKMLXLanguage`, a dense decoder (grouped-query attention, rotary
  embeddings, SwiGLU) with a key-value cache and sampling, at measured parity against Qwen3 0.6B, 1.7B,
  and 4B, including sharded checkpoints.
- `NFKMLXHybridLanguage` — the Qwen3.5 / 3.6 / 3.8 hybrid decoder (gated delta-rule recurrence with
  full attention every fourth layer). Built and verified structurally against the released Qwen3.8-27B
  checkpoint; its numerics are unmeasured, because the smallest release does not fit on the machine
  that built it.
- `NFKMLXDeepSeek` — the DeepSeek V4 decoder (multi-head latent attention over a mixture of experts),
  with `NFKMLXDeepSeekQuantization` decoding the block-scaled fp8 and 4-bit formats the release is
  stored in.
  Built and verified structurally against the released Flash checkpoint; unmeasured numerically, and
  verified more weakly than the hybrid because that release is quantized.
- `NFKMLXGemmaLanguage` — the Gemma 4 text decoder (per-layer input embeddings, sliding and full
  attention at different head widths, soft-capped logits). E2B and E4B at reference parity; the config
  guard is exact, so the family's other members are refused rather than run wrong.
- The Gemma 4 **mixture of experts** (the 26B-A4B family, `enable_moe_block`) now runs: every layer's
  dense feed-forward gains a parallel routed-expert branch, and the two normed halves are summed. The
  router normalizes without a scale, applies a learned per-channel scale and the `hidden^-0.5` factor,
  softmaxes, keeps the top `k`, renormalizes them, and multiplies by a learned per-expert scale; the
  experts are a fused gate-and-up projection and a down projection, dispatched through `gatherMM`. At
  reference parity against transformers' own `Gemma4ForCausalLM` at a tiny configuration exercising
  both layer kinds (`run_reference.py gemma4_moe`, `IK_PARITY_GEMMA4_MOE`): every hidden state exact
  layer by layer, logit cosine 0.9999999999996. Two geometry facts were load-bearing: the per-layer
  input embedding is a fixed 262144 rows (`vocab_size_per_layer_input`), not the token vocabulary
  (the two coincide on the E-series, which is why reading the token vocabulary there was invisible),
  and a full-attention layer runs a 512-wide head where the sliding layers run their own `head_dim`.
  The mixture is read from `config.json` through `NFKMLXGemmaLanguage.configuration(fromHuggingFace:)`,
  the same entry the dense sizes use. (The Gemma 4 decoder does not yet expose a public
  `NFKInferenceBackend` factory — a pre-existing gap for the whole family, dense sizes included.)
- The Gemma 4 **12B unified text decoder** (`gemma4_unified_text`) runs, a different architecture from
  the E-series: no per-layer input embeddings and no mixture, only the sandwich block with a per-layer
  scalar, over the same attention (learned query/key norms, a scale-free value norm, per-layer head
  widths, the proportional rotary on the full layers). `NFKMLXGemma4UnifiedNet` reuses the E-series
  attention and feed-forward directly. At reference parity against transformers' own
  `Gemma4UnifiedForCausalLM` at a tiny configuration (`run_reference.py gemma4_unified`): every layer
  exact, logit cosine 0.9999999999995, on the first numeric run.
- The Gemma 4 **vision encoder** (`Gemma4VisionModel`) runs, the image tower of the tri-modal release:
  a patch embedder (one linear projection plus a learned 2-D position embedding) and the sandwich block
  run bidirectionally with no rotary. `NFKMLXGemma4VisionNet`; the projections are clippable linears
  (a linear under a `.linear` key). At reference parity against transformers' own encoder output
  (`run_reference.py gemma4_vision`): cosine 0.9999999850. The **pooler** runs too (`softTokens`) — a
  position-based average pool plus `√hidden` scaling — at parity against the full model output (pooled
  cosine 0.9999999835).
- The Gemma 4 **audio Conformer** (`Gemma4AudioModel`) runs, the most complex tower: a 2-D convolutional
  subsampler, then Conformer layers — a macaron feed-forward, a blocked relative-position attention
  (Transformer-XL rel-shift, a per-dimension softplus scale, a logit softcap), a causal light depthwise
  convolution, a second macaron feed-forward, sandwich norms — and an output projection.
  `NFKMLXGemma4AudioNet` runs end to end from the mel features, building its own sliding-window blocked
  mask, at reference parity against transformers' own `Gemma4AudioModel` (`run_reference.py gemma4_audio`,
  subsampler cosine 0.9999999999999845, full tower 0.999999999999996). All four Gemma 4 architectures
  (MoE, unified, vision, audio) are at reference parity, and both towers run their full forward to soft
  tokens.
- The Gemma 4 decoders **generate text through a backend**: `NFKMLXGemmaLanguage.backend(directoryURL:)`
  / `@objc gemmaBackendWithDirectoryURL:error:` reads a release directory and dispatches on its config
  model type (E-series and 26B-A4B mixture, or the 12B unified). Gemma runs prefill-only (no key-value
  cache), so generation re-runs the growing sequence each step. The tokenizer is `NFKMLXGemmaTokenizer`
  (gained `decode`/`id(forToken:)`); a message list builds Gemma's turn format from special-token ids.
  Measured end to end on the released E2B: "The capital of France is" → " Paris."
- The Gemma 4 tri-modal **input adapters and fusion** are implemented. `NFKMLXGemma4ImageProcessor`
  turns a `CGImage` into the tower's flattened patches and positions (aspect-ratio-preserving resize,
  `(row, column, channel)` patches, `(x, y)` positions; the resize is a documented CoreGraphics
  approximation, the layout exact). `NFKMLXGemma4AudioFeatureExtractor` turns raw audio into the log-mel
  features (semicausal framing, Hann window, magnitude FFT, HTK mel bank, `log(mel + 1e-3)`), at
  reference parity (`run_reference.py gemma4_mel`, cosine 0.9999999999928). `NFKMLXGemma4MultimodalEmbedder`
  projects a tower's soft tokens into the decoder's space at parity (`gemma4_embedder`, cosine
  0.9999999999999927), and `NFKMLXGemma4Fusion.fuse` splices them at the placeholder positions.
  `NFKMLXGemma4ConditionalGeneration` wires the whole chain end to end — image/audio and a
  placeholder-carrying prompt in, a generated continuation out — running the processors, towers, and
  embedders, splicing the soft tokens, and generating prefill-only over the fused embeddings (the
  decoder gained an `embed` / `logits(fromEmbeddings:tokens:)` seam so the main stream is pre-spliced
  while the per-layer identity reads the placeholder-padded ids). **The full chain is at reference parity
  on the RELEASED E2B weights, end to end** (`run_reference.py gemma4_conditional_real`): an image fills
  four placeholder tokens and the fused sequence's logits match transformers' own
  `Gemma4ForConditionalGeneration` at logit cosine 0.999999999959, argmax 8/8. The E2B release IS the
  full tri-modal model (its 10 GB checkpoint carries the vision tower, audio Conformer, both embedders,
  and the decoder), so the vision and audio towers were verified on the real weights too
  (`gemma4_vision_real`, `gemma4_audio_real`, each ≥ 0.99999999999). Two bugs only the real weights
  exposed: the vision attention applies a **2-D rope** (base 100) the tiny test could not see (2e-8 at
  head_dim 8, but 0.77 on the real tower), and both towers train their `Gemma4ClippableLinear`
  projections with **finite clamp bounds** (`use_clipped_linears`, a QAT artifact) that are load-bearing.
- DeepSeek V4 arithmetic measured against transformers' own implementation at a tiny all-sliding
  configuration — every layer exact, logits 0.9999999999 — which found and fixed the head collapse
  being built as the wrong class. The release's fp8/fp4 storage decodes exactly; DeepSeek V4 Pro is
  accounted structurally.
- A complete trained text-to-speech voice: the espnet FastSpeech2 conformer acoustic model at
  reference parity on its released LJSpeech weights (durations exact, mel 0.9999999999), chained
  through its paired HiFi-GAN and the release's own phoneme vocabulary by `NFKMLXVoice`. The
  end-to-end test synthesizes "hello world" and the package's own Whisper transcribes it back.
- HiFi-GAN vocoder at reference parity on the released UNIVERSAL_V1 weights (0.9999999999); the
  converter fuses the release's weight normalization and reads the espnet-paired layout too.
- Whisper reads transformers-layout checkpoints (`model.encoder.layers.N.self_attn.q_proj` …) as
  well as OpenAI-layout ones.
- Whisper timestamped decoding — `transcribeWithTimestamps`, and `emitsTimestamps` on the backend,
  which adds `NSArray<NFKAudioSegment *>` under `NFKOutputSegments`. Off by default: it is a
  different decode, not a different reading of one.
- Additional sizes in ported families, each at measured parity: Whisper small / medium / large-v3,
  YOLOv8 l / x, SwinIR ×8 and the lightweight ×2 (a different upsampler), SAM 2 base_plus and large,
  and Gemma 4 E4B (measured at the bf16 the release ships in). DeepSeek V4 Pro is verified
  structurally — its 149782-tensor index accounted for exactly, never run.
- Bring-your-own-model backends: module, matting, tensor, diffusion, speech, and video.
- Video as a produced modality: `NFKMLXVideoBackend` reads and writes `NFKVideoAsset`, with clip-level
  frame interpolation (RIFE) and super-resolution (BasicVSR) over whole sequences.
- Face detection and alignment, so `NFKMLXCodeFormer` restores a photograph rather than only a
  pre-aligned crop. Two detectors: `NFKMLXRetinaFace` (the reference pipeline's own, at measured
  parity, 1.7 MB) and `NFKMLXVisionFaceDetector` (no weights, no download).
- On-device fine-tuning: `NFKMLXTrainer`, LoRA, training data helpers, and recipes for Zero-DCE,
  SegFormer, CLIP probes, and Whisper.
- `NFKMLXModelRegistry`, `NFKMLXHub`, and direct `@objc` factories, so an Objective-C consumer builds
  and runs a model without writing Swift.
- `NFKMLXRandom`, `NFKMLXGPU`, and `NFKMLXDevice` expose MLX's global runtime knobs to Objective-C.
- Every model's weight loader reads through `NFKMLXWeights.loadCheckpoint`, so a checkpoint written by
  fine-tuning reloads through the model's existing factory without being transposed twice.

### InferKitFoundationModels (companion)

- `NFKFoundationModelsBackend` over Apple's on-device system language model, with streaming, multi-turn
  transcripts, runtime-schema tool calling, and structured output.

### Distribution

- Source through Swift Package Manager and CocoaPods, from the same files.
- Prebuilt XCFrameworks for the MLX companion in static and dynamic variants, arm64, with macOS, iOS
  device, and iOS simulator slices. The static variant carries the Metal library each slice's consumer
  ships. Linking recipes per consumer shape are in `Docs/installation.md`.
