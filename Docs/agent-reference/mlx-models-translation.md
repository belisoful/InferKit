<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# InferKitMLX: text translation

The text-to-text translators, the SentencePiece reader and the encoder-decoder runtime they share,
and the contract they answer. The contract itself lives in the core (`NFKParameterSourceLanguage`,
`NFKParameterTargetLanguage`, `NFKCapabilityTranslation`) beside Apple's own translator in
`InferKitAppleSwift`; these are the open-weight engines behind it.

## The contract

- Text in under `NFKInputPrompt`; the translation out under `NFKOutputText`.
- `NFKParameterTargetLanguage` (BCP-47) is required; a request without it fails with
  `kNFKError_InferenceMissingInput`. `NFKParameterSourceLanguage` is optional: a per-pair Marian release
  knows its source, MADLAD needs none, and M2M-100 detects it with `NLLanguageRecognizer` when absent.
- A target the model does not produce, or a source a per-pair model does not read, fails with
  `kNFKError_InferenceUnsupported` naming the model's pair.
- The decode is tuned per request through `NFKMLXTranslationParameterKey`: `beamCount`
  (`NFKMLXParameterBeamCount`), `lengthPenalty`, and `splitsSentences`. The defaults are each release's
  own generation config (Marian 4 beams with `renormalize_logits`, M2M-100 5 beams with early stopping,
  MADLAD greedy). `NFKParameterMaxTokens` caps the output.
- Input is translated paragraph by paragraph (split on line breaks, separators kept), or sentence by
  sentence under `splitsSentences` (`NLTokenizer`). Off by default so the parity path is one sequence.
- `NFKMLXTranslationProvider` registers ahead of the core's default chain for `NFKCapabilityTranslation`
  (through `registerAll`) and serves M2M-100 418M when that release is in the default hub cache; it
  declines otherwise, which lets Apple's translator answer.

## Shared runtime

- `NFKMLXSentencePieceModel` / `NFKMLXSentencePieceSegmenter` / `NFKMLXSentencePieceTokenizer` (`@objc`,
  an `NFKTokenizer` subclass) — a `.model` / `.spm` reader over a minimal protobuf walker (pieces with
  score and type, `trainer_spec.model_type` / `byte_fallback` / `unk_id`, `normalizer_spec` name and
  flags), and the two segmenters: unigram Viterbi with the reference's `kUnkPenalty` of 10 below the
  lowest score, and BPE by merging the highest-scoring adjacent pair (leftmost on ties). `nmt_nfkc` is
  NFKC (`precomposedStringWithCompatibilityMapping`) plus the NMT rules (controls dropped, every space
  class to a plain space); `identity` leaves text alone. Extra whitespace is collapsed and trimmed when
  the model asks; the dummy prefix is a leading `▁`. Unknown runs fuse into one unknown piece, or into
  `<0xHH>` byte pieces under byte fallback. A release that numbers its vocabulary apart from the model
  file (Marian's and M2M-100's `vocab.json`) maps pieces through the tokenizer's table.
  **Trap:** Swift keys a dictionary by canonical equivalence, so a `vocab.json` with NFD and NFC
  spellings of one piece parses to fewer entries than Python counts (M2M-100: 127998 for 128004). Any
  base derived from a table's count is wrong; M2M-100's marker base comes from `vocab_size − 100 − 8`.
- `NFKMLXSeq2SeqNet` (`NFKMLXSeq2SeqConfiguration`) — the BART-family encoder-decoder: Marian, M2M-100
  and NLLB, BART and mBART by flags read from `config.json` (`model_type` picks positions, pre/post norm,
  final `layer_norm`, `layernorm_embedding`, `final_logits_bias`). Positions are Marian's
  `pos / 10000^(2i/d)` table, fairseq's `10000^(-i/(half−1))` table read from `pad + 1` with the padding
  row zero, or a learned table read at `+2`. Module keys are the transformers keys with `model.`
  stripped; the loader drops the tied `embed_tokens` and `lm_head` copies and a stored sinusoid table.
  The sinusoid table is held behind a plain class so the parameter walk does not count it as a weight.
  `encode(embeddings:)` takes already-scaled embeddings for a multimodal consumer (Florence-2 enters
  here). A decoder-only configuration (`encoderLayers` 0; `model_type` `trocr`) builds no encoder and
  reads an outside memory through `decode(_:memory:cache:)`; `crossAttentionWidth` sizes the
  cross-attention's key and value projections to that memory (TrOCR's 768 under a 1024 decoder) and
  `untiedOutputProjection` reads an `lm_head` of its own. The loader also strips a `decoder.model.`
  prefix, supplies `shared` from the first `embed_tokens` when the checkpoint has none, and maps
  `output_projection` to `lm_head` for an untied head. `NFKMLXSeq2SeqCache` holds per-layer self
  keys/values and the encoder projections computed once; `reorder` carries beams forward.
- `NFKMLXT5Seq2SeqNet` (`NFKMLXMADLADConfiguration`) — T5 `T5ForConditionalGeneration` over the
  `NFKT5Stack` encoder the LTX text encoder ports, plus a decoder with a causal relative bias
  (`bidirectional=False`: only distances into the past count, every bucket serves them), an
  `EncDecAttention` sublayer, and an untied `lm_head` (a tied release projects through the embedding
  scaled by `dModel^-0.5`). The 3B release stores the embedding only as `decoder.embed_tokens.weight`;
  the loader maps the first `embed_tokens` seen to `shared` and drops the rest.
- `NFKMLXSeq2SeqDecoder` / `NFKMLXSeq2SeqDecoding` / `NFKMLXSeq2SeqDecodable` — greedy and beam search
  over any cached one-step decoder. The beam search is transformers' vectorized `_beam_search`:
  `2 × beams` candidates per step, a hypothesis ending outside the top `beams` ranks dropped, a finished
  score of `sum / (generated length including the end token)^penalty`, the bank capped at `beams`, and
  the stop rule `worst kept ≥ best running / (generated length)^penalty` (or `beams` finished under
  early stopping). A forced first token (M2M-100's target marker), suppressed ids (Marian's pad), and
  Marian's `renormalize_logits` are decode options.
- `NFKMLXTranslationBackend` (`@objc`) — the `NFKInferenceBackend` face over an `NFKMLXTranslator`.

## Models

- `NFKMLXMarian` (`@objc`) / `NFKMLXMarianTranslator` — OPUS-MT (`MarianMTModel`, Helsinki-NLP,
  CC-BY-4.0 or Apache-2.0 per pair): a 6 + 6 post-normalized transformer at 512 with swish, scaled
  embeddings, Marian sinusoids, and a `final_logits_bias`; a source and a target unigram SentencePiece
  model over a shared `vocab.json`; the decoder starts from the pad id, which is also suppressed
  (`bad_words_ids`). A group release names its target with a `>>xxx<<` token at the front of the source;
  the translator maps a BCP-47 tag to ISO 639-3 (with a script suffix when the group has one). Loads from
  a release directory (`backendWithDirectoryURL:`), by repo (`backendWithRepo:revision:cacheDirectoryURL:`),
  or by pair (`backendWithSourceLanguage:targetLanguage:cacheDirectoryURL:` → `Helsinki-NLP/opus-mt-<s>-<t>`),
  each with an asynchronous peer; registered as `opus-mt`. At reference parity on
  `Helsinki-NLP/opus-mt-en-de` (`pytorch_model.bin` through the native reader): every tokenization of
  the five probe sentences id-exact (NFKC, whitespace, `ﬁ` and `½` included), encoder cosine 0.99999994,
  teacher-forced logit cosine 0.99999994 with argmax 14/14, greedy and 4-beam outputs token-exact
  ("Der schnelle Braunfuchs springt über den faulen Hund."), and the training loss 0.2073196 against
  0.20728716.
- `NFKMLXM2M100` (`@objc`, `NFKMLXM2M100Variant` `.m418M` / `.m1_2B` / `.small100`) /
  `NFKMLXM2M100Translator` — M2M-100 (`M2M100ForConditionalGeneration`, Meta, MIT): a pre-normalized
  12 + 12 transformer at 1024 with ReLU, fairseq sinusoids, and `layer_norm` on both stacks, over a
  128k SentencePiece BPE vocabulary numbered by `vocab.json`; `__xx__` markers follow the table (100
  fairseq codes in order, then 8 filler words). The source leads with its language marker and ends with
  `</s>`; the decoder starts from `</s>` and the target marker is forced as its first token. SMaLL-100
  (`alirezamsh/small100`, MIT, 12 + 3 layers) moves the target marker onto the source and starts the
  decoder plain. BCP-47 tags map by primary subtag (`nb`/`nn` → `no`, `fil` → `tl`, `iw` → `he`).
  Registered as `m2m100` (418M) and `small100`. At reference parity on `facebook/m2m100_418M`: every
  probe tokenization id-exact, encoder cosine 1.0000004, teacher-forced logit cosine 1.0000001 with
  argmax 18/18, greedy and 5-beam outputs token-exact, and the training loss 0.30732855 against
  0.30730444. SMaLL-100 and the 1.2B share the reader and the network by configuration; neither is
  measured numerically.
- `NFKMLXMADLAD` (`@objc`) / `NFKMLXMADLADTranslator` — MADLAD-400 3B-MT (`T5ForConditionalGeneration`,
  Google, Apache-2.0): 32 + 32 T5 layers at 1024 with 16 heads of 128 and an 8192-wide gated-GELU FFN,
  a 256k unigram vocabulary, and 493 `<2xx>` target markers as user-defined pieces. Tokenization follows
  the release's fast tokenizer rather than the raw SentencePiece model: runs of two or more spaces
  collapse to one, the whole `<2xx> text` string is prefixed with `▁` and segmented with the markers as
  pieces (so `▁` and `<2de>` come out as two ids), no byte fallback, `</s>` closes. A BCP-47 tag maps to
  `<2xx>` by primary subtag with its script where the release distinguishes one (`zh-Hant` →
  `<2zh_Hant>`). The 3B release is 11.8 GB of float32; `half` loads it as bfloat16 (the registry does),
  and float32 is what the parity is measured at. Registered as `madlad400-3b-mt`; the 7B is a
  configuration (`NFKMLXMADLADConfiguration.mt7B`), not measured. At reference parity on
  `google/madlad400-3b-mt`: every probe tokenization id-exact, encoder cosine 1.0, teacher-forced logit
  cosine 1.0000001 with argmax 16/16, greedy and 4-beam outputs token-exact, and the training loss
  1.0395428 against 1.0395255.

- `NFKMLXTranslateGemma` (`@objc`) / `NFKMLXTranslateGemmaTranslator` — TranslateGemma
  (`Gemma3ForConditionalGeneration`, Google, Gemma terms; 4B, 12B, 27B, gated): Gemma 3 fine-tuned for
  translation and driven by a chat template whose one user item names the source and target languages.
  The network is the shipped `NFKMLXGemma3Model`; the template is rendered in Swift rather than through
  the Jinja renderer because the release's `chat_template.jinja` carries a 580-entry language table and
  multi-line implicit string concatenation. The table is parsed from that file (`"code": "Name"` pairs
  before the first `-%}`), so a tag the release names (`de`, `de-DE`, `zh-Hant`) resolves as written and
  any other BCP-47 tag falls back to primary-with-script, then the primary subtag. The prompt is
  `<bos><start_of_turn>user\nYou are a professional S (sc) to T (tc) translator. …\nProduce only the T
  translation … Please translate the following S text into T:\n\n\n<text | trim><end_of_turn>\n<start_of_turn>model\n`;
  decoding is greedy (`temperature 0`) to `<end_of_turn>` / `<eos>`, so the beam knobs do not apply;
  `NFKParameterMaxTokens` caps it. Loads at `.float32` (parity) or `.checkpoint` (the release's bfloat16,
  which the registry uses); the download factory pulls the shard index and its shards. Registered as
  `translategemma`. At reference parity on `google/translategemma-4b-it`: the rendered template's ids
  exact against `apply_chat_template`, logits at the last 16 prompt positions cosine 1.0000001 with
  argmax 16/16, the greedy translation token-exact ("Der schnelle braune Fuchs springt über den faulen
  Hund."), and the supervised fine-tuning loss 0.011133467 against 0.011132836. The 12B
  (`google/translategemma-12b-it`, 48 layers at 3840, 16 heads of 256 over 8 KV heads) is measured
  bfloat16 against bfloat16: the release is 24 GB and this machine holds 32 GB, so the reference runs at
  the release's own precision (`TRANSLATEGEMMA_DTYPE=bfloat16`) and the port loads at `.checkpoint`.
  Template ids exact; the greedy translation token-exact over 48 tokens ("Der schnelle, braune Fuchs
  springt über den faulen Hund."); logits at the last 16 prompt positions cosine 0.99842066 with argmax
  16/16; the last position's state after every layer against the reference's `hidden_states` drifts
  smoothly with the worst cosine 0.9994581 at the final layer, which is the two accumulation orders and
  not a seam; the masked SFT loss 0.4313063 against 0.4847855, the drift amplified by a 262k-way
  cross-entropy over a dozen positions. A float32 reference for the 12B needs 48 GB and is not
  producible here. The 27B is the same reader on its geometry, not measured.

## Customization

All three ship LoRA on the decoder's query and value projections (`decoder.layers.N.*.q_proj` /
`.v_proj`; T5 `decoder.block.N.*.q` / `.v`), the encoder frozen, through `NFKMLXMarian.fineTune`,
`NFKMLXM2M100.fineTune`, and `NFKMLXMADLAD.fineTune` over `NFKMLXTranslationObjective`, and
TranslateGemma through `NFKMLXTranslateGemma.fineTune` over `NFKMLXTranslateGemmaObjective` (Gemma 3's
`layers.N.self_attn.q_proj` / `.v_proj`; the model turn's tokens scored, the prompt positions masked, which
is the reference's `labels=-100` over the prompt; the merged decoder reloads into a `NFKMLXGemma3Net`
through `NFKMLXWeights.apply` and `translator(decoder:directoryURL:)` wraps it): teacher forcing
with the target shifted right behind the start token and a mean cross-entropy over every target
position, which is the reference's `labels=` loss with no padding. The objective is measured on the
released weights in the parity tests (the three losses above, within 3e-5). `network(directoryURL:)`
builds the trainable network; after `NFKMLXLoRA.merge(into:)` and `NFKMLXWeights.save`, Marian and
M2M-100 reload through `network(directoryURL:)` (a directory holding `model.safetensors`) and MADLAD
through `NFKMLXT5Seq2SeqNet.loadWeights(from:)`; `translator(net:directoryURL:)` wraps an adapted
network with the release's tokenizers. The round trip is tested on tiny geometries.

## Oracle and validation

`run_reference.py marian|m2m100|madlad --checkpoint <release directory>` under the `llmvenv`
environment records the probe tokenizations, the encoder output, the greedy and beam outputs, the
teacher-forced logits over the greedy output, and the `labels=` loss. MADLAD's oracle uses the fast
tokenizer (`use_fast=True`) because the slow one needs a protobuf build the environment lacks; the
fast one is also what the port mirrors. Keys: `IK_VAL_MARIAN` / `IK_PARITY_MARIAN`,
`IK_VAL_M2M100` / `IK_PARITY_M2M100`, `IK_VAL_MADLAD` / `IK_PARITY_MADLAD`. `run_reference.py translategemma`
runs under the gemma interpreter and records the template ids, the last 16 positions' logits, the greedy
continuation, the last position's state after every layer, and the masked SFT loss:
`IK_VAL_TRANSLATEGEMMA` / `IK_PARITY_TRANSLATEGEMMA`, and for the 12B (recorded with
`TRANSLATEGEMMA_DTYPE=bfloat16`) `IK_VAL_TRANSLATEGEMMA_12B` / `IK_PARITY_TRANSLATEGEMMA_12B`. The 12B
test compares at `.checkpoint` with bfloat16 tolerances (logit cosine > 0.995, per-layer state > 0.99,
loss within 0.1); the greedy continuation is still asserted exact.

## Not ported, and why

- NLLB-200 (`facebook/nllb-200-*`) is the M2M-100 architecture and would load through
  `NFKMLXSeq2SeqNet` unchanged, but its weights are CC-BY-NC-4.0.
- SeamlessM4T is CC-BY-NC; TowerInstruct is CC-BY-NC; Hunyuan-MT and Seed-X carry their own licenses.
  TranslateGemma is gated: a machine whose token has not accepted the Gemma terms on the model page gets
  a `403` on `config.json`, which is what the download factory surfaces.
