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
  lowest score, and BPE by merging the highest-scoring adjacent pair (leftmost on ties). Normalization
  is `NFKMLXSentencePieceNormalizer`, which runs the model's own precompiled character map (the Darts
  trie in `normalizer_spec`, longest prefix match, user-defined pieces passed through) and then the
  spec's whitespace rules (`remove_extra_whitespaces`, `add_dummy_prefix`, `escape_whitespaces`);
  `identity` carries no map and leaves text alone. The map is per model: M2M-100's turns a zero-width
  joiner into a space, the OPUS-MT maps keep it; every map composes NFD sequences and folds
  compatibility forms, and none reorders combining marks, so Foundation NFKC cannot stand in for it
  (`nmtNFKC` remains only as the fallback for a hand-built proto). Unknown runs fuse into one unknown
  piece, or into `<0xHH>` byte pieces under byte fallback. A release that numbers its vocabulary apart
  from the model file (Marian's and M2M-100's `vocab.json`) maps pieces through the tokenizer's table.
  **Trap:** a Swift `String` key compares by canonical equivalence, and SentencePiece compares bytes.
  Vocabularies carry both spellings of a piece as distinct entries (fatha+shadda and shadda+fatha, a
  precomposed and a decomposed nukta letter, NFC and NFD Latin), so every piece table here is keyed by
  the exact scalar sequence (`NFKMLXSentencePieceSegmenter.key`) and `vocab.json` is read through
  `NSDictionary` into pairs, never `as? [String: Int]`, which also hid 6 of M2M-100's 128004 entries.
  The language probes (ar, hi, and the NFD vi/ko sentences) found this; the Gemma tokenizer in
  `NFKMLXEmbeddingGemma.swift` had the same defect in its merge table and takes the same keys.
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
  0.30730444. The 1.2B (`facebook/m2m100_1.2B`, `pytorch_model.bin`, 5 GB) at reference parity on the
  same probes: tokenizations id-exact, encoder cosine 0.9999998, logit cosine 1.0 with argmax 18/18,
  greedy and beam outputs token-exact, loss 0.373035 against 0.37298885. SMaLL-100
  (`alirezamsh/small100`, the release's own `SMALL100Tokenizer`: target marker on the source, decoder
  started plain) at reference parity: tokenizations id-exact, encoder cosine 1.0000001, logit cosine
  0.99999976 with argmax 16/16, greedy and beam token-exact, loss 0.7012323 against 0.70121324.
- `NFKMLXNLLB` (`@objc`, `NFKMLXNLLBVariant` `.distilled600M` / `.m1_3B` / `.distilled1_3B` / `.m3_3B`) /
  `NFKMLXNLLBTranslator` — NLLB-200 (`M2M100ForConditionalGeneration` + `NllbTokenizer`, Meta,
  CC-BY-NC-4.0): the M2M-100 network (`model_type` `m2m_100`, 12 + 12 at 1024 for the 600M, 24 + 24 for
  the 1.3Bs, 24 + 24 at 2048 for the 3.3B) over a 256k SentencePiece BPE vocabulary that the release
  numbers as fairseq did: `<s>` `<pad>` `</s>` `<unk>` at 0 to 3, every model piece at its id plus one,
  the 202 `xxx_Xxxx` codes after the vocabulary in the order `special_tokens_map.json` lists them
  (`eng_Latn` 256047), `<mask>` last; `vocab_size` 256206 leaves two rows unused. The table is built
  from the model file and that list (`NFKMLXNLLB.releaseTable`), no `vocab.json`. The source leads
  with its code and ends with `</s>`; the decoder starts from `</s>` with the target code forced; the
  decode drops markers and the four specials and trims, as `skip_special_tokens` does. A BCP-47 tag
  maps through `NFKMLXNLLBTranslator.individualLanguage` (NLLB's picks for macrolanguages: `ar` →
  `arb`, `fa` → `pes`, `no` → `nob`, `ms` → `zsm`, `sw` → `swh`, …) then ISO 639-3, with the tag's
  script, the implied script (`zh` → `zho_Hans`, `zh-TW` → `zho_Hant`), or the release's one script
  for that language; a tag that names a script the release lacks (`sr-Latn`) is unsupported, as is a
  language it writes in two scripts when the tag names neither (`ace`). The releases name no beam
  count and transformers' default is 1, so the default decode is greedy; `NFKMLXTranslationParameterKey.beamCount`
  turns on a beam search, and the records hold a 5-beam one beside the greedy decode. Registered as
  `nllb-200` (the distilled 600M). At reference parity on `facebook/nllb-200-distilled-600M`: every probe tokenization id-exact, encoder cosine 1.0000002, teacher-forced logit cosine 0.99999994 with argmax 18/18, greedy and 5-beam outputs token-exact and text-exact ("Der schnelle braune Fuchs springt über den faulen Hund."), training loss 0.36211884 against 0.361941. The distilled 1.3B (`facebook/nllb-200-distilled-1.3B`, 5.5 GB, 24 + 24 at 1024) at reference parity: encoder cosine 1.0000001, logit cosine 0.9999999 with argmax 18/18, generations and text exact, loss 0.33422005 against 0.3340959. The 1.3B (`facebook/nllb-200-1.3B`) at reference parity: encoder cosine 0.9999999, logit cosine 1.0000001 with argmax 18/18, generations and text exact, loss 0.35767236 against 0.35754162. The 3.3B (`facebook/nllb-200-3.3B`, 17.6 GB of float32 in three `pytorch_model-*.bin` shards under `pytorch_model.bin.index.json`, which the seq2seq loader reads) at reference parity: encoder cosine 1.0000001, logit cosine 1.0 with argmax 18/18, generations and text exact, loss 0.32635206 against 0.32632384.
- `NFKMLXMADLAD` (`@objc`) / `NFKMLXMADLADTranslator` — MADLAD-400 3B-MT (`T5ForConditionalGeneration`,
  Google, Apache-2.0): 32 + 32 T5 layers at 1024 with 16 heads of 128 and an 8192-wide gated-GELU FFN,
  a 256k unigram vocabulary, and 493 `<2xx>` target markers as user-defined pieces. Tokenization follows
  the release's fast tokenizer rather than the raw SentencePiece model: runs of two or more spaces
  collapse to one, the whole `<2xx> text` string is prefixed with `▁` and segmented with the markers as
  pieces (so `▁` and `<2de>` come out as two ids), no byte fallback, `</s>` closes. A BCP-47 tag maps to
  `<2xx>` by primary subtag with its script where the release distinguishes one (`zh-Hant` →
  `<2zh_Hant>`). The 3B release is 11.8 GB of float32; `half` loads it as bfloat16 (the registry does),
  and float32 is what the parity is measured at. Registered as `madlad400-3b-mt`. The 7B (`google/madlad400-7b-mt`, 33 GB of
  float32 in 7 shards, `NFKMLXMADLADConfiguration.mt7B`: 48 + 48 at 2048) is measured bfloat16 against
  bfloat16 (`MADLAD_DTYPE=bfloat16`; the port's `half` converts shards in groups as they are read, about
  15 GB resident, 18.4 GB peak): tokenizations id-exact, encoder cosine 0.9999317, logit cosine 0.9999195 with argmax 17/17, greedy and beam outputs token-exact ("Der schnelle braune Fuchs springt über den faulen Hund."), loss 0.1675079 against 0.16796875; its T5 blocks and cached attention round as transformers' do at bfloat16 (`mlx-models-dit-generation.md`). At reference parity on
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
  cross-entropy over a dozen positions. Those 12B figures were measured with the model held, which
  swapped; the test now streams the decoder (`.streamed`, 16 of 48 layers held) through the same path and
  reads greedy token-exact, logit cosine 0.9990946 with argmax 16/16, and worst layer state 0.9996596 at
  layer 48 (285 s, 136.7 GB read). A float32 reference for the 12B needs 48 GB and is not producible here.
  The 27B (`google/translategemma-27b-it`, 55 GB of bfloat16, 62 layers at 5376) fits neither side whole:
  the reference reads each decoder layer as its forward reaches it (`run_reference.py
  translategemma_layerwise`, 1687 s, 11.95 GB peak) and the port streams the decoder at `.checkpoint`
  (15 of 62 layers held). bfloat16 against bfloat16: template ids exact; the greedy translation
  token-exact ("Der flinke braune Fuchs springt über den faulen Hund."); logits at the last 16 prompt
  positions cosine 0.9991661 with argmax 16/16; worst layer state 0.9996508 at layer 62; the masked
  SFT loss 0.35301155 against 0.33496428 (855 and 1121 s across two Commit Checks, the store read at
  0.69 GB/s). The 4B drafting for either size, held at 4 bits, keeps the output and cuts the passes:
  the 12B produced its 15 tokens in 6 passes, keeping 14 of 20 proposals, in 93 s; the 27B its 14
  tokens in 6 passes, keeping 13 of 20, in 183 s, both runs in the 18 GB footprint bucket.

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
`TRANSLATEGEMMA_DTYPE=bfloat16`) `IK_VAL_TRANSLATEGEMMA_12B` / `IK_PARITY_TRANSLATEGEMMA_12B`.

Every translation runner takes its pair from `TRANSLATION_SOURCE_LANG` / `TRANSLATION_TARGET_LANG`
(default `en` / `de`); `run_marian` reads the pair from the release's `tokenizer_config.json` instead
and needs `MARIAN_TARGET_CODE` (the code inside `>>…<<`) on a group release. The probe sentences are
`TRANSLATION_PROBES`, one list per source language (en, ja, zh, ar, vi, th, hi, ko), and the loss
sentence is `TRANSLATION_TARGETS[target]`; the Swift tests hold the same table. A record also carries
the reference's decoded greedy and beam text as UTF-8 (`greedy_text`, `beam_text`), which the tests hold
byte for byte against `translate`. Further runners: `nllb` (NllbTokenizer with `NLLB_CODES` mapping the probe tags to
`xxx_Xxxx`), `small100` (imports the release's own
`tokenization_small100.py`), `madlad` with `MADLAD_DTYPE=bfloat16` for the 7B, and
`translategemma_layerwise`, which records what `run_translategemma` records for a release too large to
hold whole: the model is built on the meta device and each decoder layer's tensors are read from the
release (`F_NOCACHE`, in reads under 1 GiB, since macOS refuses longer ones) as the forward reaches the
layer and freed after it, so `generate` and the `labels=` loss run through transformers' own entry points.
On the 4B at bfloat16 its record equals `run_translategemma`'s bit for bit (42 keys). Probe records are keyed `IK_PARITY_<FAMILY>_<SRC>_<TGT>` (`IK_PARITY_MARIAN_JA_EN`,
`IK_PARITY_MADLAD_EN_ZH_HANT`), and a probe test skips the pairs whose record is absent. The 12B
figures above were measured with the whole model resident, which swapped on this 32 GB machine; the
12B and 27B tests now stream the decoder (`residency: .streamed`) and run the logits, every layer's state,
the greedy continuation, and the loss through the same code as the 4B. A streamed decoder decodes the
greedy ids for the printed translation instead of generating twice.

## Language probes

Every parity probe above is English → German. The probes below hold each family on the languages
that exercise a tokenizer differently: Japanese (full-width forms, the ideographic space, half-width
katakana, no spaces), Chinese in both scripts (marker resolution, CJK punctuation), Arabic (right to
left, diacritics, presentation forms), Vietnamese and Korean in NFC and in NFD (canonical
equivalence), Thai (no spaces, no sentence punctuation), Hindi (combining vowel signs, a zero-width
joiner, the danda), and an emoji line for byte fallback. The sentences are `TRANSLATION_PROBES` in
`run_reference.py` and `probes` in the test file; a record holds the ids of every probe sentence, the
seams of sentence 1, the greedy and beam ids and their decoded text, and the loss on
`TRANSLATION_TARGETS[target]`. Measured 2026-10-02 (records `IK_PARITY_<FAMILY>_<SRC>_<TGT>`):

- OPUS-MT, nine releases (ja, zh, ar, vi, th, hi, ko → en; en → zh as `cmn_Hans` and `cmn_Hant`):
  every probe tokenization id-exact, encoder and logit cosines 0.9999998 to 1.0000002, greedy and
  beam ids and decoded text exact, losses within 1e-2.
- M2M-100 418M, nine pairs (the seven into English, en → zh, en → ja): tokenizations id-exact, encoder
  and logit cosines 0.9999999 to 1.0000004, generations and text exact.
- MADLAD-400 3B, ten pairs (zh-Hant added): tokenizations id-exact, encoder cosines 0.99999976 to
  1.0000004, logit cosines 0.99999976 to 1.0000002, generations and text exact.
- NLLB-200 600M, ten pairs (zh-Hant as `zho_Hant`): tokenizations id-exact, encoder cosines 0.99999994 to 1.0000002, logit cosines 0.99999964 to 1.0000004, greedy and 5-beam ids and decoded text exact, losses within 1e-2; `zh-Hant` decodes to Traditional characters under `zho_Hant`.
- TranslateGemma 4B, ten pairs: template ids exact, last-16 logit cosines 0.99999946 to 1.0000008,
  greedy continuation and text exact, losses within 1e-2. The 4B writes Simplified characters for
  `zh-Hant` (the template names both scripts "Chinese"), which the port reproduces.

What the probes found, all before the model: Swift `String` keys merging canonically equivalent
pieces (the Arabic marks, the Devanagari nukta, the NFD sentences; the Gemma tokenizer too), the
hand-written `nmt_nfkc` table differing from the models' own maps (the zero-width joiner), the
whitespace flag SentencePiece resets per piece when `remove_extra_whitespaces` is off (InternLM2),
and OPUS-MT's union `vocab.json` naming characters the source model lacks. Each is fixed and pinned by
a weight-free test; the model math matched wherever the ids matched.

## Measuring the sizes

How the four by-configuration releases were measured (2026-10-02); the assets came through the IO
Coordinator and every run through the Testing Coordinator.

- `facebook/m2m100_1.2B` (24 + 24 at 1024, ~5 GB float32): `run_reference.py m2m100 --checkpoint` on
  its directory unchanged, under `IK_VAL_M2M100_1_2B` / `IK_PARITY_M2M100_1_2B`; the port at float32.
  Fits alongside its reference.
- `alirezamsh/small100` (12 + 3, ~1.3 GB): the release's tokenizer is its own `tokenization_small100.py`
  (`SMALL100Tokenizer`, target marker on the source, `tgt_lang` only), so the oracle needs a `small100`
  runner that imports that file from the directory; the record otherwise matches `run_m2m100`. Keys
  `IK_VAL_SMALL100` / `IK_PARITY_SMALL100`.
- `google/madlad400-7b-mt` (48 + 48 at 2048, about 33 GB float32): float32 does not fit beside anything
  on 32 GB, so the reference records at bfloat16 (`MADLAD_DTYPE=bfloat16`, added to `run_madlad`) and the
  port loads `half`, held to bf16 tolerances as the TranslateGemma 12B is. Each side runs alone at
  about 17 GB. Keys `IK_VAL_MADLAD_7B` / `IK_PARITY_MADLAD_7B`. `madlad400-7b-mt-bt` is the same
  geometry and is not separately measured.
- `google/translategemma-27b-it` (62 layers at 5376, about 54 GB bfloat16, gated): neither side fits
  whole, and the model is dense. The port streams the decoder layers it cannot hold (`.streamed`), and
  the reference runs `translategemma_layerwise`, the whole-model record built one layer at a time.
  Keys `IK_VAL_TRANSLATEGEMMA_27B` / `IK_PARITY_TRANSLATEGEMMA_27B` (the earlier first-14-layers record
  is kept as `.v1`).

## Not yet ported

The toolkit is license agnostic: a weight license is the consumer's to read and comply with, and it does
not block a port (user decision, 2026-10-02). Each entry below names the port cost, not a bar.

- SeamlessM4T v2 (`facebook/seamless-m4t-v2-large`, CC-BY-NC-4.0): its text-to-text path is an
  NLLB-style 24 + 24 encoder-decoder over the same 256k vocabulary; the speech paths add the shipped
  W2V-BERT encoder, a UnitY text-to-unit decoder, and a HiFi-GAN unit vocoder. Text-to-text is a
  medium port; the whole model is a large one.
- TowerInstruct 7B / 13B (CC-BY-NC-4.0, Llama-2 decoders) and Tower+ 9B (Gemma 2 decoder): the
  decoder is shipped, so each is a chat template over `NFKMLXLanguageNet`, the TranslateGemma pattern.
- Hunyuan-MT 7B (Tencent Hunyuan license, Hunyuan dense decoder) and Seed-X 7B (OpenMDW, Mistral
  decoder): the same template-over-decoder pattern once the decoder's reader exists.
- TranslateGemma is gated: a machine whose token has not accepted the Gemma terms on the model page
  gets a `403` on `config.json`, which is what the download factory surfaces.
