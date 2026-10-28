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
  MADLAD, NLLB, Seed-X, TranslateGemma, and HY-MT greedy). `NFKParameterMaxTokens` caps the output.
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
  `max(2, 1 + end tokens) × beams` candidates per step, the log-softmax taken over float32 logits, a
  hypothesis ending outside the top `beams` ranks dropped, a finished score of
  `sum / (generated length including the end token)^penalty`, every top-`beams` candidate finished at the
  step the limit allows last whether or not it ends, the bank capped at `beams`, and the stop rule
  `worst kept ≥ best running / (generated length)^penalty` (or `beams` finished under early stopping). A
  forced first token (M2M-100's target marker), suppressed ids (Marian's pad), Marian's
  `renormalize_logits`, and end ids beyond `endToken` (`additionalEndTokens`) are decode options. One
  core, `beamSearch(prefix:logits:decoding:next:)`, takes the first step's logits and a closure that keeps
  the chosen cache rows and feeds the next tokens, so a decoder-only model drives it too:
  `NFKMLXLanguageNet.beamSearch(prompt:decoding:)` and `NFKMLXGemma3Model.beamSearch(tokens:decoding:)`
  run the prompt once at batch 1, copy its cache rows to every beam (`NFKMLXKeyValueCache.reorder`,
  `NFKMLXGemma3Cache.reorder`), and step the beams as one batch. The prompt is the prefix, which the
  n-gram ban reads as transformers' processors read the prompt, and which the length a score is divided
  by leaves out.
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
  At its released float32 the 7B fits this machine on neither side. The reference reads each block as
  its forward reaches it (`run_reference.py madlad_layerwise`) and the port streams the blocks
  `.automatic` cannot hold (`mlx-companion.md`, "Residency"). Float32 against float32: tokenizations
  id-exact, encoder cosine 1.0000001, logit cosine 1.0 with argmax 17/17, greedy and 4-beam outputs
  token-exact ("Der schnelle braune Fuchs springt über den faulen Hund."), loss 0.1659758 against
  0.16605085. On this 32 GB machine the port streamed the 47 encoder blocks past block 0 and 10 of the
  48 decoder blocks: 63.4 GB of encoder reads at 1.76 GB/s and 181.5 GB of decoder reads at 2.32 GB/s
  over the test, 332 s with an 18.0 GB peak footprint.

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
  decoding is greedy (`temperature 0`) to `<end_of_turn>` / `<eos>` by default, and `beamCount` above 1
  runs `NFKMLXGemma3Model.beamSearch(tokens:decoding:)` with both as end tokens, so each step keeps three
  candidates per beam as the release's two-id generation config makes transformers keep;
  `NFKParameterMaxTokens` caps it. Loads at `.float32` (parity) or `.checkpoint` (the release's bfloat16,
  which the registry uses); the download factory pulls the shard index and its shards. Registered as
  `translategemma`. At reference parity on `google/translategemma-4b-it`: the rendered template's ids
  exact against `apply_chat_template`, logits at the last 16 prompt positions cosine 1.0000001 with
  argmax 16/16, the greedy translation token-exact ("Der schnelle braune Fuchs springt über den faulen
  Hund."), and the supervised fine-tuning loss 0.011133467 against 0.011132836. Its four-beam search
  against transformers' `generate(num_beams=4)` at float32: the best hypothesis token-exact and
  text-exact on the five probe sentences and on the search cut at six tokens, and the scores within
  6.8e-6 of the reference's.
  The 12B (`google/translategemma-12b-it`, 48 layers at 3840, 16 heads of 256 over 8 KV heads) is measured
  bfloat16 against bfloat16: the release is 24 GB and this machine holds 32 GB, so the reference runs at
  the release's own precision (`TRANSLATEGEMMA_DTYPE=bfloat16`) and the port loads at `.checkpoint`.
  Template ids exact; the greedy translation token-exact over 48 tokens ("Der schnelle, braune Fuchs
  springt über den faulen Hund."); logits at the last 16 prompt positions cosine 0.99842066 with argmax
  16/16; the last position's state after every layer against the reference's `hidden_states` drifts
  smoothly with the worst cosine 0.9994581 at the final layer, which is the two accumulation orders and
  not a seam; the masked SFT loss 0.4313063 against 0.4847855, the drift amplified by a 262k-way
  cross-entropy over a dozen positions. Those 12B figures were measured with the model held, which
  swapped; the test now streams the decoder (`.streamed`, 16 of 48 layers held) through the same path and
  reads the greedy continuation token-exact under teacher forcing, logit cosine 0.9990946 with argmax 16/16,
  and worst layer state 0.9996596 at layer 48 (114 s, 22.0 GB read). A float32 reference for the 12B needs
  48 GB and is not producible here.
  The 27B (`google/translategemma-27b-it`, 55 GB of bfloat16, 62 layers at 5376) fits neither side whole:
  the reference reads each decoder layer as its forward reaches it (`run_reference.py
  translategemma_layerwise`, 1687 s, 11.95 GB peak) and the port streams the decoder at `.checkpoint`
  (15 of 62 layers held). bfloat16 against bfloat16: template ids exact; the greedy translation
  token-exact under teacher forcing ("Der flinke braune Fuchs springt über den faulen Hund."); logits at
  the last 16 prompt positions cosine 0.9991661 with argmax 16/16; worst layer state 0.9996508 at layer
  62; the masked SFT loss 0.35301155 against 0.33496428 (173 s, 116.4 GB read). The 4B drafting for either
  size, held at 4 bits, keeps the output and cuts the passes:
  the 12B produced its 15 tokens in 6 passes, keeping 14 of 20 proposals, in 93 s; the 27B its 14
  tokens in 6 passes, keeping 13 of 20, in 183 s, both runs in the 18 GB footprint bucket.

- `NFKMLXSeedX` (`@objc`) / `NFKMLXSeedXTranslator` — Seed-X (ByteDance Seed, OpenMDW; PPO 7B and
  Instruct 7B, `NFKMLXSeedXVariant`): a `MistralForCausalLM` (32 layers at 4096, 8 key-value heads, a
  4096-position sliding window, a 65,269-piece vocabulary) trained to translate among 28 languages. It
  loads through the language family (`NFKMLXLanguage.loadedReleaseWithTokenizer`) and generates with
  `NFKMLXLanguageNet.generate(prompt:options:)`, greedy by default as the release's generation config
  leaves it, stopping at `</s>`; `beamCount` above 1 runs `NFKMLXLanguageNet.beamSearch(prompt:decoding:)`
  instead. The prompt is the model
  card's, with no chat template: `<s>Translate the following <Source> sentence into <Target>:\n<text>
  <tag>`, the names as the card's table spells them (`Norwegian Bokmal`) and the tag plain text, not an
  added token. A BCP-47 tag resolves by its primary subtag. The card's example searches four beams. The
  tokenizer is the Llama-style form of the Mistral BPE: a
  `Prepend("▁")` and `Replace(" ", "▁")` normalizer with no pre-tokenizer, so every run of text between
  added tokens takes a `▁`, even one that opens with a space (`mlx-models-language.md`). Registered as
  `seed-x`, loaded at the release's bfloat16 (15 GB); a `.float32` load is 30 GB and streams where it
  does not fit. It takes the language family's residencies, the precision ladder, and runtime weight
  quantization through the catalog's load options; the translation backend returns each paragraph whole
  and reads no cache parameter, and no smaller release shares the 65,269-piece vocabulary to draft for
  it. The release opens its continuation with `<s>`, which the translation's text leaves out, as the
  card's `skip_special_tokens` decode does. Seed-X-PPO-7B, bfloat16 against bfloat16: prompt ids exact
  on the five probe sentences, logit cosine 0.99993867 with argmax 24/25 over the prompt positions,
  worst last-position layer state 0.9999275 at layer 31, the greedy translation token-exact ("Der
  schnelle braune Fuchs springt über den faulen Hund."), and the fine-tuning loss 1.3389877 against
  1.3389881. Its four-beam search against transformers' `generate(num_beams=4)` at bfloat16: the best
  hypothesis token-exact and text-exact on the five probe sentences and on the search cut at six
  tokens, and each score within 0.0108962 of the float32 reference's teacher-forced score of the same
  hypothesis, where the reference's own bfloat16 scores sit up to 0.0183652 from it.
  At float32 against a reference that reads each layer as its forward reaches it: prompt ids
  exact, logit cosine 0.9999999 with argmax 25/25, worst layer state 0.99999976 at layer 31, the greedy
  translation token-exact, and the loss 1.3519824 against 1.3519785; 16 of 32 layers streamed, 167.9 GB
  read at 1.94 GB/s, an MLX peak of 17.7 GB. Seed-X-Instruct-7B, bfloat16 against bfloat16: prompt ids
  exact, logit cosine 0.9997266 with argmax 23/25, worst layer state 0.9998638 at layer 31, the greedy
  translation token-exact ("Der schnelle braune Fuchs springt über den faulen Hund."), and the loss
  0.84905034 against 0.8545502.

- `NFKMLXHYMT` (`@objc`) / `NFKMLXHYMTTranslator` — HY-MT (Tencent; HY-MT1.5 under the Tencent HY Community
  License, which does not cover the European Union, the United Kingdom, or South Korea, and Hy-MT2 under
  Apache-2.0; 1.8B and 7B each, `NFKMLXHYMTVariant`, the generation `NFKMLXHYMTRelease`):
  `HunYuanDenseV1ForCausalLM` decoders (`hunyuan_v1_dense`; the 1.8B 32 layers at 2048 with 4 key-value heads
  and a 120,818-piece vocabulary, the 7B 32 layers at 4096 with 8 key-value heads and 128,167 pieces, both
  with tied embeddings) trained to translate among 38 languages. The two generations share their geometry,
  tokenizers, and chat templates byte for byte and differ in the instruction their cards prompt with. It
  loads through the language family, which reads Hunyuan's per-head query and key norms after the rotary
  embedding and its `alpha`-raised rotary base (`mlx-models-language.md`), and generates greedily, where the
  release's generation config samples (temperature 0.7, top-k 20, top-p 0.8, repetition penalty 1.05);
  `beamCount` above 1 runs `NFKMLXLanguageNet.beamSearch(prompt:decoding:)`. The prompt is the card's default
  instruction, which names the target alone: the Chinese one with the target's Chinese name when either
  side is Chinese (`zh`, `zh-Hant`, `yue`), the English one with its English name otherwise, so an undetected
  source takes the English instruction. HY-MT1.5's template puts one line break before the text and Hy-MT2's
  two. The instruction goes inside the release's chat template, rendered in Swift: the 1.8B form
  `<｜hy_begin▁of▁sentence｜><｜hy_User｜>…<｜hy_Assistant｜>` and the 7B form `<|startoftext|>…<|extra_0|>`.
  HY-MT1.5's card renders the 1.8B prompt with `add_generation_prompt=False`, which closes the user turn
  with `<｜hy_place▁holder▁no▁8｜>`; the template's fine-tuning layout opens the assistant turn instead, so
  the translator does too (`opensAssistantTurn`), and the two forms translate the probe alike. Decoding
  stops at the end tokens `generation_config.json` names (the 7B names two). A BCP-47 tag resolves by its
  primary subtag, Chinese by its script (`zh-TW` → `zh-Hant`), and `fil` reads as `tl`. The 1.8B tokenizer
  splits in three stages, digits in runs of at most three, then CJK and kana runs, then a word pattern
  (the core's `hunyuan` pre-tokenization); the 7B's is Qwen2's split with `ignore_merges`. Registered as
  `hy-mt1.5` and `hy-mt2`, loaded at the release's bfloat16 (4.1 GB and 16.1 GB); a `.float32` 7B is 32 GB
  and streams. It takes the language family's residencies, the precision ladder, and runtime weight
  quantization; the translation backend reads no cache parameter, and the 1.8B cannot draft for the 7B,
  whose vocabulary differs. Every figure below is English to German against transformers 4.57.6 unless
  noted, with the prompt ids exact on the five probe sentences and the greedy translation token-exact.
  HY-MT1.5-1.8B at float32: logit cosine 0.99999994 with argmax 24/24, worst last-position layer state
  0.9999997 at layer 3, and the fine-tuning loss 0.073117785 against 0.07311491 ("Der schnelle braune Fuchs
  springt über den trägen Hund."). Chinese to English at float32, the Chinese instruction and the CJK
  split: 0.99999964 with argmax 29/29, worst layer 0.9999996 at layer 24, loss 0.86784077 against
  0.8678408 ("The agile brown fox jumped over the lazy dog."). The closed user turn at float32: argmax
  24/24, worst layer 0.99999964 at layer 4, loss 0.07896699 against 0.07896289, the same translation.
  At bfloat16 against bfloat16 (the reference's default SDPA attention): HY-MT1.5-1.8B logit cosine
  0.9993807 with argmax 23/24, worst layer 0.9998556 at layer 18, loss 0.07741504 against 0.07265944;
  Hy-MT2-1.8B 0.9993057 with argmax 33/34, worst layer 0.99989164 at layer 19, loss 0.016963959 against
  0.015993766 ("Der schnelle braune Fuchs springt über den faulen Hund."); HY-MT1.5-7B 0.99981064 with
  argmax 23/23, worst layer 0.9999002 at layer 31, loss 0.310143 against 0.31012538 ("Die schnelle braune
  Füchsin springt über den faulen Hund."); Hy-MT2-7B argmax 33/33, worst layer 0.999898 at layer 31, loss
  0.0030132532 against 0.0028200727. A bfloat16 stream against another diverges past its first flipped
  element: the reference's eager and SDPA runs sit 2.9e-4 apart in `1 - cosine` at the logits, more than
  either sits from float32 (1.8e-4), so the bfloat16 rounding is held by `NFKMLXBFloat16ParityTests`
  against eager attention instead (`model-parity.md`, "Half precision against the half-precision
  reference"): every block alone at most 0.024 of the accumulated floor and 0.142 of its own block floor, the logits 1.07 to 1.34 times the reference's
  own distance from float32, and each piece of layers 0, 16, and 31 of the 1.8B, the rotary embedding and
  the norms after it included, within 2.2e-9 of the reference on its own input. HY-MT1.5-7B at float32
  against a reference that reads each layer as its forward reaches it: logit cosine 1.0 with argmax 23/23,
  worst layer 0.9999996 at layer 25, loss 0.30692232 against 0.30692142; 16 of 32 layers streamed, 146.6 GB
  read at 3.84 GB/s, an MLX peak of 17.5 GB. Hy-MT2-1.8B's four-beam search against transformers'
  `generate(num_beams=4)` at float32: the best hypothesis token-exact and text-exact on the five probe
  sentences and on the search cut at six tokens, every score within 5.3e-5 of the reference's. Both
  tokenizers encode 14 cases as the tokenizers library does, among them CJK and kana runs beside Latin
  words, digit runs, full-width forms, and punctuation leading Latin and non-Latin words.

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
network with the release's tokenizers. The round trip is tested on tiny geometries. Seed-X adapts
through `NFKMLXSeedX.fineTune`, the language family's `NFKMLXLanguage.fineTune` over
`NFKMLXCausalLanguageObjective` (`self_attn.q_proj` / `.v_proj`): the example is the prompt's ids and the
target's, its pieces and `</s>`, as the release's `RM_demo.py` concatenates a pair, and only the target
is scored, which is the reference's `labels=-100` over the prompt. The merged decoder reloads through
`NFKMLXLanguage.network(weightsURL:configuration:)`, and `translator(decoder:directoryURL:)` puts the
release's tokenizer back. HY-MT adapts the same way through `NFKMLXHYMT.fineTune`: the example is the
prompt's ids, the assistant turn opened, and the target's pieces closed by the release's first end token,
as the chat template closes an assistant turn; `translator(decoder:directoryURL:release:)` puts the
release's tokenizer and template back.

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

`run_reference.py seedx --checkpoint <release directory>` (llmvenv) records the Seed-X seams: the
card's prompt ids for every probe sentence, the logits at every prompt position, the last position's
state after every layer but the last, the greedy continuation and its text, and the loss on
`TRANSLATION_TARGETS[target]` with the prompt masked. `SEEDX_DTYPE` picks the precision (bfloat16 by
default); `SEEDX_LAYERWISE=1` builds `MistralForCausalLM` on the meta device and reads each decoder layer
as the forward reaches it (`_layerwise_release`, which reads a single-file release too), for float32.
Keys `IK_VAL_SEEDX_PPO` / `IK_PARITY_SEEDX_PPO` / `IK_PARITY_SEEDX_PPO_F32` and `IK_VAL_SEEDX_INSTRUCT` /
`IK_PARITY_SEEDX_INSTRUCT`.

`run_reference.py hymt --checkpoint <release directory>` (llmvenv) records the HY-MT seams as `seedx`
records Seed-X's, the prompt rendered through the release's own `apply_chat_template` from an instruction
the oracle writes from the card, and the greedy decode run with the release's sampling and repetition
penalty off (transformers applies `repetition_penalty` under `do_sample=False`). `HYMT_RELEASE` picks the
generation's instruction (`1.5` by default, or `2`), `HYMT_ASSISTANT=0` closes the 1.8B user turn without
opening the assistant turn, `HYMT_DTYPE` picks the precision (bfloat16 by default), and `HYMT_LAYERWISE=1`
reads each layer as the forward reaches it, the head taking the embedding it ties to. `hymt_beam` records
the four-beam search through `_decoder_beam_record`, which now passes `generate` overrides through, here
`repetition_penalty=1.0`, and `hymt_tokenizer` records the tokenizers library's ids for
`HYMT_TOKENIZER_CASES`. Keys `IK_VAL_HYMT15_1_8B`, `IK_VAL_HYMT15_7B`, `IK_VAL_HYMT2_1_8B`, `IK_VAL_HYMT2_7B`;
records `IK_PARITY_HYMT15_1_8B` (and `_F32`, `_ZH`, `_CLOSED`), `IK_PARITY_HYMT2_1_8B` (and `_BEAM`),
`IK_PARITY_HYMT15_7B` (and `_F32`), `IK_PARITY_HYMT2_7B`, and `IK_PARITY_HYMT_TOKENIZER_1_8B` / `_7B`. The
bfloat16 harness reads `hf_layer_probe` records under `records/` (`hymt15_1_8b`, `hymt2_1_8b`, and
`hymt15_7b`, `_bf16_probe` and `_f32_probe`), made with the release's translation prompt in
`IK_PROBE_PROMPT_FILE`. transformers 4.57.6 warns of "an incorrect regex pattern" when it loads Hy-MT2's
tokenizer: the warning follows the `transformers_version` a release's `config.json` names (4.57.6 lies in
the window it checks) and leaves the tokenizer unchanged, and the two generations' ids are identical.

`run_reference.py seedx_beam` (llmvenv, `SEEDX_DTYPE`, bfloat16 by default) and `translategemma_beam`
(gemma interpreter, `TRANSLATEGEMMA_DTYPE`, float32 by default) record the beam search through
`_decoder_beam_record`: transformers' own `generate` with four beams and `do_sample=False`, over the
prompt of every probe sentence (64 and 48 new tokens, the greedy records' limits), holding each prompt's
ids, the best hypothesis's continuation with its end token, its text, and its `sequences_scores`, and the
same search for sentence 1 cut at six tokens, where the limit ends every top hypothesis. Keys
`IK_PARITY_SEEDX_PPO_BEAM` and `IK_PARITY_TRANSLATEGEMMA_BEAM`. A float32 search's scores hold to 1e-3 of
the record's. A bfloat16 search's scores are measured from float32: `seedx_beam_scores`
(`SEEDX_DTYPE=float32 SEEDX_LAYERWISE=1`, `SEEDX_BEAM_RECORD` naming the beam record) teacher-forces each
recorded hypothesis in one forward and takes its mean log-probability, the score `generate` gives it, and
a bfloat16 score holds where it sits no further from that value than the record's own scores do
(`IK_PARITY_SEEDX_PPO_BEAM_F32`).

Every translation runner takes its pair from `TRANSLATION_SOURCE_LANG` / `TRANSLATION_TARGET_LANG`
(default `en` / `de`); `run_marian` reads the pair from the release's `tokenizer_config.json` instead
and needs `MARIAN_TARGET_CODE` (the code inside `>>…<<`) on a group release. The probe sentences are
`TRANSLATION_PROBES`, one list per source language (en, ja, zh, ar, vi, th, hi, ko), and the loss
sentence is `TRANSLATION_TARGETS[target]`; the Swift tests hold the same table. A record also carries
the reference's decoded greedy and beam text as UTF-8 (`greedy_text`, `beam_text`), which the tests hold
byte for byte against `translate`. Further runners: `nllb` (NllbTokenizer with `NLLB_CODES` mapping the probe tags to
`xxx_Xxxx`), `small100` (imports the release's own
`tokenization_small100.py`), `madlad` with `MADLAD_DTYPE=bfloat16` for the 7B, and
`madlad_layerwise`, which records what `run_madlad` records for a release too large to hold whole at its
precision: the T5 is built on the meta device, the embedding, the untied head, and both final norms are
filled, and each encoder and decoder block is read from the release (`F_NOCACHE`) as its forward reaches
it and freed after; `MADLAD_HELD_DECODER_BLOCKS` keeps that many leading decoder blocks resident, which
changes the reads and nothing recorded. On the 3B at float32 its record equals `run_madlad`'s bit for
bit (17 keys); the 7B float32 record took 177 s with 32 decoder blocks held. Then
`translategemma_layerwise`, which records what `run_translategemma` records for a release too large to
hold whole: the model is built on the meta device and each decoder layer's tensors are read from the
release (`F_NOCACHE`, in reads under 1 GiB, since macOS refuses longer ones) as the forward reaches the
layer and freed after it, so `generate` and the `labels=` loss run through transformers' own entry points.
On the 4B at bfloat16 its record equals `run_translategemma`'s bit for bit (42 keys). Probe records are keyed `IK_PARITY_<FAMILY>_<SRC>_<TGT>` (`IK_PARITY_MARIAN_JA_EN`,
`IK_PARITY_MADLAD_EN_ZH_HANT`), and a probe test skips the pairs whose record is absent. The 12B
figures above were measured with the whole model resident, which swapped on this 32 GB machine; the
12B and 27B tests now stream the decoder (`residency: .streamed`) and run every layer's state, the logits,
and the loss through the same code as the 4B, the logits projected from the last state. A streamed decoder
reads every layer it does not hold on each pass, so those tests check the greedy continuation teacher-forced
in one pass, each position's argmax the record's next token with the stop token included, and decode the
printed translation from it. A streamed test makes three passes: the states, the continuation, and the
loss. The drafted tests generate the continuation through the cache.

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

How the four by-configuration releases were measured (2026-10-02).

- `facebook/m2m100_1.2B` (24 + 24 at 1024, ~5 GB float32): `run_reference.py m2m100 --checkpoint` on
  its directory unchanged, under `IK_VAL_M2M100_1_2B` / `IK_PARITY_M2M100_1_2B`; the port at float32.
  Fits alongside its reference.
- `alirezamsh/small100` (12 + 3, ~1.3 GB): the release's tokenizer is its own `tokenization_small100.py`
  (`SMALL100Tokenizer`, target marker on the source, `tgt_lang` only), so the oracle needs a `small100`
  runner that imports that file from the directory; the record otherwise matches `run_m2m100`. Keys
  `IK_VAL_SMALL100` / `IK_PARITY_SMALL100`.
- `google/madlad400-7b-mt` (48 + 48 at 2048, about 33 GB float32): float32 does not fit beside anything
  on 32 GB, so the reference first recorded at bfloat16 (`MADLAD_DTYPE=bfloat16`, added to `run_madlad`)
  and the port loaded `half`, held to bf16 tolerances as the TranslateGemma 12B is. Each side runs alone at
  about 17 GB. Keys `IK_VAL_MADLAD_7B` / `IK_PARITY_MADLAD_7B`. At float32 the reference runs
  `madlad_layerwise` and the port streams; key `IK_PARITY_MADLAD_7B_F32`. `madlad400-7b-mt-bt` is the
  same geometry and is not separately measured.
- `google/translategemma-27b-it` (62 layers at 5376, about 54 GB bfloat16, gated): neither side fits
  whole, and the model is dense. The port streams the decoder layers it cannot hold (`.streamed`), and
  the reference runs `translategemma_layerwise`, the whole-model record built one layer at a time.
  Keys `IK_VAL_TRANSLATEGEMMA_27B` / `IK_PARITY_TRANSLATEGEMMA_27B` (the earlier first-14-layers record
  is kept as `.v1`).

## Not yet ported

The toolkit is license agnostic: a weight license is the consumer's to read and comply with, and it does
not block a port. Each entry below names the port cost, not a bar.

- SeamlessM4T v2 (`facebook/seamless-m4t-v2-large`, CC-BY-NC-4.0): its text-to-text path is an
  NLLB-style 24 + 24 encoder-decoder over the same 256k vocabulary; the speech paths add the shipped
  W2V-BERT encoder, a UnitY text-to-unit decoder, and a HiFi-GAN unit vocoder. Text-to-text is a
  medium port; the whole model is a large one.
- TowerInstruct 7B / 13B (CC-BY-NC-4.0, Llama-2 decoders) and Tower+ 9B (Gemma 2 decoder): the
  decoder is shipped, so each is a chat template over `NFKMLXLanguageNet`, the TranslateGemma pattern.
- Hunyuan-MT 7B and Hunyuan-MT-Chimera 7B (Tencent, September 2025): `hunyuan_v1_dense` with HY-MT 7B's
  tokenizer file, so the `hymt` judge reads them through Hy-MT2's entry. Neither is measured, and
  Chimera's own prompt is not rendered.
- Hy-MT2-30B-A3B (Apache-2.0, `hy_v3`, a mixture of experts, 60 GB): a new architecture.
- TranslateGemma is gated: a machine whose token has not accepted the Gemma terms on the model page
  gets a `403` on `config.json`, which is what the download factory surfaces.
