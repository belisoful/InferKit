# inferkit-convert

Converts a Hugging Face causal language model into a Core ML model directory that
`NFKCoreMLLanguageBackend` runs on device.

This is an offline developer tool. It is not part of the InferKit SwiftPM or CocoaPods build and
ships no Objective-C.

## What it produces

A directory containing:

| File | Role |
| --- | --- |
| `model.mlpackage` | A stateful Core ML model. The KV cache is Core ML state (`ct.StateType`), so decoding reuses it across steps. Requires iOS 18 / macOS 15. |
| `vocab.json`, `merges.txt` | The byte-level BPE tokenizer files. |
| `manifest.json` | The contract the runtime reads: input/logits feature names, state names, context length, token ids, and the tokenizer description. |

## Install and run

```bash
cd Tools/inferkit-convert
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python convert.py --model <hf-id-or-path> --output ./out --context 2048
```

Inspect a checkpoint's shape before converting:

```bash
python convert.py --model <hf-id-or-path> --inspect
```

Write a synthetic tiny model to exercise the runtime path without a checkpoint:

```bash
python convert.py --tiny --output ./tiny
```

## Read before converting

- **Source is PyTorch / Hugging Face, not MLX.** `coremltools` traces PyTorch; there is no
  MLX → Core ML path. If you have an MLX model, convert the Hugging Face checkpoint it was built
  from.
- **Core ML does not tokenize or loop.** The InferKit runtime owns tokenization and the
  autoregressive sampling loop. Core ML runs one forward pass per step.
- **Tokenizer support.** The runtime ships three tokenizers: byte-level BPE (GPT-2 / Qwen family),
  SentencePiece unigram (Llama / Mistral / Gemma family), and WordPiece (BERT family). A fast
  (Rust-backed) tokenizer is needed to export the unigram and wordpiece vocabularies; load with
  `use_fast=True`.
- **The KV-cache wrapper is architecture specific**, so it lives in a registry. `convert_model`
  looks up the checkpoint's `model_type` in [`architectures.py`](architectures.py) and dispatches to
  the matching `Architecture`, which derives the cache shape from the config and builds the tracing
  wrapper. An unregistered `model_type` fails with the supported list rather than emitting a
  silently-wrong graph.

## Supported architectures

```bash
python convert.py --list-architectures
```

| Family | model_type | Tokenizer | Status |
| --- | --- | --- | --- |
| RoPE decoders | `llama`, `qwen2`, `mistral`, `gemma`, `gemma2`, `phi3`, `stablelm`, `starcoder2` | Qwen2 byte-level BPE; Llama/Mistral/Gemma SentencePiece unigram — both runtime-ready | **Numerically validated**: llama, qwen2, mistral, gemma2, phi3, stablelm, starcoder2 (6/6 argmax, ~1e-3) |
| GPT-2 | `gpt2` | byte-level BPE | **Converts + numerically validated**; shares the static-cache wrapper |

The graph is fixed-shape: the model takes `input_ids` plus a `cache_position`, scatters the new
key/value into a fixed cache, and masks unwritten positions from `cache_position`. The package is a
**multifunction model** (`--prefill`, default 64): a `decode` function runs one token and a `prefill`
function runs a fixed chunk, sharing the weights and the KV-cache state, so the runtime processes a
prompt in chunks (measured ~13× faster prompt processing on Qwen2.5-0.5B) and decodes token by token.
`--prefill 0` produces a single-function decode-only package.

## Validation status

- **Real-model conversion: validated.** The real **Qwen2.5-0.5B** (~940 MB) and **gpt2** (~240 MB)
  both convert to stateful `.mlpackage`s and generate through `NFKCoreMLLanguageBackend`.
- **Architecture spot-check: 8/8 faithful.** Tiny random models for `qwen2`, `gpt2`, `llama`,
  `mistral`, `gemma2`, `phi3`, `stablelm`, `starcoder2` are **numerically faithful to PyTorch**
  (token-by-token with state: 6/6 argmax, max logit diff ~1–3e-3 from the fp16 cache) on
  transformers 4.57 (Qwen2/GPT-2 also on 4.46).
- **Runtime + pipeline mechanics: validated end to end.** `--tiny` builds a real stateful
  `.mlpackage`, and `NFKCoreMLLanguageBackend` loads it, runs the `MLState` prefill/decode loop,
  samples, and returns text (see the opt-in `NFKCoreMLLanguageBackendLiveTests`, run with
  `INFERKIT_TEST_MODEL_DIR` set to a converted directory).
- **Batched prefill: validated.** The multifunction package (decode + prefill sharing one KV state)
  processes a ~400-token prompt in 0.27 s vs 3.67 s token-by-token on Qwen2.5-0.5B-Instruct, with
  identical output. The runtime falls back to CPU+GPU for the prefill function when the Neural
  Engine cannot compile it, and to decode-only when prefill cannot load at all.
- **int8 quantization: validated.** `--quantize int8` halves the package (Qwen2.5-0.5B 943 MB →
  473 MB) and greedy generation is unchanged.
- **Tokenizers: validated** by unit tests (byte-level BPE, unigram, WordPiece).
- **Mixture-of-experts (e.g. `qwen2_moe`): not supported.** The sparse expert routing does not lower
  to a static Core ML graph (dynamic per-expert dispatch / unbind over a symbolic count).
- **transformers version.** The wrapper supports both the pre-4.54 Cache API and the 4.54+ redesign,
  so no version pin is required. Newer redesigns may need the cache wrapper revisited.

**The runtime contract.** A converted model takes `input_ids` and, when the architecture sets
`position_feature`, a 1-D `cache_position` (the absolute index of each new token). The causal mask is
derived inside the model from `cache_position`; the runtime supplies no mask. `NFKCoreMLLanguageBackend`
feeds `cache_position` whenever the manifest names a `positionFeature`.

**Add an architecture.** Subclass `Architecture` (or `_StaticCacheArchitecture` for a Cache-based
decoder), set `model_types`, override `describe` / `build_wrapper` where the family differs, and
`register(YourArchitecture())`. The shape and feature contracts are covered by `test_architectures.py`,
which runs without torch:

```bash
python -m unittest test_architectures
```

## manifest.json

```json
{
  "modelType": "causal-lm",
  "model": "model.mlpackage",
  "inputFeature": "input_ids",
  "logitsFeature": "logits",
  "positionFeature": "cache_position",
  "stateNames": ["k_cache", "v_cache"],
  "prefill": { "function": "prefill", "length": 64 },
  "contextLength": 2048,
  "vocabSize": 151936,
  "eosTokenId": 151643,
  "bosTokenId": -1,
  "tokenizer": { "type": "bpe-bytelevel", "vocab": "vocab.json", "merges": "merges.txt" }
}
```

`eosTokenId` / `bosTokenId` are `-1` when the model defines none. `tokenizer.type` selects the
runtime tokenizer subclass; `bpe-bytelevel` is the one shipped today.
