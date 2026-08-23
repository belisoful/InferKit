# Running a language model on device with Core ML


InferKit runs a language model locally in two steps: convert once, then run.

1. **Convert** a Hugging Face causal-LM checkpoint to a Core ML model directory with the offline tool
   in [`Tools/inferkit-convert`](../Tools/inferkit-convert). It writes a stateful `model.mlpackage`
   (the KV cache is Core ML state), the tokenizer files, and a `manifest.json` naming the model's
   features and tokenizer. The source is PyTorch, not MLX; `coremltools` traces PyTorch. The tool is
   Python and needs its own packages (`torch`, `transformers`, `coremltools`, `numpy`), installed
   once into a virtual environment:

   ```bash
   cd Tools/inferkit-convert
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   python convert.py --model <hf-id-or-path> --output ../../Local/models/<name>
   ```

   The [tool's README](../Tools/inferkit-convert/README.md) documents the options, the supported
   architectures, and the manifest contract. Keep converted models under the git-ignored `Local/`.
2. **Run** the directory through `NFKCoreMLLanguageBackend`. Core ML runs one forward pass; the
   backend owns tokenization and the autoregressive sampling loop (temperature, top-k, top-p,
   repetition penalty, seed, stop sequences). `submitInferenceJobForRequest:` streams the text as it
   generates: read `job.partialResult` in the job's `progressHandler`. A request supplies either
   `NFKInputPrompt` (a string) or
   `NFKInputMessages` (an OpenAI-style array); for an instruct model, the converter captures the
   chat template so messages render with the model's own role markers.

`NFKTokenizer` is a class cluster: `tokenizerForManifest:directory:error:` returns the subclass named
by the manifest — byte-level BPE (`NFKByteLevelBPETokenizer`, GPT-2 / Qwen family), SentencePiece
unigram (`NFKUnigramTokenizer`, Llama / Mistral / Gemma family), or WordPiece (`NFKWordPieceTokenizer`,
BERT family). A new tokenizer family is a new subclass plus one line in the factory.

The converted package is a multifunction model: a `decode` function runs one token, and a `prefill`
function runs a fixed chunk of prompt tokens (default 64), both writing the same KV-cache state. The
backend prefills a prompt in chunks and finishes the remainder token by token, which processes a
~400-token prompt in ~0.3 s instead of ~3.7 s on a 0.5B model. Throughput is set by `computeUnits`
(default `MLComputeUnitsAll`, the fastest); `MLComputeUnitsCPUAndNeuralEngine` can fail to load a
stateful model because the KV-cache scatter does not compile for the Neural Engine.
