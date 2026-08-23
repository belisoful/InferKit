#!/usr/bin/env python3
"""Convert a Hugging Face causal language model to an InferKit Core ML model directory.

The output directory holds:
  - model.mlpackage   a stateful Core ML model whose KV cache is Core ML state (iOS 18 / macOS 15).
  - vocab.json        the byte-level BPE vocabulary (token -> id).
  - merges.txt        the ranked BPE merges.
  - manifest.json     the contract NFKCoreMLLanguageBackend reads (feature names, token ids, tokenizer).

Important, read before using:
  - The conversion source is a PyTorch / Hugging Face checkpoint, not MLX weights. coremltools
    traces PyTorch; there is no MLX -> Core ML path. If you have an MLX model, convert the Hugging
    Face checkpoint it was built from.
  - Tokenization and the generation loop are outside Core ML's scope; the InferKit runtime owns them.
    The runtime ships a byte-level BPE tokenizer (GPT-2 / Qwen family). A SentencePiece model exports
    here, but the runtime does not decode it yet.
  - The KV-cache-to-state wrapping is architecture specific. This tool handles the common
    attention-cache shape and documents where a model needs adaptation. Inspect first with --inspect.

Usage:
  python convert.py --model <hf-id-or-path> --output <dir> [--context 2048] [--quantize int8]
  python convert.py --model <hf-id-or-path> --inspect
  python convert.py --tiny --output <dir>          # a synthetic model for local runtime smoke tests
"""

import argparse
import json
import os
import sys

import architectures


def log(message):
    print(f"[inferkit-convert] {message}", file=sys.stderr)


# --------------------------------------------------------------------------------------------------
# Tokenizer + manifest export (works for any Hugging Face tokenizer; no model load required)
# --------------------------------------------------------------------------------------------------

def export_tokenizer(model, output_dir):
    """Writes the tokenizer files and returns the manifest 'tokenizer' section plus token ids."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer_type, files = _write_tokenizer_files(tokenizer, output_dir)

    special = {}
    for name in ("bos_token", "eos_token", "pad_token", "unk_token"):
        token = getattr(tokenizer, name, None)
        token_id = getattr(tokenizer, f"{name}_id", None)
        if token is not None and token_id is not None:
            special[str(token)] = int(token_id)
    for token, token_id in _added_special_tokens(tokenizer):
        special[str(token)] = int(token_id)

    section = {"type": tokenizer_type, **files}
    if special:
        section["specialTokens"] = special
    chat_template = _extract_chat_template(tokenizer)
    if chat_template is not None:
        section["chatTemplate"] = chat_template

    return section, {
        "vocabSize": int(getattr(tokenizer, "vocab_size", len(tokenizer))),
        "eosTokenId": _int_or(-1, getattr(tokenizer, "eos_token_id", None)),
        "bosTokenId": _int_or(-1, getattr(tokenizer, "bos_token_id", None)),
    }


def _write_tokenizer_files(tokenizer, output_dir):
    model = _backend_model(tokenizer)
    if model is not None and model.get("type") == "Unigram":
        return _write_unigram(model, tokenizer, output_dir)
    if model is not None and model.get("type") == "WordPiece":
        return _write_wordpiece(model, tokenizer, output_dir)

    saved = tokenizer.save_vocabulary(output_dir)
    saved_names = {os.path.basename(path) for path in saved if path}
    if "vocab.json" in saved_names and "merges.txt" in saved_names:
        return "bpe-bytelevel", {"vocab": "vocab.json", "merges": "merges.txt"}
    if "tokenizer.model" in saved_names:
        log("SentencePiece slow tokenizer; load it with use_fast=True to export a unigram vocab")
        return "sentencepiece", {"model": "tokenizer.model"}

    tokenizer.save_pretrained(output_dir)
    return "unknown", {"tokenizer": "tokenizer.json"}


def _write_unigram(model, tokenizer, output_dir):
    """Writes the unigram vocab the runtime NFKUnigramTokenizer reads."""
    payload = {
        "vocab": [[piece, float(score)] for piece, score in model.get("vocab", [])],
        "unkId": int(model.get("unk_id") or 0),
        "byteFallback": bool(model.get("byte_fallback", False)),
        "addDummyPrefix": _add_dummy_prefix(tokenizer),
    }
    path = os.path.join(output_dir, "unigram.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    log(f"wrote {path}")
    return "unigram", {"vocab": "unigram.json"}


def _write_wordpiece(model, tokenizer, output_dir):
    """Writes the wordpiece vocab the runtime NFKWordPieceTokenizer reads."""
    payload = {
        "vocab": model.get("vocab", {}),
        "unkToken": model.get("unk_token", "[UNK]"),
        "continuingSubwordPrefix": model.get("continuing_subword_prefix", "##"),
        "lowercase": _do_lowercase(tokenizer),
        "maxCharsPerWord": int(model.get("max_input_chars_per_word", 100)),
    }
    path = os.path.join(output_dir, "wordpiece.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    log(f"wrote {path}")
    return "wordpiece", {"vocab": "wordpiece.json"}


def _do_lowercase(tokenizer):
    if getattr(tokenizer, "do_lower_case", None) is not None:
        return bool(tokenizer.do_lower_case)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    try:
        spec = json.loads(backend.to_str()).get("normalizer") or {}
    except Exception:
        return True
    if spec.get("type") == "BertNormalizer":
        return bool(spec.get("lowercase", True))
    return True


def _extract_chat_template(tokenizer):
    """Extracts per-role markers from an instruct tokenizer's chat template by probing it with
    sentinels. Returns a descriptor the runtime renders, or None when the template is absent or does
    not decompose into prefix/content/suffix turns (the runtime then falls back to a plain format)."""
    if getattr(tokenizer, "chat_template", None) is None:
        return None
    U, U2, A, S = "\x00u\x00", "\x00v\x00", "\x00a\x00", "\x00s\x00"

    def render(messages, gen=False):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=gen)

    try:
        uu = render([{"role": "user", "content": U}, {"role": "user", "content": U2}])
        mid, user_suffix = uu.split(U, 1)[1].split(U2, 1)
        user_prefix = mid[len(user_suffix):] if mid.startswith(user_suffix) else mid

        gen, nogen = render([{"role": "user", "content": U}], True), render([{"role": "user", "content": U}])
        generation_prompt = gen[len(nogen):] if gen.startswith(nogen) else ""

        after_user = render([{"role": "user", "content": U}, {"role": "assistant", "content": A}]).split(U, 1)[1]
        after_user = after_user[len(user_suffix):] if after_user.startswith(user_suffix) else after_user
        assistant_prefix, assistant_suffix = after_user.split(A, 1)

        sys_before = render([{"role": "system", "content": S}, {"role": "user", "content": U}]).split(U, 1)[0]
        if user_prefix and sys_before.endswith(user_prefix):
            sys_before = sys_before[: -len(user_prefix)]
        system_prefix, system_suffix = sys_before.split(S, 1)

        default_system = ""
        user_only = render([{"role": "user", "content": U}]).split(U, 1)[0]
        if system_prefix and system_suffix and user_only.startswith(system_prefix):
            inner = user_only[len(system_prefix):]
            if system_suffix in inner:
                default_system = inner.split(system_suffix, 1)[0]

        return {
            "system": [system_prefix, system_suffix],
            "user": [user_prefix, user_suffix],
            "assistant": [assistant_prefix, assistant_suffix],
            "generationPrompt": generation_prompt,
            "defaultSystem": default_system,
        }
    except Exception:
        return None


def _backend_model(tokenizer):
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return None
    try:
        return json.loads(backend.to_str()).get("model")
    except Exception:
        return None


def _add_dummy_prefix(tokenizer):
    backend = getattr(tokenizer, "backend_tokenizer", None)
    try:
        spec = json.loads(backend.to_str()).get("pre_tokenizer") or {}
    except Exception:
        return True
    if spec.get("type") == "Metaspace":
        if "add_prefix_space" in spec:
            return bool(spec["add_prefix_space"])
        return spec.get("prepend_scheme", "always") != "never"
    return True


def _added_special_tokens(tokenizer):
    added = getattr(tokenizer, "added_tokens_encoder", None) or {}
    return [(token, token_id) for token, token_id in added.items()]


def _int_or(default, value):
    return int(value) if value is not None else default


def write_manifest(output_dir, tokenizer_section, token_stats, context_length, model_section):
    manifest = {
        "modelType": "causal-lm",
        "model": "model.mlpackage",
        "contextLength": int(context_length),
        "vocabSize": token_stats["vocabSize"],
        "eosTokenId": token_stats["eosTokenId"],
        "bosTokenId": token_stats["bosTokenId"],
        "tokenizer": tokenizer_section,
        **model_section,
    }
    path = os.path.join(output_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    log(f"wrote {path}")
    return manifest


# --------------------------------------------------------------------------------------------------
# Model conversion (PyTorch -> stateful Core ML)
# --------------------------------------------------------------------------------------------------

def inspect_model(model):
    """Prints the checkpoint's config so a caller can see the cache shape before converting."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model)
    fields = ("model_type", "architectures", "hidden_size", "num_hidden_layers",
              "num_attention_heads", "num_key_value_heads", "vocab_size", "max_position_embeddings")
    for field in fields:
        log(f"{field}: {getattr(config, field, None)}")


def convert_model(model, output_dir, context_length, quantize, prefill_length):
    """Traces the model with a stateful KV cache and writes model.mlpackage.

    Dispatches to the architecture registered for the checkpoint's model_type. Each architecture
    derives its cache shape from the config, builds the tracing wrapper, and declares the Core ML
    state and features. With a prefill_length, the package is a multifunction model: a "decode"
    function (one token, the default) and a "prefill" function (prefill_length tokens) share the
    weights and the KV-cache state, so a prompt runs in chunks instead of token by token.
    Returns the manifest section describing the model's features.
    """
    import tempfile
    import torch
    import coremltools as ct
    from transformers import AutoModelForCausalLM, AutoConfig

    config = AutoConfig.from_pretrained(model)
    architecture = architectures.resolve(config.model_type)
    log(f"architecture: {config.model_type} -> {type(architecture).__name__}")

    torch_model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32).eval()
    wrapper = architecture.build_wrapper(torch_model, config, context_length)

    def convert_length(seq_len):
        # check_trace is off because the cache state changes between trace runs.
        example = (torch.zeros((1, seq_len), dtype=torch.int32),
                   torch.arange(seq_len, dtype=torch.int32))
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, example, check_trace=False)
        import numpy as np
        converted = ct.convert(
            traced,
            inputs=[ct.TensorType(name=architecture.input_feature, shape=(1, seq_len), dtype=np.int32),
                    ct.TensorType(name=architecture.position_feature, shape=(seq_len,), dtype=np.int32)],
            outputs=architecture.output_types(),
            states=architecture.state_types(config, context_length),
            minimum_deployment_target=ct.target.iOS18,
            compute_units=ct.ComputeUnit.ALL,
            convert_to="mlprogram",
        )
        if quantize == "int8":
            from coremltools.optimize.coreml import linear_quantize_weights, OpLinearQuantizerConfig, OptimizationConfig
            config8 = OptimizationConfig(global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8"))
            converted = linear_quantize_weights(converted, config8)
        return converted

    path = os.path.join(output_dir, "model.mlpackage")
    section = architecture.manifest_section()

    if prefill_length > 1:
        with tempfile.TemporaryDirectory() as scratch:
            decode_path = os.path.join(scratch, "decode.mlpackage")
            prefill_path = os.path.join(scratch, "prefill.mlpackage")
            convert_length(1).save(decode_path)
            log("decode function converted")
            convert_length(prefill_length).save(prefill_path)
            log("prefill function converted")
            descriptor = ct.utils.MultiFunctionDescriptor()
            descriptor.add_function(decode_path, src_function_name="main", target_function_name="decode")
            descriptor.add_function(prefill_path, src_function_name="main", target_function_name="prefill")
            descriptor.default_function_name = "decode"
            ct.utils.save_multifunction(descriptor, path)
        section["prefill"] = {"function": "prefill", "length": prefill_length}
    else:
        convert_length(1).save(path)

    log(f"wrote {path}")
    return section


# --------------------------------------------------------------------------------------------------
# Synthetic tiny model (for local runtime smoke tests, no checkpoint needed)
# --------------------------------------------------------------------------------------------------

def build_tiny(output_dir):
    """Writes a tiny stateful model plus a toy tokenizer, so the runtime path can be exercised."""
    import numpy as np
    import torch
    import coremltools as ct

    vocab = 32
    hidden = 8

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(vocab, hidden)
            self.head = torch.nn.Linear(hidden, vocab)
            # A running state the model reads and writes, so the conversion exercises Core ML state.
            # Core ML states are fp16.
            self.register_buffer("k_cache", torch.zeros(1, hidden, dtype=torch.float16))

        def forward(self, input_ids):
            hidden_states = self.embed(input_ids.long())                    # [1, seq, hidden]
            summary = hidden_states.mean(dim=1).to(torch.float16)          # [1, hidden]
            self.k_cache += summary                                         # in-place write to state
            biased = hidden_states + self.k_cache.unsqueeze(1).to(hidden_states.dtype)  # read state
            return self.head(biased)                                        # [1, seq, vocab]

    model = Tiny().eval()
    traced = torch.jit.trace(model, torch.zeros((1, 1), dtype=torch.int32))
    seq = ct.RangeDim(lower_bound=1, upper_bound=64, default=1)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_ids", shape=(1, seq), dtype=np.int32)],
        outputs=[ct.TensorType(name="logits")],
        states=[ct.StateType(wrapped_type=ct.TensorType(shape=(1, hidden), dtype=np.float16), name="k_cache")],
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    mlmodel.save(os.path.join(output_dir, "model.mlpackage"))

    vocab_map = {chr(ord("a") + i): i for i in range(26)}
    vocab_map.update({" a": 26, " b": 27, "<eos>": 31})
    with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as handle:
        json.dump(vocab_map, handle)
    with open(os.path.join(output_dir, "merges.txt"), "w", encoding="utf-8") as handle:
        handle.write("#version: 0.2\n")

    write_manifest(
        output_dir,
        tokenizer_section={"type": "bpe-bytelevel", "vocab": "vocab.json", "merges": "merges.txt",
                           "specialTokens": {"<eos>": 31}},
        token_stats={"vocabSize": vocab, "eosTokenId": 31, "bosTokenId": -1},
        context_length=64,
        model_section={"inputFeature": "input_ids", "logitsFeature": "logits", "stateNames": ["k_cache"]},
    )
    log("wrote synthetic tiny model")


# --------------------------------------------------------------------------------------------------

def main(argv):
    parser = argparse.ArgumentParser(description="Convert a Hugging Face causal LM to an InferKit Core ML directory.")
    parser.add_argument("--model", help="Hugging Face model id or local checkpoint path")
    parser.add_argument("--output", help="output directory")
    parser.add_argument("--context", type=int, default=2048, help="maximum context length")
    parser.add_argument("--quantize", choices=["int8"], help="optional weight quantization")
    parser.add_argument("--prefill", type=int, default=64,
                        help="prefill chunk length for the multifunction model; 0 or 1 disables (default 64)")
    parser.add_argument("--inspect", action="store_true", help="print the model config and exit")
    parser.add_argument("--tiny", action="store_true", help="write a synthetic tiny model for smoke tests")
    parser.add_argument("--list-architectures", action="store_true", help="list supported model_type values and exit")
    args = parser.parse_args(argv)

    if args.list_architectures:
        for model_type in architectures.supported_types():
            print(model_type)
        return 0

    if args.inspect:
        if not args.model:
            parser.error("--inspect needs --model")
        inspect_model(args.model)
        return 0

    if not args.output:
        parser.error("--output is required")
    os.makedirs(args.output, exist_ok=True)

    if args.tiny:
        build_tiny(args.output)
        return 0

    if not args.model:
        parser.error("--model is required")

    tokenizer_section, token_stats = export_tokenizer(args.model, args.output)
    model_section = convert_model(args.model, args.output, args.context, args.quantize, args.prefill)
    write_manifest(args.output, tokenizer_section, token_stats, args.context, model_section)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
