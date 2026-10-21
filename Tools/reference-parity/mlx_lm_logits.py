#!/usr/bin/env python3
"""A release mlx_lm published, run by mlx_lm itself: the half-precision reference for the release loader.

An mlx-community release runs in the precision its scales are stored at, which is how mlx_lm computes
it. `run_reference.py mlx_quantized_release` measures what the packed weights mean (transformers on
them dequantized, at float32); this records what mlx_lm computes from them, so the loader is held to
both: the network at float32, and mlx_lm's own arithmetic at the stored precision.

Run under the `mlxlm` interpreter of Tools/validation-assets/manifest.json:

    mlx_lm_logits.py <output.safetensors> --checkpoint <release directory> --record <mlx_quantized_release record>

The prompt's token ids are read from the `mlx_quantized_release` record, so both references see the
same input. The record carries those `tokens`, the logits for every position as `output` (float32), and
the greedy `continuation` of 16 tokens, each step fed through mlx_lm's own prompt cache.
"""

import argparse
import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", help="path to write the .safetensors record")
    parser.add_argument("--checkpoint", required=True, help="the release directory")
    parser.add_argument("--record", required=True, help="the mlx_quantized_release record whose tokens to read")
    parser.add_argument("--steps", type=int, default=16, help="greedy continuation length")
    args = parser.parse_args()

    tokens = mx.load(args.record)["tokens"].astype(mx.int32)
    model, _ = load(args.checkpoint)

    logits = model(tokens[None])[0].astype(mx.float32)
    cache = make_prompt_cache(model)
    step = model(tokens[None], cache=cache)[:, -1, :]
    continuation = []
    for _ in range(args.steps):
        token = mx.argmax(step, axis=-1)
        continuation.append(token.item())
        step = model(token[None], cache=cache)[:, -1, :]

    mx.save_safetensors(args.output, {
        "tokens": tokens,
        "output": logits,
        "continuation": mx.array(continuation, dtype=mx.int32),
    })
    print(f"wrote {args.output}: {tokens.size} tokens, logits {tuple(logits.shape)}, continuation {continuation[:6]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
