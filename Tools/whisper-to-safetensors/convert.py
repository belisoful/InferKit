#!/usr/bin/env python3
"""Convert an OpenAI Whisper checkpoint to safetensors for InferKitMLX (NFKMLXWhisper).

InferKitMLX also reads the raw checkpoint directly (its native reader, InferKit 0.3.0), so
this converter is optional: it remains the offline path for producing a portable
safetensors file.

MLX loads safetensors/npz, not the `.pt` OpenAI Whisper ships. This tool extracts `model_state_dict`
and writes safetensors. The module names follow OpenAI Whisper exactly (`encoder`/`decoder`,
`blocks.N.attn.query/key/value/out`, `attn_ln`, `cross_attn`, `mlp.0`/`mlp.2`, `token_embedding`,
`positional_embedding`), so no renaming is needed; the Swift loader transposes the Conv1d weights.

This targets the OpenAI `whisper` package format. The HF transformers format (`model.encoder.layers.
N.self_attn.q_proj…`) uses different names and needs a remap (sweep item). The mel filterbank in the
module is HTK-triangular and audio is assumed 16 kHz — both sweep items for exact parity.

Usage:
    python convert.py tiny.pt whisper-tiny.safetensors

Weights: `pip install -U openai-whisper` then the cached `~/.cache/whisper/*.pt`, or download from
https://github.com/openai/whisper

Requires: torch, safetensors
"""

import argparse
import sys

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to a Whisper .pt")
    parser.add_argument("output", help="path to write .safetensors")
    parser.add_argument("--half", action="store_true", help="store weights as float16")
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not (isinstance(state, dict) and all(isinstance(v, torch.Tensor) for v in state.values())):
        raise SystemExit("unrecognized checkpoint: expected an OpenAI Whisper .pt with model_state_dict")

    dtype = torch.float16 if args.half else torch.float32
    tensors = {name: value.to(dtype).contiguous().clone() for name, value in state.items()}
    save_file(tensors, args.output)
    print(f"wrote {len(tensors)} tensors ({dtype}) to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
