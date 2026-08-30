#!/usr/bin/env python3
"""Convert a CodeFormer checkpoint to safetensors for InferKitMLX.

InferKitMLX also reads the raw checkpoint directly (its native reader, InferKit 0.3.0), so
this converter is optional: it remains the offline path for producing a portable
safetensors file.

InferKitMLX's `NFKMLXCodeFormer` loads a safetensors checkpoint; MLX's `loadArrays` reads
safetensors/npz, not PyTorch `.pth`. This tool rewrites a checkpoint into safetensors, preserving
PyTorch convolution layout `[out, in, kH, kW]` (the Swift loader transposes to MLX's channels-last
layout at load).

The reference CodeFormer checkpoint (`codeformer.pth`) nests its VQGAN encoder/generator as
heterogeneous `encoder.blocks.N.*` / `generator.blocks.N.*` module lists, a quantizer
(`quantize.embedding.weight`), the Transformer (`ft_layers.N.*`, `position_emb`, `feat_emb`,
`idx_pred_layer.*`), and the controllable feature transformation (`fuse_convs_dict.*`). InferKitMLX's
`NFKMLXCodeFormerNet` groups the encoder/generator into named residual and up/down stages and predicts
codes without the CFT fusion, so its parameter names do not line up with the reference one-to-one.
Aligning them is a validation-sweep task; pass `--list-keys` to print the checkpoint's names.

Usage:
    python convert.py codeformer.pth --list-keys
    python convert.py codeformer.pth codeformer.safetensors     # after a name remap is worked out

Get the weights from the CodeFormer release:
    https://github.com/sczhou/CodeFormer

Requires: torch, safetensors  (pip install torch safetensors)
"""

import argparse
import sys

import torch
from safetensors.torch import save_file


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("params_ema", "params", "state_dict"):
            inner = checkpoint.get(key)
            if isinstance(inner, dict):
                return inner
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise SystemExit("unrecognized checkpoint: expected a CodeFormer state dict")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to codeformer.pth")
    parser.add_argument("output", nargs="?", help="path to write .safetensors")
    parser.add_argument("--list-keys", action="store_true", help="print parameter names and exit")
    parser.add_argument("--half", action="store_true", help="store weights as float16")
    args = parser.parse_args()

    state = extract_state_dict(load_checkpoint(args.input))

    if args.list_keys:
        for name in sorted(state):
            print(f"{name}\t{tuple(state[name].shape)}")
        return

    if not args.output:
        raise SystemExit("output path required unless --list-keys")

    dtype = torch.float16 if args.half else torch.float32
    tensors = {name: value.to(dtype).contiguous().clone()
               for name, value in state.items() if value.is_floating_point()}
    save_file(tensors, args.output)
    print(f"wrote {len(tensors)} tensors ({dtype}) to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
