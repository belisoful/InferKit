#!/usr/bin/env python3
"""Convert a Zero-DCE (DCE-Net) .pth checkpoint to safetensors for InferKitMLX.

InferKitMLX's `NFKMLXZeroDCE` loads a safetensors checkpoint; MLX's `loadArrays` reads safetensors/npz,
not PyTorch `.pth`. This tool rewrites the release into safetensors, preserving the reference DCE-Net
parameter names (`e_conv1.*` … `e_conv7.*`) and PyTorch convolution layout `[out, in, kH, kW]`. The
Swift loader transposes convolution weights to MLX's channels-last layout at load, so this tool does no
transposition.

Usage:
    python convert.py Epoch99.pth zero-dce.safetensors
    python convert.py Epoch99.pth zero-dce.safetensors --half

Get the weights from the Zero-DCE release:
    https://github.com/Li-Chongyi/Zero-DCE

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
        inner = checkpoint.get("state_dict")
        if isinstance(inner, dict):
            return inner
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise SystemExit("unrecognized checkpoint: expected a DCE-Net state dict")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to a DCE-Net .pth")
    parser.add_argument("output", help="path to write .safetensors")
    parser.add_argument("--half", action="store_true", help="store weights as float16")
    args = parser.parse_args()

    state = extract_state_dict(load_checkpoint(args.input))
    if "e_conv1.weight" not in state:
        print("warning: 'e_conv1.weight' not found; keys do not look like DCE-Net", file=sys.stderr)

    dtype = torch.float16 if args.half else torch.float32
    tensors = {name: value.to(dtype).contiguous().clone()
               for name, value in state.items() if value.is_floating_point()}
    save_file(tensors, args.output)
    print(f"wrote {len(tensors)} tensors ({dtype}) to {args.output}")


if __name__ == "__main__":
    main()
