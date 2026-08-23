#!/usr/bin/env python3
"""Convert a NAFNet checkpoint to safetensors for InferKitMLX (NFKMLXNAFNet).

MLX loads safetensors/npz, not PyTorch `.pth`. This tool rewrites the release into safetensors (the
Swift loader transposes 4-D convolution weights) and renames the few attributes whose nesting differs
from the module: `middle_blks.` → `middle.`, `ups.N.0.` → `ups.N.` (the conv inside the up Sequential),
and `sca.1.` → `sca.` (the conv inside the channel-attention Sequential). Everything else
(`intro`, `ending`, `encoders.N.M`, `decoders.N.M`, `downs.N`, `conv1`…`conv5`, `norm1/2`, `beta`,
`gamma`) already matches.

Usage:
    python convert.py NAFNet-SIDD-width32.pth nafnet.safetensors
    python convert.py in.pth out.safetensors --half

Weights: https://github.com/megvii-research/NAFNet  (release .pth files store the net under `params`)

Requires: torch, safetensors
"""

import argparse
import re
import sys

import torch
from safetensors.torch import save_file


def rename(key):
    key = key.replace("middle_blks.", "middle.")
    key = re.sub(r"\bups\.(\d+)\.0\.", r"ups.\1.", key)
    key = re.sub(r"\bsca\.1\.", "sca.", key)
    return key


def load_state_dict(path):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for name in ("params", "state_dict", "model"):
            inner = obj.get(name)
            if isinstance(inner, dict):
                obj = inner
                break
    if not (isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values())):
        raise SystemExit("unrecognized checkpoint: expected a state dict of tensors")
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to the NAFNet .pth")
    parser.add_argument("output", help="path to write .safetensors")
    parser.add_argument("--half", action="store_true", help="store weights as float16")
    args = parser.parse_args()

    state = load_state_dict(args.input)
    dtype = torch.float16 if args.half else torch.float32
    tensors = {rename(name): value.to(dtype).contiguous().clone() for name, value in state.items()}
    if len(tensors) != len(state):
        print("warning: a rename collided; some keys were dropped", file=sys.stderr)
    save_file(tensors, args.output)
    print(f"wrote {len(tensors)} tensors ({dtype}) to {args.output}")


if __name__ == "__main__":
    main()
