#!/usr/bin/env python3
"""Convert a U²-Net checkpoint to safetensors for InferKitMLX (NFKMLXU2Net).

MLX loads safetensors/npz, not PyTorch `.pth`. This tool rewrites the release into safetensors,
transposing nothing (the Swift loader transposes 4-D convolution weights) but **renaming** each RSU
block's `rebnconvN` / `rebnconvNd` convolutions to the module's `enc.<i>` / `dec.<j>` array keys, so
the file loads directly. Stage, side, and `outconv` names are preserved.

Usage:
    python convert.py u2net.pth u2net.safetensors
    python convert.py u2netp.pth u2netp.safetensors --half

Weights: https://github.com/xuebinqin/U-2-Net  (u2net.pth ~176 MB, u2netp.pth ~4.7 MB)

Requires: torch, safetensors
"""

import argparse
import sys

import torch
from safetensors.torch import save_file

# RSU height per stage (encoder and decoder stages are each an RSU).
HEIGHTS = {
    "stage1": 7, "stage2": 6, "stage3": 5, "stage4": 4, "stage5": 4, "stage6": 4,
    "stage5d": 4, "stage4d": 4, "stage3d": 5, "stage2d": 6, "stage1d": 7,
}


def rename(key):
    """stageX.rebnconvN(.…) -> stageX.enc.(N-1)(.…); stageX.rebnconvNd -> stageX.dec.(H-1-N)."""
    parts = key.split(".")
    stage = parts[0]
    if stage not in HEIGHTS or len(parts) < 2:
        return key
    inner = parts[1]
    height = HEIGHTS[stage]
    if inner == "rebnconvin":
        return key
    if inner.startswith("rebnconv") and inner.endswith("d"):
        index = int(inner[len("rebnconv"):-1])
        new_inner = f"dec.{height - 1 - index}"
    elif inner.startswith("rebnconv"):
        index = int(inner[len("rebnconv"):])
        new_inner = f"enc.{index - 1}"
    else:
        return key
    return ".".join([stage, new_inner] + parts[2:])


def load_state_dict(path):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        obj = torch.load(path, map_location="cpu")
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    if not (isinstance(state, dict) and all(isinstance(v, torch.Tensor) for v in state.values())):
        raise SystemExit("unrecognized checkpoint: expected a state dict of tensors")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to u2net.pth / u2netp.pth")
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
