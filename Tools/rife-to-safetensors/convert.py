#!/usr/bin/env python3
"""Convert a RIFE (HDv3) flownet checkpoint to safetensors for InferKitMLX (NFKMLXRIFE).

InferKitMLX also reads the raw checkpoint directly (its native reader, InferKit 0.3.0), so
this converter is optional: it remains the offline path for producing a portable
safetensors file.

MLX loads safetensors/npz, not PyTorch `.pkl`/`.pth`. This tool rewrites the IFNet weights into
safetensors (the Swift loader transposes 4-D convolution weights) and renames the reference's nested
`Sequential` names to the module's keys:

    module.               -> (stripped)
    block{N}.             -> blocks.{N}.
    conv0.{i}.0 / .1      -> conv0.{i}.conv / .prelu       (Conv2d / PReLU in each conv Sequential)
    convblock.{i}.0 / .1  -> convblock.{i}.conv / .prelu
    lastconv              -> lastconv                       (unchanged)

The teacher block (`block_tea`) and any refine net are dropped — inference needs only the three IFBlocks.
RIFE has several incompatible versions; this targets HDv3 (three blocks, c = 240/150/90). Confirm the
block count / channels match `NFKMLXRIFENet` for the checkpoint you use.

Usage:
    python convert.py flownet.pkl rife.safetensors

Weights: https://github.com/megvii-research/ECCV2022-RIFE  (RIFE HDv3 `flownet.pkl`)

Requires: torch, safetensors
"""

import argparse
import re
import sys

import torch
from safetensors.torch import save_file


def rename(key):
    if key.startswith("module."):
        key = key[len("module."):]
    key = re.sub(r"^block(\d+)\.", r"blocks.\1.", key)
    key = re.sub(r"\bconv0\.(\d+)\.0\.", r"conv0.\1.conv.", key)
    key = re.sub(r"\bconv0\.(\d+)\.1\.", r"conv0.\1.prelu.", key)
    key = re.sub(r"\bconvblock\.(\d+)\.0\.", r"convblock.\1.conv.", key)
    key = re.sub(r"\bconvblock\.(\d+)\.1\.", r"convblock.\1.prelu.", key)
    return key


def load_state_dict(path):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise SystemExit("unrecognized checkpoint")
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to RIFE flownet.pkl / .pth")
    parser.add_argument("output", help="path to write .safetensors")
    parser.add_argument("--half", action="store_true", help="store weights as float16")
    args = parser.parse_args()

    state = load_state_dict(args.input)
    dtype = torch.float16 if args.half else torch.float32
    tensors = {}
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        stripped = name[len("module."):] if name.startswith("module.") else name
        if stripped.startswith("block_tea") or stripped.startswith("contextnet") or stripped.startswith("unet"):
            continue                                            # teacher / refine nets are not used at inference
        tensors[rename(name)] = value.to(dtype).contiguous().clone()

    save_file(tensors, args.output)
    print(f"wrote {len(tensors)} tensors ({dtype}) to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
