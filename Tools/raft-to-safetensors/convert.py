#!/usr/bin/env python3
"""Convert a RAFT checkpoint to safetensors for InferKitMLX (NFKMLXRAFT).

InferKitMLX also reads the raw checkpoint directly (its native reader, InferKit 0.3.0), so
this converter is optional: it remains the offline path for producing a portable
safetensors file.

MLX loads safetensors/npz, not PyTorch `.pth`. This tool rewrites the weights into safetensors (the
Swift loader transposes 4-D convolution weights) and renames the reference's names to the module keys:

    module.                     -> (stripped, DataParallel prefix)
    update_block.               -> update.
    …downsample.0 / .1          -> …downsample / …norm3   (Conv / Norm in the shortcut Sequential)
    …flow_head.conv1 / conv2    -> …flow_head.0 / .1
    …mask.0 / .2                -> …mask.0 / .1            (the ReLU at index 1 has no weights)

`fnet` / `cnet` / their `layer{1,2,3}.N`, the residual `conv1/2`+`norm1/2`, and the motion-encoder /
GRU names already match. The Swift module mirrors the reference's per-encoder `norm_fn`: `fnet` uses a
parameter-free InstanceNorm (the checkpoint carries nothing for it) and `cnet` uses BatchNorm (weight,
bias, and running statistics, all carried through here). RAFT has version differences; confirm the
loaded key set in the sweep.

Usage:
    python convert.py raft-things.pth raft.safetensors

Weights: https://github.com/princeton-vl/RAFT

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
    key = key.replace("update_block.", "update.")
    key = re.sub(r"\bdownsample\.0\.", "downsample.", key)
    key = re.sub(r"\bdownsample\.1\.", "norm3.", key)
    key = key.replace("flow_head.conv1.", "flow_head.0.").replace("flow_head.conv2.", "flow_head.1.")
    key = re.sub(r"\bmask\.2\.", "mask.1.", key)
    return key


def load_state_dict(path):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        obj = torch.load(path, map_location="cpu")
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    if not (isinstance(state, dict)):
        raise SystemExit("unrecognized checkpoint")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to the RAFT .pth")
    parser.add_argument("output", help="path to write .safetensors")
    parser.add_argument("--half", action="store_true", help="store weights as float16")
    args = parser.parse_args()

    state = load_state_dict(args.input)
    dtype = torch.float16 if args.half else torch.float32
    tensors = {}
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        tensors[rename(name)] = value.to(dtype).contiguous().clone()
    save_file(tensors, args.output)
    print(f"wrote {len(tensors)} tensors ({dtype}) to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
