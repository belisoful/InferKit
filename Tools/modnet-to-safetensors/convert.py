#!/usr/bin/env python3
"""Convert a MODNet checkpoint to safetensors for InferKitMLX.

InferKitMLX's `NFKMLXMODNet` loads a safetensors checkpoint; MLX's `loadArrays` reads safetensors/npz,
not PyTorch `.pth`. This tool rewrites a checkpoint into safetensors, preserving PyTorch convolution
layout `[out, in, kH, kW]` (the Swift loader transposes to MLX's channels-last layout at load).

The reference MODNet checkpoint (`modnet_photographic_portrait_matting.ckpt`) wraps a MobileNetV2
backbone (`backbone.model.*`) and the low-resolution / high-resolution / fusion branches
(`lr_branch.*`, `hr_branch.*`, `f_branch.*`), and is often nested under `state_dict` with a `module.`
prefix from DataParallel. InferKitMLX's `NFKMLXMODNetNet` implements the three-branch structure with a
compact depthwise-separable encoder, so its parameter names do not line up with the reference
one-to-one. Aligning them is a validation-sweep task; pass `--list-keys` to print the checkpoint's
names.

Usage:
    python convert.py modnet_photographic_portrait_matting.ckpt --list-keys
    python convert.py modnet.ckpt modnet.safetensors            # after a name remap is worked out

Get the weights from the MODNet release:
    https://github.com/ZHKKKe/MODNet

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
            checkpoint = inner
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            # Strip a DataParallel 'module.' prefix if present.
            return {name[len("module."):] if name.startswith("module.") else name: value
                    for name, value in checkpoint.items()}
    raise SystemExit("unrecognized checkpoint: expected a MODNet state dict")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to a MODNet .ckpt/.pth")
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
