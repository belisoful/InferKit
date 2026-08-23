#!/usr/bin/env python3
"""Convert a BiSeNet checkpoint to safetensors for InferKitMLX.

InferKitMLX's `NFKMLXBiSeNet` loads a safetensors checkpoint; MLX's `loadArrays` reads safetensors/npz,
not PyTorch `.pth`. This tool rewrites a checkpoint into safetensors, preserving PyTorch convolution
layout `[out, in, kH, kW]` (the Swift loader transposes to MLX's channels-last layout at load).

The reference BiSeNet checkpoint uses a ResNet/Xception context path (`cp.*`), a spatial path (`sp.*`),
attention refinement modules (`arm16` / `arm32`), and a feature-fusion module (`ffm.*`). InferKitMLX's
`NFKMLXBiSeNetNet` uses a compact strided-convolution backbone and groups the paths as `spatial.N` /
`context.N`, so the names do not line up one-to-one. Aligning them is a validation-sweep task; pass
`--list-keys` to print the checkpoint's names.

Usage:
    python convert.py bisenet_cityscapes.pth --list-keys
    python convert.py bisenet.pth bisenet.safetensors           # after a name remap is worked out

Get the weights from a BiSeNet release (e.g. CoinCheung/BiSeNet).

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
    raise SystemExit("unrecognized checkpoint: expected a BiSeNet state dict")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to a BiSeNet .pth")
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
               for name, value in state.items()
               if value.is_floating_point() and not name.endswith("num_batches_tracked")}
    save_file(tensors, args.output)
    print(f"wrote {len(tensors)} tensors ({dtype}) to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
