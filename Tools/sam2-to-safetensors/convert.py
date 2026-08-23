#!/usr/bin/env python3
"""Convert a SAM 2 checkpoint to safetensors for InferKitMLX (NFKMLXSAM2).

The released `.pt` wraps the state dict under a `model` key. Names pass through — the module keys
follow the reference, and the Swift loader handles the convolution transposes — so this only unwraps
and rewrites.

Usage:
    python convert.py sam2_hiera_base_plus.pt sam2-base-plus.safetensors
    python convert.py sam2_hiera_base_plus.pt --list-keys

Weights: https://dl.fbaipublicfiles.com/segment_anything_2/072824/<name>.pt
(sam2_hiera_tiny, sam2_hiera_base_plus, sam2_hiera_large).

Requires: torch, safetensors
"""

import argparse

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="the released .pt")
    parser.add_argument("output", nargs="?", help="the safetensors to write")
    parser.add_argument("--list-keys", action="store_true", help="print the checkpoint keys and exit")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = state.get("model", state)

    if args.list_keys:
        for key in sorted(state):
            print(key, tuple(state[key].shape))
        return

    if not args.output:
        parser.error("output path required unless --list-keys")
    save_file({key: value.contiguous() for key, value in state.items()}, args.output)
    print(f"wrote {args.output}: {len(state)} tensors")


if __name__ == "__main__":
    main()
