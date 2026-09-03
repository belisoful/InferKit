#!/usr/bin/env python3
"""Normalize the LTX-Video VAE checkpoint to safetensors for InferKitMLX.

The released LTX-Video VAE (`Lightricks/LTX-Video`, `vae/diffusion_pytorch_model.safetensors`) is ALREADY
a PyTorch-layout safetensors, and InferKitMLX's `NFKMLXLTXVideoVAE` loader reads it directly (transposing
the 5-D Conv3d weights `[out, in, kT, kH, kW]` → MLX's `[out, kT, kH, kW, in]`; the causal-conv wrapper
keeps the reference's `.conv` key, so the names match with no remap). This tool is the offline path that
re-saves the tensors into a clean safetensors, for a store that keeps one converted file per model.

Usage:
    python convert.py diffusion_pytorch_model.safetensors ltx-vae.safetensors

Requires: torch, safetensors.
"""

import argparse
import sys

from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="the released LTX-Video VAE safetensors")
    parser.add_argument("output", nargs="?", help="path to write the safetensors file")
    parser.add_argument("--list-keys", action="store_true", help="print the tensor keys and exit")
    args = parser.parse_args()

    tensors = load_file(args.input)
    if args.list_keys:
        for key in sorted(tensors):
            print(key, tuple(tensors[key].shape))
        return 0
    if not args.output:
        raise SystemExit("an output path is required unless --list-keys is given")

    save_file({k: v.contiguous() for k, v in tensors.items()}, args.output,
              metadata={"model": "ltx-video-vae", "format": "pytorch-layout"})
    print(f"wrote {args.output}: {len(tensors)} tensors", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
