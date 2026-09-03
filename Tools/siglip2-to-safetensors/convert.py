#!/usr/bin/env python3
"""Normalize a SigLIP 2 checkpoint to safetensors for InferKitMLX.

The released `google/siglip2-base-patch16-224` weights are ALREADY a safetensors in PyTorch layout, and
InferKitMLX's `NFKMLXSigLIP2` loader reads them directly (transposing the 4-D patch convolution and
mapping the `vision_model.`/`text_model.` prefixes at load), so no conversion is needed to consume the
model — point the loader at the released file. This tool is the offline path that re-saves the tensors
into a clean safetensors (dropping any non-tensor metadata), for a store that keeps one converted file
per model.

Usage:
    python convert.py model.safetensors siglip2.safetensors

Requires: torch, safetensors.
"""

import argparse
import sys

from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="the released SigLIP 2 model.safetensors")
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
              metadata={"model": "siglip2-base-patch16-224", "format": "pytorch-layout"})
    print(f"wrote {args.output}: {len(tensors)} tensors", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
