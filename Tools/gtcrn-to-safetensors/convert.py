#!/usr/bin/env python3
"""Convert a GTCRN checkpoint to safetensors for InferKitMLX.

InferKitMLX also reads the released `.tar`/`.pth` directly (its native torch reader, InferKit 0.3.0),
so this converter is optional: it remains the offline path for producing a portable safetensors file,
and the `--list-keys` output is what confirms the module key layout the Swift loader targets.

The released weights (`Xiaobin-Rong/gtcrn`, e.g. `checkpoints/model_trained_on_dns3.tar`) hold the GTCRN
state dict, possibly under a `model` key. This tool extracts it and writes it to safetensors in PyTorch
layout (4-D convolution weights stay `[out, in, kH, kW]`); InferKitMLX's `NFKMLXGTCRN` loader folds the
GRUs and transposes the convolution weights at load.

`torch.load` reads the checkpoint with torch alone; the `gtcrn` package is the parity oracle's
dependency, not the converter's.

Usage:
    python convert.py model_trained_on_dns3.tar --list-keys
    python convert.py model_trained_on_dns3.tar gtcrn.safetensors

Requires: torch, safetensors.
"""

import argparse
import sys

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="the released GTCRN checkpoint")
    parser.add_argument("output", nargs="?", help="path to write the safetensors file")
    parser.add_argument("--list-keys", action="store_true", help="print the state-dict keys and exit")
    args = parser.parse_args()

    blob = torch.load(args.input, map_location="cpu", weights_only=False)
    state = blob.get("model", blob) if isinstance(blob, dict) else blob

    if args.list_keys:
        for key in sorted(state):
            print(key, tuple(state[key].shape))
        return 0
    if not args.output:
        raise SystemExit("an output path is required unless --list-keys is given")

    tensors = {key: value.contiguous() for key, value in state.items() if torch.is_tensor(value)}
    save_file(tensors, args.output, metadata={"model": "gtcrn", "format": "pytorch-layout"})
    print(f"wrote {args.output}: {len(tensors)} tensors", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
