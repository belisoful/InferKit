#!/usr/bin/env python3
"""Convert a SNAC (Multi-Scale Neural Audio Codec) checkpoint to safetensors for InferKitMLX.

InferKitMLX also reads the released checkpoint directly (its native torch reader, InferKit 0.3.0), so
this converter is optional: it remains the offline path for a portable safetensors file, and the byte
oracle the native reader is held to.

The released `hubertsiuzdak/snac_24khz` weights are a torch state dict whose convolutions are
weight-normalized through torch's parametrization API (`…parametrizations.weight.original0` = g,
`original1` = v). This tool writes the state dict to safetensors in PyTorch Conv1d layout, keeping those
pairs; InferKitMLX's `NFKMLXSNAC` loader fuses `g·v/‖v‖`, transposes to MLX's layout (depthwise and
transposed convolutions included), and remaps the reference's nested `nn.Sequential` names.

`torch.load` reads the checkpoint with torch alone — the `snac` package is the parity oracle's
dependency, not the converter's.

Usage:
    python convert.py pytorch_model.bin --list-keys
    python convert.py pytorch_model.bin snac.safetensors

Requires: torch, safetensors.
"""

import argparse
import sys

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="the released SNAC checkpoint (pytorch_model.bin or .safetensors)")
    parser.add_argument("output", nargs="?", help="path to write the safetensors file")
    parser.add_argument("--list-keys", action="store_true", help="print the state-dict keys and exit")
    args = parser.parse_args()

    if args.input.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(args.input)
    else:
        blob = torch.load(args.input, map_location="cpu", weights_only=False)
        state = blob.get("state_dict", blob) if isinstance(blob, dict) else blob

    if args.list_keys:
        for key in sorted(state):
            print(key, tuple(state[key].shape))
        return 0
    if not args.output:
        raise SystemExit("an output path is required unless --list-keys is given")

    tensors = {key: value.contiguous() for key, value in state.items() if torch.is_tensor(value)}
    save_file(tensors, args.output, metadata={"model": "snac", "format": "pytorch-layout"})
    print(f"wrote {args.output}: {len(tensors)} tensors", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
