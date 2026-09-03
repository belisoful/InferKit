#!/usr/bin/env python3
"""Combine a TAESD encoder + decoder checkpoint into one safetensors for InferKitMLX.

The released TAESD weights are TWO files (`taesd_encoder.pth`, `taesd_decoder.pth`), each a plain
`nn.Sequential` state dict with numeric keys. This tool loads both and writes a single safetensors with
`encoder.N.*` / `decoder.N.*` keys in PyTorch Conv2d layout; InferKitMLX's `NFKMLXTAESD` loader transposes
the 4-D convolution weights to MLX's layout at load. The `[Module]`-array layout makes the numeric
Sequential keys match with no name remap.

The `input` argument is the encoder `.pth`; the decoder is taken from its sibling `taesd_decoder.pth`, or
downloaded from `madebyollin/taesd` when a sibling is not present (so `fetch.py`, which downloads only the
encoder, still produces a complete file).

Usage:
    python convert.py taesd_encoder.pth taesd.safetensors

Requires: torch, safetensors (huggingface_hub only when the decoder must be downloaded).
"""

import argparse
import os
import sys

import torch
from safetensors.torch import save_file


def decoder_path(encoder_path):
    sibling = os.path.join(os.path.dirname(encoder_path), "taesd_decoder.pth")
    if os.path.exists(sibling):
        return sibling
    from huggingface_hub import hf_hub_download
    return hf_hub_download("madebyollin/taesd", "taesd_decoder.pth")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="the released taesd_encoder.pth")
    parser.add_argument("output", nargs="?", help="path to write the safetensors file")
    parser.add_argument("--list-keys", action="store_true", help="print the combined keys and exit")
    args = parser.parse_args()

    encoder = torch.load(args.input, map_location="cpu", weights_only=True)
    decoder = torch.load(decoder_path(args.input), map_location="cpu", weights_only=True)
    combined = {f"encoder.{k}": v.contiguous() for k, v in encoder.items()}
    combined.update({f"decoder.{k}": v.contiguous() for k, v in decoder.items()})

    if args.list_keys:
        for key in sorted(combined):
            print(key, tuple(combined[key].shape))
        return 0
    if not args.output:
        raise SystemExit("an output path is required unless --list-keys is given")

    save_file(combined, args.output, metadata={"model": "taesd", "format": "pytorch-layout"})
    print(f"wrote {args.output}: {len(combined)} tensors", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
