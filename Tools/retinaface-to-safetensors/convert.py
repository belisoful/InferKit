#!/usr/bin/env python3
"""Convert a RetinaFace .pth release to safetensors for NFKMLXRetinaFace.

Names pass through unchanged: the positional `nn.Sequential` remap lives in Swift
(`NFKMLXRetinaFace.remapReferenceKey`), where the module's own layout is known.

`num_batches_tracked` counters and the backbone's unused ImageNet classifier (`body.fc`, `body.avg`)
are dropped — they are not parameters of the detector.

    python3 convert.py detection_mobilenet0.25_Final.pth retinaface.safetensors [--list-keys]
"""

import argparse
import sys

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--list-keys", action="store_true", help="print the checkpoint's keys and exit")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    # A release trained under DataParallel carries a `module.` prefix.
    state = {key[len("module."):] if key.startswith("module.") else key: value
             for key, value in state.items()}

    if args.list_keys:
        for key, value in state.items():
            print(f"{key}\t{tuple(value.shape)}")
        return 0

    if not args.output:
        parser.error("an output path is required unless --list-keys is given")

    dropped = 0
    tensors = {}
    for key, value in state.items():
        if key.endswith("num_batches_tracked") or key.startswith(("body.fc", "body.avg")):
            dropped += 1
            continue
        tensors[key] = value.contiguous().to(torch.float32)

    save_file(tensors, args.output, metadata={"format": "pt"})
    print(f"wrote {args.output}: {len(tensors)} tensors ({dropped} dropped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
