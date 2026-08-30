#!/usr/bin/env python3
"""Convert a Segment Anything (SAM) checkpoint to safetensors for InferKitMLX (NFKMLXSAM).

InferKitMLX also reads the raw checkpoint directly (its native reader, InferKit 0.3.0), so
this converter is optional: it remains the offline path for producing a portable
safetensors file.

MLX loads safetensors/npz, not PyTorch `.pth`. This tool extracts the state dict and writes safetensors;
the Swift loader transposes 4-D convolution weights.

InferKitMLX's SAM simplifies the image encoder (global attention with absolute positions — the
reference adds windowed attention and decomposed relative-position embeddings `rel_pos_h`/`rel_pos_w`)
and uses flat neck / decoder names rather than the reference `Sequential` nesting. So a real SAM
checkpoint needs a `remap`, and the relative-position / windowing weights are unused — a deeper
validation-sweep task. `--list-keys` prints the checkpoint keys to build the map.

Usage:
    python convert.py sam_vit_b.pth sam.safetensors
    python convert.py sam_vit_b.pth --list-keys

Weights: https://github.com/facebookresearch/segment-anything (and SAM 2:
https://github.com/facebookresearch/sam2 — its Hiera encoder needs its own module).

Requires: torch, safetensors
"""

import argparse
import sys

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to a SAM .pth")
    parser.add_argument("output", nargs="?", help="path to write .safetensors")
    parser.add_argument("--half", action="store_true", help="store weights as float16")
    parser.add_argument("--list-keys", action="store_true", help="print keys and exit")
    args = parser.parse_args()

    obj = torch.load(args.input, map_location="cpu")
    state = obj.get("model", obj) if isinstance(obj, dict) else obj
    if not (isinstance(state, dict) and all(isinstance(v, torch.Tensor) for v in state.values())):
        raise SystemExit("unrecognized checkpoint: expected a SAM state dict")

    if args.list_keys:
        for key in sorted(state):
            print(f"{key}\t{tuple(state[key].shape)}")
        sys.exit(0)
    if not args.output:
        raise SystemExit("output path required (or pass --list-keys)")

    dtype = torch.float16 if args.half else torch.float32
    tensors = {name: value.to(dtype).contiguous().clone() for name, value in state.items()}
    save_file(tensors, args.output)
    print(f"wrote {len(tensors)} tensors ({dtype}) to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
