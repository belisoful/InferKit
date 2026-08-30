#!/usr/bin/env python3
"""Convert a Demucs checkpoint to safetensors for InferKitMLX.

InferKitMLX also reads the raw checkpoint directly (its native reader, InferKit 0.3.0), so
this converter is optional: it remains the offline path for producing a portable
safetensors file.

MLX loads safetensors/npz, not PyTorch `.th`/`.pt`. This tool extracts the model state and writes
safetensors; the Swift loader transposes the Conv1d / transposed-conv weights and translates the
reference key names in `remapReferenceKey`, so no renaming happens here.

Both generations convert through this one path, because both store their tensors under `state`:

- Demucs v2 and the speech denoiser → `NFKMLXDemucs` / `NFKMLXDenoiser` (the time-domain U-Net,
  bidirectional bottleneck included).
- Demucs v4 (`htdemucs`) → `NFKMLXHTDemucs` (the hybrid transformer model, 533 tensors).

`--list-keys` prints the checkpoint keys and shapes, which is how a new release's geometry is read off
rather than guessed.

Usage:
    python convert.py demucs.th demucs.safetensors
    python convert.py htdemucs.th htdemucs.safetensors
    python convert.py demucs.th --list-keys

Weights: https://github.com/facebookresearch/demucs

Requires: torch, safetensors
"""

import argparse
import sys

import torch
from safetensors.torch import save_file


def load_state_dict(path):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Released Demucs checkpoints pickle their training configuration alongside the tensors, which
        # torch >= 2.6 refuses to unpickle under the safe default.
        obj = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state", "state_dict", "model"):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            obj = obj[key]
            break
    if not (isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values())):
        raise SystemExit("unrecognized checkpoint: expected a state dict of tensors")
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to a Demucs checkpoint (v2, v4, or the denoiser)")
    parser.add_argument("output", nargs="?", help="path to write .safetensors")
    parser.add_argument("--half", action="store_true", help="store weights as float16")
    parser.add_argument("--list-keys", action="store_true", help="print keys and exit")
    args = parser.parse_args()

    state = load_state_dict(args.input)
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
