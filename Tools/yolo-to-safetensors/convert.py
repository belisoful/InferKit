#!/usr/bin/env python3
"""Convert a YOLO checkpoint to safetensors for InferKitMLX.

InferKitMLX's `NFKMLXYOLO` loads a safetensors checkpoint; MLX's `loadArrays` reads safetensors/npz,
not PyTorch `.pt`. This tool rewrites a checkpoint into safetensors, preserving PyTorch convolution
layout `[out, in, kH, kW]` (the Swift loader transposes to MLX's channels-last layout at load).

The reference Ultralytics YOLOv8 checkpoint nests a CSPDarknet backbone, a PAN-FPN neck, and a
decoupled multi-scale head under `model.model.N.*`, with distribution-focal box regression.
InferKitMLX's `NFKMLXYOLONet` is an anchor-free single-scale detector with a strided-convolution
backbone, so its parameter names do not line up with the reference one-to-one. Aligning them (and the
multi-scale / DFL decode) is a validation-sweep task; pass `--list-keys` to print the checkpoint's
names.

Usage:
    python convert.py yolov8n.pt --list-keys
    python convert.py yolov8n.pt yolo.safetensors               # after a name remap is worked out

Get the weights from Ultralytics:
    https://github.com/ultralytics/ultralytics

Requires: torch, safetensors  (pip install torch safetensors)
"""

import argparse
import sys

import torch
from safetensors.torch import save_file


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint):
    # Ultralytics wraps the model object under 'model'; take its state_dict.
    if isinstance(checkpoint, dict):
        model = checkpoint.get("model")
        if hasattr(model, "state_dict"):
            return model.state_dict()
        inner = checkpoint.get("state_dict")
        if isinstance(inner, dict):
            return inner
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()
    raise SystemExit("unrecognized checkpoint: expected a YOLO checkpoint")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to a YOLO .pt")
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
               for name, value in state.items() if value.is_floating_point()}
    save_file(tensors, args.output)
    print(f"wrote {len(tensors)} tensors ({dtype}) to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
