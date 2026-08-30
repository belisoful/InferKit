#!/usr/bin/env python3
"""Convert a Robust Video Matting checkpoint to safetensors for InferKitMLX.

InferKitMLX also reads the raw checkpoint directly (its native reader, InferKit 0.3.0), so
this converter is optional: it remains the offline path for producing a portable
safetensors file.

InferKitMLX's `NFKMLXRVM` loads a safetensors checkpoint; MLX's `loadArrays` reads safetensors/npz,
not PyTorch `.pth`. This tool rewrites a checkpoint into safetensors, preserving PyTorch convolution
layout `[out, in, kH, kW]` (the Swift loader transposes to MLX's channels-last layout at load).

The reference RVM checkpoint (`rvm_mobilenetv3.pth`) uses a torchvision MobileNetV3-Large backbone
(`backbone.features.N.*`, with batch norm and squeeze-and-excitation), an LR-ASPP module, a recurrent
decoder, and a deep guided filter. InferKitMLX's `NFKMLXRVMNet` implements the recurrent-matting
architecture with a compact depthwise-separable encoder, so its parameter names do not line up with
the reference one-to-one. Aligning them (mapping the MobileNetV3 backbone and the decoder/DGF keys to
the module's names) is a validation-sweep task; pass `--list-keys` to print the checkpoint's names as a
starting point.

Usage:
    python convert.py rvm_mobilenetv3.pth --list-keys
    python convert.py rvm_mobilenetv3.pth rvm.safetensors      # after a name remap is worked out

Get the weights from the Robust Video Matting release:
    https://github.com/PeterL1n/RobustVideoMatting

Requires: torch, safetensors  (pip install torch safetensors)
"""

import argparse
import sys

import torch
from safetensors.torch import save_file


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        inner = checkpoint.get("state_dict")
        if isinstance(inner, dict):
            return inner
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise SystemExit("unrecognized checkpoint: expected an RVM state dict")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to rvm_mobilenetv3.pth")
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
