#!/usr/bin/env python3
"""Convert a fast-neural-style TransformerNet .pth checkpoint to safetensors for InferKitMLX.

InferKitMLX's `NFKMLXStyleTransfer` loads a safetensors checkpoint; MLX's `loadArrays` reads
safetensors/npz, not PyTorch `.pth`. This tool rewrites a trained style checkpoint into safetensors,
preserving the reference `TransformerNet` parameter names (`conv1.conv2d.*`, `in1.*`,
`res1.conv1.conv2d.*`, `deconv3.conv2d.*`) and PyTorch convolution layout `[out, in, kH, kW]`. The
Swift loader transposes convolution weights to MLX's channels-last layout at load, so this tool does
no transposition.

The released checkpoints (pytorch/examples fast_neural_style) were saved with an older InstanceNorm
that tracked running statistics; the current model uses affine-only InstanceNorm, so this tool drops
the deprecated `running_mean` / `running_var` / `num_batches_tracked` keys.

Usage:
    python convert.py mosaic.pth mosaic.safetensors
    python convert.py candy.pth candy.safetensors --half        # store float16

Get the weights from the pytorch/examples fast_neural_style saved models:
    https://github.com/pytorch/examples/tree/main/fast_neural_style

Requires: torch, safetensors  (pip install torch safetensors)
"""

import argparse
import sys

import torch
from safetensors.torch import save_file

DEPRECATED_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


def extract_state_dict(checkpoint):
    """Return the tensor state dict. TransformerNet checkpoints are a raw state dict, but tolerate a
    common nesting under 'state_dict'."""
    if isinstance(checkpoint, dict):
        inner = checkpoint.get("state_dict")
        if isinstance(inner, dict):
            return inner
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise SystemExit("unrecognized checkpoint: expected a TransformerNet state dict")


def load_checkpoint(path):
    """Load with weights_only when available (safer), falling back for older checkpoints."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to a TransformerNet .pth")
    parser.add_argument("output", help="path to write .safetensors")
    parser.add_argument("--half", action="store_true",
                        help="store weights as float16 (smaller; feed float16 input to match)")
    args = parser.parse_args()

    state = extract_state_dict(load_checkpoint(args.input))
    if "conv1.conv2d.weight" not in state:
        print("warning: 'conv1.conv2d.weight' not found; keys do not look like TransformerNet", file=sys.stderr)

    dropped = [name for name in state if name.endswith(DEPRECATED_SUFFIXES)]
    for name in dropped:
        del state[name]
    if dropped:
        print(f"dropped {len(dropped)} deprecated InstanceNorm running-stats tensors", file=sys.stderr)

    dtype = torch.float16 if args.half else torch.float32
    tensors = {name: value.to(dtype).contiguous().clone() for name, value in state.items()}
    save_file(tensors, args.output)
    print(f"wrote {len(tensors)} tensors ({dtype}) to {args.output}")


if __name__ == "__main__":
    main()
