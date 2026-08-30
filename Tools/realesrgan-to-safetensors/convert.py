#!/usr/bin/env python3
"""Convert a Real-ESRGAN RRDBNet .pth checkpoint to safetensors for InferKitMLX.

InferKitMLX also reads the raw checkpoint directly (its native reader, InferKit 0.3.0), so
this converter is optional: it remains the offline path for producing a portable
safetensors file.

InferKitMLX's `NFKMLXRealESRGAN` loads a safetensors checkpoint; MLX's `loadArrays` reads
safetensors/npz, not PyTorch `.pth`. This tool rewrites the official release into safetensors,
preserving the reference RRDBNet parameter names (`conv_first.*`, `body.N.rdbM.convK.*`, `conv_last.*`)
and PyTorch convolution layout `[out, in, kH, kW]`. The Swift loader transposes convolution weights to
MLX's channels-last layout at load, so this tool does no transposition.

Usage:
    python convert.py RealESRGAN_x4plus.pth RealESRGAN_x4plus.safetensors
    python convert.py RealESRGAN_x4plus.pth out.safetensors --half   # store float16

Get the weights (about 67 MB) from the official release:
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

Requires: torch, safetensors  (pip install torch safetensors)
"""

import argparse
import sys

import torch
from safetensors.torch import save_file


def extract_state_dict(checkpoint):
    """Return the tensor state dict. BasicSR nests it under 'params_ema' (preferred) or 'params'."""
    if isinstance(checkpoint, dict):
        for key in ("params_ema", "params"):
            inner = checkpoint.get(key)
            if isinstance(inner, dict):
                return inner
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise SystemExit("unrecognized checkpoint: expected a 'params_ema'/'params' dict or a raw state dict")


def load_checkpoint(path):
    """Load with weights_only when available (safer), falling back for older checkpoints."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # weights_only unsupported (old torch) or the pickle carries non-tensor objects.
        return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to RealESRGAN_x4plus.pth")
    parser.add_argument("output", help="path to write .safetensors")
    parser.add_argument("--half", action="store_true",
                        help="store weights as float16 (smaller; feed float16 input to match)")
    args = parser.parse_args()

    state = extract_state_dict(load_checkpoint(args.input))
    if "conv_first.weight" not in state:
        print("warning: 'conv_first.weight' not found; keys do not look like RRDBNet", file=sys.stderr)

    dtype = torch.float16 if args.half else torch.float32
    tensors = {name: value.to(dtype).contiguous().clone() for name, value in state.items()}
    save_file(tensors, args.output)
    print(f"wrote {len(tensors)} tensors ({dtype}) to {args.output}")


if __name__ == "__main__":
    main()
