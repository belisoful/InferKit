#!/usr/bin/env python3
"""Convert a LaMa checkpoint to safetensors for InferKitMLX (NFKMLXLaMa).

InferKitMLX also reads the raw checkpoint directly (its native reader, InferKit 0.3.0), so
this converter is optional: it remains the offline path for producing a portable
safetensors file.

The big-lama release is a training checkpoint whose generator weights sit under `generator.*` in a
flat `nn.Sequential` (`generator.model.N.*`). This tool extracts the generator tensors and writes
safetensors, preserving names and PyTorch convolution layout `[out, in, kH, kW]`. The Swift loader
transposes convolution weights to MLX's channels-last layout.

The module uses descriptive names (`init_conv`, `down.N`, `blocks.N`, `up.N`) rather than the flat
`model.N` index, so map the reference names with the `remap` on `NFKMLXLaMa.loadWeights`. This is a
validation-sweep task; `--list-keys` prints the checkpoint's generator keys to build that map.

Usage:
    python convert.py big-lama.ckpt big-lama.safetensors
    python convert.py big-lama.ckpt --list-keys

Weights: https://huggingface.co/smartywu/big-lama  (or the official release)

Requires: torch, safetensors
"""

import argparse
import sys

import torch
from safetensors.torch import save_file


def load_generator(path):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # torch >= 2.6 defaults weights_only to True, so the fallback must say so explicitly;
        # a bare call here would repeat the attempt that just failed.
        obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    if not isinstance(state, dict):
        raise SystemExit("unrecognized checkpoint")
    generator = {k[len("generator."):]: v for k, v in state.items()
                 if k.startswith("generator.") and isinstance(v, torch.Tensor)}
    if not generator:
        # Some exports store the generator state directly.
        generator = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
    if not generator:
        raise SystemExit("no generator tensors found")
    return generator


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to the LaMa .ckpt/.pt")
    parser.add_argument("output", nargs="?", help="path to write .safetensors")
    parser.add_argument("--half", action="store_true", help="store weights as float16")
    parser.add_argument("--list-keys", action="store_true", help="print generator keys and exit")
    args = parser.parse_args()

    generator = load_generator(args.input)
    if args.list_keys:
        for key in sorted(generator):
            print(f"{key}\t{tuple(generator[key].shape)}")
        sys.exit(0)
    if not args.output:
        raise SystemExit("output path required (or pass --list-keys)")

    dtype = torch.float16 if args.half else torch.float32
    tensors = {name: value.to(dtype).contiguous().clone() for name, value in generator.items()}
    save_file(tensors, args.output)
    print(f"wrote {len(tensors)} generator tensors ({dtype}) to {args.output}")


if __name__ == "__main__":
    main()
