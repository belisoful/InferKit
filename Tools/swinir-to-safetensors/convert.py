#!/usr/bin/env python3
"""Convert a SwinIR checkpoint to safetensors for InferKitMLX.

InferKitMLX also reads the raw checkpoint directly (its native reader, InferKit 0.3.0), so
this converter is optional: it remains the offline path for producing a portable
safetensors file.

InferKitMLX's `NFKMLXSwinIR` loads a safetensors checkpoint; MLX's `loadArrays` reads safetensors/npz,
not PyTorch `.pth`. This tool rewrites a checkpoint into safetensors, preserving PyTorch convolution
layout `[out, in, kH, kW]` (the Swift loader transposes to MLX's channels-last layout at load).

The reference SwinIR checkpoint nests each block under `layers.N.residual_group.blocks.M.*` (attention
`attn.qkv`, `attn.relative_position_bias_table`, MLP `mlp.fc1`/`fc2`), a per-layer `layers.N.conv`,
then `conv_after_body`, `conv_before_upsample.0`, `upsample.N`, and `conv_last`; it also stores a
`relative_position_index` buffer per block (InferKitMLX recomputes this, so it is dropped). InferKitMLX's
`NFKMLXSwinIRNet` groups blocks as `layers.N.blocks.M.*` with `mlp_fc1`/`mlp_fc2`, so the names do not
line up one-to-one. Aligning them (and non-power-of-two upsampling) is a validation-sweep task; pass
`--list-keys` to print the checkpoint's names.

Usage:
    python convert.py 001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth --list-keys
    python convert.py swinir.pth swinir.safetensors             # after a name remap is worked out

Get the weights from the SwinIR release:
    https://github.com/JingyunLiang/SwinIR

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
        for key in ("params", "params_ema", "state_dict", "model"):
            inner = checkpoint.get(key)
            if isinstance(inner, dict):
                return inner
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise SystemExit("unrecognized checkpoint: expected a SwinIR state dict")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to a SwinIR .pth")
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

    # relative_position_index is a recomputed integer buffer, not a learned parameter.
    dtype = torch.float16 if args.half else torch.float32
    tensors = {name: value.to(dtype).contiguous().clone()
               for name, value in state.items()
               if value.is_floating_point() and "relative_position_index" not in name}
    save_file(tensors, args.output)
    print(f"wrote {len(tensors)} tensors ({dtype}) to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
