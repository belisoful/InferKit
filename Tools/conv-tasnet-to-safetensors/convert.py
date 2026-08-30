#!/usr/bin/env python3
"""Convert a Conv-TasNet checkpoint to safetensors for InferKitMLX.

InferKitMLX also reads the raw checkpoint directly (its native reader, InferKit 0.3.0), so
this converter is optional: it remains the offline path for producing a portable
safetensors file.

InferKitMLX's `NFKMLXConvTasNet` loads a safetensors checkpoint; MLX's `loadArrays` reads
safetensors/npz, not PyTorch `.pth`. This tool rewrites a checkpoint into safetensors, preserving
PyTorch 1-D convolution layout `[out, in, k]` (the Swift loader transposes to MLX's `[out, k, in]` at
load). The transposed-convolution decoder weight is 4-D `[in, out, kH, 1]` in InferKitMLX's singleton-
width implementation; permute it to `[out, in, kH, 1]` so the loader's single 4-D transpose applies.

The reference Asteroid / conv-tasnet checkpoint nests the separator as `masker.*` /
`separator.network.*` with a `bottleneck`, a `TCN` of `SeparableConv` blocks, and `encoder` / `decoder`
1-D convolutions; the global layer norms carry `gamma` / `beta`. InferKitMLX's `NFKMLXConvTasNetNet`
groups these into `encoder` / `blocks.N` / `mask_conv` / `decoder`, so the names do not line up
one-to-one. Aligning them is a validation-sweep task; pass `--list-keys` to print the checkpoint's names.

Usage:
    python convert.py conv_tasnet.pth --list-keys
    python convert.py conv_tasnet.pth conv-tasnet.safetensors    # after a name remap is worked out

Get the weights from an Asteroid Conv-TasNet release:
    https://github.com/asteroid-team/asteroid

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
        for key in ("state_dict", "model_state_dict", "model"):
            inner = checkpoint.get(key)
            if isinstance(inner, dict):
                return inner
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise SystemExit("unrecognized checkpoint: expected a Conv-TasNet state dict")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to a Conv-TasNet .pth")
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
    tensors = {}
    for name, value in state.items():
        if not value.is_floating_point():
            continue
        # The decoder transposed conv is [in, out, k]; give it a singleton width and [out, in] order.
        if "decoder" in name and name.endswith("weight") and value.dim() == 3:
            value = value.permute(1, 0, 2).unsqueeze(-1)
        tensors[name] = value.to(dtype).contiguous().clone()
    save_file(tensors, args.output)
    print(f"wrote {len(tensors)} tensors ({dtype}) to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
