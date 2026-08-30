#!/usr/bin/env python3
"""Convert a SegFormer checkpoint to safetensors for InferKitMLX.

InferKitMLX also reads the raw checkpoint directly (its native reader, InferKit 0.3.0), so
this converter is optional: it remains the offline path for producing a portable
safetensors file.

InferKitMLX's `NFKMLXSegFormer` loads a safetensors checkpoint; MLX's `loadArrays` reads
safetensors/npz. This tool rewrites a checkpoint into safetensors, preserving PyTorch convolution
layout `[out, in, kH, kW]` (the Swift loader transposes to MLX's channels-last layout at load).

The Hugging Face SegFormer checkpoint nests the encoder as `segformer.encoder.patch_embeddings.N.*`,
`segformer.encoder.block.N.M.*` (attention `attention.self.query`/`.key`/`.value`, `attention.self.sr`,
Mix-FFN `mlp.dwconv`), stage norms `segformer.encoder.layer_norm.N`, and the decode head
`decode_head.linear_c.N`, `decode_head.linear_fuse`, `decode_head.classifier`. InferKitMLX's
`NFKMLXSegFormerNet` groups these into `stageN.*` / `linear_c.*` and fuses q/kv, so the names do not
line up one-to-one. Aligning them (and the decode-head BN) is a validation-sweep task; pass
`--list-keys` to print the checkpoint's names.

Usage:
    python convert.py pytorch_model.bin --list-keys
    python convert.py pytorch_model.bin segformer.safetensors    # after a name remap is worked out

Get the weights from a Hugging Face SegFormer model (e.g. nvidia/segformer-b0-finetuned-ade-512-512).

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
    raise SystemExit("unrecognized checkpoint: expected a SegFormer state dict")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to a SegFormer .bin/.pth")
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
