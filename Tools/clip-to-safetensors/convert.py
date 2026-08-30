#!/usr/bin/env python3
"""Convert an OpenAI CLIP checkpoint to safetensors for InferKitMLX.
InferKitMLX also reads the raw checkpoint directly (its native reader, InferKit 0.3.0), so
this converter is optional: it remains the offline path for producing a portable
safetensors file.

InferKitMLX's `NFKMLXCLIP` loads a safetensors checkpoint; MLX's `loadArrays` reads safetensors/npz.
The OpenAI CLIP release ships a TorchScript/JIT `.pt` whose weights follow the reference names
(`visual.conv1.weight`, `visual.transformer.resblocks.N.attn.in_proj_weight`, `token_embedding.weight`,
`text_projection`, `logit_scale`). This tool extracts the `state_dict` and writes safetensors,
preserving those names and the PyTorch convolution layout `[out, in, kH, kW]`. The Swift loader
transposes the 4-D patch-embedding convolution to MLX's channels-last layout at load, so this tool
does no transposition.

A Hugging Face `CLIPModel` uses different key names (`vision_model.*`, `text_model.*`); this tool
targets the original OpenAI layout. Pass `--list-keys` to print the checkpoint's parameter names for a
remap sweep against the module's expected names.

Usage:
    python convert.py ViT-B-32.pt ViT-B-32.safetensors
    python convert.py ViT-B-32.pt --list-keys

Get the weights from the OpenAI CLIP release (loaded via `clip.load`, or the raw JIT archive).

Requires: torch, safetensors  (pip install torch safetensors)
"""

import argparse
import sys

import torch
from safetensors.torch import save_file


def extract_state_dict(checkpoint):
    """Return the tensor state dict from a state dict, a nested 'state_dict', or a JIT module."""
    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        inner = checkpoint.get("state_dict")
        if isinstance(inner, dict):
            return inner
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise SystemExit("unrecognized checkpoint: expected a CLIP state dict or module")


def load_checkpoint(path):
    try:
        return torch.jit.load(path, map_location="cpu")
    except Exception:
        pass
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to an OpenAI CLIP .pt")
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
    if "visual.conv1.weight" not in state:
        print("warning: 'visual.conv1.weight' not found; keys do not look like OpenAI CLIP", file=sys.stderr)

    dtype = torch.float16 if args.half else torch.float32
    # Keep only float tensors (a JIT archive can carry non-parameter buffers).
    tensors = {name: value.to(dtype).contiguous().clone()
               for name, value in state.items() if value.is_floating_point()}
    save_file(tensors, args.output)
    print(f"wrote {len(tensors)} tensors ({dtype}) to {args.output}")


if __name__ == "__main__":
    main()
