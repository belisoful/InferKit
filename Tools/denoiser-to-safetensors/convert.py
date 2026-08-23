#!/usr/bin/env python3
"""Convert a Demucs-denoiser (speech enhancement) checkpoint to safetensors for InferKitMLX.

InferKitMLX's `NFKMLXDenoiser` reuses the Demucs U-Net (`NFKMLXDemucsNet`) with a single output
channel, and loads through the same weight path (`NFKMLXDemucs.loadWeights`): Conv1d weights are
transposed `[out, in, k]` -> MLX `[out, k, in]`, and the transposed-conv decoder weights are given a
singleton width. This tool rewrites a checkpoint into safetensors, preserving PyTorch layout and
performing the decoder axis/width fix so the Swift loader's uniform transpose applies.

The reference denoiser checkpoint (facebookresearch/denoiser) nests encoder/decoder `nn.Sequential`
blocks (`encoder.N.0`, `decoder.N.2`, an `lstm`, ...); InferKitMLX omits the LSTM bottleneck and names
blocks `encoder.N.conv1` / `decoder.N.convt`, so the names do not line up one-to-one. Aligning them is
a validation-sweep task; pass `--list-keys` to print the checkpoint's names.

Usage:
    python convert.py dns64.th --list-keys
    python convert.py dns64.th denoiser.safetensors             # after a name remap is worked out

Get the weights from the denoiser release:
    https://github.com/facebookresearch/denoiser

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
        for key in ("state", "state_dict", "model"):
            inner = checkpoint.get(key)
            if isinstance(inner, dict):
                return inner
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise SystemExit("unrecognized checkpoint: expected a denoiser state dict")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to a denoiser checkpoint")
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
        # Decoder transposed conv [in, out, k] -> [out, in, k, 1] (singleton width for the MLX impl).
        if "decoder" in name and name.endswith("weight") and value.dim() == 3 and "convt" in name:
            value = value.permute(1, 0, 2).unsqueeze(-1)
        tensors[name] = value.to(dtype).contiguous().clone()
    save_file(tensors, args.output)
    print(f"wrote {len(tensors)} tensors ({dtype}) to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
