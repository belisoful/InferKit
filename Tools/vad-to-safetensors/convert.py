#!/usr/bin/env python3
"""Convert a MarbleNet VAD checkpoint to safetensors for InferKitMLX.

InferKitMLX's `NFKMLXVAD` loads a safetensors checkpoint; MLX's `loadArrays` reads safetensors/npz.
This tool rewrites a checkpoint into safetensors, preserving PyTorch Conv1d layout `[out, in, k]` (the
Swift loader transposes to MLX's `[out, k, in]` at load).

The reference MarbleNet checkpoint (NVIDIA NeMo) stores time-channel separable blocks under
`encoder.encoder.N.mconv.*` with batch norm, plus a `decoder` classifier. InferKitMLX's `NFKMLXVADNet`
groups these into `stem_conv` / `blocks.N.dwconv` / `blocks.N.pwconv` / `head`, so the names do not
line up one-to-one. Aligning them (and the log-mel/MFCC front-end normalization) is a validation-sweep
task; pass `--list-keys` to print the checkpoint's names.

Usage:
    python convert.py vad_marblenet.ckpt --list-keys
    python convert.py vad_marblenet.ckpt vad.safetensors        # after a name remap is worked out

Get the weights from NVIDIA NeMo (nvidia/vad_marblenet).

Requires: torch, safetensors  (pip install torch safetensors)
"""

import argparse
import os
import tarfile
import tempfile
import sys

import torch
from safetensors.torch import save_file


def load_checkpoint(path):
    # A NeMo release ships as a `.nemo`, which is a tar of `model_config.yaml` and
    # `model_weights.ckpt`; torch cannot open the archive itself.
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            member = next((m for m in archive.getmembers()
                           if m.name.endswith("model_weights.ckpt")), None)
            if member is None:
                raise SystemExit(f"{path} is a tar but carries no model_weights.ckpt")
            with tempfile.TemporaryDirectory() as scratch:
                archive.extract(member, scratch)
                return load_checkpoint(os.path.join(scratch, member.name))
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # torch 2.6 made `weights_only` default to True, so the fallback has to say so explicitly
        # or it fails the same way the first attempt did.
        return torch.load(path, map_location="cpu", weights_only=False)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        inner = checkpoint.get("state_dict")
        if isinstance(inner, dict):
            return inner
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise SystemExit("unrecognized checkpoint: expected a MarbleNet state dict")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to a MarbleNet checkpoint")
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
               for name, value in state.items()
               if value.is_floating_point() and not name.endswith("num_batches_tracked")}
    save_file(tensors, args.output)
    print(f"wrote {len(tensors)} tensors ({dtype}) to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
