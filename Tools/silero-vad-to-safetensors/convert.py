#!/usr/bin/env python3
"""Convert a Silero VAD v6 (snakers4) TorchScript model to safetensors for InferKitMLX.

InferKitMLX also reads the released TorchScript model directly (its native reader, InferKit 0.3.0),
so this converter is optional: it remains the offline path for producing a portable safetensors file,
and the byte oracle the native reader is held to.

The released `silero_vad.jit` is a TorchScript module whose `state_dict()` carries a 16 kHz branch under
`_model.*` and an 8 kHz branch under `_model_8k.*`. This tool keeps the 16 kHz branch's `_model.*`
tensors in PyTorch Conv1d layout `[out, in, k]`; InferKitMLX's `NFKMLXSileroVAD` loader remaps the names
and transposes to MLX's `[out, k, in]` at load. The 8 kHz branch is dropped, as this port runs the
16 kHz model.

`torch.jit.load` reads the model with torch alone — the `silero-vad` package is not needed for the
weights (it is the parity oracle's dependency, not the converter's).

Usage:
    python convert.py silero_vad.jit --list-keys
    python convert.py silero_vad.jit silero-vad.safetensors

Requires: torch, safetensors.
"""

import argparse
import sys

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="the released silero_vad.jit")
    parser.add_argument("output", nargs="?", help="path to write the safetensors file")
    parser.add_argument("--list-keys", action="store_true", help="print the state-dict keys and exit")
    args = parser.parse_args()

    state = torch.jit.load(args.input, map_location="cpu").state_dict()

    if args.list_keys:
        for key in sorted(state):
            print(key, tuple(state[key].shape))
        return 0
    if not args.output:
        raise SystemExit("an output path is required unless --list-keys is given")

    # Keep only the 16 kHz branch. "_model." is not a prefix of "_model_8k." (the seventh character is
    # "_", not "."), so this excludes the 8 kHz weights.
    branch = {key: value.contiguous() for key, value in state.items() if key.startswith("_model.")}
    if not branch:
        raise SystemExit("no _model.* tensors found; the release layout may have changed")

    save_file(branch, args.output, metadata={"model": "silero_vad", "format": "pytorch-layout",
                                              "branch": "16k"})
    print(f"wrote {args.output}: {len(branch)} tensors", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
