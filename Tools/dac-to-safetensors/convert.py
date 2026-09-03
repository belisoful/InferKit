#!/usr/bin/env python3
"""Convert a Descript Audio Codec (DAC) checkpoint to safetensors for InferKitMLX.

InferKitMLX also reads the released `.pth` directly (its native torch reader, InferKit 0.3.0), so this
converter is optional: it remains the offline path for producing a portable safetensors file, and the
byte oracle the native reader is held to.

The released weights (`dac.utils.download`, e.g. `weights_44khz_8kbps_0.0.1.pth`) are a torch save whose
`state_dict` holds every convolution weight-NORMALIZED (`weight_g`/`weight_v`). This tool extracts the
state dict and writes it to safetensors in PyTorch Conv1d layout, keeping the weight-norm pairs;
InferKitMLX's `NFKMLXDAC` loader fuses `g·v/‖v‖` and transposes to MLX's layout at load, and remaps the
reference's nested `nn.Sequential` names (`encoder.block.N`, `decoder.model.N`,
`quantizer.quantizers.N`).

`torch.load` reads the checkpoint with torch alone — the `descript-audio-codec` package is the parity
oracle's dependency, not the converter's.

Usage:
    python convert.py weights_44khz_8kbps_0.0.1.pth --list-keys
    python convert.py weights_44khz_8kbps_0.0.1.pth dac.safetensors

Requires: torch, safetensors.
"""

import argparse
import sys

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="the released DAC .pth")
    parser.add_argument("output", nargs="?", help="path to write the safetensors file")
    parser.add_argument("--list-keys", action="store_true", help="print the state-dict keys and exit")
    args = parser.parse_args()

    blob = torch.load(args.input, map_location="cpu", weights_only=False)
    state = blob.get("state_dict", blob) if isinstance(blob, dict) else blob

    if args.list_keys:
        for key in sorted(state):
            print(key, tuple(state[key].shape))
        return 0
    if not args.output:
        raise SystemExit("an output path is required unless --list-keys is given")

    tensors = {key: value.contiguous() for key, value in state.items() if torch.is_tensor(value)}
    save_file(tensors, args.output, metadata={"model": "dac", "format": "pytorch-layout"})
    print(f"wrote {args.output}: {len(tensors)} tensors", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
