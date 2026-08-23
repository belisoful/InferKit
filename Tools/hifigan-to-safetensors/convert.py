#!/usr/bin/env python3
"""Convert a HiFi-GAN generator checkpoint (jik876 layout) to safetensors for InferKitMLX
(NFKMLXHiFiGANNet).

The released `g_*` file wraps the state dict under a `generator` key and stores every convolution
WEIGHT-NORMALIZED — a magnitude `weight_g` and a direction `weight_v` instead of a plain weight. The
reference fuses them at inference (`remove_weight_norm`), and this converter does the same:
`weight = g * v / ||v||`, the norm per slice of dim 0. The transposed upsampling convolutions are
renamed under the `conv` child the Swift module nests them in; the Swift loader transposes layouts.

Usage:
    python convert.py g_02500000 hifigan-universal.safetensors
    python convert.py g_02500000 --list-keys

Weights: the official UNIVERSAL_V1 / LJ_V1 releases of https://github.com/jik876/hifi-gan
(mirrored at huggingface.co/csdc-atl/hifigan-universal_v1).

Requires: torch, safetensors
"""

import argparse

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="the released g_* file")
    parser.add_argument("output", nargs="?", help="the safetensors to write")
    parser.add_argument("--list-keys", action="store_true", help="print the checkpoint keys and exit")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = state.get("generator", state)
    # The espnet-paired release (fastspeech2_conformer_with_hifigan) carries the generator under a
    # `vocoder.` prefix with `upsampler.N` naming and its weight norm ALREADY fused; accept it too.
    if any(key.startswith("vocoder.") for key in state):
        state = {key[len("vocoder."):].replace("upsampler.", "ups."): value
                 for key, value in state.items() if key.startswith("vocoder.")}

    if args.list_keys:
        for key in sorted(state):
            print(key, tuple(state[key].shape))
        return
    if not args.output:
        parser.error("output path required unless --list-keys")

    fused = {}
    for key, value in state.items():
        if key.endswith(".weight_g"):
            base = key[: -len(".weight_g")]
            g, v = state[key], state[base + ".weight_v"]
            norm = v.norm(dim=tuple(range(1, v.dim())), keepdim=True)
            fused[base + ".weight"] = (g * v / norm).contiguous()
        elif key.endswith(".weight_v"):
            continue
        else:
            fused[key] = value.contiguous()

    renamed = {}
    for key, value in fused.items():
        name = key
        if name.startswith("ups."):
            parts = name.split(".")                     # ups.N.weight -> ups.N.conv.weight
            name = ".".join(parts[:2] + ["conv"] + parts[2:])
        renamed[name] = value
    save_file(renamed, args.output)
    print(f"wrote {args.output}: {len(renamed)} tensors")


if __name__ == "__main__":
    main()
