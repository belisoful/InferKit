#!/usr/bin/env python3
"""Convert a StoRM Lightning checkpoint to the EMA safetensors InferKitMLX reads.

StoRM (sp-uhh/storm) trains a StochasticRegenerationModel — a discriminative predictor (`denoiser_net`)
and a score network (`score_net`), both NCSN++ — with a parameter EMA over both. Inference runs on the
EMA weights, which `model.eval()` copies in (`ema.copy_to(self.parameters())`). This tool applies that
and dumps the `denoiser_net.*` / `score_net.*` state, whose keys are the names the Swift port mirrors.

The output stays in PyTorch layout (4-D convolution weights `[out, in, kH, kW]`); `NFKMLXStoRM`'s loader
transposes them at load, and every other weight is <= 2-D and passes through.

Set IK_STORM_SRC to the cloned repository (holding `sgmse/`) so `sgmse.model.StochasticRegenerationModel`
imports.

Usage:
    IK_STORM_SRC=~/storm python convert.py storm_wsj0_reverb.ckpt --list-keys
    IK_STORM_SRC=~/storm python convert.py storm_wsj0_reverb.ckpt storm.safetensors

Requires: torch, torch_ema, pytorch_lightning, safetensors, and the storm sources.
"""

import argparse
import os
import sys

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="the released StoRM Lightning .ckpt")
    parser.add_argument("output", nargs="?", help="path to write the safetensors file")
    parser.add_argument("--list-keys", action="store_true", help="print the state-dict keys and exit")
    args = parser.parse_args()

    sys.path.insert(0, os.environ.get("IK_STORM_SRC", "."))
    from sgmse.model import StochasticRegenerationModel

    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

    model = StochasticRegenerationModel.load_from_checkpoint(args.input, map_location="cpu",
                                                             base_dir="/tmp", batch_size=1, num_workers=0)
    model.eval()                                        # torch_ema copies the EMA weights into both nets
    state = {k: v for k, v in model.state_dict().items()
             if k.startswith("denoiser_net.") or k.startswith("score_net.")}

    if args.list_keys:
        for key in sorted(state):
            print(key, tuple(state[key].shape))
        return 0
    if not args.output:
        raise SystemExit("an output path is required unless --list-keys is given")

    tensors = {key: value.contiguous() for key, value in state.items() if torch.is_tensor(value)}
    save_file(tensors, args.output, metadata={"model": "storm", "format": "pytorch-layout"})
    print(f"wrote {args.output}: {len(tensors)} tensors", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
