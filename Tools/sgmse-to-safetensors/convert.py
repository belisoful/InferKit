#!/usr/bin/env python3
"""Convert an SGMSE+ Lightning checkpoint to the EMA safetensors InferKitMLX reads.

SGMSE+ (sp-uhh/sgmse) trains with a parameter EMA and runs inference on the EMA weights, which live in
the checkpoint under `checkpoint['ema']` as a flat shadow-parameter list rather than a named state dict.
This tool lets `torch_ema` apply that EMA the way inference does — `ScoreModel.eval()` calls
`ema.copy_to(self.dnn.parameters())` — and then dumps `model.dnn.state_dict()`, whose keys are the
NCSN++ names the Swift port mirrors (`all_modules.N.*`, `output_layer.*`).

The output stays in PyTorch layout (4-D convolution weights `[out, in, kH, kW]`); `NFKMLXSGMSE`'s loader
transposes them to `[out, kH, kW, in]` at load, and the `NIN` `W` (`[in, out]`) and every Linear /
GroupNorm / Gaussian-Fourier weight are <= 2-D and pass through.

Set IK_SGMSE_SRC to the cloned repository (holding `sgmse/`) so `sgmse.model.ScoreModel` imports.

Usage:
    IK_SGMSE_SRC=~/sgmse python convert.py train_wsj0_reverb.ckpt --list-keys
    IK_SGMSE_SRC=~/sgmse python convert.py train_wsj0_reverb.ckpt sgmse.safetensors

Requires: torch, torch_ema, pytorch_lightning, safetensors, and the sgmse sources.
"""

import argparse
import os
import sys

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="the released SGMSE+ Lightning .ckpt")
    parser.add_argument("output", nargs="?", help="path to write the safetensors file")
    parser.add_argument("--list-keys", action="store_true", help="print the dnn state-dict keys and exit")
    args = parser.parse_args()

    sys.path.insert(0, os.environ.get("IK_SGMSE_SRC", "."))
    from sgmse.model import ScoreModel

    # The Lightning checkpoint pickles the data module; torch >= 2.6 defaults weights_only=True and
    # refuses it. Restore the pre-2.6 behavior for this trusted, first-party checkpoint.
    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

    model = ScoreModel.load_from_checkpoint(args.input, map_location="cpu",
                                            base_dir="/tmp", batch_size=1, num_workers=0)
    model.eval()                                        # torch_ema copies the EMA weights into model.dnn
    state = model.dnn.state_dict()

    if args.list_keys:
        for key in sorted(state):
            print(key, tuple(state[key].shape))
        return 0
    if not args.output:
        raise SystemExit("an output path is required unless --list-keys is given")

    tensors = {key: value.contiguous() for key, value in state.items() if torch.is_tensor(value)}
    save_file(tensors, args.output, metadata={"model": "sgmse", "format": "pytorch-layout"})
    print(f"wrote {args.output}: {len(tensors)} tensors", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
