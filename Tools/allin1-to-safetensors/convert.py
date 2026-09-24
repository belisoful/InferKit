#!/usr/bin/env python3
"""Convert an All-In-One music structure checkpoint into safetensors for InferKitMLX.

The released checkpoints (`taejunkim/allinone`, MIT) are `torch.save` dictionaries holding a
`state_dict` and the training `config`. The tensors transfer unchanged; the Swift loader transposes
the four-dimensional convolution weights into MLX's layout at load.

One tensor is added that the checkpoint does not carry: `frontend.filterbank`, madmom's logarithmic
filterbank as a `[bins, bands]` matrix. The model was trained on madmom's filtered logarithmic
spectrogram, and the filterbank is a constant of the configuration (12 bands per octave from 30 Hz to
17 kHz, each filter normalized), so taking it from madmom itself is the only way to be sure of the
band edges rather than re-deriving them. That makes the converted checkpoint self-contained: the
consumer needs no madmom.

Usage:
    python convert.py harmonix-fold0-0vra4ys2.pth allin1-harmonix-fold0.safetensors
    python convert.py --list-keys harmonix-fold0-0vra4ys2.pth

Requires: torch, madmom, safetensors. Runs under the `allin1` oracle environment.
"""

import argparse
import sys

import numpy as np
import torch
from safetensors.torch import save_file


def filterbank(config):
    """madmom's `LogarithmicFilterbank` for the model's own front-end settings."""
    from madmom.audio.filters import LogarithmicFilterbank
    bin_frequencies = np.fft.fftfreq(config["window_size"], 1.0 / config["sample_rate"])[: config["window_size"] // 2]
    bank = LogarithmicFilterbank(bin_frequencies,
                                 num_bands=config["num_bands"],
                                 fmin=config["fmin"],
                                 fmax=config["fmax"],
                                 norm_filters=True,
                                 unique_filters=True)
    return np.ascontiguousarray(np.asarray(bank), dtype=np.float32)


def convert(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    config = dict(checkpoint["config"])

    expected = {"sample_rate": 44100, "window_size": 2048, "hop_size": 441, "num_bands": 12,
                "fmin": 30, "fmax": 17000, "dim_input": 81, "dim_embed": 24, "depth": 11,
                "num_heads": 2, "kernel_size": 5}
    for key, value in expected.items():
        if key in config and config[key] != value:
            raise SystemExit(f"the checkpoint's {key} is {config[key]}, not the released {value}; "
                             f"this converter targets the released Harmonix geometry")

    tensors = {key: value.contiguous().float() for key, value in state.items()}
    bank = filterbank({k: config.get(k, expected[k]) for k in expected})
    if bank.shape != (config.get("window_size", 2048) // 2, config.get("dim_input", 81)):
        raise SystemExit(f"the filterbank came out {bank.shape}, not "
                         f"{(config.get('window_size', 2048) // 2, config.get('dim_input', 81))}")
    tensors["frontend.filterbank"] = torch.from_numpy(bank)

    # The post-processing thresholds are tuned per fold (0.21 and 0.22 differ between the released
    # checkpoints), and the beat decoder trims the track to the span that reaches its threshold, so
    # they travel with the weights rather than as a default in the code.
    tensors["postprocess.thresholds"] = torch.tensor(
        [float(config.get("best_threshold_beat", config["threshold_beat"])),
         float(config.get("best_threshold_downbeat", config["threshold_downbeat"]))])
    return tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="a released All-In-One .pth")
    parser.add_argument("output", nargs="?", help="path to write the safetensors file")
    parser.add_argument("--list-keys", action="store_true", help="print the keys and exit")
    args = parser.parse_args()

    tensors = convert(args.input)
    if args.list_keys:
        for key, value in tensors.items():
            print(f"{key:62s} {list(value.shape)}")
        return 0
    if args.output is None:
        parser.error("an output path is required unless --list-keys is given")

    save_file(tensors, args.output)
    total = sum(value.numel() for value in tensors.values())
    print(f"wrote {args.output}: {len(tensors)} tensors, {total} parameters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
