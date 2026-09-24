#!/usr/bin/env python3
"""Convert the released hFT-Transformer checkpoint into safetensors for InferKitMLX.

`sony/hFT-Transformer` (MIT) releases `model_016_003.pkl`, which is not a `torch.save` archive: it is
a plain pickle of the live `Model_SPEC2MIDI` module, saved on CUDA with an old torch that pickles
storages by value. Reading it therefore needs the model's own class definitions on the path (set
IK_HFT_SRC to a directory holding `model/model_spec2midi.py`) and a patch that forces the nested
storage load onto the CPU, since that inner `torch.load` takes no `map_location`.

Two tensors are added that the checkpoint does not carry, because the reference builds them at
runtime from torchaudio: the analysis window and the mel filterbank (`n_fft` 2048, `n_mels` 256,
Slaney-normalized, HTK scale). Taking them from torchaudio is the only way to be sure they are the
ones the model was trained against, and it makes the converted file self-contained.

The reference's module names (`encoder_spec2midi.*`, `decoder_spec2midi.*`) are shortened to the
module's own (`encoder.*`, `decoder.*`).

Usage:
    IK_HFT_SRC=~/.inferkit-validation/reference-sources/hft \
        python convert.py model_016_003.pkl hft-transformer-maestro.safetensors

Requires: torch, torchaudio, safetensors.
"""

import argparse
import io
import os
import pickle
import sys

import torch
import torch.storage as torch_storage
from safetensors.torch import save_file


def load_checkpoint(path):
    """The released pickle, on the CPU."""
    source = os.environ.get("IK_HFT_SRC")
    if source:
        sys.path.insert(0, source)
    # The nested load inside `_load_from_bytes` takes no map_location, so it is patched rather than
    # passed; without this the read fails on a machine with no CUDA.
    torch_storage._load_from_bytes = lambda data: torch.load(io.BytesIO(data), map_location="cpu",
                                                             weights_only=False)
    with open(path, "rb") as handle:
        return pickle.load(handle)


def front_end(sample_rate=16000, n_fft=2048, n_mels=256):
    """The window and filterbank the reference's `torchaudio.transforms.MelSpectrogram` builds."""
    import torchaudio
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, win_length=n_fft, hop_length=256,
        pad_mode="constant", n_mels=n_mels, norm="slaney")
    return transform.spectrogram.window.contiguous(), transform.mel_scale.fb.contiguous()


def rename(key):
    if key.startswith("encoder_spec2midi."):
        return "encoder." + key[len("encoder_spec2midi."):]
    if key.startswith("decoder_spec2midi."):
        return "decoder." + key[len("decoder_spec2midi."):]
    return key


def convert(path):
    model = load_checkpoint(path)
    state = model.state_dict()
    tensors = {rename(key): value.detach().cpu().contiguous().float() for key, value in state.items()}

    window, filterbank = front_end()
    tensors["frontend.window"] = window.float()
    tensors["frontend.filterbank"] = filterbank.float()

    expected = tensors["encoder.tok_embedding_freq.weight"].shape
    if expected[0] != 256:
        raise SystemExit(f"the checkpoint's hidden width is {expected[0]}, not the released 256")
    return tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="the released model_016_003.pkl")
    parser.add_argument("output", nargs="?", help="path to write the safetensors file")
    parser.add_argument("--list-keys", action="store_true", help="print the keys and exit")
    args = parser.parse_args()

    tensors = convert(args.input)
    if args.list_keys:
        for key, value in tensors.items():
            print(f"{key:60s} {list(value.shape)}")
        return 0
    if args.output is None:
        parser.error("an output path is required unless --list-keys is given")

    save_file(tensors, args.output)
    total = sum(value.numel() for value in tensors.values())
    print(f"wrote {args.output}: {len(tensors)} tensors, {total} parameters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
