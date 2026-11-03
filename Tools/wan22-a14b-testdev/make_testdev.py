#!/usr/bin/env python3
"""Builds a Wan 2.2 A14B test-dev release: the diffusers release's own layout, file names, key names, and
stored dtypes, at a small geometry with seeded random weights.

The A14B releases are about 126 GB each (two 14B experts at float32), so the port's end-to-end parity is
measured on this directory instead: diffusers' `WanPipeline` (T2V-A14B) or `WanImageToVideoPipeline`
(I2V-A14B) and the InferKitMLX generator both load it, and
`run_reference.py wan22_a14b_pipeline --checkpoint <out>` records the reference.

Every component is built by its own reference class (`WanTransformer3DModel` for both experts,
`AutoencoderKLWan`, transformers' `UMT5EncoderModel`) from the release's own `config.json` with only its
size keys reduced, and saved by that class's `save_pretrained`, so a key the release names is the key this
directory names. What stays at the released value:

- the umT5 vocabulary (256,384), so the release's own tokenizer is used unchanged;
- the autoencoder's latent width, compression, and latent statistics, which the pipeline reads;
- the transformers' channel counts (16 in for T2V, 36 for I2V; 16 out), patch, and rotary settings;
- `boundary_ratio` in `model_index.json`, which hands the schedule to `transformer_2`.

The two experts draw different seeds, so a step run by the wrong expert shows. The scheduler, tokenizer,
and `model_index.json` are the release's own files, copied.

Usage (the `wananimate` oracle env):
    make_testdev.py --release <dir of the release's small files> --out <dir>

Run it once over T2V-A14B's small files and once over I2V-A14B's.
"""

import argparse
import hashlib
import json
import os
import shutil

import torch

# Reduced size keys, per component config. Every other key keeps the released value. A head width of 24
# keeps the three rotary axes even (8, 8, 8).
TRANSFORMER = {"num_attention_heads": 2, "attention_head_dim": 24, "num_layers": 2, "ffn_dim": 96, "text_dim": 32}
TEXT_ENCODER = {"d_model": 32, "num_heads": 2, "d_kv": 16, "d_ff": 64, "num_layers": 2}
VAE = {"base_dim": 8}

COPIED = ["model_index.json", "scheduler/scheduler_config.json"]
COPIED_DIRECTORIES = ["tokenizer"]

# The released components' stored types: both experts and the autoencoder at float32, umT5 at bfloat16.
DTYPES = {"transformer": torch.float32, "transformer_2": torch.float32, "text_encoder": torch.bfloat16,
          "vae": torch.float32}

SEEDS = {"transformer": 201, "transformer_2": 202, "text_encoder": 203, "vae": 204}
SCALES = {"transformer": 0.05, "transformer_2": 0.05, "text_encoder": 0.1, "vae": 0.05}


def reduced(path, overrides):
    with open(path) as handle:
        config = json.load(handle)
    for key, value in overrides.items():
        if key not in config:
            raise SystemExit(f"{path} has no {key}; the release's config changed")
        config[key] = value
    return config


def randomized(model, name):
    """Every floating tensor drawn from a seeded normal; LayerNorm and RMS scales around one, so the
    activations neither vanish nor saturate through the stack."""
    generator = torch.Generator().manual_seed(SEEDS[name])
    state = model.state_dict()
    for key in sorted(state):
        tensor = state[key]
        if not tensor.is_floating_point():
            continue
        noise = torch.randn(tensor.shape, generator=generator, dtype=torch.float32)
        is_scale = key.endswith((".gamma", "norm.weight", "norm1.weight", "norm2.weight", "norm3.weight",
                                 "layer_norm.weight", "norm_q.weight", "norm_k.weight", "final_layer_norm.weight"))
        state[key] = (1 + 0.1 * noise if is_scale else SCALES[name] * noise).to(tensor.dtype)
    model.load_state_dict(state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True, help="the release's small files, in its layout")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    release, out = arguments.release, arguments.out
    os.makedirs(out, exist_ok=False)

    from diffusers import AutoencoderKLWan, WanTransformer3DModel
    from transformers import UMT5Config, UMT5EncoderModel

    for relative in COPIED:
        os.makedirs(os.path.dirname(os.path.join(out, relative)) or out, exist_ok=True)
        shutil.copyfile(os.path.join(release, relative), os.path.join(out, relative))
    for relative in COPIED_DIRECTORIES:
        shutil.copytree(os.path.join(release, relative), os.path.join(out, relative))

    models = {name: WanTransformer3DModel.from_config(reduced(os.path.join(release, name, "config.json"), TRANSFORMER))
              for name in ("transformer", "transformer_2")}
    models["vae"] = AutoencoderKLWan.from_config(reduced(os.path.join(release, "vae/config.json"), VAE))
    models["text_encoder"] = UMT5EncoderModel(UMT5Config(**reduced(
        os.path.join(release, "text_encoder/config.json"), TEXT_ENCODER)))

    for name, model in models.items():
        randomized(model.float(), name)
        model.to(DTYPES[name]).save_pretrained(os.path.join(out, name), safe_serialization=True)

    manifest = {}
    for root, _, files in os.walk(out):
        for file in sorted(files):
            path = os.path.join(root, file)
            with open(path, "rb") as handle:
                manifest[os.path.relpath(path, out)] = hashlib.sha256(handle.read()).hexdigest()
    with open(os.path.join(out, "testdev-manifest.json"), "w") as handle:
        json.dump({"geometry": {"transformer": TRANSFORMER, "text_encoder": TEXT_ENCODER, "vae": VAE},
                   "seeds": SEEDS, "sha256": manifest}, handle, indent=2, sort_keys=True)
    print(json.dumps({k: v[:16] for k, v in manifest.items()}, indent=1))


if __name__ == "__main__":
    main()
