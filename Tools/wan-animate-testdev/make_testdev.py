#!/usr/bin/env python3
"""Builds the Wan 2.2 Animate 2 test-dev release: the diffusers release's own layout, file names, key
names, and stored dtypes, at a small geometry with seeded random weights.

The released pipeline is about 50 GB and runs on no 32 GiB machine, so the port's end-to-end parity is
measured on this directory instead: diffusers' modular `WanAnimate2Blocks` and the InferKitMLX backend
both load it, and `run_reference.py wan_animate_pipeline --checkpoint <out>` records the reference.

Every component is built by its own reference class (`WanAnimate2Transformer3DModel`,
`AutoencoderKLWan`, transformers' `UMT5EncoderModel` and `CLIPVisionModel`) from the release's own
`config.json` with only its size keys reduced, and saved by that class's `save_pretrained`, so a key the
release names is the key this directory names. What stays at the released value:

- the umT5 vocabulary (256,384), so the release's own tokenizer is used unchanged;
- the image encoder's width (1,280), which the transformer's `img_emb` projector reads, its 224-pixel
  input, and its 14-pixel patches;
- the autoencoder's latent width, compression, and latent statistics, which the pipeline reads;
- the transformer's text length and every rotary and reference-offset setting;
- the transformer's 40 blocks reduced to 10, so the block the unconditional pass skips (index 9) exists.

The scheduler, tokenizer, image processor, and both index files are the release's own files, copied.

Usage (the `wananimate` oracle env):
    make_testdev.py --release <dir of the release's small files> --out <dir>

Run it once over the standard release's small files and once over the distilled release's; the two
directories differ in the scheduler and the index files.
"""

import argparse
import hashlib
import json
import os
import shutil

import torch

# Reduced size keys, per component config. Every other key keeps the released value.
TRANSFORMER = {"dim": 64, "num_heads": 2, "num_layers": 10, "ffn_dim": 96, "text_dim": 32}
TEXT_ENCODER = {"d_model": 32, "num_heads": 2, "d_kv": 16, "d_ff": 64, "num_layers": 2}
IMAGE_ENCODER = {"num_hidden_layers": 3, "intermediate_size": 1280}
VAE = {"base_dim": 8}

COPIED = ["model_index.json", "modular_model_index.json", "configuration.json",
          "scheduler/scheduler_config.json"]
COPIED_DIRECTORIES = ["tokenizer", "image_processor"]

# The released components' stored types: the transformer, umT5, and the image encoder at bfloat16 (the
# transformer's config states none; its 14B parameters take 32.8 GB), the autoencoder at float32.
DTYPES = {"transformer": torch.bfloat16, "text_encoder": torch.bfloat16, "image_encoder": torch.bfloat16,
          "vae": torch.float32}

SEEDS = {"transformer": 101, "text_encoder": 102, "image_encoder": 103, "vae": 104}
SCALES = {"transformer": 0.05, "text_encoder": 0.1, "image_encoder": 0.02, "vae": 0.05}


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
                                 "layer_norm.weight", "layer_norm1.weight", "layer_norm2.weight",
                                 "layrnorm.weight", "layernorm.weight", "norm_q.weight", "norm_k.weight",
                                 "norm_added_k.weight", "final_layer_norm.weight")) \
            or (key.startswith("img_emb.proj.") and key.endswith(("0.weight", "4.weight")))
        state[key] = (1 + 0.1 * noise if is_scale else SCALES[name] * noise).to(tensor.dtype)
    model.load_state_dict(state)
    return model


def prefixed(path, prefix):
    """Rewrites a saved file with every key under `prefix`. transformers 5 saves a `CLIPVisionModel`
    without the `vision_model.` level that the release, saved by transformers 4.57, names, and loads
    either form."""
    from safetensors.torch import load_file, save_file
    tensors = load_file(path)
    if all(key.startswith(prefix) for key in tensors):
        return
    save_file({key if key.startswith(prefix) else prefix + key: value for key, value in tensors.items()}, path,
              metadata={"format": "pt"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True, help="the release's small files, in its layout")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    release, out = arguments.release, arguments.out
    os.makedirs(out, exist_ok=False)

    from diffusers import AutoencoderKLWan, WanAnimate2Transformer3DModel
    from transformers import CLIPVisionConfig, CLIPVisionModel, UMT5Config, UMT5EncoderModel

    for relative in COPIED:
        os.makedirs(os.path.dirname(os.path.join(out, relative)) or out, exist_ok=True)
        shutil.copyfile(os.path.join(release, relative), os.path.join(out, relative))
    for relative in COPIED_DIRECTORIES:
        shutil.copytree(os.path.join(release, relative), os.path.join(out, relative))

    transformer = WanAnimate2Transformer3DModel.from_config(
        reduced(os.path.join(release, "transformer/config.json"), TRANSFORMER))
    vae = AutoencoderKLWan.from_config(reduced(os.path.join(release, "vae/config.json"), VAE))
    text_encoder = UMT5EncoderModel(UMT5Config(**reduced(os.path.join(release, "text_encoder/config.json"),
                                                         TEXT_ENCODER)))
    image_encoder = CLIPVisionModel(CLIPVisionConfig(**reduced(os.path.join(release, "image_encoder/config.json"),
                                                               IMAGE_ENCODER)))

    for name, model in (("transformer", transformer), ("vae", vae), ("text_encoder", text_encoder),
                        ("image_encoder", image_encoder)):
        randomized(model.float(), name)
        model.to(DTYPES[name]).save_pretrained(os.path.join(out, name), safe_serialization=True)
    prefixed(os.path.join(out, "image_encoder", "model.safetensors"), "vision_model.")

    manifest = {}
    for root, _, files in os.walk(out):
        for file in sorted(files):
            path = os.path.join(root, file)
            with open(path, "rb") as handle:
                manifest[os.path.relpath(path, out)] = hashlib.sha256(handle.read()).hexdigest()
    with open(os.path.join(out, "testdev-manifest.json"), "w") as handle:
        json.dump({"geometry": {"transformer": TRANSFORMER, "text_encoder": TEXT_ENCODER,
                                "image_encoder": IMAGE_ENCODER, "vae": VAE},
                   "seeds": SEEDS, "sha256": manifest}, handle, indent=2, sort_keys=True)
    print(json.dumps({k: v[:16] for k, v in manifest.items()}, indent=1))


if __name__ == "__main__":
    main()
