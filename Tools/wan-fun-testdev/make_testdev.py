#!/usr/bin/env python3
"""Builds a Wan 2.2 Fun test-dev release: the original Wan layout the alibaba-pai Fun releases ship (their
file names, key names, and stored dtypes) at a small geometry with seeded random weights.

The Fun releases (Control, InP, Control-Camera, at 5B and A14B) are 24 to 71 GB, so the port's end-to-end
parity is measured on this directory: VideoX-Fun's `Wan2_2FunControlPipeline` / `Wan2_2FunInpaintPipeline`
and the InferKitMLX generator both load it, and `run_reference.py wan_fun_pipeline --checkpoint <out>`
records the reference.

Every component is built by VideoX-Fun's own class and saved as the release stores it:

- the transformer (`Wan2_2Transformer3DModel`) from the release's own `config.json`, the size keys
  reduced (a 5B InP release keeps its released 3072 width and 24 heads, the value VideoX-Fun's Inpaint
  pipeline reads to take its 5B path), its state dict under the original names in `diffusion_pytorch_model.safetensors` at bfloat16
  (`high_noise_model/` and `low_noise_model/` for an A14B release, on different seeds);
- the autoencoder (`AutoencoderKLWan3_8` for the 5B releases, `AutoencoderKLWan` for A14B) as
  `Wan2.2_VAE.pth` / `Wan2.1_VAE.pth` without the wrapper's `model.` prefix, at float32. VideoX-Fun builds
  the Wan 2.1 autoencoder at its released width and the Wan 2.2 decoder at its released 256, so only the
  Wan 2.2 encoder width (`c_dim`) is reduced;
- umT5 (`WanT5EncoderModel`) at 32 wide over the released 256,384-token vocabulary, as
  `models_t5_umt5-xxl-enc-bf16.pth` at bfloat16; the release's `google/umt5-xxl` tokenizer is copied.

What stays at the released value: every channel count the pipeline wires (in, out, reference, adapter),
the patch, the autoencoder's latent width and statistics, and the vocabulary.

Usage (the `videoxfun` oracle env, with `IK_VIDEOX_FUN_SRC` set):
    make_testdev.py --release <dir of the release's small files> --out <dir>
"""

import argparse
import hashlib
import json
import os
import shutil
import sys

import torch

TRANSFORMER = {"dim": 48, "num_heads": 2, "num_layers": 2, "ffn_dim": 96, "text_dim": 32}
TEXT_ENCODER = {"vocab": 256384, "dim": 32, "dim_attn": 32, "dim_ffn": 64, "num_heads": 2, "num_layers": 2,
                "num_buckets": 32, "shared_pos": False, "dropout": 0.0}
WAN22_ENCODER_WIDTH = 16

SEEDS = {"transformer": 301, "high_noise_model": 302, "low_noise_model": 303, "text_encoder": 304, "vae": 305}
SCALES = {"transformer": 0.05, "text_encoder": 0.1, "vae": 0.05}
SCALE_SUFFIXES = (".gamma", "norm.weight", "norm1.weight", "norm2.weight", "norm3.weight", "norm_q.weight",
                  "norm_k.weight")


def videox_fun():
    source = os.environ.get("IK_VIDEOX_FUN_SRC")
    if not source:
        raise SystemExit("set IK_VIDEOX_FUN_SRC")
    sys.path.insert(0, source)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import vfun_shims  # noqa: F401  (stands in for triton, which the CPU float path never reaches)
    import videox_fun.models as models
    return models


def randomized(model, seed, scale):
    """Every floating tensor drawn from a seeded normal; norm scales around one, so the activations
    neither vanish nor saturate through the stack."""
    generator = torch.Generator().manual_seed(seed)
    state = model.state_dict()
    for key in sorted(state):
        tensor = state[key]
        if not tensor.is_floating_point():
            continue
        noise = torch.randn(tensor.shape, generator=generator, dtype=torch.float32)
        state[key] = (1 + 0.1 * noise if key.endswith(SCALE_SUFFIXES) else scale * noise).to(tensor.dtype)
    model.load_state_dict(state)
    return model


def reduced_config(path):
    with open(path) as handle:
        config = json.load(handle)
    overrides = dict(TRANSFORMER)
    # VideoX-Fun's Inpaint pipeline takes its 5B path (per-token timesteps, the masked-latent blend) where
    # `transformer.config.dim == 3072`, so a 5B InP release keeps its released width and heads.
    in_dim = config.get("in_dim", config.get("in_channels"))
    if config.get("dim") == 3072 and in_dim == 100 and not config.get("add_control_adapter", False):
        overrides = {key: value for key, value in overrides.items() if key not in ("dim", "num_heads")}
    for key, value in overrides.items():
        if key == "text_dim" and key not in config:
            config[key] = value
            continue
        if key not in config:
            raise SystemExit(f"{path} has no {key}; the release's config changed")
        config[key] = value
        if key == "dim" and "hidden_size" in config:
            config["hidden_size"] = value
    return config


def write_transformer(models, release, out, name, seed):
    from safetensors.torch import save_file
    config = reduced_config(os.path.join(release, name, "config.json") if name else os.path.join(release, "config.json"))
    directory = os.path.join(out, name) if name else out
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "config.json"), "w") as handle:
        json.dump(config, handle, indent=2)
    keys = {key: value for key, value in config.items() if not key.startswith("_")}
    keys.setdefault("in_channels", keys.get("in_dim"))
    keys.setdefault("hidden_size", keys.get("dim"))
    model = models.Wan2_2Transformer3DModel(**{k: v for k, v in keys.items()
                                               if k in models.Wan2_2Transformer3DModel.__init__.__code__.co_varnames})
    randomized(model.float(), seed, SCALES["transformer"])
    state = {key: value.to(torch.bfloat16).contiguous() for key, value in model.state_dict().items()
             if not key.startswith("img_emb.")}
    save_file(state, os.path.join(directory, "diffusion_pytorch_model.safetensors"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True, help="the release's small files, in its layout")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    release, out = arguments.release, arguments.out
    os.makedirs(out, exist_ok=False)
    models = videox_fun()

    a14b = os.path.isdir(os.path.join(release, "high_noise_model"))
    if a14b:
        for name in ("high_noise_model", "low_noise_model"):
            write_transformer(models, release, out, name, SEEDS[name])
    else:
        write_transformer(models, release, out, None, SEEDS["transformer"])

    if a14b:
        vae = models.AutoencoderKLWan(latent_channels=16)
        vae_file = "Wan2.1_VAE.pth"
    else:
        vae = models.AutoencoderKLWan3_8(latent_channels=48, c_dim=WAN22_ENCODER_WIDTH,
                                         dim_mult=[1, 2, 4, 4], temperal_downsample=[False, True, True])
        vae_file = "Wan2.2_VAE.pth"
    randomized(vae.model.float(), SEEDS["vae"], SCALES["vae"])
    torch.save(vae.model.state_dict(), os.path.join(out, vae_file))

    text_encoder = models.WanT5EncoderModel(**TEXT_ENCODER)
    randomized(text_encoder.float(), SEEDS["text_encoder"], SCALES["text_encoder"])
    torch.save({key: value.to(torch.bfloat16) for key, value in text_encoder.state_dict().items()},
               os.path.join(out, "models_t5_umt5-xxl-enc-bf16.pth"))
    shutil.copytree(os.path.join(release, "google"), os.path.join(out, "google"))
    for extra in ("configuration.json",):
        if os.path.exists(os.path.join(release, extra)):
            shutil.copyfile(os.path.join(release, extra), os.path.join(out, extra))

    manifest = {}
    for root, _, files in os.walk(out):
        for file in sorted(files):
            path = os.path.join(root, file)
            with open(path, "rb") as handle:
                manifest[os.path.relpath(path, out)] = hashlib.sha256(handle.read()).hexdigest()
    with open(os.path.join(out, "testdev-manifest.json"), "w") as handle:
        json.dump({"geometry": {"transformer": TRANSFORMER, "text_encoder": TEXT_ENCODER,
                                "wan22_encoder_width": WAN22_ENCODER_WIDTH},
                   "seeds": SEEDS, "sha256": manifest}, handle, indent=2, sort_keys=True)
    print(json.dumps({k: v[:16] for k, v in manifest.items()}, indent=1))


if __name__ == "__main__":
    main()
