#!/usr/bin/env python3
"""Run a model's reference implementation and record its input and output for InferKit to compare against.

Visual plausibility is not reference quality. A model can load every parameter, run end to end, and still
be wrong — SAM tracked its subject convincingly while missing its entire input normalization, and Depth
Anything produced a correct-looking map because its output is min-max normalized. The only check that
catches that class of bug is running the reference on the same input and comparing numerically.

Protocol: this script writes ONE safetensors file per model containing both

    input_image   the raw RGB plate, [H, W, 3] float32 in 0...1 (no preprocessing applied)
    output        the reference implementation's result

InferKit's `NFKMLXReferenceParityTests` reads the same file, feeds `input_image` to the MLX model, and
compares against `output`. Sharing the input tensor rather than an image file keeps image decoding out of
the comparison, while each side still applies its own preprocessing — so a missing normalization shows up
as a mismatch rather than hiding.

Usage:
    python run_reference.py clip   out/clip-reference.safetensors
    python run_reference.py depth  out/depth-reference.safetensors

A training objective is validated the same way. `zero_dce_losses` records the tensors the reference
scored alongside its four loss values, so both sides score identical inputs:

    python run_reference.py zero_dce_losses out/zero-dce-losses.safetensors --size 64

Requires: torch, safetensors, transformers.
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file


def deterministic_image(height=224, width=224, seed=7):
    """A fixed pseudo-random RGB plate in 0...1. Structured enough to exercise a real forward, and
    identical on both sides of the comparison."""
    generator = np.random.default_rng(seed)
    # Round the block grid UP so the result is exactly height × width — rounding down silently returns a
    # smaller plate, which would reintroduce a resize on the reference side and spoil an isolation test.
    blocks = ((height + 7) // 8, (width + 7) // 8)
    base = generator.random((blocks[0], blocks[1], 3), dtype=np.float32)
    # Smooth blocks give the model spatial structure rather than pixel noise.
    image = np.repeat(np.repeat(base, 8, axis=0), 8, axis=1)
    return np.ascontiguousarray(image[:height, :width, :])


def subject_image(height=320, width=320, seed=7):
    """A plate with an actual subject: a textured ellipse on a smooth background.

    A saliency or matting model has nothing to find in unstructured blocks — the reference's own map
    comes out uniformly near zero, and comparing two near-constant maps measures nothing. This gives
    both sides a foreground to separate.
    """
    generator = np.random.default_rng(seed)
    rows = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    columns = np.linspace(0, 1, width, dtype=np.float32)[None, :, None]
    background = 0.25 + 0.3 * rows + 0.15 * columns * np.array([1.0, 0.6, 0.2], dtype=np.float32)

    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    inside = (((y - height / 2) / (height * 0.3)) ** 2 + ((x - width / 2) / (width * 0.22)) ** 2) <= 1.0
    texture = generator.random((height, width, 3), dtype=np.float32) * 0.2 + 0.75
    image = np.where(inside[..., None], texture, background)
    return np.ascontiguousarray(np.clip(image, 0, 1).astype(np.float32))


def run_clip(image):
    """OpenAI CLIP ViT-B/32 image embedding, L2-normalized, through transformers.

    IK_CLIP_REPO names a different tower of the same architecture — MetaCLIP publishes its towers as
    `CLIPModel` repositories (`facebook/metaclip-b32-400m`), which is what makes them a weights change
    rather than a port.
    """
    from transformers import CLIPModel, CLIPImageProcessor

    repo = os.environ.get("IK_CLIP_REPO", "openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained(repo).eval()
    processor = CLIPImageProcessor.from_pretrained(repo)
    # The processor expects HWC uint8-like input; hand it the raw plate and let it normalize.
    inputs = processor(images=(image * 255).astype(np.uint8), return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    return features[0].contiguous()


# The released encoder sizes, as `DepthAnythingV2` itself configures them.
_DEPTH_VARIANTS = {
    "Small": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "Base": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "Large": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def _depth_model(checkpoint):
    """Build `DepthAnythingV2` from the authors' own package and load a released `.pth`.

    This drove `transformers` until that package stopped registering the `depth_anything` model type,
    at which point every size raised `KeyError` and the oracle could no longer be re-run at all. The
    original repository is the durable source — and the pattern every other model here already uses.
    IK_REF_SRC holds a `depth_anything_v2/` directory (the real package, so its own relative imports
    resolve); its parent goes on the path rather than being synthesized.
    """
    sys.path.insert(0, _reference_source())
    from depth_anything_v2.dpt import DepthAnythingV2

    variant = os.environ.get("IK_DEPTH_VARIANT", "Small")
    model = DepthAnythingV2(**_DEPTH_VARIANTS[variant])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return model.eval()


def _depth_input(image):
    """ImageNet normalization at the plate's own size. The port runs a fixed 518×518, so feeding a
    518×518 plate keeps every resize out of the comparison."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    deviation = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(((image - mean) / deviation).transpose(2, 0, 1)).unsqueeze(0)


def run_depth(image, checkpoint):
    """Depth Anything V2 relative depth, min-max normalized to 0...1 (as InferKit emits it)."""
    model = _depth_model(checkpoint)
    with torch.no_grad():
        depth = model(_depth_input(image))[0]                  # [h, w]
    depth = (depth - depth.min()) / torch.clamp(depth.max() - depth.min(), min=1e-6)
    return depth.contiguous()


def run_depth_encoder(image, checkpoint):
    """The encoder's hooked feature maps, to localize a mismatch to encoder vs DPT head.

    Comparing only final outputs says *that* two implementations differ, not *where*. These are the
    seam between the DINOv2 backbone and the DPT head — and the layer whose missing final LayerNorm
    was the original Depth Anything bug.
    """
    model = _depth_model(checkpoint)
    with torch.no_grad():
        features = model.pretrained.get_intermediate_layers(
            _depth_input(image), model.intermediate_layer_idx[model.encoder], return_class_token=True)
    return features[0][0][0].contiguous()                       # [tokens, C], class token excluded


def run_depth3(image, checkpoint):
    """Depth Anything 3 (DA3-SMALL) monocular depth, seam by seam.

    The `depth_anything_3` package is not in transformers; IK_REF_SRC holds it (the pip wheel's own
    `depth_anything_3/` importable directory). Every released tensor is built and loaded strictly: the
    backbone (DinoV2), both DualDPT branches, the camera decoder, and the camera encoder. The image is
    fed to the backbone directly (no ImageNet normalization), so the Swift parity test feeds the same
    `input_image` tensor and the comparison isolates the network. The seams (the four hooked features,
    the head's four resized stage features, the fused map, the pre-exp logits, the four aux pyramid
    levels, the ray logits, the camera token, the pose encoding, and the camera encoder's tokens) are
    recorded in `_extra` for localization; the returned `output` is the exp-depth map.
    """
    sys.path.insert(0, _reference_source())
    from safetensors.torch import load_file
    from depth_anything_3.model.dinov2.dinov2 import DinoV2
    from depth_anything_3.model.dualdpt import DualDPT
    from depth_anything_3.model.cam_dec import CameraDec
    from depth_anything_3.model.cam_enc import CameraEnc

    size = image.shape[0]
    # IK_DEPTH3_VARIANT selects the released size; each one's numbers come from its own config.json
    # (`depth-anything/DA3-<SIZE>`): large hooks later blocks and starts its rotary/qknorm/camera
    # alternation at block 8, and every size widens the DualDPT with the backbone.
    variant = os.environ.get("IK_DEPTH3_VARIANT", "small")
    geometry = {
        "small": dict(name="vits", out_layers=[5, 7, 9, 11], start=4, dim_in=768, features=64,
                      out_channels=[48, 96, 192, 384]),
        "base": dict(name="vitb", out_layers=[5, 7, 9, 11], start=4, dim_in=1536, features=128,
                     out_channels=[96, 192, 384, 768]),
        "large": dict(name="vitl", out_layers=[11, 15, 19, 23], start=8, dim_in=2048, features=256,
                      out_channels=[256, 512, 1024, 1024]),
    }[variant]
    hooks = geometry["out_layers"]
    net = DinoV2(name=geometry["name"], out_layers=hooks, alt_start=geometry["start"],
                 qknorm_start=geometry["start"], rope_start=geometry["start"], cat_token=True)
    head = DualDPT(dim_in=geometry["dim_in"], output_dim=2, features=geometry["features"],
                   out_channels=geometry["out_channels"], head_names=("depth", "ray"))
    # dim_in is the concatenated camera token, dim_out the backbone width; both come from config.json.
    cam_dec = CameraDec(dim_in=geometry["dim_in"])
    cam_enc = CameraEnc(dim_out=geometry["dim_in"] // 2)
    state = load_file(checkpoint) if checkpoint.endswith(".safetensors") else torch.load(checkpoint, map_location="cpu")

    def subtree(prefix):
        return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}

    # Strict where the release is complete, so a tensor with no counterpart fails here rather than
    # silently. The head is the exception: the release carries `output_conv2_aux.0.2` (the aux head's
    # channel LayerNorm) and no such tensor for levels 1-3, and the reference's own loader
    # (`utils/model_loading.py`) loads `strict=False`, so those three run at the `nn.LayerNorm` init of
    # weight 1 and bias 0. Inference reads level 3, so the ray head's normalization is unweighted.
    net.load_state_dict(subtree("model.backbone."), strict=True)
    missing, unexpected = head.load_state_dict(subtree("model.head."), strict=False)
    expected_missing = {f"scratch.output_conv2_aux.{i}.2.{p}" for i in (1, 2, 3) for p in ("weight", "bias")}
    assert set(missing) == expected_missing, sorted(set(missing) ^ expected_missing)
    assert not unexpected, unexpected
    cam_dec.load_state_dict(subtree("model.cam_dec."), strict=True)
    cam_enc.load_state_dict(subtree("model.cam_enc."), strict=True)
    net.eval()
    head.eval()
    cam_dec.eval()
    cam_enc.eval()

    img = torch.from_numpy(image).permute(2, 0, 1)[None, None].float()   # [1, 1, 3, H, W]
    extra = {}
    with torch.no_grad():
        feats, _ = net.pretrained.get_intermediate_layers(img, hooks, cam_token=None)
        for i, (ft, cam) in enumerate(feats):
            extra[f"hook{i}"] = ft[0, 0].contiguous()
            extra[f"camera_token{i}"] = cam[0, 0].contiguous()
        # The head's depth branch, step by step (see NFKMLXDepthAnything3.NFKDA3Head.seams).
        from depth_anything_3.model.utils.head_utils import custom_interpolate
        ph = pw = size // 14
        flat = [f[0].reshape(1, ph * pw, f[0].shape[-1]) for f in feats]
        resized = []
        for si, x in enumerate(flat):
            x = head.norm(x).permute(0, 2, 1).reshape(1, x.shape[-1], ph, pw)
            x = head._add_pos_embed(head.projects[si](x), size, size)
            x = head.resize_layers[si](x)
            resized.append(x)
            extra[f"stage{si}"] = x[0].contiguous()
        sc = head.scratch
        o = sc.refinenet4(sc.layer4_rn(resized[3]), size=sc.layer3_rn(resized[2]).shape[2:])
        o = sc.refinenet3(o, sc.layer3_rn(resized[2]), size=sc.layer2_rn(resized[1]).shape[2:])
        o = sc.refinenet2(o, sc.layer2_rn(resized[1]), size=sc.layer1_rn(resized[0]).shape[2:])
        o = sc.refinenet1(o, sc.layer1_rn(resized[0]))
        o = sc.output_conv1(o)
        extra["fused"] = o[0].contiguous()
        o = custom_interpolate(o, (size, size), mode="bilinear", align_corners=True)
        o = head._add_pos_embed(o, size, size)
        logits = sc.output_conv2(o)
        extra["logits"] = logits[0].contiguous()
        depth = torch.exp(logits[0, 0])

        # The aux (ray) branch: its own fusion chain over the same reassembled pyramid, then the
        # per-level neck, then the final level's head. Only the finest level is returned, and it is
        # never interpolated to the image size.
        a = sc.refinenet4_aux(sc.layer4_rn(resized[3]), size=sc.layer3_rn(resized[2]).shape[2:])
        aux_list = [a]
        a = sc.refinenet3_aux(a, sc.layer3_rn(resized[2]), size=sc.layer2_rn(resized[1]).shape[2:])
        aux_list.append(a)
        a = sc.refinenet2_aux(a, sc.layer2_rn(resized[1]), size=sc.layer1_rn(resized[0]).shape[2:])
        aux_list.append(a)
        a = sc.refinenet1_aux(a, sc.layer1_rn(resized[0]))
        aux_list.append(a)
        aux_list = [sc.output_conv1_aux[i](x) for i, x in enumerate(aux_list)]
        for i, x in enumerate(aux_list):
            extra[f"aux{i}"] = x[0].contiguous()
        last_aux = head._add_pos_embed(aux_list[-1], size, size)
        extra["aux_pos"] = last_aux[0].contiguous()
        stack = sc.output_conv2_aux[-1]
        step = stack[0](last_aux)
        extra["aux_conv0"] = step[0].contiguous()
        step = stack[2](stack[1](step))
        extra["aux_norm"] = step.permute(0, 3, 1, 2)[0].contiguous()
        ray_logits = sc.output_conv2_aux[-1](last_aux)
        extra["ray_logits"] = ray_logits[0].contiguous()
        fmap = ray_logits.permute(0, 2, 3, 1)
        extra["ray"] = fmap[0, ..., :-1].contiguous()
        extra["ray_conf"] = (torch.exp(fmap[0, ..., -1]) + 1).contiguous()

        # The camera decoder reads the last hook's camera token.
        pose_enc = cam_dec(feats[-1][1])
        extra["pose_enc"] = pose_enc[0].contiguous()

        # The camera encoder turns a known pose into the token the backbone reads in its place. A
        # deterministic rotation about the y axis with a translation and a plain pinhole intrinsic is
        # enough to exercise the whole path.
        from depth_anything_3.model.utils.transform import extri_intri_to_pose_encoding
        from depth_anything_3.utils.geometry import affine_inverse
        angle = 0.3
        rotation = torch.tensor([[math.cos(angle), 0.0, math.sin(angle)],
                                 [0.0, 1.0, 0.0],
                                 [-math.sin(angle), 0.0, math.cos(angle)]], dtype=torch.float32)
        translation = torch.tensor([0.2, -0.1, 1.5], dtype=torch.float32)
        ext = torch.eye(4, dtype=torch.float32)[None, None].clone()
        ext[0, 0, :3, :3] = rotation
        ext[0, 0, :3, 3] = translation
        ixt = torch.eye(3, dtype=torch.float32)[None, None].clone()
        ixt[0, 0, 0, 0] = 320.0
        ixt[0, 0, 1, 1] = 300.0
        ixt[0, 0, 0, 2] = size / 2
        ixt[0, 0, 1, 2] = size / 2
        c2ws = affine_inverse(ext)
        pose_encoding = extri_intri_to_pose_encoding(c2ws, ixt, (size, size))
        extra["cam_enc_extrinsic"] = ext[0, 0].contiguous()
        extra["cam_enc_intrinsic"] = ixt[0, 0].contiguous()
        extra["cam_enc_pose_encoding"] = pose_encoding[0, 0].contiguous()
        extra["cam_enc_tokens"] = cam_enc(ext, ixt, (size, size))[0, 0].contiguous()

    globals()["_extra"] = extra
    return depth.contiguous()


def run_segformer(image):
    """SegFormer-B0 (ADE20k) class logits at the decode head's native quarter resolution."""
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    name = "nvidia/segformer-b0-finetuned-ade-512-512"
    processor = SegformerImageProcessor.from_pretrained(name)
    model = SegformerForSemanticSegmentation.from_pretrained(name).eval()
    inputs = processor(images=(image * 255).astype(np.uint8), return_tensors="pt", do_resize=False)
    with torch.no_grad():
        logits = model(**inputs).logits                         # [1, classes, h/4, w/4]
    return logits[0].permute(1, 2, 0).contiguous()              # [h/4, w/4, classes] to match NHWC


def run_swinir(image, checkpoint):
    """SwinIR classical SR, using the reference `network_swinir.py` from the SwinIR repository.

    IK_SWINIR_SCALE picks the release (4 by default; 3 is the non-power-of-two upsampler).

    Set IK_SWINIR_SRC to the directory holding a downloaded `network_swinir.py` (it is a single
    self-contained file; it needs `timm`).
    """
    import os
    import sys

    sys.path.insert(0, os.environ.get("IK_SWINIR_SRC", "."))
    from network_swinir import SwinIR

    # IK_SWINIR_SCALE selects the release. The upsampler differs by more than a factor: x4 runs two
    # ×2 pixel-shuffle stages, x3 runs one ×3 stage, so a checkpoint fits only its own scale.
    scale = int(os.environ.get("IK_SWINIR_SCALE", "4"))
    # IK_SWINIR_LIGHT selects the lightweight release, which is narrower, shallower, and reconstructs
    # through `pixelshuffledirect` — one convolution and one shuffle, with no surrounding convolutions.
    light = os.environ.get("IK_SWINIR_LIGHT", "") == "1"
    # IK_SWINIR_REAL selects a real-world release: `m` is the classical geometry with the
    # `nearest+conv` tail, `l` is wider (240), deeper (nine groups), eight heads, and the `3conv`
    # residual connection.
    real = os.environ.get("IK_SWINIR_REAL", "")
    if real == "m":
        model = SwinIR(upscale=scale, in_chans=3, img_size=64, window_size=8, img_range=1.0,
                       depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                       mlp_ratio=2, upsampler="nearest+conv", resi_connection="1conv").eval()
    elif real == "l":
        model = SwinIR(upscale=scale, in_chans=3, img_size=64, window_size=8, img_range=1.0,
                       depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240, num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                       mlp_ratio=2, upsampler="nearest+conv", resi_connection="3conv").eval()
    elif light:
        model = SwinIR(upscale=scale, in_chans=3, img_size=64, window_size=8, img_range=1.0,
                       depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                       mlp_ratio=2, upsampler="pixelshuffledirect", resi_connection="1conv").eval()
    else:
        model = SwinIR(upscale=scale, in_chans=3, img_size=48, window_size=8, img_range=1.0,
                       depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                       mlp_ratio=2, upsampler="pixelshuffle", resi_connection="1conv").eval()
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state.get("params_ema", state.get("params", state)), strict=True)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)      # [1, 3, H, W]
    with torch.no_grad():
        output = model(tensor)
    return output[0].permute(1, 2, 0).contiguous()                      # [scale·H, scale·W, 3] NHWC


def run_convtasnet(image, checkpoint):
    """Asteroid Conv-TasNet separation of a deterministic mono waveform, `[speakers, samples]`.

    The plate argument is unused: this is an audio model, so the record's `input_image` carries the
    waveform as a `[1, samples, 1]` tensor that the Swift side reads back.
    """
    from asteroid.models import ConvTasNet

    model = ConvTasNet.from_pretrained(checkpoint).eval()
    samples = 16000
    time = np.arange(samples, dtype=np.float32) / 16000.0
    # Two tones plus a little noise: enough structure for separation to do something measurable.
    generator = np.random.default_rng(3)
    wave = (0.4 * np.sin(2 * np.pi * 220 * time) + 0.3 * np.sin(2 * np.pi * 587 * time)
            + 0.02 * generator.standard_normal(samples).astype(np.float32)).astype(np.float32)
    globals()["_extra"] = {"waveform": torch.from_numpy(wave).contiguous()}
    with torch.no_grad():
        estimates = model(torch.from_numpy(wave).reshape(1, 1, -1))
    return estimates[0].contiguous()                            # [speakers, samples]


def _capture_init(init):
    """The reference decorator that records a module's construction arguments."""
    import functools

    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)
    return __init__


def _center_trim(tensor, reference):
    """The reference's `center_trim`, trimming a tensor to a length around its middle."""
    if hasattr(reference, "size"):
        reference = reference.size(-1)
    delta = tensor.size(-1) - reference
    if delta < 0:
        raise ValueError("tensor must be larger than reference")
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor


def _import_reference(source, package, name, siblings=(), injected=None):
    """Import one reference file from `source` as a member of a synthetic `package`.

    A reference module reaches for its siblings (`from .utils import capture_init`), and those siblings
    import dependencies the model itself never uses — demucs's `utils` pulls in `diffq`. `siblings` names
    files to load from `source` alongside the target; `injected` supplies the rest as ready-made modules,
    so only what the forward pass needs has to exist.
    """
    import importlib.util
    import os
    import types

    shell = types.ModuleType(package)
    shell.__path__ = [source]
    sys.modules[package] = shell
    for member, module in (injected or {}).items():
        sys.modules[f"{package}.{member}"] = module

    def load(member):
        spec = importlib.util.spec_from_file_location(f"{package}.{member}",
                                                      os.path.join(source, f"{member}.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{package}.{member}"] = module
        spec.loader.exec_module(module)
        return module

    for member in siblings:
        load(member)
    return load(name)


def run_demucs(image, checkpoint):
    """Demucs v2 music separation of a deterministic stereo waveform, `[stems, channels, samples]`.

    Set IK_DEMUCS_SRC to a directory holding `model.py` from the demucs v2 branch (it needs `julius`).
    The record's `waveform` carries the `[channels, samples]` mix the Swift side reads back.

    The reference pads to `valid_length` and center-trims the result, because a `context` of 3 leaves the
    decoder longer than the mix it started from. `NFKMLXDemucsNet.separate` does both internally, so the
    two are compared over the same window.
    """
    import os
    import types
    import torch.nn.functional as F

    utils = types.ModuleType("demucs_v2.utils")
    utils.capture_init, utils.center_trim = _capture_init, _center_trim
    module = _import_reference(os.environ.get("IK_DEMUCS_SRC", "."), "demucs_v2", "model",
                               injected={"utils": utils})
    model = module.Demucs(sources=["drums", "bass", "other", "vocals"], channels=64).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)

    samples = 22050
    time = np.arange(samples, dtype=np.float32) / 44100.0
    generator = np.random.default_rng(5)
    # A chord plus a periodic transient: harmonic content for the tonal stems, an onset for the drums.
    left = (0.3 * np.sin(2 * np.pi * 110 * time) + 0.2 * np.sin(2 * np.pi * 330 * time)
            + 0.2 * np.sin(2 * np.pi * 440 * time))
    right = (0.3 * np.sin(2 * np.pi * 110 * time + 0.4) + 0.25 * np.sin(2 * np.pi * 554 * time))
    click = ((np.arange(samples) % 5512) < 40).astype(np.float32) * 0.3
    wave = np.stack([left + click, right + click]).astype(np.float32)
    wave += 0.01 * generator.standard_normal(wave.shape).astype(np.float32)

    length = wave.shape[-1]
    padded = F.pad(torch.from_numpy(wave)[None], (0, model.valid_length(length) - length))
    with torch.no_grad():
        estimates = model(padded)
    globals()["_extra"] = {"waveform": torch.from_numpy(wave).contiguous()}
    return _center_trim(estimates, length)[0].contiguous()      # [stems, channels, samples]


def run_htdemucs_bag(image, checkpoint):
    """The fine-tuned Hybrid Transformer Demucs release (`htdemucs_ft`): four checkpoints of the base
    geometry, each contributing the stem it was fine-tuned for. `checkpoint` is a directory holding
    the four released files; they are read in the release's own order (drums, bass, other, vocals)
    and combined as `demucs.apply.BagOfModels` with its one-hot weights does — each stem is the
    weighted mean of the models' estimates, and the weights select one model per stem.
    """
    import functools

    torch.load = functools.partial(torch.load, weights_only=False)
    import demucs.states

    names = ["f7e0c4bc-ba3fe64a.th", "d12395a8-e57c48e6.th", "92cfc3b6-ef3bcb9c.th", "04573f0d-f3cf25b2.th"]
    models = []
    for name in names:
        model = demucs.states.load_model(os.path.join(checkpoint, name)).eval()
        model.use_train_segment = False
        models.append(model)

    samples = 44100
    time = np.arange(samples, dtype=np.float32) / 44100.0
    generator = np.random.default_rng(11)
    left = (0.3 * np.sin(2 * np.pi * 110 * time) + 0.2 * np.sin(2 * np.pi * 330 * time)
            + 0.2 * np.sin(2 * np.pi * 440 * time))
    right = (0.3 * np.sin(2 * np.pi * 110 * time + 0.4) + 0.25 * np.sin(2 * np.pi * 554 * time))
    click = ((np.arange(samples) % 5512) < 40).astype(np.float32) * 0.3
    wave = np.stack([left + click, right + click]).astype(np.float32)
    wave += 0.01 * generator.standard_normal(wave.shape).astype(np.float32)
    mix = torch.from_numpy(wave)[None]

    sources = len(models[0].sources)
    weights = torch.eye(sources)                                        # model i -> stem i
    estimate = torch.zeros(1, sources, 2, samples)
    totals = torch.zeros(sources)
    with torch.no_grad():
        for model, weight in zip(models, weights):
            estimate += model(mix) * weight.view(1, sources, 1, 1)
            totals += weight
    estimate /= totals.view(1, sources, 1, 1)
    globals()["_extra"] = {"waveform": torch.from_numpy(wave).contiguous()}
    return estimate[0].contiguous()                                     # [sources, channels, samples]


def run_htdemucs(image, checkpoint):
    """Hybrid Transformer Demucs (Demucs v4) separation of a deterministic stereo clip.

    `demucs` 4.0.1 is installed and `demucs.states.load_model` builds the network straight from the
    released checkpoint, so nothing is vendored here. It predates torch 2.6, so `torch.load` has to be
    patched to `weights_only=False` before the archive will open.

    `use_train_segment` is turned off so the record runs at the clip's own length. It is a padding
    policy, not a weight or a shape: with it on the reference zero-pads every input up to the 7.8-second
    training segment, which would make the record a hundred megabytes and the comparison no sharper.

    The record carries the branch seams as well as the waveform, so a failure says which half is wrong:
    `spectrogram` is the complex-as-channels input, `bottleneck_in`/`bottleneck_out` bracket the
    cross-transformer, and `freq_out`/`time_out` are the two branches' final predictions.
    """
    import functools

    torch.load = functools.partial(torch.load, weights_only=False)
    import demucs.states

    model = demucs.states.load_model(checkpoint).eval()
    model.use_train_segment = False

    samples = 44100
    time = np.arange(samples, dtype=np.float32) / 44100.0
    generator = np.random.default_rng(11)
    left = (0.3 * np.sin(2 * np.pi * 110 * time) + 0.2 * np.sin(2 * np.pi * 330 * time)
            + 0.2 * np.sin(2 * np.pi * 440 * time))
    right = (0.3 * np.sin(2 * np.pi * 110 * time + 0.4) + 0.25 * np.sin(2 * np.pi * 554 * time))
    click = ((np.arange(samples) % 5512) < 40).astype(np.float32) * 0.3
    wave = np.stack([left + click, right + click]).astype(np.float32)
    wave += 0.01 * generator.standard_normal(wave.shape).astype(np.float32)
    mix = torch.from_numpy(wave)[None]

    seams = {}
    handles = [
        model.encoder[3].register_forward_hook(
            lambda m, i, o: seams.__setitem__("bottleneck_in", o.detach())),
        model.crosstransformer.register_forward_hook(
            lambda m, i, o: seams.__setitem__("bottleneck_out", o[0].detach())),
        model.decoder[3].register_forward_hook(
            lambda m, i, o: seams.__setitem__("freq_out", o[0].detach())),
        model.tdecoder[3].register_forward_hook(
            lambda m, i, o: seams.__setitem__("time_out", o[0].detach())),
    ]
    with torch.no_grad():
        estimates = model(mix)
    for handle in handles:
        handle.remove()

    with torch.no_grad():
        spectrogram = model._magnitude(model._spec(mix))
    globals()["_extra"] = {
        "waveform": torch.from_numpy(wave).contiguous(),
        # Channels last, as the port holds them.
        "spectrogram": spectrogram[0].permute(1, 2, 0).contiguous(),
        "bottleneck_in": seams["bottleneck_in"][0].permute(1, 2, 0).contiguous(),
        "bottleneck_out": seams["bottleneck_out"][0].permute(1, 2, 0).contiguous(),
        "freq_out": seams["freq_out"][0].permute(1, 2, 0).contiguous(),
        "time_out": seams["time_out"][0].permute(1, 0).contiguous(),
    }
    return estimates[0].contiguous()                            # [stems, channels, samples]


def run_sd_unet(image, checkpoint):
    """A Stable Diffusion UNet forward, `[height, width, 4]` epsilon.

    diffusers cannot be installed beside the other oracles here — it needs a newer transformers than the
    4.33.3 the Whisper / CLIP / SegFormer records were measured against — so it lives in its own venv and
    this mode runs under that interpreter. Set IK_SD_CONFIG to the model's `unet/config.json`.

    The text conditioning is supplied as a tensor rather than encoded here, and the record carries it:
    the port takes a context and the caller brings the encoder, so the text tower is deliberately not on
    this path.
    """
    import json
    import os
    from diffusers import UNet2DConditionModel

    with open(os.environ["IK_SD_CONFIG"]) as handle:
        config = json.load(handle)
    config = {k: v for k, v in config.items() if not k.startswith("_")}
    from safetensors.torch import load_file
    model = UNet2DConditionModel.from_config(config).eval()
    # A half-precision release loads into the float32 model, so both sides compute the same way.
    model.load_state_dict({k: v.float() for k, v in load_file(checkpoint).items()}, strict=True)

    generator = np.random.default_rng(3)
    size = int(os.environ.get("IK_SD_SIZE", "32"))
    latent = torch.from_numpy(
        generator.standard_normal((1, config["in_channels"], size, size)).astype(np.float32))
    context = torch.from_numpy(
        generator.standard_normal((1, 77, config["cross_attention_dim"])).astype(np.float32))
    timestep = torch.tensor([201], dtype=torch.long)
    extra = {"latent": latent[0].permute(1, 2, 0).contiguous(),
             "context": context[0].contiguous(),
             "timestep": timestep.to(torch.float32)}
    kwargs = {}
    if config.get("num_class_embeds"):
        labels = torch.tensor([17], dtype=torch.long)
        kwargs["class_labels"] = labels
        extra["class_label"] = labels.to(torch.float32)
    if config.get("addition_embed_type") == "text_time":
        # The pooled text embedding and the size descriptor SDXL folds into its timestep.
        pooled_width = (config["projection_class_embeddings_input_dim"]
                        - config["addition_time_embed_dim"] * 6)
        pooled = torch.from_numpy(generator.standard_normal((1, pooled_width)).astype(np.float32))
        time_ids = torch.tensor([[size * 8, size * 8, 0, 0, size * 8, size * 8]], dtype=torch.float32)
        kwargs["added_cond_kwargs"] = {"text_embeds": pooled, "time_ids": time_ids}
        extra["pooled"] = pooled[0].contiguous()
        extra["time_ids"] = time_ids[0].contiguous()
    with torch.no_grad():
        out = model(latent, timestep, encoder_hidden_states=context, **kwargs).sample
    globals()["_extra"] = extra
    return out[0].permute(1, 2, 0).contiguous()                 # [height, width, 4]


def run_sd_vae(image, checkpoint):
    """A Stable Diffusion autoencoder round trip, `[height, width, 3]`.

    The record carries the encoder's latent mean as well as the decoded image, so a failure says which
    half diverged. Set IK_SD_CONFIG to the model's `vae/config.json`.
    """
    import json
    import os
    from diffusers import AutoencoderKL

    with open(os.environ["IK_SD_CONFIG"]) as handle:
        config = json.load(handle)
    config = {k: v for k, v in config.items() if not k.startswith("_")}
    from safetensors.torch import load_file
    model = AutoencoderKL.from_config(config).eval()
    # The SD 1.5 autoencoders predate the attention rename that `from_pretrained` patches up on the
    # way in. Renaming here keeps the load strict, which is what proves the geometry matches.
    renames = {".query.": ".to_q.", ".key.": ".to_k.", ".value.": ".to_v.",
               ".proj_attn.": ".to_out.0."}
    state = {}
    for key, value in load_file(checkpoint).items():
        for old_part, new_part in renames.items():
            if old_part in key and "attentions" in key:
                key = key.replace(old_part, new_part)
        state[key] = value
    model.load_state_dict(state, strict=True)

    plate = image                                               # [H, W, 3] in 0...1, from --size/--plate
    x = torch.from_numpy(plate).permute(2, 0, 1)[None] * 2 - 1  # the reference works in -1...1
    with torch.no_grad():
        posterior = model.encode(x).latent_dist
        latent = posterior.mean
        decoded = model.decode(latent).sample
    # The plate is already the record's `input_image`; repeating it here would make two entries share
    # one storage, which safetensors refuses to write.
    globals()["_extra"] = {"latent": latent[0].permute(1, 2, 0).contiguous()}
    return decoded[0].permute(1, 2, 0).contiguous()             # [height, width, 3]


def run_sd_scheduler(image):
    """The DDIM sampler's schedule and per-step update, independent of any network.

    The InferKitMLX networks are at parity, but the loop that iterates them was never measured. This
    drives diffusers' own `DDIMScheduler` over a fixed sequence of latents and model outputs, so a
    comparison isolates the sampler's arithmetic from the UNet entirely.

    The configuration is the released `scheduler/scheduler_config.json` for SD 1.5 inpainting, whose
    `steps_offset` of 1 and `set_alpha_to_one: false` both move the schedule.

    Runs under the diffusers virtual environment, like the other `sd_*` modes.
    """
    import os
    from diffusers import DDIMScheduler

    steps = int(os.environ.get("IK_SD_STEPS", "20"))
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                              beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False, steps_offset=1,
                              prediction_type="epsilon")
    scheduler.set_timesteps(steps)

    generator = np.random.default_rng(7)
    shape = (1, 4, 8, 8)
    latent = torch.from_numpy(generator.standard_normal(shape).astype(np.float32))
    latents_in, predictions, latents_out = [], [], []
    for t in scheduler.timesteps:
        prediction = torch.from_numpy(generator.standard_normal(shape).astype(np.float32))
        latents_in.append(latent[0].permute(1, 2, 0).contiguous())
        predictions.append(prediction[0].permute(1, 2, 0).contiguous())
        latent = scheduler.step(prediction, t, latent).prev_sample
        latents_out.append(latent[0].permute(1, 2, 0).contiguous())

    clean = torch.from_numpy(generator.standard_normal(shape).astype(np.float32))
    noise = torch.from_numpy(generator.standard_normal(shape).astype(np.float32))
    noised = scheduler.add_noise(clean, noise, scheduler.timesteps[:1])

    globals()["_extra"] = {
        "timesteps": scheduler.timesteps.to(torch.float32),
        "alphas_cumprod": scheduler.alphas_cumprod.to(torch.float32),
        "latents_in": torch.stack(latents_in),
        "predictions": torch.stack(predictions),
        "clean": clean[0].permute(1, 2, 0).contiguous(),
        "noise": noise[0].permute(1, 2, 0).contiguous(),
        "noised": noised[0].permute(1, 2, 0).contiguous(),
    }
    return torch.stack(latents_out)                             # [steps, height, width, 4]


# The prompts the tokenizer and text-encoder records are measured on. The Swift side declares the same
# list, so a change to either without the other shows up as a mismatch rather than passing quietly.
SD_PROMPTS = [
    "a photograph of an astronaut riding a horse",
    "A PHOTO, of  a   cat!!! (highly detailed), 8k",
    "",
    "  spaced   out  text  ",
    # Written with escapes so the exact code points are visible: byte-level BPE splits a multi-byte
    # character into bytes, which is a real case for the port to reproduce.
    "2024 was a year; na\u00efve caf\u00e9 \u2014 r\u00e9sum\u00e9",
]


def run_sd_tokenizer(image):
    """The ids CLIPTokenizer produces for SD_PROMPTS, one entry per prompt.

    The text encoder embeds ids rather than text, so the tokenizer is a separate seam: a prompt that
    tokenizes differently reaches the model as a different sentence, and the embedding comparison
    alone cannot tell that apart from a wrong weight.

    Set IK_SD_TOKENIZER to a release's `tokenizer/` directory. Runs under the diffusers virtual
    environment, like the other `sd_*` modes.
    """
    import os
    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(os.environ["IK_SD_TOKENIZER"])
    extra = {}
    for index, prompt in enumerate(SD_PROMPTS):
        # The ids for the text alone; the start and end markers are the model input's, added below.
        raw = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt))
        extra[f"prompt_{index}"] = torch.tensor(raw, dtype=torch.int32)
    padded = tokenizer(SD_PROMPTS, padding="max_length", max_length=77, truncation=True,
                       return_tensors="pt")["input_ids"]
    extra["markers"] = torch.tensor([tokenizer.bos_token_id, tokenizer.eos_token_id,
                                     tokenizer.pad_token_id], dtype=torch.int32)
    globals()["_extra"] = extra
    return padded.to(torch.int32).contiguous()                  # [prompts, 77]


def run_sd_text_encoder(image, checkpoint):
    """A Stable Diffusion text encoder's hidden states for SD_PROMPTS[0], `[77, width]`.

    `--checkpoint` is the release's `text_encoder` DIRECTORY, so the reference builds itself from the
    config that ships beside the weights. A tower carrying a projection (SDXL's second) is built as
    `CLIPTextModelWithProjection` and the record adds its pooled embedding.

    The record carries the reference's own token ids: the port embeds ids, not text, so the tokenizer
    is measured separately by `sd_tokenizer` rather than folded into this number.

    Both hidden states are recorded. A release reads one of them — SD 1.x and 2.x take the last, after
    the final layer normalization; SDXL takes the penultimate, before it — and recording both means a
    disagreement says which convention diverged rather than only that something did.

    Set IK_SD_TOKENIZER to the release's `tokenizer/`. Runs under the diffusers virtual environment.
    """
    import json
    import os
    from transformers import CLIPTokenizer

    with open(os.path.join(checkpoint, "config.json")) as handle:
        config = json.load(handle)
    projected = "CLIPTextModelWithProjection" in config.get("architectures", [])
    if projected:
        from transformers import CLIPTextModelWithProjection as Model
    else:
        from transformers import CLIPTextModel as Model
    # A release published only in half precision names its files `.fp16.safetensors`.
    variant = None if os.path.exists(os.path.join(checkpoint, "model.safetensors")) else "fp16"
    model = Model.from_pretrained(checkpoint, torch_dtype=torch.float32, variant=variant).eval()

    tokenizer = CLIPTokenizer.from_pretrained(os.environ["IK_SD_TOKENIZER"])
    encoded = tokenizer(SD_PROMPTS[0], padding="max_length", max_length=77, truncation=True,
                        return_tensors="pt")
    with torch.no_grad():
        out = model(encoded["input_ids"], output_hidden_states=True)

    extra = {"tokens": encoded["input_ids"][0].to(torch.int32).contiguous(),
             "penultimate": out.hidden_states[-2][0].contiguous()}
    if projected:
        extra["pooled"] = out.text_embeds[0].contiguous()
    globals()["_extra"] = extra
    return out.last_hidden_state[0].contiguous()                # [77, width]


def run_sd_text_to_image(image, checkpoint):
    """A whole Stable Diffusion text-to-image run, `[height, width, 3]` in 0...1.

    `--checkpoint` is the release DIRECTORY, so the reference assembles itself from the same files the
    port loads. The sampler is swapped to DDIM, which is the one InferKitMLX implements; the released
    configuration's `steps_offset` and `set_alpha_to_one` carry over through `from_config`.

    The record carries the initial latent. Matching a random source across two implementations proves
    nothing about either, so the port starts from the reference's own noise and the comparison is of
    the text encoder, the UNet, the sampler, and the autoencoder end to end.

    IK_SD_STEPS sets the step count and IK_SD_SIZE the latent side. Runs under the diffusers virtual
    environment, like the other `sd_*` modes.
    """
    import json
    import os
    from diffusers import DDIMScheduler, DiffusionPipeline

    with open(os.path.join(checkpoint, "model_index.json")) as handle:
        index = json.load(handle)
    extra = {} if index["_class_name"].startswith("StableDiffusionXL") else {
        "safety_checker": None, "requires_safety_checker": False}
    variant = None if os.path.exists(
        os.path.join(checkpoint, "unet", "diffusion_pytorch_model.safetensors")) else "fp16"
    pipeline = DiffusionPipeline.from_pretrained(checkpoint, torch_dtype=torch.float32,
                                                 variant=variant, **extra)
    # DDIM is the sampler InferKitMLX implements. A release whose own scheduler counts its schedule
    # down from the end of the training range keeps that spacing through `from_config`.
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)

    steps = int(os.environ.get("IK_SD_STEPS", "4"))
    side = int(os.environ.get("IK_SD_SIZE", "32"))
    guidance = float(os.environ.get("IK_SD_GUIDANCE", "7.5"))
    # A release with `force_zeros_for_empty_prompt` conditions on zeros when NO negative prompt is
    # supplied, and on the embedding of an empty sentence when one is. IK_SD_NEGATIVE picks which.
    negative = None if os.environ.get("IK_SD_NEGATIVE") == "none" else ""
    generator = np.random.default_rng(11)
    channels = pipeline.unet.config.in_channels
    latent = torch.from_numpy(generator.standard_normal((1, channels, side, side)).astype(np.float32))

    # The latent after each step, so a whole-picture mismatch says WHICH step diverged rather than
    # only that something did.
    trace = []

    def capture(pipe, index, timestep, kwargs):
        trace.append(kwargs["latents"][0].permute(1, 2, 0).contiguous().clone())
        return kwargs

    with torch.no_grad():
        result = pipeline(prompt=SD_PROMPTS[0], negative_prompt=negative, num_inference_steps=steps,
                          guidance_scale=guidance, latents=latent, height=side * 8, width=side * 8,
                          output_type="np", callback_on_step_end=capture)
    # The conditioning and the first guided prediction, so a divergence separates the text tower from
    # the sampler from the UNet without another run.
    extra = {"latent": latent[0].permute(1, 2, 0).contiguous(),
             "latents": torch.stack(trace),
             "settings": torch.tensor([steps, side, guidance, 1 if negative is None else 0], dtype=torch.float32)}
    two_towers = hasattr(pipeline, "text_encoder_2")
    with torch.no_grad():
        if two_towers:
            conditional, unconditional, pooled, unpooled = pipeline.encode_prompt(
                prompt=SD_PROMPTS[0], device="cpu", num_images_per_prompt=1,
                do_classifier_free_guidance=guidance > 1, negative_prompt=negative)
            extra["pooled"] = pooled[0].contiguous()
            added = {"text_embeds": pooled,
                     "time_ids": torch.tensor([[side * 8, side * 8, 0, 0, side * 8, side * 8]],
                                              dtype=torch.float32)}
        else:
            conditional, unconditional = pipeline.encode_prompt(
                SD_PROMPTS[0], device="cpu", num_images_per_prompt=1,
                do_classifier_free_guidance=guidance > 1, negative_prompt=negative)[:2]
            added = None
        pipeline.scheduler.set_timesteps(steps)
        first = pipeline.scheduler.timesteps[0]
        kwargs = {} if added is None else {"added_cond_kwargs": added}
        if guidance > 1:
            both = torch.cat([unconditional, conditional])
            if added is not None:
                kwargs["added_cond_kwargs"] = {"text_embeds": torch.cat([unpooled, pooled]),
                                               "time_ids": torch.cat([added["time_ids"]] * 2)}
            predicted = pipeline.unet(torch.cat([latent] * 2), first,
                                      encoder_hidden_states=both, **kwargs).sample
            prediction = predicted[0] + guidance * (predicted[1] - predicted[0])
        else:
            prediction = pipeline.unet(latent, first, encoder_hidden_states=conditional,
                                       **kwargs).sample[0]

    extra["context"] = conditional[0].contiguous()
    extra["uncontext"] = (unconditional[0].contiguous() if unconditional is not None
                          else torch.zeros_like(conditional[0]))
    extra["first_prediction"] = prediction.permute(1, 2, 0).contiguous()
    globals()["_extra"] = extra
    return torch.from_numpy(result.images[0]).contiguous()      # [height, width, 3] in 0...1


def run_denoiser(image, checkpoint):
    """facebookresearch/denoiser speech enhancement of a deterministic noisy clip, `[samples]`.

    Set IK_DENOISER_SRC to a directory holding `demucs.py` and `resample.py` from the denoiser
    repository. The released files are bare state dicts, so the one thing dns48 and dns64 differ in —
    the hidden width — is read off the first encoder weight rather than guessed.

    This model is `NFKMLXDemucsNet` in its speech configuration — the same encoder, bottleneck, and
    decoder the music model uses — so this record also guards the shared module against a change made
    for one family breaking the other.
    """
    import os
    import types

    utils = types.ModuleType("denoiser.utils")
    utils.capture_init = _capture_init
    module = _import_reference(os.environ.get("IK_DENOISER_SRC", "."), "denoiser", "demucs",
                               siblings=["resample"], injected={"utils": utils})
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("state", state)
    model = module.Demucs(hidden=state["encoder.0.0.weight"].shape[0]).eval()
    model.load_state_dict(state, strict=True)

    samples = 16000
    time = np.arange(samples, dtype=np.float32) / 16000.0
    generator = np.random.default_rng(9)
    # A voiced-like harmonic stack under broadband noise: something for the model to actually remove.
    speech = sum(0.3 / (h + 1) * np.sin(2 * np.pi * 140 * (h + 1) * time) for h in range(5))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * time)
    wave = (speech * envelope + 0.05 * generator.standard_normal(samples)).astype(np.float32)

    with torch.no_grad():
        cleaned = model(torch.from_numpy(wave).reshape(1, 1, -1))
    globals()["_extra"] = {"waveform": torch.from_numpy(wave).contiguous()}
    return cleaned[0, 0].contiguous()                           # [samples]


def _reference_source():
    """The directory holding the single-file reference implementations, from IK_REF_SRC."""
    import os
    return os.environ.get("IK_REF_SRC", ".")


def run_u2net(image, checkpoint):
    """U²-Net saliency, `[H, W]` in 0...1, from the reference `model/u2net.py`.

    The reference's eval script resizes to 320×320 before the network; feeding a 320×320 plate makes
    that resize an identity, so only the network and its normalization are compared.
    """
    module = _import_reference(_reference_source(), "u2net_ref", "u2net")
    # `u2netp` is the light network, a separate class in the reference rather than a configuration.
    light = os.environ.get("IK_U2NET_VARIANT", "full") == "light"
    model = (module.U2NETP(3, 1) if light else module.U2NET(3, 1)).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)

    # `ToTensorLab`: scale by the plate's own maximum, then normalize with ImageNet statistics.
    scaled = image / max(image.max(), 1e-8)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    deviation = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = torch.from_numpy(((scaled - mean) / deviation).transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        saliency = model(tensor)[0]
    return saliency[0, 0].contiguous()                          # [H, W]


def run_isnet(image, checkpoint):
    """IS-Net (DIS) dichotomous segmentation on a released checkpoint, from the DIS `models/isnet.py`.

    The successor to U²-Net by the same authors: the same Residual U-blocks behind a stride-2 stem,
    wider stages, and six separate side maps with no fusion convolution. The reference inference
    resizes to 1024x1024, scales to [0,1], and normalizes with mean 0.5 and unit standard deviation;
    feeding a 1024x1024 plate makes the resize an identity, so only the network is compared.

    IK_REF_SRC holds a `dis/` directory with `isnet.py`. The record carries the stem and the deepest
    encoder stage beside the five coarser side maps, which is what separates a stem mistake from a
    decoder one.
    """
    module = _import_reference(os.path.join(_reference_source(), "dis"), "isnet_ref", "isnet")
    model = module.ISNetDIS(3, 1).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)

    seams = {}
    handles = [
        model.conv_in.register_forward_hook(lambda m, i, o: seams.__setitem__("stem", o)),
        model.stage1.register_forward_hook(lambda m, i, o: seams.__setitem__("stage1", o)),
        model.stage6.register_forward_hook(lambda m, i, o: seams.__setitem__("stage6", o)),
    ]

    # The reference scales by 255 and then normalizes with mean 0.5, standard deviation 1.0.
    tensor = torch.from_numpy(image - 0.5).permute(2, 0, 1)[None].float()
    with torch.no_grad():
        sides, _ = model(tensor)
    for handle in handles:
        handle.remove()

    extra = {"stem": seams["stem"][0].permute(1, 2, 0).contiguous(),
             "stage1": seams["stage1"][0].permute(1, 2, 0).contiguous(),
             "stage6": seams["stage6"][0].permute(1, 2, 0).contiguous()}
    for index in range(1, 6):
        extra[f"side{index + 1}"] = sides[index][0, 0].contiguous()
    globals()["_extra"] = extra
    return sides[0][0, 0].contiguous()                          # [H, W]


def run_adain(image, checkpoint):
    """AdaIN arbitrary style transfer, from naoto0804's `net.py` and `function.py`.

    `--checkpoint` is the released decoder; IK_ADAIN_VGG names the released normalized VGG beside it.
    IK_REF_SRC holds an `adain/` directory with `net.py` and `function.py` (they import each other
    flatly, so the directory itself goes on the path). Both images reach the encoder unnormalized: the
    released VGG's first 1x1 convolution carries the normalization.

    The style plate is a rolled and channel-rotated copy of the content, which gives the transfer real
    statistics to move while keeping both sides reading identical pixels. The record carries the two
    feature maps and the normalized features beside the decoded image, so a mismatch says whether the
    encoder, the normalization, or the decoder is wrong.
    """
    sys.path.insert(0, os.path.join(_reference_source(), "adain"))
    import net as adain_net
    from function import adaptive_instance_normalization

    vgg = adain_net.vgg
    vgg.load_state_dict(torch.load(os.environ["IK_ADAIN_VGG"], map_location="cpu", weights_only=True))
    vgg = torch.nn.Sequential(*list(vgg.children())[:31]).eval()
    decoder = adain_net.decoder
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # A decoder trained through a wrapper module carries its attribute name on every key.
    state = {k[len("net."):] if k.startswith("net.") else k: v for k, v in state.items()}
    decoder.load_state_dict(state)
    decoder = decoder.eval()

    style_plate = np.ascontiguousarray(np.roll(image, 11, axis=1)[..., ::-1])
    content = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    style = torch.from_numpy(style_plate.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        content_features = vgg(content)
        style_features = vgg(style)
        normalized = adaptive_instance_normalization(content_features, style_features)
        decoded = decoder(normalized)

    globals()["_extra"] = {
        "style_image": torch.from_numpy(style_plate).contiguous(),
        "content_features": content_features[0].permute(1, 2, 0).contiguous(),
        "style_features": style_features[0].permute(1, 2, 0).contiguous(),
        "normalized": normalized[0].permute(1, 2, 0).contiguous(),
    }
    return decoded[0].permute(1, 2, 0).contiguous()


def run_hat(image, checkpoint):
    """HAT super-resolution on a released checkpoint, from XPixelGroup's own `hat_arch.py`.

    IK_REF_SRC holds a `hat/` directory with `hat_arch.py`. The file imports basicsr for a registry
    decorator and two initializer helpers, none of which affect a loaded model, so they are shimmed
    rather than installing the training framework. IK_HAT_GROUPS and IK_HAT_DIM name the release's
    geometry (HAT-L is 12 groups at 180 channels).

    The record carries the first block, the first group's overlapping cross-attention, the first
    group, and the whole deep-feature trunk beside the upscaled image, which separates the window
    attention from the overlapping attention from the reconstruction.
    """
    import types

    registry = types.ModuleType("basicsr.utils.registry")

    class _Registry:
        def register(self, *args, **kwargs):
            def identity(cls):
                return cls
            return identity(args[0]) if args and callable(args[0]) else identity

    registry.ARCH_REGISTRY = _Registry()
    arch_util = types.ModuleType("basicsr.archs.arch_util")
    arch_util.to_2tuple = lambda value: value if isinstance(value, tuple) else (value, value)
    arch_util.trunc_normal_ = lambda tensor, **kwargs: tensor
    for name, module in [("basicsr", types.ModuleType("basicsr")),
                         ("basicsr.utils", types.ModuleType("basicsr.utils")),
                         ("basicsr.utils.registry", registry),
                         ("basicsr.archs", types.ModuleType("basicsr.archs")),
                         ("basicsr.archs.arch_util", arch_util)]:
        sys.modules.setdefault(name, module)

    module = _import_reference(os.path.join(_reference_source(), "hat"), "hat_ref", "hat_arch")
    groups = int(os.environ.get("IK_HAT_GROUPS", "12"))
    dimensions = int(os.environ.get("IK_HAT_DIM", "180"))
    model = module.HAT(img_size=64, patch_size=1, in_chans=3, embed_dim=dimensions,
                       depths=(6,) * groups, num_heads=(6,) * groups, window_size=16,
                       compress_ratio=3, squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5,
                       mlp_ratio=2., upsampler="pixelshuffle", resi_connection="1conv",
                       upscale=4, img_range=1.).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("params_ema", state.get("params", state))
    model.load_state_dict(state, strict=True)

    seams = {}
    group = model.layers[0]
    handles = [
        model.conv_first.register_forward_hook(lambda m, i, o: seams.__setitem__("shallow", o)),
        group.residual_group.blocks[0].register_forward_hook(lambda m, i, o: seams.__setitem__("block0", o)),
        group.residual_group.overlap_attn.register_forward_hook(lambda m, i, o: seams.__setitem__("ocab0", o)),
        group.register_forward_hook(lambda m, i, o: seams.__setitem__("group0", o)),
    ]

    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        features = model.forward_features(model.conv_first(tensor - model.mean.type_as(tensor)))
        upscaled = model(tensor)
    for handle in handles:
        handle.remove()

    tokens = dimensions
    globals()["_extra"] = {
        "shallow": seams["shallow"][0].permute(1, 2, 0).contiguous(),
        "block0": seams["block0"][0].reshape(64, 64, tokens).contiguous(),
        "ocab0": seams["ocab0"][0].reshape(64, 64, tokens).contiguous(),
        "group0": seams["group0"][0].reshape(64, 64, tokens).contiguous(),
        "features": features[0].permute(1, 2, 0).contiguous(),
        "rpi_oca": model.relative_position_index_OCA.to(torch.int32).contiguous(),
        "rpi_sa": model.relative_position_index_SA.to(torch.int32).contiguous(),
    }
    return upscaled[0].permute(1, 2, 0).contiguous()


def run_raft(image, checkpoint):
    """RAFT optical flow between a plate and a shifted copy of it, from princeton-vl's own `core/`.

    IK_REF_SRC holds a `raft/` directory with `raft.py`, `corr.py`, `extractor.py`, `update.py`, and
    `utils/utils.py` (the package imports its siblings flatly, so its directory goes on the path rather
    than through `_import_reference`). `--checkpoint` is `raft-things.pth`, whose keys carry the
    `module.` prefix of the training wrapper.

    The record carries `frame1` (the shifted copy both sides read) and both of the reference's outputs:
    `output` is the eighth-resolution flow the recurrent update produces, and `flow_up` the
    convex-mask upsampling of it. Comparing the low-resolution field first separates the network from
    the upsampler.
    """
    import os

    sys.path.insert(0, os.path.join(_reference_source(), "raft"))
    from raft import RAFT

    class Arguments:
        small = False
        dropout = 0
        alternate_corr = False
        mixed_precision = False

        def __contains__(self, name):
            return hasattr(self, name)

    model = RAFT(Arguments())
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict({name.replace("module.", "", 1): value for name, value in state.items()},
                          strict=True)
    model.eval()

    # A known translation gives the network real motion to find; rolling keeps both frames identical
    # in content, so any disagreement is the implementation rather than the input.
    shift = (5, 7)
    second = np.roll(np.roll(image, shift[0], axis=0), shift[1], axis=1)
    frames = [torch.from_numpy(np.ascontiguousarray(plate)).permute(2, 0, 1)[None] * 255.0
              for plate in (image, second)]
    with torch.no_grad():
        # Both sides must run the same number of recurrent updates; six is the port's default.
        low, up = model(frames[0], frames[1], iters=6, test_mode=True)
    globals()["_extra"] = {"frame1": torch.from_numpy(np.ascontiguousarray(second)),
                           "flow_up": up[0].permute(1, 2, 0).contiguous()}
    return low[0].permute(1, 2, 0).contiguous()                 # [H/8, W/8, 2] to match NHWC


def run_audio_tagger(image, checkpoint):
    """PANNs Cnn14 AudioSet tag probabilities, `[classes]`, from the reference `models.py`.

    IK_REF_SRC holds `models.py` (saved as `panns_models.py`) and its sibling `pytorch_utils.py` from
    qiuqiangkong/audioset_tagging_cnn; the reference needs `torchlibrosa`. The record's `waveform`
    carries the clip both sides read, `features` the reference's own mel spectrogram, and `embedding`
    the 2048-wide clip vector the classifier reads — three seams, so a front-end mismatch, a network
    mismatch, and a classifier mismatch are told apart rather than summed into one number.
    """
    import os

    source = _reference_source()
    sys.path.insert(0, source)                                  # models.py imports pytorch_utils flatly
    module = _import_reference(source, "panns_ref", "panns_models")
    model = module.Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
                         fmin=50, fmax=14000, classes_num=527).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"],
                          strict=True)

    rate, samples = 32000, 64000
    time = np.arange(samples, dtype=np.float32) / rate
    generator = np.random.default_rng(19)
    # A voiced-like harmonic stack, then noise: a clip with real structure in both halves.
    tone = sum(0.3 / (h + 1) * np.sin(2 * np.pi * 220 * (h + 1) * time) for h in range(6))
    wave = np.where(time < 1.0, tone, 0.1 * generator.standard_normal(samples)).astype(np.float32)

    with torch.no_grad():
        batch = torch.from_numpy(wave)[None]
        mel = model.logmel_extractor(model.spectrogram_extractor(batch))    # [1, 1, frames, mels]
        output = model(batch)
    globals()["_extra"] = {"waveform": torch.from_numpy(wave).contiguous(),
                           "features": mel[0, 0].contiguous(),
                           "embedding": output["embedding"][0].contiguous()}
    return output["clipwise_output"][0].contiguous()            # [classes]


def run_pose(image, checkpoint):
    """SimpleBaseline joint heatmaps, `[H/4, W/4, joints]`, from the reference `pose_resnet.py`.

    IK_REF_SRC holds `pose_resnet.py` from microsoft/human-pose-estimation.pytorch. `--checkpoint` is the
    mmpose ResNet-50 COCO release, whose keys are the reference's under a `backbone.`/`head.` prefix, so
    the strict load doubles as proof that the two architectures are the same one.

    The network is fully convolutional, so it takes the square plate as it stands; feeding it directly
    keeps the person-crop resize out of the comparison and leaves the network and its normalization.
    """
    import os
    import types as _types

    module = _import_reference(_reference_source(), "pose_ref", "pose_resnet")
    extra = _types.SimpleNamespace(DECONV_WITH_BIAS=False, NUM_DECONV_LAYERS=3,
                                   NUM_DECONV_FILTERS=[256, 256, 256], NUM_DECONV_KERNELS=[4, 4, 4],
                                   FINAL_CONV_KERNEL=1)
    cfg = _types.SimpleNamespace(MODEL=_types.SimpleNamespace(NUM_JOINTS=17, EXTRA=extra))
    model = module.PoseResNet(module.Bottleneck, [3, 4, 6, 3], cfg).eval()

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pose-to-safetensors"))
    import convert as pose_convert                              # its mmengine stub, not a second copy
    state = pose_convert.extract_state_dict(pose_convert.load_checkpoint(checkpoint))
    trimmed = {name.split(".", 1)[1]: value for name, value in state.items()
               if name.startswith(("backbone.", "head."))}
    model.load_state_dict(trimmed, strict=True)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    deviation = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = torch.from_numpy(((image - mean) / deviation).transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        heatmaps = model(tensor)
    return heatmaps[0].permute(1, 2, 0).contiguous()            # [H/4, W/4, joints] to match NHWC


def run_rvm(image, checkpoint):
    """Robust Video Matting first-frame alpha, `[H, W]` in 0...1, from the reference `MattingNetwork`.

    IK_REF_SRC holds an `rvm/` directory with the repository's `model/` files (`model.py` and its
    siblings; it needs torchvision). The record's `foreground` carries the composited full-resolution
    foreground, and `alpha_downsampled` / `foreground_downsampled` a second pass at
    `downsample_ratio=0.5`, which routes through the deep-guided-filter refiner the full-resolution
    pass never touches — so the refiner is measured separately from the core network.
    """
    import os

    module = _import_reference(os.path.join(_reference_source(), "rvm"), "rvm_ref", "model",
                               siblings=["mobilenetv3", "resnet", "lraspp", "decoder",
                                         "fast_guided_filter", "deep_guided_filter"])
    # IK_RVM_VARIANT selects the released encoder (mobilenetv3 by default, or resnet50).
    model = module.MattingNetwork(os.environ.get("IK_RVM_VARIANT", "mobilenetv3")).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)

    src = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        fgr, pha, *_ = model(src)
        fgr_small, pha_small, *_ = model(src, downsample_ratio=0.5)
    globals()["_extra"] = {"foreground": fgr[0].permute(1, 2, 0).contiguous(),
                           "alpha_downsampled": pha_small[0, 0].contiguous(),
                           "foreground_downsampled": fgr_small[0].permute(1, 2, 0).contiguous()}
    return pha[0, 0].contiguous()


def run_yolo(image, checkpoint):
    """YOLOv8 pre-suppression predictions, `[anchors, 4 + classes]`, from ultralytics' own model.

    The decoded tensor is the right seam: box centers and sizes in pixels plus sigmoid class
    probabilities for every anchor across the three strides, before any thresholding or NMS — so a
    wrong DFL decode or anchor grid cannot hide behind a lucky suppression. Feed a square plate
    (`--size 640 --image <photo>`) so the reference's letterboxing is an identity.
    """
    from ultralytics import YOLO

    model = YOLO(checkpoint)
    net = model.model.float().eval()
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        predictions = net(tensor)
    decoded = predictions[0] if isinstance(predictions, (list, tuple)) else predictions
    return decoded[0].transpose(0, 1).contiguous()              # [anchors, 4 + classes]



def run_yolo_loss(image):
    """YOLO's training objective, ultralytics' own `v8DetectionLoss`, on identical head outputs.

    A three-class YOLOv8n built from the package's bundled `yolov8n.yaml` supplies the loss its
    strides, `reg_max`, and `get_cfg()`'s gains; no weights are needed, because the head outputs are
    synthesized: seeded class logits, and box distributions peaked near bin 2 so every predicted box
    overlaps the boxes it lands in (a zero-metric candidate would make the assigner's top-k depend on
    `torch.topk`'s unspecified tie order). The batch has two 64-pixel images: three boxes in the first
    (two overlapping, so an anchor is claimed twice, and one narrower than the first stride, so the
    assigner grows it) and one in the second. Runs on the interpreter that has ultralytics installed.
    """
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.loss import v8DetectionLoss

    classes, size, batch = 3, 64, 2
    model = DetectionModel("yolov8n.yaml", nc=classes, verbose=False)
    model.args = get_cfg()
    criterion = v8DetectionLoss(model)
    strides = [int(s) for s in model.model[-1].stride.tolist()]
    feats = [torch.zeros(batch, 1, size // s, size // s) for s in strides]
    anchors = sum((size // s) ** 2 for s in strides)
    reg_max = criterion.reg_max

    generator = torch.Generator().manual_seed(41)
    bins = torch.arange(reg_max, dtype=torch.float32)
    peaked = -0.5 * (bins - 2.2) ** 2                                      # [reg_max]
    boxes = peaked.view(1, 1, reg_max, 1) + 0.3 * torch.randn(batch, 4, reg_max, anchors, generator=generator)
    boxes = boxes.reshape(batch, 4 * reg_max, anchors)
    scores = torch.randn(batch, classes, anchors, generator=generator)

    truth = torch.tensor([[0, 0, 10, 12, 46, 50],
                          [0, 1, 20, 18, 56, 58],
                          [0, 2, 40, 6, 45, 14],
                          [1, 1, 8, 30, 60, 62]], dtype=torch.float32)            # image, class, x1, y1, x2, y2
    xyxy = truth[:, 2:]
    xywh = torch.cat(((xyxy[:, :2] + xyxy[:, 2:]) / 2, xyxy[:, 2:] - xyxy[:, :2]), 1) / size
    batch_dict = {"batch_idx": truth[:, 0], "cls": truth[:, 1:2], "bboxes": xywh}
    preds = {"boxes": boxes, "scores": scores, "feats": feats}

    (fg_mask, _, _, _, _), _, _ = criterion.get_assigned_targets_and_loss(preds, batch_dict)
    total, items = criterion.loss(preds, batch_dict)
    globals()["_extra"] = {
        "box_distribution": boxes.permute(0, 2, 1).contiguous(),              # [batch, anchors, 4 · reg_max]
        "class_logits": scores.permute(0, 2, 1).contiguous(),                 # [batch, anchors, classes]
        "targets": truth.contiguous(),
        "strides": torch.tensor(strides, dtype=torch.int32),
        "image_size": torch.tensor([size], dtype=torch.int32),
        "components": torch.stack([items["box_loss"], items["cls_loss"], items["dfl_loss"]]).float(),
        "foreground": fg_mask.reshape(-1).nonzero().reshape(-1).to(torch.int32).contiguous(),
    }
    return total.sum().reshape(1).contiguous()



def run_yolo_e2e_loss(image):
    """The end-to-end generations' objective, ultralytics' own `E2ELoss`, for YOLOv10n (DFL) and YOLO26n
    (`reg_max` 1, an L1 on the side distances), each three-class from its bundled yaml.

    Each model's criterion (`model.init_criterion()`) scores synthesized one-to-many and one-to-one head
    outputs over two 64-pixel images with the boxes `yolo_loss` uses; the run is 4 epochs, and the
    total is recorded at epoch 0 and again after one `update()`, where the one-to-many weight has moved
    from 0.8. Each branch's own `v8DetectionLoss` terms are recorded too. Runs on the interpreter that
    has ultralytics installed.
    """
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel

    classes, size, batch = 3, 64, 2
    truth = torch.tensor([[0, 0, 10, 12, 46, 50],
                          [0, 1, 20, 18, 56, 58],
                          [0, 2, 40, 6, 45, 14],
                          [1, 1, 8, 30, 60, 62]], dtype=torch.float32)
    xyxy = truth[:, 2:]
    xywh = torch.cat(((xyxy[:, :2] + xyxy[:, 2:]) / 2, xyxy[:, 2:] - xyxy[:, :2]), 1) / size
    batch_dict = {"batch_idx": truth[:, 0], "cls": truth[:, 1:2], "bboxes": xywh}
    extra = {"targets": truth.contiguous(), "image_size": torch.tensor([size], dtype=torch.int32)}
    generator = torch.Generator().manual_seed(43)
    totals = []
    for prefix, cfg in [("v10", "yolov10n.yaml"), ("y26", "yolo26n.yaml")]:
        model = DetectionModel(cfg, nc=classes, verbose=False)
        model.args = get_cfg()
        model.args.epochs = 4
        criterion = model.init_criterion()
        strides = [int(s) for s in model.model[-1].stride.tolist()]
        feats = [torch.zeros(batch, 1, size // s, size // s) for s in strides]
        anchors = sum((size // s) ** 2 for s in strides)
        reg_max = criterion.one2many.reg_max

        def head():
            if reg_max > 1:
                bins = torch.arange(reg_max, dtype=torch.float32)
                boxes = -0.5 * (bins - 2.2).view(1, 1, reg_max, 1) ** 2 \
                    + 0.3 * torch.randn(batch, 4, reg_max, anchors, generator=generator)
            else:
                boxes = 2 + 0.3 * torch.rand(batch, 4, 1, anchors, generator=generator)
            return {"boxes": boxes.reshape(batch, 4 * reg_max, anchors),
                    "scores": torch.randn(batch, classes, anchors, generator=generator), "feats": feats}

        many, one = head(), head()
        preds = {"one2many": many, "one2one": one}
        first = criterion(preds, batch_dict)[0].sum()
        criterion.update()
        second = criterion(preds, batch_dict)[0].sum()
        _, many_items = criterion.one2many.loss(many, batch_dict)
        _, one_items = criterion.one2one.loss(one, batch_dict)
        names = criterion.one2many.loss_names
        for branch, preds_branch, items in [("many", many, many_items), ("one", one, one_items)]:
            extra[f"{prefix}_{branch}_distribution"] = preds_branch["boxes"].permute(0, 2, 1).contiguous()
            extra[f"{prefix}_{branch}_logits"] = preds_branch["scores"].permute(0, 2, 1).contiguous()
            extra[f"{prefix}_{branch}_components"] = torch.stack([items[n] for n in names]).float()
        extra[f"{prefix}_strides"] = torch.tensor(strides, dtype=torch.int32)
        extra[f"{prefix}_totals"] = torch.stack([first, second]).float()
        totals += [first, second]
    globals()["_extra"] = extra
    return torch.stack(totals).float().contiguous()

def run_yolo_training_setup(image):
    """ultralytics' training setup for a three-class YOLOv8n, from the trainer's own methods.

    `BaseTrainer.build_optimizer` runs unbound on a stand-in trainer (`optimizer="auto"`, a 2-image
    batch, 20 iterations) and reports its groups; `_setup_scheduler` and `_get_warmup_iterations` give
    the schedule for 4 epochs of 5 batches, and the per-iteration warm-up is `_do_train`'s own
    `np.interp` expression (trainer.py lines 469-481 at 8.4.120) evaluated over the run; `ModelEMA`
    averages a small module through three updates. Runs on the interpreter that has ultralytics installed.
    """
    import types
    from ultralytics.cfg import get_cfg
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import ModelEMA

    classes, batch, epochs, per_epoch = 3, 2, 4, 5
    model = DetectionModel("yolov8n.yaml", nc=classes, verbose=False)
    args = get_cfg()
    stand_in = types.SimpleNamespace(args=args, data={"nc": classes})
    accumulate = max(round(args.nbs / batch), 1)
    decay = args.weight_decay * batch * accumulate / args.nbs
    optimizer = BaseTrainer.build_optimizer(stand_in, model, name="auto", lr=args.lr0, momentum=args.momentum,
                                            decay=decay, iterations=epochs * per_epoch)
    groups = {}
    for group in optimizer.param_groups:
        names = [n for n, p in model.named_parameters() if any(p is q for q in group["params"])]
        kept = [n for n in names if ".dfl" not in n]                        # the trainer always freezes .dfl
        groups[group["param_group"]] = (len(kept), group["lr"], group["weight_decay"])

    stand_in.epochs, stand_in.optimizer = epochs, optimizer
    BaseTrainer._setup_scheduler(stand_in)
    warmup = BaseTrainer._get_warmup_iterations(stand_in, per_epoch)
    rates = []
    for ni in range(epochs * per_epoch):
        epoch = ni // per_epoch
        scale = stand_in.lf(epoch)
        if ni < warmup:
            scale = float(np.interp(ni, [0, warmup], [0.0, scale]))
        rates.append(scale)

    ema_net = torch.nn.Linear(3, 2)
    torch.nn.init.constant_(ema_net.weight, 1.0)
    torch.nn.init.constant_(ema_net.bias, 0.0)
    ema = ModelEMA(ema_net)
    for step in range(3):
        with torch.no_grad():
            ema_net.weight.fill_(float(step + 2))
            ema_net.bias.fill_(float(-(step + 1)))
        ema.update(ema_net)

    globals()["_extra"] = {
        "group_counts": torch.tensor([groups["weight"][0], groups["bn"][0], groups["bias"][0]], dtype=torch.int32),
        "group_decays": torch.tensor([groups["weight"][2], groups["bn"][2], groups["bias"][2]], dtype=torch.float64),
        "rate": torch.tensor([groups["weight"][1]], dtype=torch.float64),
        "schedule": torch.tensor(rates, dtype=torch.float64),
        "ema_weight": ema.ema.weight.detach().reshape(-1).contiguous(),
        "ema_bias": ema.ema.bias.detach().contiguous(),
    }
    return torch.tensor([float(warmup)])

def run_yolo_generation(image, checkpoint):
    """The pre-suppression predictions of a YOLOv9 / v10 / 11 / v12 / YOLO26 release, from ultralytics.

    `run_yolo` reads the model's own return value, which the end-to-end generations already reduce to
    300 rows. This mode takes the seam one step earlier instead: the head's inputs are captured, the
    branch it predicts from is re-run, and the decoded `[anchors, 4 + classes]` tensor is recorded. That
    is the same seam for every generation, so one comparison covers them all, and it keeps a wrong
    distribution-focal decode or anchor grid from hiding behind a top-k selection.

    The record also carries three backbone stage outputs, so a mismatch localizes to a stage rather
    than to "the network". Runs on the interpreter that has ultralytics installed.
    """
    from ultralytics import YOLO

    model = YOLO(checkpoint)
    net = model.model.float().eval()
    head = net.model[-1]

    captured = {}
    handle = head.register_forward_pre_hook(lambda module, args: captured.__setitem__("feats", args[0]))
    stages = {}
    watched = [index for index in (2, 4, 9) if index < len(net.model) - 1]
    handles = [handle] + [
        net.model[index].register_forward_hook(
            lambda m, i, o, index=index: stages.__setitem__(f"stage{index}", o))
        for index in watched
    ]

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        net(tensor)
        feats = list(captured["feats"])
        branch = head.one2one if getattr(head, "end2end", False) else head.one2many
        decoded = head._inference(head.forward_head(feats, **branch))

    for handle in handles:
        handle.remove()

    extra = {name: value[0].permute(1, 2, 0).contiguous() for name, value in stages.items()}
    extra["reg_max"] = torch.tensor(head.reg_max, dtype=torch.int32)
    extra["end2end"] = torch.tensor(int(bool(getattr(head, "end2end", False))), dtype=torch.int32)
    globals()["_extra"] = extra
    return decoded[0].transpose(0, 1).contiguous()              # [anchors, 4 + classes]


def run_codeformer(image, checkpoint):
    """CodeFormer restoration, `[H, W, 3]` in 0...1, from sczhou/CodeFormer's own architecture files.

    IK_REF_SRC holds a `codeformer/` directory with `codeformer_arch.py` and `vqgan_arch.py`; the
    basicsr registry/logger they import are stubbed. Runs the real inference settings (`w=0.5`,
    `adain=True`). The record's `logits` carries the code-prediction seam `[tokens, codebook]` — the
    continuous tensor that localizes a mismatch to encoder+transformer vs generator, robust to
    argmax near-ties the final image is not.
    """
    import os
    import types

    registry = types.ModuleType("basicsr.utils.registry")
    registry.ARCH_REGISTRY = type("Registry", (), {"register": staticmethod(lambda: (lambda cls: cls))})()
    utils = types.ModuleType("basicsr.utils")
    utils.get_root_logger = lambda *args, **kwargs: None
    sys.modules.update({"basicsr": types.ModuleType("basicsr"), "basicsr.utils": utils,
                        "basicsr.utils.registry": registry})

    source = os.path.join(_reference_source(), "codeformer")
    module = _import_reference(source, "basicsr.archs", "codeformer_arch", siblings=["vqgan_arch"])
    model = module.CodeFormer(fix_modules=None).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("params_ema", state.get("params", state)), strict=True)

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        out, logits, _ = model(tensor, w=0.5, adain=True)
    globals()["_extra"] = {"logits": logits[0].contiguous()}
    return torch.clamp((out[0].permute(1, 2, 0) + 1) / 2, 0, 1).contiguous()


def run_yolo_detections(image, checkpoint):
    """YOLOv8 detections on a NON-SQUARE frame, through ultralytics' own `predict` — letterboxing,
    suppression, and the mapping back to original-image coordinates included.

    `run_yolo` compares the raw prediction tensor on a square plate, where letterboxing is an
    identity. This mode is the other half: a 16:9 frame, where the reference scales by the smaller
    ratio, pads to a stride multiple with gray, and then undoes both when it reports boxes. The record
    carries `plate` (the non-square frame, `[H, W, 3]`), `output` (normalized xyxy boxes), `classes`,
    and `confidences`.

    Ultralytics treats a numpy array as BGR and flips it internally, so the plate is handed over
    reversed — passing RGB would silently detect on colour-swapped pixels.
    """
    from PIL import Image
    from ultralytics import YOLO

    source = os.environ.get("IK_YOLO_IMAGE")
    if not source:
        raise SystemExit("yolo_detections needs IK_YOLO_IMAGE (a photo to letterbox)")
    photo = Image.open(source).convert("RGB").resize((640, 360), Image.BILINEAR)
    plate = np.ascontiguousarray(np.asarray(photo).astype(np.float32) / 255.0)

    model = YOLO(checkpoint)
    results = model.predict(np.ascontiguousarray((plate * 255).astype(np.uint8)[..., ::-1]),
                            verbose=False)[0]
    globals()["_extra"] = {"plate": torch.from_numpy(plate).contiguous(),
                           "classes": results.boxes.cls.to(torch.int32).contiguous(),
                           "confidences": results.boxes.conf.contiguous()}
    return results.boxes.xyxyn.contiguous()                     # [detections, 4] normalized xyxy


def run_lama(image, checkpoint):
    """LaMa (big-lama) inpainting, `[H, W, 3]` in 0...1, from advimman's own `ffc.py`.

    IK_REF_SRC holds a `lama/` directory with `ffc.py`. Its three `saicinpainting` siblings are
    stubbed: `get_activation` is the only one the generator actually reaches (big-lama ends in a
    sigmoid), while the spatial-transform wrapper and squeeze-excitation are unused at this config.
    The geometry comes from big-lama's own `config.yaml`: 18 blocks, ratio 0.75 through the trunk and
    at the last downsample only.

    The record carries `mask` (1 where the region is regenerated), `raw` (the generator's own output,
    before compositing — the seam that isolates the network from the paste-back) and `output` (the
    composited result the port returns).
    """
    import types
    from torch import nn

    base = types.ModuleType("saicinpainting.training.modules.base")

    def get_activation(kind="tanh"):
        return {"tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(), False: nn.Identity()}[kind]

    base.get_activation = get_activation
    base.BaseDiscriminator = nn.Module
    spatial = types.ModuleType("saicinpainting.training.modules.spatial_transform")
    spatial.LearnableSpatialTransformWrapper = nn.Module
    squeeze = types.ModuleType("saicinpainting.training.modules.squeeze_excitation")
    squeeze.SELayer = nn.Module
    for name in ["saicinpainting", "saicinpainting.training", "saicinpainting.training.modules"]:
        package = types.ModuleType(name)
        package.__path__ = []                                   # a plain module cannot host submodules
        sys.modules[name] = package
    utils = types.ModuleType("saicinpainting.utils")
    utils.get_shape = lambda t: tuple(t.shape)
    sys.modules["saicinpainting.utils"] = utils
    sys.modules["saicinpainting.training.modules.base"] = base
    sys.modules["saicinpainting.training.modules.spatial_transform"] = spatial
    sys.modules["saicinpainting.training.modules.squeeze_excitation"] = squeeze

    module = _import_reference(os.path.join(_reference_source(), "lama"), "lama_ref", "ffc")
    model = module.FFCResNetGenerator(
        input_nc=4, output_nc=3, ngf=64, n_downsampling=3, n_blocks=18, add_out_act="sigmoid",
        init_conv_kwargs={"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False},
        downsample_conv_kwargs={"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False},
        resnet_conv_kwargs={"ratio_gin": 0.75, "ratio_gout": 0.75, "enable_lfu": False}).eval()

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    model.load_state_dict({name[len("generator."):]: value for name, value in state.items()
                           if name.startswith("generator.")}, strict=True)

    # A rectangle over the middle of the frame: a real hole, not a token one.
    height, width = image.shape[:2]
    mask = np.zeros((height, width, 1), dtype=np.float32)
    mask[height // 3: height * 2 // 3, width // 3: width * 2 // 3] = 1.0

    plate = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0)
    stacked = torch.cat([plate * (1 - mask_tensor), mask_tensor], dim=1)
    with torch.no_grad():
        predicted = model(stacked)
    composited = mask_tensor * predicted + (1 - mask_tensor) * plate
    globals()["_extra"] = {"mask": torch.from_numpy(mask[..., 0]).contiguous(),
                           "raw": predicted[0].permute(1, 2, 0).contiguous()}
    return composited[0].permute(1, 2, 0).contiguous()


def run_nafnet(image, checkpoint):
    """NAFNet restoration, `[H, W, 3]` in 0...1, from megvii-research's own `NAFNet_arch.py`.

    IK_REF_SRC holds a `nafnet/` directory with `NAFNet_arch.py` (saved as `nafnet_arch.py`). Its two
    basicsr imports are supplied here: `LayerNorm2d` is the reference's own channel-wise normalization
    (a LayerNorm over the channel axis of an NCHW tensor), and `Local_Base` is only used by the
    TLSC variant, so a bare class is enough. The released SIDD width-32 checkpoint is the denoising
    model, which is what the module's default configuration sizes.
    """
    import types
    from torch import nn

    class LayerNorm2d(nn.Module):
        def __init__(self, channels, eps=1e-6):
            super().__init__()
            self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
            self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
            self.eps = eps

        def forward(self, x):
            mu = x.mean(1, keepdim=True)
            var = (x - mu).pow(2).mean(1, keepdim=True)
            y = (x - mu) / (var + self.eps).sqrt()
            return self.weight[None, :, None, None] * y + self.bias[None, :, None, None]

    arch_util = types.ModuleType("basicsr.models.archs.arch_util")
    arch_util.LayerNorm2d = LayerNorm2d
    local_arch = types.ModuleType("basicsr.models.archs.local_arch")
    local_arch.Local_Base = type("Local_Base", (), {})
    for name in ["basicsr", "basicsr.models", "basicsr.models.archs"]:
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    sys.modules["basicsr.models.archs.arch_util"] = arch_util
    sys.modules["basicsr.models.archs.local_arch"] = local_arch

    module = _import_reference(os.path.join(_reference_source(), "nafnet"), "nafnet_ref", "nafnet_arch")
    # The released geometries, from the repository's own option files. SIDD (denoise) spreads its
    # blocks through the middle; GoPro (deblur) puts twenty-eight of them in the last encoder stage.
    geometries = {
        "SIDD": {"width": 32, "middle_blk_num": 12, "enc_blk_nums": [2, 2, 4, 8], "dec_blk_nums": [2, 2, 2, 2]},
        "GoPro": {"width": 32, "middle_blk_num": 1, "enc_blk_nums": [1, 1, 1, 28], "dec_blk_nums": [1, 1, 1, 1]},
        "REDS": {"width": 64, "middle_blk_num": 1, "enc_blk_nums": [1, 1, 1, 28], "dec_blk_nums": [1, 1, 1, 1]},
        "SIDD64": {"width": 64, "middle_blk_num": 12, "enc_blk_nums": [2, 2, 4, 8], "dec_blk_nums": [2, 2, 2, 2]},
        "GoPro64": {"width": 64, "middle_blk_num": 1, "enc_blk_nums": [1, 1, 1, 28], "dec_blk_nums": [1, 1, 1, 1]},
    }
    model = module.NAFNet(img_channel=3, **geometries[os.environ.get("IK_NAFNET_VARIANT", "SIDD")]).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("params", state.get("state_dict", state)), strict=True)

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        restored = model(tensor)
    return restored[0].permute(1, 2, 0).contiguous()


def run_rife(image, checkpoint):
    """RIFE frame interpolation, `[H, W, 3]` in 0...1, from the released HDv3 `IFNet_HDv3.py`.

    IK_REF_SRC holds a `rife/` directory with `ifnet_hdv3.py` and `warplayer.py` (the reference
    imports `warp` from `model.warplayer`, injected here). The two frames are the plate and a shifted
    copy, so the network has real motion to find; the record's `frame1` carries the shifted copy the
    Swift side reads back.
    """
    import types

    warplayer = _import_reference(os.path.join(_reference_source(), "rife"), "rife_model", "warplayer")
    model_package = types.ModuleType("model")
    model_package.__path__ = []
    sys.modules["model"] = model_package
    sys.modules["model.warplayer"] = warplayer

    module = _import_reference(os.path.join(_reference_source(), "rife"), "rife_ref", "ifnet_hdv3")
    model = module.IFNet().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict({name.replace("module.", "", 1): value for name, value in state.items()},
                          strict=True)

    shift = (4, 6)
    second = np.roll(np.roll(image, shift[0], axis=0), shift[1], axis=1)

    # The reference network has no internal padding and simply fails on a size that is not a multiple
    # of the coarsest stride — its own inference script pads outside the model. The port pads by edge
    # replication to a multiple of 32 and crops back, so the oracle does the same here: what is then
    # compared is the network on identical input, not a padding convention the reference never fixes.
    height, width = image.shape[:2]
    padded = [np.pad(plate, ((0, -height % 32), (0, -width % 32), (0, 0)), mode="edge")
              for plate in (image, second)]
    frames = [torch.from_numpy(np.ascontiguousarray(plate)).permute(2, 0, 1)[None] for plate in padded]
    with torch.no_grad():
        _, _, merged = model(torch.cat(frames, dim=1))
    globals()["_extra"] = {"frame1": torch.from_numpy(np.ascontiguousarray(second)).contiguous()}
    return merged[2][0].permute(1, 2, 0)[:height, :width].contiguous()   # finest scale, cropped back


def run_modnet(image, checkpoint):
    """MODNet portrait matte, `[H, W]` in 0...1, from ZHKKKe's own `modnet.py`.

    IK_REF_SRC holds a `modnet/` directory with `modnet.py`, `mobilenetv2.py`, and `wrapper.py`. The
    backbone wrapper would fetch ImageNet weights on construction, so `backbone_pretrained=False`
    skips that — the released checkpoint supplies everything. Inference normalizes to -1...1 (the
    demo's `Normalize(0.5, 0.5)`), and the network's strides need sides that are multiples of 32, so
    feed a plate that already is one.
    """
    import types
    from torch import nn

    source = os.path.join(_reference_source(), "modnet")
    # `wrapper.py` imports the network from `.mobilenetv2`, and `modnet.py` reaches for
    # `.backbones.SUPPORTED_BACKBONES` — a package `__init__` that is not one of the three files, so
    # it is synthesized here from the wrapper's own class.
    mobilenet = _import_reference(source, "modnet_ref", "mobilenetv2")
    sys.modules["modnet_ref.backbones.mobilenetv2"] = mobilenet
    wrapper = _import_reference(source, "modnet_ref", "wrapper",
                                injected={"mobilenetv2": mobilenet})
    backbones = types.ModuleType("modnet_ref.backbones")
    backbones.__path__ = []
    backbones.SUPPORTED_BACKBONES = {"mobilenetv2": wrapper.MobileNetV2Backbone}
    sys.modules["modnet_ref.backbones"] = backbones

    module = _import_reference(source, "modnet_ref", "modnet")
    model = module.MODNet(backbone_pretrained=False).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    model.load_state_dict({name.replace("module.", "", 1): value for name, value in state.items()},
                          strict=True)

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        _, _, matte = model(tensor, inference=True)
    return matte[0, 0].contiguous()                             # [H, W] alpha


def run_bisenet(image, checkpoint):
    """BiSeNetV1 class logits, `[H, W, classes]`, from CoinCheung's own `bisenetv1.py`.

    IK_REF_SRC holds a `bisenet/` directory with `bisenetv1.py` and `resnet.py`. The ResNet-18 would
    fetch torchvision's ImageNet weights on construction, so its loader is neutralized — the released
    checkpoint supplies everything. Comparing logits rather than the label map keeps the check
    sensitive: an argmax hides every difference too small to flip a pixel.
    """
    import types
    from torch import nn

    modules = types.ModuleType("lib.models.resnet")
    for name in ["lib", "lib.models"]:
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package

    source = os.path.join(_reference_source(), "bisenet")
    resnet = _import_reference(source, "bisenet_ref", "resnet")
    resnet.Resnet18.init_weight = lambda self: None             # skip the torchvision download
    sys.modules["lib.models.resnet"] = resnet
    module = _import_reference(source, "bisenet_ref", "bisenetv1", injected={"resnet": resnet})

    # `aux_mode="eval"` omits the two auxiliary heads, which the released checkpoint carries, so the
    # model is built in the mode that has them and simply run in eval; the first output is the one
    # inference uses.
    model = module.BiSeNetV1(n_classes=19, aux_mode="train").eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("state_dict", state), strict=True)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    deviation = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = ((torch.from_numpy(image).permute(2, 0, 1) - mean) / deviation).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)[0]
    return logits[0].permute(1, 2, 0).contiguous()              # [H, W, classes] to match NHWC


def run_whisper(image, checkpoint):
    """Whisper tiny transcription, from openai-whisper itself.

    `--checkpoint` is the released `tiny.pt`. The record carries the `waveform` both sides read, the
    reference's own **log-mel** (the seam between the front end and the encoder — a mel mismatch and a
    network mismatch look identical at the token level), and `output` as the generated token ids. The
    clip is a deterministic harmonic sweep rather than speech: the tokens it produces are arbitrary,
    but both implementations must produce the SAME arbitrary tokens.
    """
    import whisper

    model = whisper.load_model(checkpoint) if os.path.exists(checkpoint) else whisper.load_model("tiny")
    model.eval()
    # The prompt and the suppression boundary come from the model's OWN tokenizer: large-v3 carries one
    # more language token than every earlier size, which shifts `<|transcribe|>` and `<|notimestamps|>`.
    reference_tokenizer = whisper.tokenizer.get_tokenizer(
        model.is_multilingual, num_languages=getattr(model, "num_languages", 99),
        language="en", task="transcribe")
    prompt_ids = list(reference_tokenizer.sot_sequence_including_notimestamps)

    rate, seconds = 16000, 4.0
    time = np.arange(int(rate * seconds), dtype=np.float32) / rate
    generator = np.random.default_rng(29)
    # A voiced-like stack with a slow sweep, then noise: enough structure to move the decoder off its
    # prompt without depending on a real recording.
    sweep = sum(0.3 / (h + 1) * np.sin(2 * np.pi * (140 + 40 * time) * (h + 1) * time) for h in range(4))
    wave = np.where(time < 3.0, sweep, 0.05 * generator.standard_normal(time.shape)).astype(np.float32)

    padded = whisper.pad_or_trim(torch.from_numpy(wave))
    # large-v3 produces 128 mel bands where every earlier size produces 80, and its encoder's first
    # convolution takes that many channels; the default would feed it the wrong shape.
    mel = whisper.log_mel_spectrogram(padded, n_mels=model.dims.n_mels)

    # `model.decode` is not plain greedy: it also applies SuppressBlank, a curated non-speech token
    # list, and timestamp rules. The port suppresses the special/timestamp range wholesale, so running
    # the reference's own decoder under the PORT's rule is what isolates the network — otherwise a
    # decoding-policy difference is indistinguishable from a weights or attention bug. The policy gap
    # itself is real and recorded separately.
    prompt = prompt_ids                                         # <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
    tokens = list(prompt)
    with torch.no_grad():
        audio = model.encoder(mel.unsqueeze(0))
        first_logits = None
        for _ in range(32):
            logits = model.decoder(torch.tensor([tokens]), audio)[0, -1].clone()
            if first_logits is None:
                first_logits = logits.clone()
            logits[reference_tokenizer.sot:] = float("-inf")     # the port's suppression, applied here
            nxt = int(torch.argmax(logits).item())
            if nxt == 50257:                                    # <|endoftext|>
                break
            tokens.append(nxt)

    # The reference's OWN rules, which the port now implements: its curated non-speech set masked at
    # every step, and SuppressBlank (a space and an immediate end of text) at the first sampled
    # position only. Recorded beside the port-rule tokens so the two policies are compared directly
    # rather than described.
    non_speech = list(reference_tokenizer.non_speech_tokens)
    space_token = reference_tokenizer.encode(" ")[0]

    ruled = list(prompt)
    with torch.no_grad():
        for step in range(32):
            logits = model.decoder(torch.tensor([ruled]), audio)[0, -1].clone()
            logits[reference_tokenizer.sot:] = float("-inf")     # specials and timestamps
            logits[non_speech] = float("-inf")                  # SuppressTokens, the "-1" default
            if step == 0:                                       # SuppressBlank
                logits[space_token] = float("-inf")
                logits[reference_tokenizer.eot] = float("-inf")
            nxt = int(torch.argmax(logits).item())
            if nxt == reference_tokenizer.eot:
                break
            ruled.append(nxt)

    # The TIMESTAMPED decode, which is a different prompt as well as a different rule set: the segment
    # times only exist when `<|notimestamps|>` is left OUT of the prompt, and the timestamp range then
    # has to stay unmasked. `ApplyTimestampRules` is what orders the result — timestamps in pairs,
    # never decreasing, and a timestamp forced at the opening position.
    timestamp_begin = reference_tokenizer.timestamp_begin
    no_timestamps = reference_tokenizer.no_timestamps
    timed_prompt = [t for t in prompt_ids if t != no_timestamps]
    sample_begin = len(timed_prompt)
    max_initial_index = round(1.0 / (30.0 / model.dims.n_audio_ctx))     # max_initial_timestamp 1.0 s

    timed = list(timed_prompt)
    with torch.no_grad():
        for _ in range(32):
            logits = model.decoder(torch.tensor([timed]), audio)[0, -1].clone()
            logits[no_timestamps] = float("-inf")
            logits[reference_tokenizer.sot: timestamp_begin] = float("-inf")   # specials, not times
            logits[non_speech] = float("-inf")
            if len(timed) == sample_begin:
                logits[space_token] = float("-inf")
                logits[reference_tokenizer.eot] = float("-inf")

            seq = timed[sample_begin:]
            last_was_timestamp = len(seq) >= 1 and seq[-1] >= timestamp_begin
            penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= timestamp_begin
            if last_was_timestamp:
                if penultimate_was_timestamp:
                    logits[timestamp_begin:] = float("-inf")
                else:
                    logits[: reference_tokenizer.eot] = float("-inf")
            stamps = [t for t in seq if t >= timestamp_begin]
            if stamps:
                last = stamps[-1] if (last_was_timestamp and not penultimate_was_timestamp) else stamps[-1] + 1
                logits[timestamp_begin: last] = float("-inf")
            if len(timed) == sample_begin:
                logits[: timestamp_begin] = float("-inf")
                logits[timestamp_begin + max_initial_index + 1:] = float("-inf")
            # If the timestamps hold more probability together than any single text token, take one.
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            if logprobs[timestamp_begin:].logsumexp(dim=-1) > logprobs[:timestamp_begin].max():
                logits[:timestamp_begin] = float("-inf")

            nxt = int(torch.argmax(logits).item())
            if nxt == reference_tokenizer.eot:
                break
            timed.append(nxt)

    globals()["_extra"] = {"prompt": torch.tensor(prompt_ids, dtype=torch.int32).contiguous(),
                           "waveform": torch.from_numpy(wave).contiguous(),
                           "features": mel.contiguous(),                 # [mels, frames]
                           "first_logits": first_logits.contiguous(),    # the decoder seam
                           "non_speech_tokens": torch.tensor(sorted(non_speech), dtype=torch.int32).contiguous(),
                           "space_token": torch.tensor([space_token], dtype=torch.int32).contiguous(),
                           "ruled_tokens": torch.tensor(ruled[len(prompt):], dtype=torch.int32).contiguous(),
                           "timestamp_prompt": torch.tensor(timed_prompt, dtype=torch.int32).contiguous(),
                           "timestamp_begin": torch.tensor([timestamp_begin], dtype=torch.int32).contiguous(),
                           "timed_tokens": torch.tensor(timed[sample_begin:], dtype=torch.int32).contiguous()}
    return torch.tensor(tokens[len(prompt):], dtype=torch.int32).contiguous()



def run_retinaface(image, checkpoint):
    """RetinaFace mobile0.25 detection, from facexlib's own RetinaFace.

    This is the detector the CodeFormer reference pipeline uses, so matching it is what makes an
    aligned crop here the same crop the reference produces.

    The record carries the network's PRE-SUPPRESSION outputs — `output` is the box offsets, with
    `scores` and `landmark_offsets` beside them — because a detection-level comparison alone cannot
    separate a network mismatch from a decoding or threshold difference. `detections` carries the
    reference's own decoded, suppressed result for the end-to-end check.
    """
    import numpy as np
    import torch
    from facexlib.detection.retinaface import RetinaFace

    model = RetinaFace(network_name="mobile0.25", device=torch.device("cpu"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    # The reference reads BGR 0...255 through OpenCV and subtracts a per-channel mean.
    rgb = (image * 255.0).astype(np.float32)
    bgr = rgb[:, :, ::-1].copy()
    tensor = torch.from_numpy(bgr).permute(2, 0, 1).unsqueeze(0) - model.mean_tensor

    with torch.no_grad():
        loc, conf, landmarks, priors = model._RetinaFace__detect_faces(tensor)
        detections = model.detect_faces((bgr).astype(np.float32), conf_threshold=0.8, nms_threshold=0.4)

    globals()["_extra"] = {
        "scores": conf.squeeze(0).contiguous(),
        "landmark_offsets": landmarks.squeeze(0).contiguous(),
        "priors": priors.contiguous(),
        "detections": torch.from_numpy(np.asarray(detections, dtype=np.float32)).contiguous(),
    }
    return loc.squeeze(0).contiguous()



def run_qwen3(image, checkpoint):
    """Qwen3 dense decoder logits, from transformers' own Qwen3ForCausalLM.

    `--checkpoint` is the released model DIRECTORY. Needs transformers >= 4.51, which is newer than the
    one the vision oracles run under; `oracle_environments` in the validation manifest records where
    that interpreter lives.

    The record carries the token ids both sides read, the logits for every position of the prompt (the
    prefill, which is where an attention or normalization mistake shows), and the greedy continuation
    the model produces — a token-level match is the end-to-end check and the logits say where a
    mismatch came from.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # E2B fits at float32; E4B's 16 GB of bf16 weights would double past this machine's RAM, so
    # IK_GEMMA_DTYPE=bfloat16 runs the oracle at the released precision instead. The Swift side then
    # loads at `.checkpoint` precision, and the parity threshold is the half-precision one.
    dtype = getattr(torch, os.environ.get("IK_GEMMA_DTYPE", "float32"))
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=dtype).eval()

    prompt = "The capital of France is"
    ids = tokenizer(prompt, return_tensors="pt").input_ids

    with torch.no_grad():
        logits = model(ids).logits[0]                       # [tokens, vocabulary]
        generated = model.generate(ids, max_new_tokens=16, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
    continuation = generated[0, ids.shape[1]:]

    globals()["_extra"] = {
        "tokens": ids[0].to(torch.int32).contiguous(),
        "continuation": continuation.to(torch.int32).contiguous(),
    }
    return logits.contiguous()


# The query/document pair the Qwen3-Embedding record is measured on. The Swift side declares the same
# strings, so the tokenizer-agreement check reproduces the reference's ids from the text.
_EMBED_TASK = "Given a web search query, retrieve relevant passages that answer the query"
_EMBED_QUERY = f"Instruct: {_EMBED_TASK}\nQuery:What is the capital of China?"
_EMBED_DOCUMENT = "The capital of China is Beijing."


def run_qwen3_embedding(image, checkpoint):
    """Qwen3-Embedding embeddings, from the model card's own transformers recipe.

    `--checkpoint` is the released `Qwen/Qwen3-Embedding-0.6B` directory. The embedder is the base
    `Qwen3Model` (AutoModel, no lm_head) read at its last hidden state, pooled at the LAST token — the
    tokenizer appends `<|endoftext|>`, whose position is what the pooling reads — and L2-normalized. It
    needs transformers >= 4.51, the same interpreter the qwen3 mode uses.

    The record carries the query embedding as `output` and the document embedding beside it, plus the
    reference's own token ids for both: the network check feeds those ids and compares the embedding,
    while a separate tokenizer-agreement check reproduces the ids from the shared text.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
    model = AutoModel.from_pretrained(checkpoint, dtype=torch.float32).eval()

    def embed(text):
        batch = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state          # [1, tokens, hidden]
        pooled = hidden[:, -1]                                  # last-token pooling, single sequence
        embedding = F.normalize(pooled, p=2, dim=1)[0]
        return embedding, batch["input_ids"][0]

    query_embedding, query_ids = embed(_EMBED_QUERY)
    document_embedding, document_ids = embed(_EMBED_DOCUMENT)

    globals()["_extra"] = {
        "query_tokens": query_ids.to(torch.int32).contiguous(),
        "document_tokens": document_ids.to(torch.int32).contiguous(),
        "document_embedding": document_embedding.contiguous(),
        # The retrieval score the two embeddings produce, for an end-to-end check that survives a
        # per-embedding drift the cosine would still call a match.
        "score": torch.dot(query_embedding, document_embedding).reshape(1).contiguous(),
    }
    return query_embedding.contiguous()


# The query/document pair the EmbeddingGemma record is measured on, each with the model's own task
# prompt prepended. The Swift side declares the same strings.
_EG_QUERY = "task: search result | query: What is the capital of China?"
_EG_DOCUMENT = "title: none | text: The capital of China is Beijing."


def run_embeddinggemma(image, checkpoint):
    """EmbeddingGemma-300M embeddings, from the sentence-transformers pipeline over transformers' own
    Gemma3TextModel.

    `--checkpoint` is the released directory (the ungated `unsloth/embeddinggemma-300m` mirror). The
    backbone is Gemma3TextModel with `use_bidirectional_attention` (no causal mask), read at its last
    hidden state, mean-pooled over every token, run through the two Dense projections (768 -> 3072 ->
    768, no bias, Identity activation) the sentence-transformers head carries, and L2-normalized. Needs
    transformers >= 4.56 (bidirectional Gemma3); the `llm` oracle interpreter has it.

    The record carries the query embedding as `output`, the document embedding beside it, both token
    id sequences, the retrieval score, and the backbone's per-layer hidden states for the query so a
    divergence is located to a layer rather than guessed from the embedding.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    from safetensors.torch import load_file

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint, dtype=torch.float32).eval()
    dense2 = load_file(os.path.join(checkpoint, "2_Dense", "model.safetensors"))["linear.weight"]
    dense3 = load_file(os.path.join(checkpoint, "3_Dense", "model.safetensors"))["linear.weight"]

    def embed(text, record_layers=False):
        batch = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = model(**batch, output_hidden_states=record_layers)
        pooled = out.last_hidden_state.mean(dim=1)          # mean pooling, single unpadded sequence
        projected = (pooled @ dense2.T) @ dense3.T          # the Dense bottleneck, no bias
        embedding = F.normalize(projected, p=2, dim=1)[0]
        return embedding, batch["input_ids"][0], (out.hidden_states if record_layers else None)

    query_embedding, query_ids, layers = embed(_EG_QUERY, record_layers=True)
    document_embedding, document_ids, _ = embed(_EG_DOCUMENT)

    extra = {
        "query_tokens": query_ids.to(torch.int32).contiguous(),
        "document_tokens": document_ids.to(torch.int32).contiguous(),
        "document_embedding": document_embedding.contiguous(),
        "score": torch.dot(query_embedding, document_embedding).reshape(1).contiguous(),
    }
    for index, hidden in enumerate(layers):
        extra[f"hidden.{index}"] = hidden[0].float().contiguous()
    globals()["_extra"] = extra
    return query_embedding.contiguous()


# The query and the two documents the reranker record is measured on: one relevant, one not, so the
# test checks the scores match AND that the reranker orders them.
_RERANK_QUERY = "What is the capital of France?"
# Long enough that the pair exceeds the 128-token local attention window (64 either side), so the
# sliding-window layers are exercised by the isolation harness rather than degenerating to global.
_RERANK_RELEVANT = ("Paris is the capital and most populous city of France. Situated on the river Seine "
                    "in the north of the country, it has been a major European centre of finance, "
                    "diplomacy, commerce, fashion, and art for centuries. The city is home to landmarks "
                    "such as the Eiffel Tower, the Louvre, and the cathedral of Notre-Dame, and its "
                    "metropolitan area is among the largest in Europe.")
_RERANK_IRRELEVANT = "The Great Barrier Reef is the world's largest coral reef system, off Australia."


def run_modernbert_reranker(image, checkpoint):
    """A ModernBERT cross-encoder reranker's relevance score, from transformers' own
    ModernBertForSequenceClassification.

    `--checkpoint` is the released `Alibaba-NLP/gte-reranker-modernbert-base` directory. The model reads
    a `[CLS] query [SEP] document [SEP]` pair through the bidirectional encoder, mean-pools, runs a
    prediction head (dense + gelu + LayerNorm), and a single-logit classifier gives the score.

    The record carries the token ids of both pairs, the relevant pair's per-layer hidden states for the
    isolation harness, and both scores, so the test checks the arithmetic and the ordering.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, dtype=torch.float32).eval()

    def score(document, record_layers=False):
        inputs = tokenizer(_RERANK_QUERY, document, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=record_layers)
        return out.logits[0], inputs["input_ids"][0], (out.hidden_states if record_layers else None)

    relevant, relevant_ids, layers = score(_RERANK_RELEVANT, record_layers=True)
    irrelevant, irrelevant_ids, _ = score(_RERANK_IRRELEVANT)

    extra = {
        "relevant_tokens": relevant_ids.to(torch.int32).contiguous(),
        "irrelevant_tokens": irrelevant_ids.to(torch.int32).contiguous(),
        "irrelevant_score": irrelevant.contiguous(),
    }
    for index, hidden in enumerate(layers):
        extra[f"hidden.{index}"] = hidden[0].float().contiguous()
    globals()["_extra"] = extra
    return relevant.contiguous()


_LAYA_STATE = ("Subject: Payouts failing for three days. Hello, my weekly payouts to my bank account have failed "
               "three times in a row since Monday with the message 'transfer rejected by the receiving bank'. "
               "I have not changed my account details, the account is open, and other merchants pay into it "
               "without any problem. Support chat told me to wait 48 hours and it has now been 72. I have "
               "staff to pay on Friday and need this resolved today. Account id 88213, plan Pro, region EU. "
               "Please escalate this to someone who can actually look at the transfer logs. Thanks, Dana.")

# The keys are in sorted order, which is how the port serializes a record, so the two sides agree.
_LAYA_RECORD = {"account": 88213, "body": "Where is my refund? It was promised in 5 business days.",
                "plan": "Pro", "priority": 2.5, "tags": ["billing", "refund"], "vip": True, "note": None}
_LAYA_RECORD = dict(sorted(_LAYA_RECORD.items()))

_LAYA_QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations",
                                "sales": None}},
    "topic": {"type": "choice", "instructions": "What is the message mainly about?",
              "criteria": ["payouts", "refunds", "login", "pricing", "bug report", "other"]},
    "severity": {"type": "score", "instructions": "How severe is the problem?", "criteria": ["low", "medium", "high"]},
    "urgent": {"type": "noul", "instructions": "The customer needs an answer today.",
               "criteria": {"true": "the customer states or implies a deadline within a day",
                            "false": "no deadline is stated or implied"}},
}


def _laya_agent(checkpoint):
    """The release's own RLAgent over a variant directory, with rl_common.py found at the release root and
    the transformers 5 `rope_parameters` copied into the fields a 4.x ModernBertConfig reads (mmBERT's
    local rotary base is 160000, which the 4.x default of 10000 would silently replace)."""
    import os
    import sys
    import transformers

    root = os.path.abspath(checkpoint)
    while not os.path.exists(os.path.join(root, "rl_common.py")):
        parent = os.path.dirname(root)
        if parent == root:
            raise SystemExit(f"rl_common.py not found at or above {checkpoint}")
        root = parent
    if root not in sys.path:
        sys.path.insert(0, root)
    original = transformers.AutoConfig.from_pretrained

    def with_rope_parameters(path, *args, **kwargs):
        config = original(path, *args, **kwargs)
        rope = getattr(config, "rope_parameters", None) or {}
        if isinstance(rope, dict):
            if "full_attention" in rope:
                config.global_rope_theta = float(rope["full_attention"]["rope_theta"])
            if "sliding_attention" in rope:
                config.local_rope_theta = float(rope["sliding_attention"]["rope_theta"])
        return config
    transformers.AutoConfig.from_pretrained = with_rope_parameters
    from rl_agent_api import RLAgent
    return RLAgent(checkpoint, device="cpu")


def run_laya(image, checkpoint):
    """Laya (convaiinnovations/laya), the open reproduction of Jev, from the release's own inference code
    (`rl_agent_api.RLAgent` over `rl_common.build_sequence` and `DecisionModel`).

    `--checkpoint` is one variant directory: the release root (ModernBERT-large), `typed-decisions`
    (the same geometry fine-tuned), or `multilingual` (mmBERT-base with Gemma's tokenizer). Four
    questions (a described 3-way choice, an undescribed 6-way choice, a 3-level score, and a noul with
    meanings) are asked about a string state and about a record state. The record carries, per question
    and state (`q{i}.*` for the string, `d{i}.*` for the record), the token ids and marker positions
    the reference builds, the raw marker logits before the temperature, the calibrated probabilities the
    API answers with, and the act probability; for the string state's first question, the encoder's
    per-layer hidden states and the decision head's output for the isolation harness; and the
    serialized record state as UTF-8 bytes, so the port's serialization is checked byte for byte.
    """
    import torch

    agent = _laya_agent(checkpoint)
    from rl_common import QTYPES, build_sequence, render_options, serialize_state, temp_bucket
    model, tok, cfg = agent.model, agent.tok, agent.cfg
    extra = {}
    keys = list(_LAYA_QUESTIONS)
    first_logits = None
    for prefix, state in (("q", _LAYA_STATE), ("d", _LAYA_RECORD)):
        answers = agent.system_one(state, _LAYA_QUESTIONS)["answers"]
        for index, qid in enumerate(keys):
            q = agent._to_internal(_LAYA_QUESTIONS[qid])
            seq, markers = build_sequence(tok, state, q, cfg["max_len"], cfg["head_max_len"])
            k = len(markers)
            assert k == len(render_options(q)), (qid, k)
            ids = torch.tensor([seq])
            attention = torch.ones_like(ids)
            with torch.no_grad():
                logits, act = model(ids, attention, torch.tensor([markers]), torch.ones(1, k, dtype=torch.bool),
                                    torch.tensor([QTYPES[q["t"]]]))
            logits = logits[0, :k].float()
            qt = QTYPES[q["t"]]
            temperature = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
            probabilities = torch.softmax(logits / temperature, -1)
            answer = answers[qid]
            extra[f"{prefix}{index}.ids"] = ids[0].to(torch.int32).contiguous()
            extra[f"{prefix}{index}.markers"] = torch.tensor(markers, dtype=torch.int32)
            extra[f"{prefix}{index}.logits"] = logits.contiguous()
            extra[f"{prefix}{index}.probabilities"] = probabilities.contiguous()
            extra[f"{prefix}{index}.temperature"] = torch.tensor([float(temperature)])
            extra[f"{prefix}{index}.act_probability"] = torch.tensor([float(answer["rl_agent"]["act_probability"])])
            if q["t"] == "choice":
                extra[f"{prefix}{index}.choice"] = torch.tensor([list(q["crit"]).index(answer["choice"])], dtype=torch.int32)
                extra[f"{prefix}{index}.confidence"] = torch.tensor([float(answer["confidence"])])
            elif q["t"] == "score":
                extra[f"{prefix}{index}.score"] = torch.tensor([float(answer["score"])])
                extra[f"{prefix}{index}.confidence"] = torch.tensor([float(answer["confidence"])])
            else:
                extra[f"{prefix}{index}.noul"] = torch.tensor([float(answer["noul"])])
            if prefix == "q" and index == 0:
                first_logits = logits
                with torch.no_grad():
                    encoded = model.encoder(input_ids=ids, attention_mask=attention, output_hidden_states=True)
                    for layer, hidden in enumerate(encoded.hidden_states):
                        extra[f"hidden.{layer}"] = hidden[0].float().contiguous()
                    h = encoded.last_hidden_state + model.type_emb(torch.tensor([qt]))[:, None, :]
                    for layer in model.head.layers:
                        h = layer(h, src_key_padding_mask=~attention.bool())
                    extra["head_out"] = h[0].float().contiguous()
    extra["record_bytes"] = torch.tensor(list(serialize_state(_LAYA_RECORD).encode("utf-8")), dtype=torch.int32)
    extra["temperature"] = torch.tensor([float(t) for t in agent.temperature])
    globals()["_extra"] = extra
    return first_logits.clone()                     # the record's output must not share storage with q0.logits


_OJD_CASES = [
    ("Customer: I was charged twice for the same order and nobody answers my emails. I want my money back now.",
     [{"type": "choice", "instructions": "Which product area is the message about?",
       "options": ["fees & charges", "pin & security", "refund & dispute", "card", "other"]},
      {"type": "score", "instructions": "How positive is the sentiment of this message?",
       "options": ["very negative", "negative", "neutral", "positive", "very positive"]},
      {"type": "noul", "instructions": "The customer is asking for a refund."}]),
    # The port serializes a record with sorted keys the way json.dumps(sort_keys=True, ensure_ascii=False)
    # writes it; the reference takes text, so it is handed that text.
    ({"ticket": 4471, "channel": "email", "body": "Mon café est froid — rembourser svp", "vip": True, "tags": ["billing", None]},
     [{"type": "noul", "instructions": "The message is written in French."},
      {"type": "choice", "instructions": "Which team?", "options": ["billing", "technical", "sales"]}]),
    (("The quarterly report shows revenue of $4.2M, up 12% year over year, while churn fell to 3.1%. "
      "Ｆｕｌｌｗｉｄｔｈ text, naïve résumé, 東京 office, and tabs\tand\nnewlines  with   spaces. ") * 12,
     [{"type": "choice", "instructions": "What is the main topic?",
       "options": ["finance", "hiring", "product", "legal", "marketing", "operations", "sales", "support", "security", "other"]},
      {"type": "score", "instructions": "How optimistic is the report?", "options": ["low", "medium", "high"]}]),
]
_OJD_TEXTS = ["Hello world", "  leading and trailing  ", "naïve café résumé", "Ｆｕｌｌｗｉｄｔｈ ＡＢＣ １２３",
              "東京タワー", "tabs\tand\nnewlines", "$4.2M (12%) -- ok?!", "don't won't I'm", "emoji 🙂 end",
              "UPPER lower MiXeD", "", "a" * 40, "x = f(y) + 3.14159e-2", "ﬁ ligature ½ ①"]


def run_open_jev_deberta(image, checkpoint):
    """open-jev-deberta-v3-large (com-kotobalabs), from the release's own `typed_decisions` package.

    `--checkpoint` is the release directory. Three cases (`c{i}.*`): a customer message with a
    5-way choice, a 5-level score, and a noul; a record state (serialized with sorted keys) with a
    noul and a 3-way choice; and a long state, cut to the 256-token budget, with a 10-way choice and a
    3-level score. Each carries the token ids, the per-token span slots (`seg`), the raw logits before
    the temperature, the probabilities, and the readout. Case 0 carries every encoder layer's hidden
    state. `batch.*` is cases 0 and 1 padded together through the collator, the padding path, with the
    objective `decision_loss` at Brier weights 1 and 0.5 against fixed golds. `text{i}` is the
    tokenizer on its own over awkward strings, and `buckets` is the log-bucketed relative position
    for offsets -600 through 600.
    """
    import json as _json
    import sys as _sys
    import torch
    _sys.path.insert(0, checkpoint)
    from typed_decisions.open_jev import OpenJev
    from typed_decisions.encoder import decision_loss
    from transformers.models.deberta_v2.modeling_deberta_v2 import make_log_bucket_position

    m = OpenJev.from_pretrained(checkpoint, device="cpu")
    model, collator = m.model, m.collator
    extra = {}
    first = None
    batch_items = []
    for index, (state, questions) in enumerate(_OJD_CASES):
        text = state if isinstance(state, str) else _json.dumps(state, ensure_ascii=False, sort_keys=True)
        qs = [m._question(i, q) for i, q in enumerate(questions)]
        if index < 2:
            batch_items.append((text, qs))
        b = collator([(text, qs)], torch.device("cpu"))
        with torch.no_grad():
            logits = model(b["input_ids"], b["attention_mask"], b["opt_pos"], b["opt_mask"], b["q_pos"], b["seg"]).float()
        probabilities = (logits / model.temperature).softmax(-1)[0]
        answers = m.decide(text, questions)
        extra[f"c{index}.ids"] = b["input_ids"][0].to(torch.int32).contiguous()
        extra[f"c{index}.seg"] = b["seg"][0].to(torch.int32).contiguous()
        extra[f"c{index}.logits"] = logits[0].masked_fill(~b["opt_mask"][0], 0).contiguous()
        extra[f"c{index}.probabilities"] = probabilities.contiguous()
        if not isinstance(state, str):
            extra[f"c{index}.state_bytes"] = torch.tensor(list(text.encode("utf-8")), dtype=torch.int32)
        for qi, (q, answer) in enumerate(zip(questions, answers)):
            key = f"c{index}.q{qi}"
            if q["type"] == "choice":
                extra[key + ".choice"] = torch.tensor([q["options"].index(answer["choice"])], dtype=torch.int32)
                extra[key + ".confidence"] = torch.tensor([float(answer["confidence"])])
            elif q["type"] == "score":
                extra[key + ".score"] = torch.tensor([float(answer["score"])])
                extra[key + ".confidence"] = torch.tensor([float(answer["confidence"])])
            else:
                extra[key + ".noul"] = torch.tensor([float(answer["noul"])])
        if index == 0:
            first = logits[0].masked_fill(~b["opt_mask"][0], 0).clone()
            with torch.no_grad():
                encoded = model.backbone(input_ids=b["input_ids"], attention_mask=b["attention_mask"], output_hidden_states=True)
            for layer, hidden in enumerate(encoded.hidden_states):
                extra[f"hidden.{layer}"] = hidden[0].float().contiguous()
    b = collator(batch_items, torch.device("cpu"))
    with torch.no_grad():
        logits = model(b["input_ids"], b["attention_mask"], b["opt_pos"], b["opt_mask"], b["q_pos"], b["seg"]).float()
    gold = torch.tensor([[2, 1, 1], [0, 0, -100]])
    extra["batch.ids"] = b["input_ids"].to(torch.int32).contiguous()
    extra["batch.logits"] = logits.masked_fill(~b["opt_mask"], 0).contiguous()
    extra["batch.gold"] = gold.to(torch.int32)
    for weight in (1.0, 0.5):
        loss, info = decision_loss(logits, gold, weight)
        extra[f"batch.loss.{weight}"] = torch.tensor([float(loss), info["ce"], info["brier"]])
    for index, text in enumerate(_OJD_TEXTS):
        extra[f"text{index}"] = torch.tensor(m.tok(text, add_special_tokens=False)["input_ids"] or [-1], dtype=torch.int32)
    offsets = torch.arange(-600, 601)
    extra["buckets"] = make_log_bucket_position(offsets, 256, 512).to(torch.int32)
    extra["temperature"] = torch.tensor([float(model.temperature)])
    globals()["_extra"] = extra
    return first


_OPEN_JEV_REQUESTS = [
    ({"customer": "Dana", "message": "I was charged twice for order 88213 and want my money back.", "plan": "Pro", "vip": True},
     {"intent": {"type": "choice", "instructions": "Choose the customer intent.",
                 "criteria": {"billing": "A payment or refund issue", "technical": "A malfunction or setup issue", "other": None}},
      "urgency": {"type": "score", "instructions": "How urgent is the request?", "criteria": ["not urgent", "somewhat urgent", "very urgent"]},
      "refund": {"type": "noul", "instructions": "The customer asks for a refund.",
                 "criteria": {"true": "the message requests money back", "false": "no refund is requested"}}}),
    ("Café au lait costs €4.50 at the 2nd location; the resérvation for 12 people is at 19:30.",
     {"topic": {"type": "choice", "instructions": "What is the text about?",
                "criteria": {"food": None, "travel": None, "sports": None, "finance": None}},
      "numbers": {"type": "noul", "instructions": "The text mentions a time of day."}}),
]


def run_open_jev(image, checkpoint):
    """Open-Jev (ZefanCai/Open-Jev-2B or -9B), through the loader's own `jev.model.DecisionModel.load`
    and `jev.api`: each candidate is its own chat-templated Yes/No prompt through the Qwen3.5 text
    model with the LoRA adapter, and a float32 head reads the last token.

    `--checkpoint` is the package's `checkpoint` directory. `OPEN_JEV_BASE` is a local copy of the
    base model at the revision `model.json` pins, `OPEN_JEV_SOURCE` the loader checkout (the folder
    holding `jev/`), and `OPEN_JEV_DTYPE` the precision (`float32`, the default, or `bfloat16`, which
    is the loader's own). The loader's `from_pretrained` calls are redirected to the local base at that
    precision; everything else is its code.

    Two requests (`r{i}.*` per compiled record): a record state with a described 3-way choice, a
    3-level score, and a noul with meanings; a text state (accents, a decomposed accent, digits) with
    a 4-way choice and a plain noul. Each record carries every candidate's token ids, the raw logits
    (a noul's are [0, s]), the probabilities at the saved temperature, and the answer. At float32 the
    first candidate carries every hidden state of the text model. `loss.*` is the training loss at the
    released Brier weight 0.1 against fixed soft targets, per record.
    """
    import json as _json
    import os as _os
    import sys as _sys
    import torch
    _sys.path.insert(0, _os.environ["OPEN_JEV_SOURCE"])
    import jev.model as jev_model
    from jev.api import compile_request, candidate_prompts, format_response
    from jev.metrics import softmax

    base = _os.environ["OPEN_JEV_BASE"]
    dtype = getattr(torch, _os.environ.get("OPEN_JEV_DTYPE", "float32"))
    real_model, real_tokenizer = jev_model.AutoModelForImageTextToText, jev_model.AutoTokenizer

    class _LocalModel:
        @staticmethod
        def from_pretrained(model_id, revision=None, torch_dtype=None, **kw):
            kw.pop("device_map", None)
            return real_model.from_pretrained(base, dtype=dtype, **kw)

    class _LocalTokenizer:
        @staticmethod
        def from_pretrained(model_id, revision=None, **kw):
            return real_tokenizer.from_pretrained(base, **kw)

    jev_model.AutoModelForImageTextToText, jev_model.AutoTokenizer = _LocalModel, _LocalTokenizer
    model = jev_model.DecisionModel.load(checkpoint, device="cpu")
    temperature = _json.load(open(_os.path.join(checkpoint, "temperature.json")))["temperature"]
    extra = {"temperature": torch.tensor([float(temperature)])}
    targets = {"choice": lambda n: [1.0 if i == 1 else 0.0 for i in range(n)],
               "score": lambda n: [0.1, 0.7, 0.2][:n], "noul": lambda n: [0.3, 0.7]}
    first = None
    index = 0
    for state, questions in _OPEN_JEV_REQUESTS:
        records = compile_request(state, questions)
        for record in records:
            key = f"r{index}"
            prompts = [model.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                           add_generation_prompt=True, enable_thinking=False)
                       for text in candidate_prompts(record)]
            for c, text in enumerate(prompts):
                extra[f"{key}.c{c}.ids"] = torch.tensor(model.tokenizer(text)["input_ids"], dtype=torch.int32)
            with torch.no_grad():
                logits = model([record])[0].float()
            probabilities = softmax(logits.tolist(), temperature=temperature)
            answer = format_response([record], [probabilities])["answers"][record["id"]]
            extra[f"{key}.logits"] = logits.contiguous().clone()
            extra[f"{key}.probabilities"] = torch.tensor(probabilities)
            if record["kind"] == "choice":
                extra[f"{key}.choice"] = torch.tensor([record["answer_keys"].index(answer["choice"])], dtype=torch.int32)
                extra[f"{key}.confidence"] = torch.tensor([float(answer["confidence"])])
            elif record["kind"] == "score":
                extra[f"{key}.score"] = torch.tensor([float(answer["score"])])
                extra[f"{key}.confidence"] = torch.tensor([float(answer["confidence"])])
            else:
                extra[f"{key}.noul"] = torch.tensor([float(answer["noul"])])
            target = torch.tensor(targets[record["kind"]](len(logits)))
            loss = -(target * logits.log_softmax(-1)).sum() + 0.1 * ((logits.softmax(-1) - target) ** 2).sum()
            extra[f"loss.{key}"] = torch.tensor([float(loss)])
            extra[f"loss.{key}.target"] = target
            if first is None:
                first = logits.clone()
                if dtype == torch.float32:
                    ids = torch.tensor([model.tokenizer(prompts[0])["input_ids"]])
                    with torch.no_grad():
                        out = model.backbone(input_ids=ids, output_hidden_states=True, use_cache=False)
                    for layer, hidden in enumerate(out.hidden_states):
                        extra[f"hidden.{layer}"] = hidden[0].float().contiguous()
                    extra["last_hidden"] = out.last_hidden_state[0].float().contiguous().clone()
            index += 1
    globals()["_extra"] = extra
    return first


def run_open_jev_deberta_budget(image, checkpoint):
    """The state budget of typed-decisions' current collator, tokens only.

    The release's bundled collator raises when the state and questions overflow `max_len`; the
    repository's current one (`TYPED_DECISIONS_SOURCE`, the folder holding `typed_decisions/`) cuts
    the state further so the questions fit. `--checkpoint` is the release directory, for its
    tokenizer and limits. One case, `budget.ids`: a long state under ten 10-way choices, which the
    bundled collator refuses and the current one fits in exactly `max_len` tokens.
    """
    import json as _json
    import os as _os
    import sys as _sys
    import torch
    from transformers import AutoTokenizer
    _sys.path.insert(0, _os.environ["TYPED_DECISIONS_SOURCE"])
    from typed_decisions.encoder import Collator
    from typed_decisions.schema import Question

    cfg = _json.load(open(_os.path.join(checkpoint, "open_jev_config.json")))
    tok = AutoTokenizer.from_pretrained(checkpoint)
    collator = Collator(tok, max_state_tokens=cfg["max_state_tokens"], max_len=cfg["max_len"])
    state, _ = _OJD_CASES[2]
    options = ["finance", "hiring", "product", "legal", "marketing", "operations", "sales", "support", "security", "other"]
    questions = [Question(f"q{i}", "choice", f"Which department owns item number {i} in this report?", options, 0)
                 for i in range(10)]
    ids = collator([(state, questions)])["input_ids"][0]
    globals()["_extra"] = {"budget.ids": ids.to(torch.int32).contiguous()}
    return ids.float().clone()


_LAYA_EPISODE = {
    # Context keys sort before "conversation", which the reference appends last, so the port's sorted
    # serialization and the reference's insertion order write the same text.
    "ctx": {"account": 88213, "channel": "chat"},
    "turns": [
        {"role": "customer", "text": "Hi, my payouts have failed three times this week."},
        {"role": "agent", "text": "Sorry to hear that. Which bank is the account with?"},
        {"role": "customer", "text": "Nordbank, and nothing changed on my side."},
        {"role": "agent", "text": "I see two rejections from the receiving bank. Did they contact you?"},
        {"role": "customer", "text": "No. I have staff to pay on Friday, this cannot wait."},
        {"role": "agent", "text": "Understood. I am escalating this to the payments team now."},
        {"role": "customer", "text": "Thank you. Will I hear back today?"},
        {"role": "agent", "text": "Yes, within two hours, and the transfer will be retried."},
    ],
    "y": 1,
}


def run_laya_episode(image, checkpoint):
    """Laya's conversation-prefix training path, from the release's own `rl_common`: an eight-turn episode
    sampled to the release's `max_prefixes` prefix lengths (`episode_prefix_lengths`), each prefix built by
    `encode_record` (the context plus the turns so far under `conversation`, cut from the left), and the
    TD(lambda) targets `td_lambda_targets` produces over the grouped prefixes at lambda 1 (the release's
    setting) and lambda 0.5 from a fixed vector of next-prefix predictions. `--checkpoint` is the release
    root; only the tokenizer and the config are read. The record carries the prefix lengths, each
    prefix's ids, markers, and serialized state bytes, the predictions, and both target tables.
    """
    import torch
    agent = _laya_agent(checkpoint)
    from rl_common import collate_items, encode_record, episode_prefix_lengths, serialize_state, td_lambda_targets

    question = {"t": "noul", "ins": "The customer's problem will be resolved by the end of the conversation.",
                "crit": {"true": "the issue is on its way to a fix", "false": "the issue stays open"}}
    record = {"kind": "episode", "ep": _LAYA_EPISODE, "qs": [question], "src": "test"}
    items = encode_record(record, agent.tok, agent.cfg, None, False)
    for item in items:
        item["rec_uid"] = 0                     # one group, which is what td_lambda_targets walks
    lengths = episode_prefix_lengths(len(_LAYA_EPISODE["turns"]), agent.cfg["max_prefixes"])
    assert len(items) == len(lengths), (len(items), lengths)
    batch = collate_items([items], agent.tok.pad_token_id)
    torch.manual_seed(5)
    p_true = torch.rand(len(items))
    lam1 = td_lambda_targets(p_true, batch, 1.0)
    lam05 = td_lambda_targets(p_true, batch, 0.5)

    extra = {"prefix_lengths": torch.tensor(lengths, dtype=torch.int32),
             "p_true": p_true.contiguous(), "targets_lambda1": lam1.contiguous(),
             "targets_lambda05": lam05.contiguous(), "outcome": torch.tensor([int(_LAYA_EPISODE["y"])], dtype=torch.int32)}
    for index, (item, length) in enumerate(zip(items, lengths)):
        state = dict(_LAYA_EPISODE["ctx"], conversation=_LAYA_EPISODE["turns"][:length])
        extra[f"p{index}.ids"] = torch.tensor(item["ids"], dtype=torch.int32)
        extra[f"p{index}.markers"] = torch.tensor(item["markers"], dtype=torch.int32)
        extra[f"p{index}.state_bytes"] = torch.tensor(list(serialize_state(state).encode("utf-8")), dtype=torch.int32)
    globals()["_extra"] = extra
    return lam05.clone()


def run_laya_loss(image, checkpoint):
    """Laya's training objective, `rl_common.proper_reward`, on identical tensors: the log score plus half
    the spherical score for every question, minus the ranked probability score for an ordinal one, over
    six rows of mixed type and cardinality with one-hot and soft targets. `--checkpoint` is the release
    root, where rl_common.py lives; no weights are loaded. The record carries the raw logits, the
    targets, the option mask, the types, the per-row reward, and the mean loss the port minimizes.
    """
    import os
    import sys
    import torch

    root = os.path.abspath(checkpoint)
    while not os.path.exists(os.path.join(root, "rl_common.py")):
        root = os.path.dirname(root)
    sys.path.insert(0, root)
    from rl_common import proper_reward

    torch.manual_seed(11)
    counts = [3, 6, 2, 4, 5, 2]
    qtypes = torch.tensor([0, 0, 1, 1, 2, 0])
    rows, width = len(counts), max(counts)
    mask = torch.zeros(rows, width, dtype=torch.bool)
    targets = torch.zeros(rows, width)
    for row, k in enumerate(counts):
        mask[row, :k] = True
        if row % 2 == 0:
            targets[row, torch.randint(k, (1,)).item()] = 1.0
        else:
            soft = torch.rand(k)
            targets[row, :k] = soft / soft.sum()
    logits = torch.randn(rows, width) * 2
    q = torch.softmax(logits.masked_fill(~mask, -1e4), -1) * mask
    reward = proper_reward(q, targets, qtypes, mask)
    loss = -reward.mean()
    globals()["_extra"] = {
        "logits": logits.contiguous(), "targets": targets.contiguous(),
        "mask": mask.to(torch.int32).contiguous(), "types": qtypes.to(torch.int32).contiguous(),
        "reward": reward.contiguous(), "loss": loss.reshape(1).contiguous(),
    }
    return reward.clone()


def run_smolvlm(image, checkpoint):
    """A SmolVLM2 vision-language forward, from transformers' own SmolVLMForConditionalGeneration.

    `--checkpoint` is the released `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` directory. The image is
    the shared plate, resized to the processor's own tiling; the SigLIP vision encoder embeds each tile,
    the pixel-shuffle connector projects them to the decoder width, and the Llama decoder reads the text
    with the projected vision tokens spliced in at the image-token positions.

    The record carries the input ids, the processor's pixel values (so the Swift side skips the image
    processor and measures the network), the vision encoder's per-tile last hidden state and the
    connector's output for the isolation harness, the logits, and the greedy continuation.

    Needs Pillow, torchvision, and num2words for the processor; the `llm` oracle interpreter has them.
    """
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(checkpoint)
    model = AutoModelForImageTextToText.from_pretrained(checkpoint, dtype=torch.float32).eval()

    pil = Image.fromarray((image * 255).astype(np.uint8))
    prompt = "User:<image>What is in this image?<end_of_utterance>\nAssistant:"
    inputs = processor(text=prompt, images=[pil], return_tensors="pt")

    pixel_values = inputs["pixel_values"]                       # [1, tiles, 3, 512, 512]
    flat = pixel_values.view(-1, *pixel_values.shape[2:])       # [tiles, 3, 512, 512]
    patch = model.config.vision_config.patch_size
    grid = flat.shape[-1] // patch
    patch_attention_mask = torch.ones(flat.shape[0], grid, grid, dtype=torch.bool)

    vm = model.model.vision_model
    with torch.no_grad():
        embeddings = vm.embeddings(pixel_values=flat, patch_attention_mask=patch_attention_mask)
        layer0 = vm.encoder.layers[0](embeddings, attention_mask=None)[0]
        vision = vm(pixel_values=flat, patch_attention_mask=patch_attention_mask).last_hidden_state
        features = model.model.connector(vision)               # [tiles, 64, decoder width]
        logits = model(**inputs).logits[0]                     # [sequence, vocabulary]
        generated = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    continuation = generated[0, inputs["input_ids"].shape[1]:]

    globals()["_extra"] = {
        "input_ids": inputs["input_ids"][0].to(torch.int32).contiguous(),
        "pixel_values": flat.contiguous(),
        "vision_embeddings": embeddings.contiguous(),
        "vision_layer0": layer0.contiguous(),
        "vision_hidden": vision.contiguous(),
        "image_features": features.contiguous(),
        "continuation": continuation.to(torch.int32).contiguous(),
        "image_token_id": torch.tensor([model.config.image_token_id], dtype=torch.int32),
    }
    return logits.contiguous()


def run_qwen3vl(image, checkpoint):
    """Qwen3-VL's vision tower, from transformers' own Qwen3VLForConditionalGeneration.

    `--checkpoint` is the released `Qwen/Qwen3-VL-2B-Instruct` directory. The 2D-rotary ViT embeds each
    patch, a merger folds 2×2 patches to the decoder width, and the deepstack heads (layers 5/11/17)
    produce three feature maps. The record carries the processor's pixel values and grid, the vision
    seams for the isolation harness (patch embedding, the added position embedding, the 2D rotary table),
    and the merged output plus the three deepstack features, plus the input ids and logits.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(checkpoint)
    model = AutoModelForImageTextToText.from_pretrained(checkpoint, dtype=torch.float32).eval()

    pil = Image.fromarray((image * 255).astype(np.uint8))
    prompt = ("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is in this image?"
              "<|im_end|>\n<|im_start|>assistant\n")
    inputs = processor(text=[prompt], images=[pil], return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    grid_thw = inputs["image_grid_thw"]

    visual = model.model.visual
    with torch.no_grad():
        patch = visual.patch_embed(pixel_values)
        positions = visual.fast_pos_embed_interpolate(grid_thw)
        rotary = visual.rot_pos_emb(grid_thw)
        embeds, deepstack = visual(pixel_values, grid_thw=grid_thw)
        logits = model(**inputs).logits[0]
        generated = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    continuation = generated[0, inputs["input_ids"].shape[1]:]

    extra = {
        "pixel_values": pixel_values.contiguous(),
        "image_grid_thw": grid_thw.to(torch.int32).contiguous(),
        "patch_embed": patch.contiguous(),
        "pos_embeds": positions.contiguous(),
        "rotary_pos_emb": rotary.contiguous(),
        "vision_output": embeds.contiguous(),
        "input_ids": inputs["input_ids"][0].to(torch.int32).contiguous(),
        "continuation": continuation.to(torch.int32).contiguous(),
    }
    for index, feature in enumerate(deepstack):
        extra[f"deepstack_{index}"] = feature.contiguous()
    globals()["_extra"] = extra
    return logits.contiguous()


def run_pixtral(image, checkpoint):
    """Pixtral 12B's vision tower, connector, and fused decoder, from transformers' own
    LlavaForConditionalGeneration.

    `--checkpoint` is the released `mistral-experimental/pixtral-12b` directory. The vision tower is a
    from-scratch 2D-rotary ViT (PixtralVisionModel): a patch convolution, an RMSNorm, and blocks of
    RMSNorm-normalized attention with a 2D rotary over the (height, width) patch grid and a SiLU-gated
    MLP. A two-layer GELU connector projects the patch features to the decoder width, and the decoder
    is a Mistral-Nemo dense stack. The vision seams are recorded in float32 for a high-precision
    isolation harness; the fused logits and the greedy continuation are recorded in the release's
    bfloat16, since fp32 for the 12B decoder does not fit. The record carries the processor's pixel
    values, the patch-embedding, ln_pre, first-layer, and last-layer vision seams, the projected image
    features, the input ids, the logits, and the continuation.
    """
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(checkpoint)
    model = AutoModelForImageTextToText.from_pretrained(checkpoint, dtype=torch.bfloat16).eval()

    pil = Image.fromarray((image * 255).astype(np.uint8))
    prompt = "<s>[INST]Describe the image in detail.[IMG][/INST]"
    inputs = processor(text=prompt, images=[pil], return_tensors="pt")

    # The fused pass in the release dtype: the 12B decoder does not fit in float32 on this machine.
    # The processor returns float32 pixels, which the bfloat16 patch convolution rejects, so the pixel
    # values are cast to the model's dtype for the fused pass.
    fused = dict(inputs)
    fused["pixel_values"] = fused["pixel_values"].to(torch.bfloat16)
    with torch.no_grad():
        logits = model(**fused).logits[0]
        generated = model.generate(**fused, max_new_tokens=16, do_sample=False)
    continuation = generated[0, inputs["input_ids"].shape[1]:]

    # The vision tower and connector, upcast to float32 for high-precision seams.
    vision_tower = model.model.vision_tower.float()
    projector = model.model.multi_modal_projector.float()
    pixel_values = inputs["pixel_values"].float()
    image_sizes = inputs.get("image_sizes")
    with torch.no_grad():
        patch_conv = vision_tower.patch_conv(pixel_values)
        patch_embeds = patch_conv.flatten(2).transpose(1, 2)         # [1, patches, hidden], row-major
        vision = vision_tower(pixel_values, image_sizes=image_sizes, output_hidden_states=True)
        vision_output = vision.last_hidden_state
        projected = projector(vision_output)

    extra = {
        "pixel_values": pixel_values.contiguous(),
        "patch_embeds": patch_embeds.contiguous(),
        "ln_pre": vision.hidden_states[0].contiguous(),
        "layer0": vision.hidden_states[1].contiguous(),
        "vision_output": vision_output.contiguous(),
        "projected": projected.contiguous(),
        "input_ids": inputs["input_ids"][0].to(torch.int32).contiguous(),
        "continuation": continuation.to(torch.int32).contiguous(),
    }
    globals()["_extra"] = extra
    return logits.float().contiguous()


def run_pixtral_tiny(image, checkpoint):
    """The whole Pixtral pipeline at a tiny random configuration, from transformers' own
    LlavaForConditionalGeneration.

    The released 12B decoder does not fit this machine at any precision the fused pass needs, so the
    fusion — the vision tower, the connector, the scatter of the projected patch features into the
    `[IMG]` positions, and the Mistral decoder over the fused sequence — is measured at a size that
    runs in float32. `--checkpoint` is a writable directory the tiny release is saved into, which the
    Swift side loads through its ordinary release builders. The record carries the pixel values, the
    image sizes, the input ids, the projected features, the vision output, and the fused logits.
    """
    import numpy as np
    import torch
    from transformers import (LlavaConfig, LlavaForConditionalGeneration, MistralConfig,
                              PixtralVisionConfig)

    torch.manual_seed(23)
    vision = PixtralVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                 num_attention_heads=2, image_size=32, patch_size=4, rope_theta=10000.0)
    # num_attention_heads is intentionally omitted so the tiny model inherits the MistralConfig default
    # (32), exercising the same absent-count path the released Pixtral config needs — 48 / 16 = 3 would
    # be the wrong derived count, as 5120 / 128 = 40 is for the release.
    text = MistralConfig(hidden_size=48, intermediate_size=64, num_hidden_layers=2,
                         num_key_value_heads=2, head_dim=16, vocab_size=64,
                         rms_norm_eps=1e-5, rope_theta=1_000_000.0, max_position_embeddings=128,
                         tie_word_embeddings=False)
    config = LlavaConfig(vision_config=vision.to_dict(), text_config=text.to_dict(),
                         image_token_index=10, vision_feature_layer=-1,
                         vision_feature_select_strategy="full", projector_hidden_act="gelu")
    model = LlavaForConditionalGeneration(config).eval().float()

    height, width = 8, 12                                        # a 2×3 patch grid → 6 image tokens
    pixel_values = torch.randn(1, 3, height, width)
    image_sizes = torch.tensor([[height, width]])
    image_ids = [10] * 6
    input_ids = torch.tensor([[1] + image_ids + [5, 7, 9, 2]], dtype=torch.long)

    with torch.no_grad():
        vision_output = model.model.vision_tower(pixel_values, image_sizes=image_sizes).last_hidden_state
        projected = model.model.multi_modal_projector(vision_output)
        logits = model(input_ids=input_ids, pixel_values=pixel_values, image_sizes=image_sizes).logits[0]

    model.save_pretrained(checkpoint)

    extra = {
        "pixel_values": pixel_values.contiguous(),
        "image_sizes": image_sizes.to(torch.int32).contiguous(),
        "input_ids": input_ids[0].to(torch.int32).contiguous(),
        "vision_output": vision_output.contiguous(),
        "projected": projected.contiguous(),
    }
    globals()["_extra"] = extra
    return logits.float().contiguous()


def run_gguf(image, checkpoint):
    """Reference dequantization of a GGUF model's tensors, from the `gguf` package.

    `--checkpoint` is a `.gguf` file. For the first tensor of each block-quant type the native reader
    implements, this records the dequantized values (capped so the record stays small) under the type's
    name. The Swift side reads the same GGUF, picks the first tensor of each type in file order — the
    same one — dequantizes it, and compares. `gguf` is the ground truth, the way llama.cpp is for the
    format.
    """
    import numpy as np
    import torch
    from gguf import GGUFReader
    from gguf.quants import dequantize

    reader = GGUFReader(checkpoint)
    cap = 262_144
    seen = set()
    extra = {}
    for tensor in reader.tensors:
        name = tensor.tensor_type.name
        if name in ("Q4_K", "Q6_K", "Q8_0", "Q5_0", "F32") and name not in seen:
            seen.add(name)
            values = dequantize(tensor.data, tensor.tensor_type).flatten()[:cap].astype(np.float32)
            extra[name] = torch.from_numpy(np.ascontiguousarray(values))
    globals()["_extra"] = extra
    return torch.zeros(1)


def _tiny_decoder_record(model, tokens, release_name=lambda key: key):
    """Logits, every hidden state, and the weights in release naming, for a tiny random model.

    `IK_TINY_DTYPE=bfloat16` runs the model at bf16 with eager attention, the record a bf16 port's
    rounding placement is held to; `IK_TINY_DTYPE=bfloat16-weights` runs float32 arithmetic on the same
    bf16-rounded weights, its floor. The pair differs in arithmetic precision alone, as a released bf16
    checkpoint's float32 and bf16 runs do.
    """
    mode = os.environ.get("IK_TINY_DTYPE")
    if mode in ("bfloat16", "bfloat16-weights"):
        model = model.to(torch.bfloat16)
        if mode == "bfloat16-weights":
            model = model.float()
        for module in model.modules():
            config = getattr(module, "config", None)
            if config is not None and hasattr(config, "_attn_implementation"):
                config._attn_implementation = "eager"
    extra = {"tokens": tokens[0].to(torch.int32).contiguous()}
    restore = None
    if os.environ.get("IK_PROBE_LAYERS"):
        restore = _probe_hooks(model, [int(x) for x in os.environ["IK_PROBE_LAYERS"].split(",")], extra)
    with torch.no_grad():
        out = model(tokens, output_hidden_states=True)
    if restore is not None:
        restore()
    for index, hidden in enumerate(out.hidden_states):
        extra[f"hidden.{index}"] = hidden[0].float().contiguous()
    for key, value in model.state_dict().items():
        extra[f"w::{release_name(key)}"] = (value.float() if value.is_floating_point()
                                            else value).contiguous()
    globals()["_extra"] = extra
    return out.logits[0].float().contiguous()


def _randomized(model, seed=11, scale=0.05):
    """`model` with every floating tensor drawn from a seeded normal at `scale`, in float32.

    `IK_DIT_DTYPE=bfloat16` then runs it at bf16, every floating input rounded to bf16 on the way in, the
    record a bf16 port's rounding placement is held to; `bfloat16-weights` runs float32 arithmetic on
    the same rounded weights and inputs, the floor for that record. The weights recorded as `w::` are
    the rounded ones either way."""
    torch.manual_seed(seed)
    state = model.state_dict()
    for key in sorted(state):
        if state[key].is_floating_point():
            state[key] = torch.randn(state[key].shape) * scale
    model.load_state_dict(state)
    model.eval().float()
    mode = os.environ.get("IK_DIT_DTYPE")
    if mode:
        # `from_pretrained` casts what the state dict holds and leaves a non-persistent buffer (a
        # sinusoid's frequencies, a rotary table) as its constructor built it, so those are restored.
        persistent = set(model.state_dict())
        kept = {name: buffer.clone() for name, buffer in model.named_buffers()
                if name not in persistent and buffer.is_floating_point()}
        model.to(torch.bfloat16)
        target = torch.bfloat16 if mode == "bfloat16" else torch.float32
        model.to(target)
        # A transformers model reads its attention implementation from its config at every call.
        if hasattr(getattr(model, "config", None), "_attn_implementation"):
            model.config._attn_implementation = "eager"
        for name, buffer in kept.items():
            owner, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(owner) if owner else model, leaf, buffer)

        def rounded(value):
            if torch.is_tensor(value) and value.is_floating_point():
                return value.to(torch.bfloat16).to(target)
            return value

        def round_inputs(module, args, kwargs):
            return tuple(rounded(a) for a in args), {k: rounded(v) for k, v in kwargs.items()}

        model.register_forward_pre_hook(round_inputs, with_kwargs=True)
        # Every submodule's first call is kept as `probe.<name>.in` / `.out` for localizing a seam.
        for name, module in model.named_modules():
            if not name:
                continue

            def keep(module, inputs, output, name=name):
                out = output[0] if isinstance(output, tuple) else output
                if f"probe.{name}.out" in _DIT_PROBES or not torch.is_tensor(out):
                    return
                _DIT_PROBES[f"probe.{name}.out"] = out.detach().float().clone().contiguous()
                if inputs and torch.is_tensor(inputs[0]):
                    _DIT_PROBES[f"probe.{name}.in"] = inputs[0].detach().float().clone().contiguous()
            module.register_forward_hook(keep)
    return model


_DIT_PROBES = {}


def run_qwen3_moe(image, checkpoint):
    """The Qwen3-MoE decoder's arithmetic, from transformers' own Qwen3MoeForCausalLM, at a tiny
    random configuration.

    The released sizes (30B-A3B and up) do not fit this machine at any precision the oracle can run,
    so the mixture-of-experts feed-forward — router softmax over every expert, top-k selection with
    the selected weights renormalized (`norm_topk_prob`), the experts' SwiGLU, the weighted sum — is
    measured at a size that does, with the dense attention around it. Every hidden state is recorded
    so a divergence is located to a layer. The weights are saved under the release's own names
    (`mlp.experts.N.gate_proj.weight`), which the Swift loader stacks. `checkpoint` is unused.
    """
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        hidden_size=64, num_hidden_layers=3, vocab_size=128, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=96, moe_intermediate_size=32,
        num_experts=8, num_experts_per_tok=2, norm_topk_prob=True, decoder_sparse_step=1,
        mlp_only_layers=[], rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=False,
        max_position_embeddings=64, attention_bias=False)
    model = _randomized(Qwen3MoeForCausalLM(config))
    tokens = torch.tensor([[3, 17, 42, 99, 7, 61, 12, 5]], dtype=torch.long)
    return _tiny_decoder_record(model, tokens)


def run_mixtral(image, checkpoint):
    """The Mixtral decoder's arithmetic, from transformers' own MixtralForCausalLM, at a tiny random
    configuration.

    Mixtral's block takes the softmax over the SELECTED experts' logits, which equals Qwen3-MoE's
    softmax-over-all followed by renormalization; this record is what holds the Swift module, which
    computes the latter form for both, to Mixtral's own arithmetic. Its tensor names differ
    (`block_sparse_moe.experts.N.w1/w3/w2`), so the record also measures the loader's rename. No
    sliding window, as the released 8x7B has none. `checkpoint` is unused.
    """
    from transformers import MixtralConfig, MixtralForCausalLM

    config = MixtralConfig(
        hidden_size=64, num_hidden_layers=3, vocab_size=128, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=32, num_local_experts=4,
        num_experts_per_tok=2, rms_norm_eps=1e-6, rope_theta=10000.0, sliding_window=None,
        max_position_embeddings=64, tie_word_embeddings=False)
    model = _randomized(MixtralForCausalLM(config), seed=13)
    tokens = torch.tensor([[3, 17, 42, 99, 7, 61, 12, 5]], dtype=torch.long)
    return _tiny_decoder_record(model, tokens)


def run_mamba2(image, checkpoint):
    """The Mamba-2 decoder's arithmetic, from transformers' own Mamba2ForCausalLM, at a tiny random
    configuration.

    Mamba-2 is the toolkit's first state-space model: every layer replaces attention with a selective
    scan — a fused input projection, a depthwise causal convolution over x/B/C, the SSD recurrence, a
    gated RMS normalization, and an output projection. The released Codestral-Mamba-7B does not fit
    this machine at float32, so the scan's arithmetic is measured at a size that does. Every hidden
    state is recorded so a divergence is located to a layer; transformers' CPU path is the naive scan,
    which is what the Swift port matches. The weights are saved under the release's own names, read
    unchanged but for squeezing the depthwise convolution `[C, 1, K]` to `[C, K]`. `checkpoint` unused.
    """
    from transformers import Mamba2Config, Mamba2ForCausalLM

    config = Mamba2Config(
        hidden_size=64, num_hidden_layers=3, vocab_size=128, num_heads=8, head_dim=16,
        state_size=16, n_groups=2, conv_kernel=4, expand=2, chunk_size=8,
        use_conv_bias=True, use_bias=False, tie_word_embeddings=False)
    model = _randomized(Mamba2ForCausalLM(config), seed=23)
    tokens = torch.tensor([[3, 17, 42, 99, 7, 61, 12, 5]], dtype=torch.long)
    return _tiny_decoder_record(model, tokens)


def run_mamba2_real(image, checkpoint):
    """Codestral-Mamba-7B logits + greedy continuation from transformers' Mamba2ForCausalLM.

    `--checkpoint` is the release directory (config.json + the sharded safetensors + tokenizer). The
    7B does not fit float32 on a 32 GB machine, so both sides run bfloat16 (as the Gemma E4B parity
    does). The record carries the prompt tokens, the prefill logits, and the greedy continuation, so a
    divergence shows as a logit cosine below the bf16 floor or a differing token.

    Parity is a function of the token ids, and the release's tokenizer needs a sentencepiece/protobuf
    stack the oracle environment does not carry, so a fixed id prompt is fed directly. Both sides read
    the same ids from the record, so the encoding is immaterial.
    """
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16).eval()

    ids = torch.tensor([[1, 1602, 4934, 322, 1148, 42, 7]], dtype=torch.long)
    with torch.no_grad():
        logits = model(ids).logits[0].float()
        generated = model.generate(ids, max_new_tokens=12, do_sample=False, pad_token_id=0)
    continuation = generated[0, ids.shape[1]:]
    globals()["_extra"] = {
        "tokens": ids[0].to(torch.int32).contiguous(),
        "continuation": continuation.to(torch.int32).contiguous(),
    }
    return logits.contiguous()


def run_granite_hybrid(image, checkpoint):
    """Granite 4.0-H (`GraniteMoeHybridForCausalLM`, IBM) at a tiny random configuration: a hybrid of
    Mamba-2 selective-scan layers and NoPE grouped-query attention layers, a gated-linear shared MLP,
    and Granite's scalar multipliers (embedding, residual, attention, logits). This config is DENSE
    (`num_local_experts` 0); the routed mixture of experts is a separate mode. The multipliers are set
    to values distinct from the released defaults so a port that hard-codes the defaults diverges.
    Every hidden state is recorded so a divergence is located to a layer. `checkpoint` unused.
    """
    from transformers import GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM

    config = GraniteMoeHybridConfig(
        hidden_size=64, num_hidden_layers=6, vocab_size=128,
        num_attention_heads=4, num_key_value_heads=2,
        mamba_n_heads=8, mamba_d_head=16, mamba_n_groups=1, mamba_d_state=16, mamba_d_conv=4,
        mamba_expand=2, mamba_chunk_size=8, shared_intermediate_size=96,
        num_local_experts=0, num_experts_per_tok=0, intermediate_size=0,
        embedding_multiplier=2.0, residual_multiplier=0.5, attention_multiplier=0.25, logits_scaling=3.0,
        layer_types=["mamba", "mamba", "mamba", "mamba", "mamba", "attention"],
        tie_word_embeddings=False)  # untied so safetensors does not refuse the aliased lm_head
    model = _randomized(GraniteMoeHybridForCausalLM(config), seed=29)
    tokens = torch.tensor([[3, 17, 42, 99, 7, 61]], dtype=torch.long)
    return _tiny_decoder_record(model, tokens)


def run_granite_hybrid_moe(image, checkpoint):
    """Granite 4.0-H at a tiny random configuration WITH the routed mixture of experts (the h-tiny /
    h-small feed-forward): every layer carries `num_local_experts` experts routed `num_experts_per_tok`
    at a time, whose output sums with the shared MLP. `block_sparse_moe.input_linear` is the fused
    gate+up projection, `output_linear` the down projection, both stored `[experts, out, in]`, and
    `router.layer` scores the experts with the softmax taken over the chosen top-k. The multipliers
    are set distinct from the released defaults. Every hidden state is recorded. `checkpoint` unused.
    """
    from transformers import GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM

    config = GraniteMoeHybridConfig(
        hidden_size=64, num_hidden_layers=6, vocab_size=128,
        num_attention_heads=4, num_key_value_heads=2,
        mamba_n_heads=8, mamba_d_head=16, mamba_n_groups=1, mamba_d_state=16, mamba_d_conv=4,
        mamba_expand=2, mamba_chunk_size=8, shared_intermediate_size=96,
        num_local_experts=8, num_experts_per_tok=2, intermediate_size=32,
        embedding_multiplier=2.0, residual_multiplier=0.5, attention_multiplier=0.25, logits_scaling=3.0,
        layer_types=["mamba", "mamba", "attention", "mamba", "mamba", "attention"],
        tie_word_embeddings=False)  # untied so safetensors does not refuse the aliased lm_head
    model = _randomized(GraniteMoeHybridForCausalLM(config), seed=31)
    tokens = torch.tensor([[3, 17, 42, 99, 7, 61]], dtype=torch.long)
    return _tiny_decoder_record(model, tokens)


def run_granite_hybrid_loss(image, checkpoint):
    """The causal language-model loss `GraniteMoeHybridForCausalLM` computes for the tiny dense config,
    for the fine-tune objective's parity. The same seed-29 weights as `granite_hybrid`, so the logits
    match that record; `labels=tokens` makes transformers apply its shifted cross-entropy. The scalar
    loss is the reference output; the logits and tokens are recorded so the port scores the objective on
    IDENTICAL logits (isolating the objective's arithmetic from the forward pass). `checkpoint` unused.
    """
    import torch
    from transformers import GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM

    config = GraniteMoeHybridConfig(
        hidden_size=64, num_hidden_layers=6, vocab_size=128,
        num_attention_heads=4, num_key_value_heads=2,
        mamba_n_heads=8, mamba_d_head=16, mamba_n_groups=1, mamba_d_state=16, mamba_d_conv=4,
        mamba_expand=2, mamba_chunk_size=8, shared_intermediate_size=96,
        num_local_experts=0, num_experts_per_tok=0, intermediate_size=0,
        embedding_multiplier=2.0, residual_multiplier=0.5, attention_multiplier=0.25, logits_scaling=3.0,
        layer_types=["mamba", "mamba", "mamba", "mamba", "mamba", "attention"],
        tie_word_embeddings=False)
    model = _randomized(GraniteMoeHybridForCausalLM(config), seed=29)
    tokens = torch.tensor([[3, 17, 42, 99, 7, 61]], dtype=torch.long)
    with torch.no_grad():
        out = model(tokens, labels=tokens)
    globals()["_extra"] = {
        "tokens": tokens[0].to(torch.int32).contiguous(),
        "logits": out.logits[0].float().contiguous(),
    }
    return torch.tensor([out.loss.item()], dtype=torch.float32)


def run_granite_hybrid_real(image, checkpoint):
    """Granite 4.0-H released decoder logits + greedy continuation from transformers, float32.

    `--checkpoint` is the release directory. granite-4.0-h-1b is the DENSE hybrid (no routed experts)
    and small enough for float32, so it gets a full numeric oracle. A fixed id prompt is fed directly
    (the tokenizer is sidestepped; parity is a function of the ids). The record carries the tokens, the
    per-layer hidden states for seam isolation, the prefill logits, and the greedy continuation.
    `IK_GRANITE_DTYPE=bfloat16` builds it at the released bf16 with eager attention instead, the
    record a bf16 load's rounding placement is held to.
    """
    import torch
    from transformers import AutoModelForCausalLM

    if os.environ.get("IK_GRANITE_DTYPE") == "bfloat16":
        model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16,
                                                     attn_implementation="eager").eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32).eval()
    ids = torch.tensor([[1602, 4934, 322, 1148, 42, 7, 55]], dtype=torch.long)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
        logits = out.logits[0].float()
        generated = model.generate(ids, max_new_tokens=12, do_sample=False, pad_token_id=0)
    continuation = generated[0, ids.shape[1]:]
    extra = {
        "tokens": ids[0].to(torch.int32).contiguous(),
        "continuation": continuation.to(torch.int32).contiguous(),
    }
    for index, hidden in enumerate(out.hidden_states):
        extra[f"hidden.{index}"] = hidden[0].float().contiguous()
    globals()["_extra"] = extra
    return logits.contiguous()


def run_nemotron_h(image, checkpoint):
    """Nemotron-H (`NemotronHForCausalLM`, NVIDIA) at a tiny random configuration: the hybrid decoder
    behind Nemotron Nano 2, interleaving Mamba-2 selective-scan layers (`linear_attention`), NoPE
    grouped-query attention layers (`full_attention`), and ReLU-squared dense feed-forward layers
    (`mlp`), one mixer per block. Unlike Granite there are no scalar multipliers and each block carries
    a single pre-norm with a plain residual add. `n_groups` is 2 so the Mamba mixer's gated output norm
    exercises its GROUPED path (HF's `Zamba2RMSNormGated`), the one departure from the ungrouped
    Codestral/Granite mixer. Every hidden state is recorded so a divergence is located to a layer.
    Requires the music oracle (transformers >= 5, which carries `NemotronHForCausalLM`). `checkpoint`
    unused.
    """
    from transformers import NemotronHConfig, NemotronHForCausalLM

    config = NemotronHConfig(
        hidden_size=64, vocab_size=128,
        mamba_num_heads=8, mamba_head_dim=16, ssm_state_size=16, n_groups=2, conv_kernel=4, chunk_size=8,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=96, mlp_hidden_act="relu2", mamba_hidden_act="silu",
        layer_norm_epsilon=1e-5, time_step_min=0.001, time_step_max=0.1, time_step_floor=1e-4,
        use_conv_bias=True, use_bias=False, mlp_bias=False, tie_word_embeddings=False,
        layers_block_type=["linear_attention", "mlp", "linear_attention", "full_attention",
                           "linear_attention", "mlp"],
        attn_implementation="eager")
    model = _randomized(NemotronHForCausalLM(config), seed=37)
    tokens = torch.tensor([[3, 17, 42, 99, 7, 61]], dtype=torch.long)
    return _tiny_decoder_record(model, tokens)


def run_nemotron_h_loss(image, checkpoint):
    """The causal language-model loss `NemotronHForCausalLM` computes for the tiny config, for the
    fine-tune objective's parity. The same seed-37 weights as `nemotron_h`, so the logits match that
    record; `labels=tokens` makes transformers apply its shifted cross-entropy. The scalar loss is the
    reference output; the logits and tokens are recorded so the port scores the objective on IDENTICAL
    logits, isolating the objective's arithmetic from the forward pass. `checkpoint` unused.
    """
    import torch
    from transformers import NemotronHConfig, NemotronHForCausalLM

    config = NemotronHConfig(
        hidden_size=64, vocab_size=128,
        mamba_num_heads=8, mamba_head_dim=16, ssm_state_size=16, n_groups=2, conv_kernel=4, chunk_size=8,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=96, mlp_hidden_act="relu2", mamba_hidden_act="silu",
        layer_norm_epsilon=1e-5, time_step_min=0.001, time_step_max=0.1, time_step_floor=1e-4,
        use_conv_bias=True, use_bias=False, mlp_bias=False, tie_word_embeddings=False,
        layers_block_type=["linear_attention", "mlp", "linear_attention", "full_attention",
                           "linear_attention", "mlp"],
        attn_implementation="eager")
    model = _randomized(NemotronHForCausalLM(config), seed=37)
    tokens = torch.tensor([[3, 17, 42, 99, 7, 61]], dtype=torch.long)
    with torch.no_grad():
        out = model(tokens, labels=tokens)
    globals()["_extra"] = {
        "tokens": tokens[0].to(torch.int32).contiguous(),
        "logits": out.logits[0].float().contiguous(),
    }
    return torch.tensor([out.loss.item()], dtype=torch.float32)


def run_nemotron_h_real(image, checkpoint):
    """Nemotron Nano 2 released decoder logits + greedy continuation from transformers.

    `--checkpoint` is the release directory (`nvidia/NVIDIA-Nemotron-Nano-9B-v2`). The 9B does not fit
    float32 on a 32 GB machine, so both sides run bfloat16 (as the Codestral-Mamba and Gemma E4B parity
    do). A fixed id prompt is fed directly (the tokenizer is sidestepped; parity is a function of the
    ids). The record carries the tokens, the per-layer hidden states for seam isolation, the prefill
    logits, and the greedy continuation. Requires the music oracle (transformers >= 5).
    """
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.bfloat16, trust_remote_code=True).eval()
    ids = torch.tensor([[1602, 4934, 322, 1148, 42, 7, 55]], dtype=torch.long)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
        logits = out.logits[0].float()
        generated = model.generate(ids, max_new_tokens=12, do_sample=False, pad_token_id=0)
    continuation = generated[0, ids.shape[1]:]
    extra = {
        "tokens": ids[0].to(torch.int32).contiguous(),
        "continuation": continuation.to(torch.int32).contiguous(),
    }
    for index, hidden in enumerate(out.hidden_states):
        extra[f"hidden.{index}"] = hidden[0].float().contiguous()
    globals()["_extra"] = extra
    return logits.contiguous()


def run_granite_speech(image, checkpoint):
    """Granite Speech 3.3-2B (`GraniteSpeechForConditionalGeneration`, IBM) at a tiny random config: a
    Conformer CTC encoder, a BLIP-2 Q-former projector, and a dense Granite decoder. Seams recorded so a
    divergence localizes: the encoder output (Conformer with Shaw relative-position block-local
    attention, macaron feed-forward, GLU convolution, mid-stack CTC skip), the projector output (the
    windowed Q-former's self- then cross-attention with learned queries), and the fused logits (audio
    features scattered into the decoder prompt at the audio-token positions, then the dense Granite
    decoder with its four scalar multipliers). BatchNorm running variance is set positive (a randomized
    negative variance would NaN both sides). `checkpoint` unused. Requires the llm oracle (transformers
    4.57.6, which carries GraniteSpeech and the dense `granite` decoder).
    """
    import torch
    from transformers import GraniteSpeechConfig, GraniteSpeechForConditionalGeneration
    from transformers.models.granite_speech.configuration_granite_speech import GraniteSpeechEncoderConfig

    enc = GraniteSpeechEncoderConfig(
        input_dim=16, hidden_dim=32, output_dim=24, num_layers=4, num_heads=2, dim_head=16,
        feedforward_mult=2, conv_expansion_factor=2, conv_kernel_size=5, context_size=8, max_pos_emb=16)
    text = dict(model_type="granite", hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, head_dim=8, intermediate_size=64, vocab_size=40, rope_theta=1e6,
                rms_norm_eps=1e-5, embedding_multiplier=2.0, residual_multiplier=0.5,
                attention_multiplier=0.25, logits_scaling=3.0, tie_word_embeddings=False,
                max_position_embeddings=64)  # untied so safetensors does not refuse the aliased lm_head
    proj = dict(model_type="blip_2_qformer", hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                intermediate_size=64, encoder_hidden_size=32, hidden_act="gelu", layer_norm_eps=1e-12)
    config = GraniteSpeechConfig(encoder_config=enc.to_dict(), text_config=text, projector_config=proj,
                                 audio_token_index=39, window_size=4, downsample_rate=2,
                                 has_lora_adapter=False)
    model = GraniteSpeechForConditionalGeneration(config).eval()
    state = model.state_dict()
    generator = torch.Generator().manual_seed(41)
    for key in sorted(state):
        tensor = state[key]
        if not tensor.is_floating_point():
            continue
        if key.endswith("running_var"):
            state[key] = torch.ones_like(tensor)
        elif key.endswith("running_mean"):
            state[key] = torch.zeros_like(tensor)
        else:
            state[key] = torch.randn(tensor.shape, generator=generator) * 0.05
    model.load_state_dict(state)
    model.eval()

    features = torch.randn(1, 12, 16, generator=generator)
    tokens = torch.tensor([[3, 17, 39, 39, 39, 39, 39, 39, 5]], dtype=torch.long)
    features_mask = torch.ones(1, 6, dtype=torch.bool)
    with torch.no_grad():
        encoder_out = model.encoder(features)
        projector_out = model.projector(encoder_out)
        out = model(input_ids=tokens, input_features=features, input_features_mask=features_mask)
    extra = {
        "tokens": tokens[0].to(torch.int32).contiguous(),
        "features": features[0].float().contiguous(),
        "encoder_out": encoder_out[0].float().contiguous(),
        "projector_out": projector_out[0].float().contiguous(),
    }
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = (value.float() if value.is_floating_point() else value).contiguous()
    globals()["_extra"] = extra
    return out.logits[0].float().contiguous()


def _probe_encoder_layers(layers, probes):
    """Hooks every submodule of an encoder's `layers`, keeping each one's first call as
    `enc.<layer>.<submodule>.in` / `.out` (`block` for the layer itself) in `probes`."""
    import torch

    def keep(value):
        return value[0].detach().float().clone().contiguous()

    for index, layer in enumerate(layers):
        for name, module in layer.named_modules():
            def hook(module, inputs, output, label=f"enc.{index}.{name or 'block'}"):
                if f"{label}.out" in probes or f"{label}.in" in probes:
                    return
                if inputs and torch.is_tensor(inputs[0]):
                    probes[f"{label}.in"] = keep(inputs[0])
                out = output[0] if isinstance(output, tuple) else output
                if torch.is_tensor(out):
                    probes[f"{label}.out"] = keep(out)
            module.register_forward_hook(hook)


def run_granite_speech_real(image, checkpoint):
    """Granite Speech 3.3-2b released decoder, float32, with the audio LoRA adapter enabled — the
    audio-active model. `--checkpoint` is the release directory (`from_pretrained` auto-loads the adapter
    it ships). Fixed random log-mel features and a prompt with the right number of audio tokens are fed
    directly (parity is a function of the inputs). The record carries the tokens, the features, the
    encoder and projector seams, the prefill logits, and the greedy continuation. Requires the llm oracle
    (transformers 4.57.6 + peft).
    """
    import torch
    from transformers import GraniteSpeechForConditionalGeneration

    # `IK_GRANITE_SPEECH_DTYPE=bfloat16` builds the model at bf16 with eager attention; `bfloat16-inputs`
    # keeps float32 arithmetic but rounds the features to bf16, the floor for that record. Both draw the
    # same features, so the pair differs in arithmetic precision alone.
    # `bfloat16-folded` folds the float32 LoRA into the decoder weights before the bf16 cast, as the port
    # loads it; transformers otherwise applies the adapter as its own bf16 branch, which alone moves the
    # logits by most of the bf16 floor.
    mode = os.environ.get("IK_GRANITE_SPEECH_DTYPE")
    dtype = torch.bfloat16 if mode == "bfloat16" else torch.float32
    model = GraniteSpeechForConditionalGeneration.from_pretrained(
        checkpoint, dtype=dtype, attn_implementation="eager" if mode else None).eval()
    if hasattr(model, "enable_adapters"):
        model.enable_adapters()                                   # the audio path the adapter serves
    if mode == "bfloat16-folded":
        with torch.no_grad():
            for module in model.modules():
                for key in getattr(module, "lora_A", {}):
                    delta = module.lora_B[key].weight @ module.lora_A[key].weight
                    module.base_layer.weight += delta * module.scaling[key]
        model.disable_adapters()
        model, dtype = model.to(torch.bfloat16), torch.bfloat16
        # `.to` casts the rotary's inverse frequencies too; a bf16 load keeps them float32.
        for module in model.modules():
            if hasattr(module, "inv_freq") and hasattr(module, "rope_init_fn"):
                inverse, _ = module.rope_init_fn(module.config, module.inv_freq.device)
                module.inv_freq = inverse.float()
                module.original_inv_freq = module.inv_freq
    torch.manual_seed(3)
    frames = 30
    features = torch.randn(1, frames, 160)
    if mode:
        features = features.to(torch.bfloat16).to(dtype)
    audioCount = ((frames + 14) // 15) * 3                        # ceil(frames / window) · queries
    audioId = model.config.audio_token_id
    ids = torch.tensor([[1, 100, 200] + [audioId] * audioCount + [300]], dtype=torch.long)
    mask = torch.ones(1, audioCount, dtype=torch.bool)
    # `IK_PROBE_ENCODER=1` records every Conformer submodule's input and output on the encoder pass,
    # keyed `enc.<layer>.<submodule>.in` / `.out`, for isolating a block piece by piece.
    probes = {}
    if os.environ.get("IK_PROBE_ENCODER"):
        _probe_encoder_layers(model.encoder.layers, probes)
    with torch.no_grad():
        encoder_out = model.encoder(features)
        recorded = dict(probes)
        projector_out = model.projector(encoder_out)
        # `IK_PROBE_LAYERS` records the named decoder layers on the prefill, as `hf_layer_probe` does.
        decoder_probes, restore = {}, None
        if os.environ.get("IK_PROBE_LAYERS"):
            layers = [int(x) for x in os.environ["IK_PROBE_LAYERS"].split(",")]
            restore = _probe_hooks(model.language_model, layers, decoder_probes)
        logits = model(input_ids=ids, input_features=features, input_features_mask=mask).logits[0].float()
        recorded.update({f"dec.{key}": value for key, value in decoder_probes.items()})
        if restore is not None:
            restore()
        generated = model.generate(input_ids=ids, input_features=features, input_features_mask=mask,
                                   max_new_tokens=8, do_sample=False)
    continuation = generated[0, ids.shape[1]:]
    globals()["_extra"] = {
        "tokens": ids[0].to(torch.int32).contiguous(),
        "features": features[0].float().contiguous(),
        "encoder_out": encoder_out[0].float().contiguous(),
        "projector_out": projector_out[0].float().contiguous(),
        "continuation": continuation.to(torch.int32).contiguous(),
        **recorded,
    }
    return logits.contiguous()


def run_voxtral(image, checkpoint):
    """Voxtral-Mini (`VoxtralForConditionalGeneration`, Mistral) at a tiny random config: a Whisper audio
    encoder, a two-layer projector that groups four encoder frames, and a Llama decoder that generates
    with the audio embeddings scattered into the prompt at the audio-token positions. Seams recorded: the
    encoder last hidden state, the projected audio embeddings, and the fused logits. `checkpoint` unused.
    Requires the llm oracle (transformers 4.57.6, which carries Voxtral).
    """
    from transformers import VoxtralConfig, VoxtralForConditionalGeneration
    from transformers.models.voxtral.configuration_voxtral import VoxtralEncoderConfig

    audio = VoxtralEncoderConfig(num_mel_bins=32, hidden_size=64, intermediate_size=256,
                                 num_attention_heads=4, num_hidden_layers=2, max_source_positions=24)
    text = dict(model_type="llama", hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, intermediate_size=128, vocab_size=40, rope_theta=1e4,
                rms_norm_eps=1e-5, tie_word_embeddings=False, max_position_embeddings=64)
    config = VoxtralConfig(audio_config=audio.to_dict(), text_config=text, audio_token_id=39,
                           projector_hidden_act="gelu")
    model = VoxtralForConditionalGeneration(config).eval()
    state = model.state_dict()
    generator = torch.Generator().manual_seed(41)
    for key in sorted(state):
        if state[key].is_floating_point():
            state[key] = torch.randn(state[key].shape, generator=generator) * 0.05
    # The encoder positional embedding is Whisper's fixed sinusoids, which the port computes rather than
    # loads; set it so the tiny oracle matches (the released model already carries the sinusoids).
    import math
    posLength, channels = state["audio_tower.embed_positions.weight"].shape
    half = channels // 2
    logTimescale = math.log(10000) / max(half - 1, 1)
    sinusoids = torch.zeros(posLength, channels)
    for t in range(posLength):
        for i in range(half):
            scaled = t * math.exp(-logTimescale * i)
            sinusoids[t, i] = math.sin(scaled)
            sinusoids[t, half + i] = math.cos(scaled)
    state["audio_tower.embed_positions.weight"] = sinusoids
    model.load_state_dict(state)
    model.eval()

    features = torch.randn(1, 32, 48, generator=generator)      # [batch, mels, 2·max_source_positions]
    with torch.no_grad():
        encoder_out = model.audio_tower(features).last_hidden_state
        audio_embeds = model.get_audio_features(features)
        audioCount = audio_embeds.shape[0]
        tokens = torch.tensor([[1, 5] + [39] * audioCount + [7]], dtype=torch.long)
        logits = model(input_ids=tokens, input_features=features).logits[0].float()
    extra = {
        "tokens": tokens[0].to(torch.int32).contiguous(),
        "features": features[0].float().contiguous(),
        "encoder_out": encoder_out[0].float().contiguous(),
        "audio_embeds": audio_embeds.float().contiguous(),
    }
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = (value.float() if value.is_floating_point() else value).contiguous()
    globals()["_extra"] = extra
    return logits.contiguous()


def run_voxtral_real(image, checkpoint):
    """Voxtral-Mini-3B released decoder, float32. `--checkpoint` is the release directory. Fixed random
    log-mel features (30 s, 128 mel bands) and a prompt with the right number of audio tokens are fed
    directly (parity is a function of the inputs). The record carries the tokens, the features, the
    encoder and projector seams, the prefill logits, and the greedy continuation. Requires the llm
    oracle (transformers 4.57.6 which carries Voxtral).
    """
    import torch
    from transformers import VoxtralForConditionalGeneration

    # `IK_VOXTRAL_DTYPE=bfloat16` builds the model at bf16 with eager attention; `bfloat16-inputs` keeps
    # float32 arithmetic on the same features rounded to bf16, the floor for that record.
    # `IK_PROBE_ENCODER=1` records the audio tower's layers piece by piece, `IK_PROBE_LAYERS` the named
    # decoder layers on the prefill.
    mode = os.environ.get("IK_VOXTRAL_DTYPE")
    dtype = torch.bfloat16 if mode == "bfloat16" else torch.float32
    model = VoxtralForConditionalGeneration.from_pretrained(
        checkpoint, dtype=dtype, attn_implementation="eager" if mode else None).eval()
    torch.manual_seed(3)
    features = torch.randn(1, 128, 3000)
    if mode:
        features = features.to(torch.bfloat16).to(dtype)
    probes = {}
    if os.environ.get("IK_PROBE_ENCODER"):
        _probe_encoder_layers(model.audio_tower.layers, probes)
    with torch.no_grad():
        encoder_out = model.audio_tower(features).last_hidden_state
        audio_embeds = model.get_audio_features(features)
    audioCount = audio_embeds.shape[0]
    audioId = model.config.audio_token_id
    ids = torch.tensor([[1] + [audioId] * audioCount + [100, 200]], dtype=torch.long)
    restore = None
    if os.environ.get("IK_PROBE_LAYERS"):
        decoder_probes = {}
        restore = _probe_hooks(model.language_model, [int(x) for x in os.environ["IK_PROBE_LAYERS"].split(",")],
                               decoder_probes)
    with torch.no_grad():
        logits = model(input_ids=ids, input_features=features).logits[0].float()
        if restore is not None:
            restore()
            probes.update({f"dec.{key}": value for key, value in decoder_probes.items()})
        generated = model.generate(input_ids=ids, input_features=features, max_new_tokens=8, do_sample=False)
    continuation = generated[0, ids.shape[1]:]
    globals()["_extra"] = {
        "tokens": ids[0].to(torch.int32).contiguous(),
        "features": features[0].float().contiguous(),
        "encoder_out": encoder_out[0].float().contiguous(),
        "audio_embeds": audio_embeds.float().contiguous(),
        "continuation": continuation.to(torch.int32).contiguous(),
        **probes,
    }
    return logits.contiguous()


def run_qwen2_moe(image, checkpoint):
    """The Qwen2-MoE decoder's arithmetic, from transformers' own Qwen2MoeForCausalLM, at a tiny
    random configuration.

    Qwen2-MoE (Qwen1.5-MoE-A2.7B, Qwen2-57B-A14B) is the routed feed-forward with a SHARED expert
    beside it: every token also runs one dense SwiGLU of `shared_expert_intermediate_size`, gated by
    `sigmoid(shared_expert_gate(x))`, and the two sums add. Its router leaves the selected weights
    UNnormalized by default (`norm_topk_prob` false), which this record keeps so it differs from the
    Qwen3-MoE record on that axis too, and its attention projections carry biases (`qkv_bias`). Every
    hidden state is recorded so a divergence is located to a layer. `checkpoint` is unused.
    """
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

    config = Qwen2MoeConfig(
        hidden_size=64, num_hidden_layers=3, vocab_size=128, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=96, moe_intermediate_size=32,
        shared_expert_intermediate_size=48, num_experts=8, num_experts_per_tok=2, norm_topk_prob=False,
        decoder_sparse_step=1, mlp_only_layers=[], rms_norm_eps=1e-6, rope_theta=10000.0,
        tie_word_embeddings=False, max_position_embeddings=64, use_sliding_window=False, qkv_bias=True)
    if os.environ.get("IK_TINY_DTYPE"):
        # This transformers picks the attention class at construction, so a bf16 record asks for eager
        # here rather than through `_tiny_decoder_record`.
        config._attn_implementation = "eager"
    model = _randomized(Qwen2MoeForCausalLM(config), seed=17)
    tokens = torch.tensor([[3, 17, 42, 99, 7, 61, 12, 5]], dtype=torch.long)
    return _tiny_decoder_record(model, tokens)


def run_gpt_oss(image, checkpoint):
    """The gpt-oss decoder's arithmetic, from transformers' own GptOssForCausalLM, at a tiny random
    configuration.

    gpt-oss differs from the other mixtures in four places, each exercised here: the layers ALTERNATE
    sliding-window and full attention (a window of 4 over 8 tokens, so the sliding layers see less
    than the full ones); every head carries a learned attention SINK, an extra softmax logit that
    drains mass without a value; the router takes the softmax over the SELECTED top-k logits, with a
    bias; and the experts run a clamped SwiGLU over an interleaved fused `gate_up_proj` with biases
    (`gate` the even columns, `up` the odd, `gate` clamped above at 7, `up` clamped to ±7,
    `(up + 1) · gate · sigmoid(1.702 · gate)`), with biases on `down_proj` too. The rotary is YaRN with
    `truncate: false`, the release's own, and the attention projections all carry biases, the output
    projection included. Eager attention is forced, since the fused kernels take no sink. Every hidden
    state is recorded so a divergence is located to a layer. `checkpoint` is unused.
    """
    from transformers import GptOssConfig, GptOssForCausalLM

    config = GptOssConfig(
        num_hidden_layers=4, num_local_experts=4, vocab_size=128, hidden_size=64, intermediate_size=32,
        head_dim=16, num_attention_heads=4, num_key_value_heads=2, sliding_window=4,
        rope_theta=150000.0, max_position_embeddings=131072, rms_norm_eps=1e-5, num_experts_per_tok=2,
        tie_word_embeddings=False)
    config._attn_implementation = "eager"
    model = _randomized(GptOssForCausalLM(config), seed=19)
    tokens = torch.tensor([[3, 17, 42, 99, 7, 61, 12, 5]], dtype=torch.long)
    return _tiny_decoder_record(model, tokens)


def run_gemma4(image, checkpoint):
    """Gemma 4 text-decoder logits, from transformers' own Gemma4 implementation.

    `--checkpoint` is the released model DIRECTORY. Gemma 4 is in no released transformers, so this
    runs under its own interpreter — see `oracle_environments` in the validation manifest.

    The record carries the token ids both sides read, the decoder's logits for the whole prompt, and
    the greedy continuation. The decoder is what this port implements; the release's vision and audio
    towers are not.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32).eval()

    prompt = "The capital of France is"
    ids = tokenizer(prompt, return_tensors="pt").input_ids

    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
        logits = out.logits[0]
        generated = model.generate(ids, max_new_tokens=12, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
    continuation = generated[0, ids.shape[1]:]

    # PER-LAYER ISOLATION. `hidden_states` is the embedding output followed by each layer's, so
    # `hidden.0` is what enters layer 0 and `hidden.N+1` is what layer N produced. Comparing them one
    # at a time says WHICH layer first diverges, which a whole-model cosine cannot.
    extra = {
        "tokens": ids[0].to(torch.int32).contiguous(),
        "continuation": continuation.to(torch.int32).contiguous(),
    }
    for index, state in enumerate(out.hidden_states):
        extra[f"hidden.{index}"] = state[0].float().contiguous()
    globals()["_extra"] = extra
    return logits.float().contiguous()



def _gemma3n_audio_tiny_config(**overrides):
    from transformers import Gemma3nAudioConfig
    settings = dict(
        hidden_size=32, conf_num_hidden_layers=2, conf_num_attention_heads=2, input_feat_size=16,
        sscp_conv_channel_size=[8, 4], conf_attention_chunk_size=4, conf_attention_context_left=3,
        conf_attention_context_right=1, conf_conv_kernel_size=3, gradient_clipping=10.0,
        conf_reduction_factor=2, conf_residual_weight=0.5, conf_attention_logit_cap=50.0,
        rms_norm_eps=1e-6, sscp_conv_group_norm_eps=1e-3, vocab_size=16)
    settings.update(overrides)
    return Gemma3nAudioConfig(**settings)


def run_gemma3n_audio(image):
    """The Gemma 3n audio encoder at a tiny random configuration, from transformers' own
    Gemma3nAudioEncoder.

    The released encoder reads NO future context (`conf_attention_context_right` is 0), so this
    configuration sets it to 1: the chunked attention's right reach, the relative-position shift over
    a span that is not purely causal, and the block mask's upper bound are all exercised here and
    nowhere else. The activation clamp is set low enough to bite, so the forward pass actually runs
    through it rather than past it. A padded tail marks the frame-validity path.

    Records the encoded sequence, the sub-sampled front end's output, the weights under the release
    naming, the mel, and the frame mask. `image` unused.
    """
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nAudioEncoder

    config = _gemma3n_audio_tiny_config()
    model = _randomized(Gemma3nAudioEncoder(config), seed=53)
    torch.manual_seed(7)
    frames = 26
    mel = torch.randn(1, frames, config.input_feat_size)
    # True marks a PADDED frame in this reference; the tail is padding.
    mask = torch.zeros(1, frames, dtype=torch.bool)
    mask[:, -5:] = True

    with torch.no_grad():
        subsampled = model.subsample_conv_projection(mel)
        out = model(audio_mel=mel, audio_mel_mask=mask)

    extra = {"mel": mel[0].float().contiguous(),
             "mel_mask": mask[0].to(torch.int32).contiguous(),
             "subsampled": subsampled[0].float().contiguous(),
             "out_mask": out.audio_mel_mask[0].to(torch.int32).contiguous()}
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return out.last_hidden_state[0].float().contiguous()


def run_gemma3n_mel(image, checkpoint):
    """Gemma 3n's audio front end, from transformers' own Gemma3nAudioFeatureExtractor.

    `--checkpoint` is the release directory, whose `preprocessor_config.json` states the geometry. A
    deterministic random waveform measures it exactly. `image` unused.
    """
    from transformers import AutoFeatureExtractor

    extractor = AutoFeatureExtractor.from_pretrained(checkpoint)
    print("frame", extractor.frame_length, "hop", extractor.hop_length, "fft", extractor.fft_length,
          "mels", extractor.feature_size)

    torch.manual_seed(23)
    waveform = torch.randn(16_000).numpy()
    features = extractor([waveform], sampling_rate=extractor.sampling_rate, return_tensors="pt")

    globals()["_extra"] = {"waveform": torch.tensor(waveform).float().contiguous(),
                           "mask": features["input_features_mask"][0].to(torch.int32).contiguous()}
    return features["input_features"][0].float().contiguous()


def run_gemma3n_audio_real(image, checkpoint):
    """The Gemma 3n audio encoder on the RELEASED weights, loaded on its own out of the tri-modal
    checkpoint (the decoder and the vision tower stay on disk).

    The encoder is a pure function of its mel, so a deterministic random mel measures it exactly; no
    audio file is needed. `--checkpoint` is the release directory. `image` unused.
    """
    import glob
    import os
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nAudioEncoder

    config = AutoConfig.from_pretrained(checkpoint).audio_config
    model = Gemma3nAudioEncoder(config).eval().float()

    prefix = "model.audio_tower."
    collected = {}
    for shard in sorted(glob.glob(os.path.join(checkpoint, "*.safetensors"))):
        for key, value in load_file(shard).items():
            if key.startswith(prefix):
                collected[key[len(prefix):]] = value.float()
    missing, unexpected = model.load_state_dict(collected, strict=False)
    print(f"audio tower: loaded {len(collected)}, missing {len(missing)}, unexpected {len(unexpected)}")
    if missing or unexpected:
        raise SystemExit(f"audio tower did not load strictly: missing {missing[:5]} unexpected {unexpected[:5]}")

    torch.manual_seed(11)
    frames = 300
    mel = torch.randn(1, frames, config.input_feat_size)
    mask = torch.zeros(1, frames, dtype=torch.bool)

    block = model.conformer[0]
    with torch.no_grad():
        subsampled = model.subsample_conv_projection(mel)
        sub_mask = torch.zeros(1, subsampled.shape[1], dtype=torch.bool)
        # The block's own seams, so a divergence lands on a branch rather than on "the block".
        after_ffw_start = block.ffw_layer_start(subsampled)
        after_attention = block.attention(after_ffw_start, sub_mask)
        after_lconv = block.lconv1d(after_attention * (~sub_mask).unsqueeze(-1).to(after_attention.dtype))
        after_ffw_end = block.ffw_layer_end(after_lconv)
        first = block(subsampled, sub_mask)
        out = model(audio_mel=mel, audio_mel_mask=mask)

    globals()["_extra"] = {"mel": mel[0].float().contiguous(),
                           "subsampled": subsampled[0].float().contiguous(),
                           "ffw_start": after_ffw_start[0].float().contiguous(),
                           "attention": after_attention[0].float().contiguous(),
                           "lconv": after_lconv[0].float().contiguous(),
                           "ffw_end": after_ffw_end[0].float().contiguous(),
                           "first_block": first[0].float().contiguous()}
    return out.last_hidden_state[0].float().contiguous()


def run_gemma3n(image, checkpoint):
    """Gemma 3n text-decoder logits on the RELEASED weights, from transformers' own Gemma3n.

    `--checkpoint` is a released model DIRECTORY (the tri-modal E2B / E4B releases; the decoder is
    loaded on its own, so the vision and audio towers stay on disk). Runs under the gemma oracle
    interpreter. `IK_GEMMA_DTYPE=bfloat16` runs both sides at the checkpoint's own precision, which is
    what the larger sizes need to fit.

    Records the token ids both sides read, the decoder's logits for the whole prompt, every hidden
    state for the per-layer isolation, and the greedy continuation. `image` unused.
    """
    import os
    import torch
    from transformers import AutoConfig, AutoTokenizer, Gemma3nForCausalLM

    dtype = torch.bfloat16 if os.environ.get("IK_GEMMA_DTYPE") == "bfloat16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    config = AutoConfig.from_pretrained(checkpoint)
    text_config = getattr(config, "text_config", config)
    # Eager attention at bf16: torch's CPU SDPA kernel approximates `exp` with its own polynomial, which
    # no port reproduces, and a bf16 record is what a port's rounding placement is held to.
    attention = "eager" if dtype == torch.bfloat16 else None
    model = Gemma3nForCausalLM.from_pretrained(checkpoint, config=text_config, dtype=dtype,
                                               attn_implementation=attention).eval()

    prompt = "The capital of France is"
    ids = tokenizer(prompt, return_tensors="pt").input_ids

    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
        logits = out.logits[0]
        generated = model.generate(ids, max_new_tokens=12, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
    continuation = generated[0, ids.shape[1]:]
    print("continuation:", tokenizer.decode(continuation))

    extra = {"tokens": ids[0].to(torch.int32).contiguous(),
             "continuation": continuation.to(torch.int32).contiguous()}
    for index, state in enumerate(out.hidden_states):
        extra[f"hidden.{index}"] = state[0].float().contiguous()
    globals()["_extra"] = extra
    return logits.float().contiguous()


def run_gemma3n_vision_real(image, checkpoint):
    """The Gemma 3n vision tower on the RELEASED weights: timm's own MobileNetV5-300M encoder, loaded
    out of the tri-modal checkpoint, plus the multimodal embedder that turns its grid into soft tokens.

    The tower is a pure function of its pixels, so a deterministic random frame in `0...1` measures it
    exactly and takes the release's own resize out of the comparison (that resize is CoreGraphics on
    the Swift side and PIL here, a documented approximation). Records the stem, each of the four
    stages, the fused grid, and the projected soft tokens. `image` unused.
    """
    import glob
    import os
    import timm
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nMultimodalEmbedder

    config = AutoConfig.from_pretrained(checkpoint)
    tower = timm.create_model("mobilenetv5_300m_enc", pretrained=False, num_classes=0).eval().float()
    embedder = Gemma3nMultimodalEmbedder(config.vision_config, config.text_config).eval().float()

    tower_prefix = "model.vision_tower.timm_model."
    embed_prefix = "model.embed_vision."
    tower_state, embed_state = {}, {}
    for shard in sorted(glob.glob(os.path.join(checkpoint, "*.safetensors"))):
        for key, value in load_file(shard).items():
            if key.startswith(tower_prefix):
                tower_state[key[len(tower_prefix):]] = value.float()
            elif key.startswith(embed_prefix):
                embed_state[key[len(embed_prefix):]] = value.float()
    missing, unexpected = tower.load_state_dict(tower_state, strict=False)
    print(f"vision tower: loaded {len(tower_state)}, missing {len(missing)}, unexpected {len(unexpected)}")
    if missing or unexpected:
        raise SystemExit(f"vision tower did not load strictly: missing {missing[:5]} unexpected {unexpected[:5]}")
    missing, unexpected = embedder.load_state_dict(embed_state, strict=False)
    print(f"embed_vision: loaded {len(embed_state)}, missing {len(missing)}, unexpected {len(unexpected)}")

    torch.manual_seed(19)
    size = config.vision_config.image_size if hasattr(config.vision_config, "image_size") else 768
    pixels = torch.rand(1, 3, size, size)

    stages = {}
    with torch.no_grad():
        x = tower.conv_stem(pixels)
        stages["stem"] = x.clone()
        captured = []
        for index, block in enumerate(tower.blocks):
            x = block(x)
            stages[f"stage{index}"] = x.clone()
            if (index + 1) in tower.msfa_indices:
                captured.append(x)
        fused = tower.msfa(captured)
        whole = tower.forward_features(pixels)
        tokens = fused.reshape(1, config.vision_config.hidden_size,
                               config.vision_soft_tokens_per_image).permute(0, 2, 1)
        tokens = tokens * (config.vision_config.hidden_size ** 0.5)
        projected = embedder(inputs_embeds=tokens)

    extra = {"pixels": pixels[0].permute(1, 2, 0).float().contiguous(),
             "fused": fused[0].permute(1, 2, 0).float().contiguous(),
             "whole": whole[0].permute(1, 2, 0).float().contiguous(),
             "projected": projected[0].float().contiguous()}
    for name, value in stages.items():
        extra[name] = value[0].permute(1, 2, 0).float().contiguous()
    for key, value in embedder.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return fused[0].permute(1, 2, 0).float().contiguous()


def run_gemma3n_conditional_real(image, checkpoint):
    """The WHOLE Gemma 3n on the released weights: the vision tower, the multimodal embedder, the
    two-stage splice into the prompt, and the decoder over the fused sequence.

    Runs the release's own processor so the prompt's token layout — the markers around the 256 image
    placeholders — is the release's rather than this port's guess, and records the processor's pixel
    values so both sides read identical pixels (the resize is PIL here and CoreGraphics on the Swift
    side, a documented approximation that would otherwise sit inside the comparison).

    Records the token ids, the pixel values, the fused input embeddings, the per-position argmax, the
    logits of the last positions (the whole matrix is a quarter of a gigabyte), and the greedy
    continuation.
    """
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor, Gemma3nForConditionalGeneration

    processor = AutoProcessor.from_pretrained(checkpoint)
    model = Gemma3nForConditionalGeneration.from_pretrained(checkpoint, dtype=torch.float32).eval()

    pil = Image.fromarray((image * 255).astype(np.uint8))
    # The ungated mirror ships no chat template, so the turn is written out with the release's own
    # markers and the processor expands the image placeholder into its soft-token run.
    prompt = (f"<start_of_turn>user\n{processor.image_token}Describe this image."
              f"<end_of_turn>\n<start_of_turn>model\n")
    inputs = processor(text=[prompt], images=[pil], return_tensors="pt")
    ids = inputs["input_ids"]
    pixels = inputs["pixel_values"]
    print("prompt tokens:", ids.shape, "image placeholders:",
          int((ids == model.config.image_token_id).sum()))

    with torch.no_grad():
        fused = model.model.get_input_embeddings()(ids)
        out = model(input_ids=ids, pixel_values=pixels)
        logits = out.logits[0]
        generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    continuation = generated[0, ids.shape[1]:]
    print("continuation:", processor.decode(continuation))

    globals()["_extra"] = {
        "tokens": ids[0].to(torch.int32).contiguous(),
        "pixels": pixels[0].permute(1, 2, 0).float().contiguous(),
        "text_embeddings": fused[0].float().contiguous(),
        "argmax": logits.argmax(-1).to(torch.int32).contiguous(),
        "last_logits": logits[-16:].float().clone().contiguous(),
        "continuation": continuation.to(torch.int32).contiguous(),
    }
    return logits[-1].float().clone().contiguous()   # `last_logits` aliases it otherwise


def run_gemma3(image, checkpoint):
    """Gemma 3 text-decoder logits, from transformers' own Gemma3 implementation.

    `--checkpoint` is a released model DIRECTORY: a text-only size (`gemma3_text`: 270M, 1B) loads as
    `Gemma3ForCausalLM`, a multimodal one (`gemma3`: 4B and up) as `Gemma3ForConditionalGeneration`
    driven text-only. Runs under the gemma oracle interpreter.

    The record carries the token ids both sides read, the decoder's logits for the whole prompt, the
    greedy continuation (which the reference decodes through its hybrid cache, so the Swift cache is
    measured by it), every hidden state for the per-layer isolation harness, the chat-templated ids of
    a one-turn conversation, and the ids of a few tokenizer probe strings.

    `IK_GEMMA_DTYPE=bfloat16` builds the reference at the released precision, eager attention, which
    is the record a bf16 port is held to; the float32 record from the same prompt is its floor.
    """
    import json
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    dtype = torch.bfloat16 if os.environ.get("IK_GEMMA_DTYPE") == "bfloat16" else torch.float32
    config = json.load(open(os.path.join(checkpoint, "config.json")))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if config.get("model_type") == "gemma3":
        model = AutoModelForImageTextToText.from_pretrained(checkpoint, dtype=dtype,
                                                            attn_implementation="eager").eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=dtype,
                                                     attn_implementation="eager").eval()

    prompt = "The capital of France is"
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
        logits = out.logits[0]
        generated = model.generate(input_ids=ids, max_new_tokens=12, do_sample=False,
                                   pad_token_id=tokenizer.pad_token_id)
    continuation = generated[0, ids.shape[1]:]

    # The rendered template already spells `<bos>`, so it is tokenized without a second one.
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Describe the sky in one sentence."}], add_generation_prompt=True,
        tokenize=False)
    extra = {
        "tokens": ids[0].to(torch.int32).contiguous(),
        "continuation": continuation.to(torch.int32).contiguous(),
        "chat_tokens": torch.tensor(tokenizer(chat, add_special_tokens=False).input_ids, dtype=torch.int32),
    }
    for index, probe in enumerate(_GEMMA3_TOKENIZER_PROBES):
        extra[f"probe.{index}"] = torch.tensor(tokenizer(probe, add_special_tokens=False).input_ids, dtype=torch.int32)
    for index, state in enumerate(out.hidden_states):
        extra[f"hidden.{index}"] = state[0].float().contiguous()
    globals()["_extra"] = extra
    return logits.float().contiguous()


def _probe_hooks(model, probed, extra):
    """Hooks every submodule of the decoder layers in `probed`, the eager attention function, and the
    module-level functions `IK_PROBE_FUNCTIONS` names, recording into `extra` under the keys
    `hf_layer_probe` documents. Returns a function that undoes the attention patch."""
    import importlib
    import torch

    decoder_layers = next(m for n, m in model.named_modules()
                          if isinstance(m, torch.nn.ModuleList) and n.endswith("layers")
                          and "vision" not in n and "audio" not in n and "embed_tokens_extend" not in n)

    def keep(value):
        return value.detach().float().clone().contiguous()

    for index in probed:
        layer = decoder_layers[index]
        for name, module in layer.named_modules():
            label = f"{index}.{name}" if name else f"{index}.block"
            def hook(module, inputs, output, label=label, whole=not name):
                if inputs and torch.is_tensor(inputs[0]):
                    extra[f"{label}.in"] = keep(inputs[0][0])
                    if whole:
                        extra[f"{label}.in.whole"] = keep(inputs[0])
                if whole and torch.is_tensor(output if not isinstance(output, tuple) else output[0]):
                    extra[f"{label}.out.whole"] = keep(output if not isinstance(output, tuple) else output[0])
                out = output[0] if isinstance(output, tuple) else output
                if torch.is_tensor(out):
                    extra[f"{label}.out"] = keep(out[0])
            module.register_forward_hook(hook)

    modeling = importlib.import_module(type(decoder_layers[0]).__module__)
    original = getattr(modeling, "eager_attention_forward", None)
    layer_of = {id(decoder_layers[i].self_attn): i for i in probed if hasattr(decoder_layers[i], "self_attn")}

    def recording(module, query, key, value, attention_mask, **kwargs):
        output, weights = original(module, query, key, value, attention_mask, **kwargs)
        index = layer_of.get(id(module))
        if index is not None:
            extra[f"{index}.attn.q"] = keep(query[0])
            extra[f"{index}.attn.k"] = keep(key[0])
            extra[f"{index}.attn.v"] = keep(value[0])
            extra[f"{index}.attn.weights"] = keep(weights[0])
            extra[f"{index}.attn.out"] = keep(output[0])
        return output, weights

    if original is not None:
        modeling.eager_attention_forward = recording
    # `IK_PROBE_FUNCTIONS` names module-level functions (a recurrence, a convolution) that no hook
    # sees; each call's tensor arguments and outputs are kept as `fn.<name>.<call>.arg<i>` / `.out<i>`.
    for function in filter(None, os.environ.get("IK_PROBE_FUNCTIONS", "").split(",")):
        def wrap(inner, name=function, calls=[0]):
            def recorded(*args, **kwargs):
                result = inner(*args, **kwargs)
                call = calls[0]
                calls[0] += 1
                for i, value in enumerate(list(args) + [kwargs[k] for k in sorted(kwargs)]):
                    if torch.is_tensor(value):
                        extra[f"fn.{name}.{call}.arg{i}"] = keep(value)
                outputs = result if isinstance(result, tuple) else (result,)
                for i, value in enumerate(outputs):
                    if torch.is_tensor(value):
                        extra[f"fn.{name}.{call}.out{i}"] = keep(value)
                return result
            return recorded
        setattr(modeling, function, wrap(getattr(modeling, function)))

    def restore():
        if original is not None:
            modeling.eager_attention_forward = original
    return restore


def _stream_loaded(loader, checkpoint):
    """`checkpoint` loaded at bf16 with every module streamed to float32 (`_stream_float32`). The
    buffers the constructor computes (an embedding scale, a rotary table) are made at bf16 by a bf16
    load, and widening cannot undo that rounding, so they are taken from a float32 construction of the
    same model whose parameters live on the meta device and cost no memory."""
    import copy
    import torch

    model = loader.from_pretrained(checkpoint, dtype=torch.bfloat16, attn_implementation="eager").eval()
    persistent = set(model.state_dict())
    register = torch.nn.Module.register_parameter

    def register_on_meta(module, name, parameter):
        register(module, name, parameter)
        if parameter is not None:
            module._parameters[name] = torch.nn.Parameter(parameter.to("meta"), requires_grad=False)

    # The loaded config records bf16 on itself and its sub-configs, and a multimodal wrapper builds
    # its language model at that type, so the float32 construction reads a copy set to float32.
    config = copy.deepcopy(model.config)
    pending = [config]
    while pending:
        current = pending.pop()
        for key in ("dtype", "torch_dtype"):
            if getattr(current, key, None) is not None:
                setattr(current, key, torch.float32)
        pending.extend(value for value in vars(current).values() if hasattr(value, "to_dict") and hasattr(value, "model_type"))
    torch.nn.Module.register_parameter = register_on_meta
    set_default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        shell = type(model)(config)
    finally:
        torch.nn.Module.register_parameter = register
        torch.set_default_dtype(set_default)
    for name, buffer in shell.named_buffers():
        if name in persistent or not buffer.is_floating_point() or buffer.device.type == "meta":
            continue
        owner, _, leaf = name.rpartition(".")
        target = model.get_submodule(owner) if owner else model
        if leaf in target._buffers:
            target._buffers[leaf] = buffer.to(torch.float32)
    del shell
    _stream_float32(model)
    return model


def _stream_float32(model):
    """Hooks every module so its own floating parameters and buffers are float32 while it runs and
    return to their stored types after. A tensor shared by two modules (a tied head) is widened by
    whichever runs, and restored when that module ends."""
    import torch

    def tensors(module):
        return [t for t in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False))
                if t.is_floating_point()]

    def widen(module, args):
        stored = []
        for t in tensors(module):
            stored.append((t, t.data.dtype))
            if t.data.dtype != torch.float32:
                t.data = t.data.float()
        module._ik_stored = stored

    def restore(module, args, output):
        for t, original in getattr(module, "_ik_stored", []):
            if t.data.dtype != original:
                t.data = t.data.to(original)
        module._ik_stored = []

    for module in model.modules():
        if tensors(module):
            module.register_forward_pre_hook(widen)
            module.register_forward_hook(restore)


def run_hf_layer_probe(image, checkpoint):
    """Every submodule's input and output inside chosen decoder layers of a transformers release.

    `--checkpoint` is a released DIRECTORY that loads as a causal LM (a multimodal Gemma 3 release
    loads as `AutoModelForImageTextToText` and is driven text-only). `IK_PROBE_LAYERS` names the
    layers (default `0`), `IK_PROBE_DTYPE=bfloat16` builds the reference at that precision, eager
    attention. Records `hidden.i` for every hidden state, then per probed layer `L`:
    `L.<module>.in` / `L.<module>.out` for every submodule (cloned at capture, since a module can
    mutate its output in place later), the block's whole input and output as `L.block.in.whole` /
    `L.block.out.whole` (Gemma 3n's residual stream carries AltUp copies ahead of the batch axis), and from the eager attention function the post-rotary
    `L.attn.q` / `L.attn.k`, the `L.attn.v`, the rounded probabilities `L.attn.weights`, and the
    weighted sum `L.attn.out`. Each seam's input is the reference's own, so a port runs a piece on it
    and counts the elements that differ. `image` unused.
    """
    import json
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    dtype = torch.bfloat16 if os.environ.get("IK_PROBE_DTYPE") == "bfloat16" else torch.float32
    # `IK_PROBE_STREAM_F32=1` computes the float32 run from a bf16 load: each module's own weights are
    # widened to float32 just before it runs and returned to their stored type after, so only one
    # module is float32 at a time. A bf16 weight widens exactly, so the arithmetic is the float32
    # model's; the peak is the bf16 model plus the largest single module.
    stream = os.environ.get("IK_PROBE_STREAM_F32") == "1" and dtype == torch.float32
    probed = [int(x) for x in os.environ.get("IK_PROBE_LAYERS", "0").split(",")]
    config = json.load(open(os.path.join(checkpoint, "config.json")))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # A multimodal Gemma 3 release only loads whole; Gemma 3n and 4 load their text decoder alone.
    loader = AutoModelForImageTextToText if config.get("model_type") == "gemma3" else AutoModelForCausalLM
    model = _stream_loaded(loader, checkpoint) if stream else loader.from_pretrained(
        checkpoint, dtype=dtype, attn_implementation="eager").eval()

    extra = {}
    restore = _probe_hooks(model, probed, extra)
    prompt = "The capital of France is"
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    restore()
    extra["tokens"] = ids[0].to(torch.int32).contiguous()
    for index, state in enumerate(out.hidden_states):
        extra[f"hidden.{index}"] = state[0].detach().float().clone().contiguous()
    globals()["_extra"] = extra
    return out.logits[0].float().contiguous()


# Strings the Swift tokenizer is held to the reference on: runs of spaces, multi-byte text, newlines
# around a marker (the double newline merges with a neighbouring one), and a chat turn with its markers.
_GEMMA3_TOKENIZER_PROBES = [
    "The capital of France is", "  two  spaces ", "na\u00efve caf\u00e9 \u2615 \u4f60\u597d", "a\nb\n\nc",
    "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n",
    "user\n\n\n<start_of_image><image_soft_token><image_soft_token><end_of_image>\n\nDescribe this.",
]


def _gemma3_tiny_config(**overrides):
    from transformers import Gemma3TextConfig
    settings = dict(
        hidden_size=64, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=96, vocab_size=131, query_pre_attn_scalar=16, sliding_window=4,
        sliding_window_pattern=3, rope_theta=1000000.0, rope_local_base_freq=10000.0,
        rope_scaling={"rope_type": "linear", "factor": 8.0}, attn_logit_softcapping=50.0,
        final_logit_softcapping=30.0, max_position_embeddings=64)
    settings.update(overrides)
    return Gemma3TextConfig(**settings)


def run_gemma3_tiny(image):
    """The Gemma 3 decoder's arithmetic at a tiny random configuration, from transformers' own
    Gemma3ForCausalLM: a 4-position sliding window over a sliding/sliding/full pattern with a
    12-token prompt (so the sliding layers see less than the full one), a linear rotary scaling on
    the full layer, an attention soft-cap, and a final logit soft-cap. Records the logits, every
    hidden state, the weights under the release naming, the greedy continuation the reference decodes
    through its cache, and the teacher-forced logits over prompt + continuation (what a cached
    step-by-step decode must reproduce). `image` unused.
    """
    from transformers import Gemma3ForCausalLM

    config = _gemma3_tiny_config()
    print("layer_types:", config.layer_types, "rope:", config.rope_parameters)
    model = _randomized(Gemma3ForCausalLM(config), seed=41)
    tokens = torch.tensor([[3, 17, 42, 99, 7, 61, 12, 5, 88, 30, 44, 120]], dtype=torch.long)
    with torch.no_grad():
        out = model(tokens, output_hidden_states=True)
        generated = model.generate(tokens, max_new_tokens=6, do_sample=False, pad_token_id=0)
        full = model(generated).logits[0]
    continuation = generated[0, tokens.shape[1]:]

    extra = {"tokens": tokens[0].to(torch.int32).contiguous(),
             "continuation": continuation.to(torch.int32).contiguous(),
             "full_logits": full.float().contiguous()}
    for index, state in enumerate(out.hidden_states):
        extra[f"hidden.{index}"] = state[0].float().contiguous()
    for key, value in model.state_dict().items():
        if key == "lm_head.weight":                                 # tied to the embedding; safetensors refuses the alias
            continue
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return out.logits[0].float().contiguous()


def run_gemma3_bidirectional_tiny(image):
    """The Gemma 3 encoder read bidirectionally (`use_bidirectional_attention`, EmbeddingGemma's
    backbone) at a tiny random configuration with a window the 12-token sequence exceeds, from
    transformers' own Gemma3TextModel. The release states the window as its full span and the
    reference turns it into the exclusive bound `span // 2 + 1` on `|q - k|` — which a short input
    never reaches, so this is where that rule is measured. Records the last hidden state, every hidden
    state, and the weights. `image` unused.
    """
    from transformers import Gemma3TextModel

    config = _gemma3_tiny_config(use_bidirectional_attention=True, sliding_window=6,
                                 rope_scaling=None, attn_logit_softcapping=None,
                                 final_logit_softcapping=None)
    print("effective sliding_window:", config.sliding_window)
    model = _randomized(Gemma3TextModel(config), seed=43)
    tokens = torch.tensor([[3, 17, 42, 99, 7, 61, 12, 5, 88, 30, 44, 120]], dtype=torch.long)
    with torch.no_grad():
        out = model(tokens, output_hidden_states=True)

    extra = {"tokens": tokens[0].to(torch.int32).contiguous()}
    for index, state in enumerate(out.hidden_states):
        extra[f"hidden.{index}"] = state[0].float().contiguous()
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return out.last_hidden_state[0].float().clone().contiguous()   # `hidden.N` aliases it otherwise


def _gemma3n_tiny_config(**overrides):
    from transformers import Gemma3nTextConfig
    layers = 6
    settings = dict(
        hidden_size=64, num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, intermediate_size=[96] * layers, vocab_size=140,
        vocab_size_per_layer_input=128, hidden_size_per_layer_input=8,
        altup_num_inputs=4, altup_active_idx=0, altup_correct_scale=True, laurel_rank=8,
        num_kv_shared_layers=2, sliding_window=4,
        layer_types=["sliding_attention", "sliding_attention", "full_attention",
                     "sliding_attention", "sliding_attention", "full_attention"],
        activation_sparsity_pattern=[0.95, 0.95, 0.0, 0.0, 0.0, 0.0],
        rope_theta=1000000.0, rope_local_base_freq=10000.0,
        final_logit_softcapping=30.0, max_position_embeddings=64)
    settings.update(overrides)
    return Gemma3nTextConfig(**settings)


def run_gemma3n_tiny(image):
    """The Gemma 3n decoder's arithmetic at a tiny random configuration, from transformers' own
    Gemma3nForCausalLM.

    Gemma 3n is a distinct architecture rather than a Gemma 3 variant, and this exercises every part
    of it that a shape cannot show: AltUp's four parallel residual copies with their per-token
    prediction and correction maps, the LAuReL low-rank detour, the per-layer input embeddings gated
    into the inactive copies, the Gaussian activation sparsity on the first two layers' feed-forward
    gates (the last four are dense, so both paths are covered), and the trailing two layers that
    compute no keys or values and reuse an earlier layer's — one sliding, one full, so the rule that
    a layer reuses the last non-shared layer OF ITS OWN KIND is measured rather than assumed. The
    12-token prompt exceeds the 4-position window, so the sliding layers see less than the full ones.

    Records the logits, every hidden state (so a divergence is located to a layer), the weights under
    the release naming, the greedy continuation the reference decodes through its cache, and the
    teacher-forced logits over prompt + continuation, which a cached step-by-step decode must
    reproduce. `image` unused.
    """
    from transformers import Gemma3nForCausalLM

    config = _gemma3n_tiny_config()
    print("layer_types:", config.layer_types, "| kv shared:", config.num_kv_shared_layers,
          "| sparsity:", config.activation_sparsity_pattern)
    model = _randomized(Gemma3nForCausalLM(config), seed=47)
    tokens = torch.tensor([[3, 17, 42, 99, 7, 61, 12, 5, 88, 30, 44, 120]], dtype=torch.long)
    with torch.no_grad():
        out = model(tokens, output_hidden_states=True)
        generated = model.generate(tokens, max_new_tokens=6, do_sample=False, pad_token_id=0)
        full = model(generated).logits[0]
    continuation = generated[0, tokens.shape[1]:]

    extra = {"tokens": tokens[0].to(torch.int32).contiguous(),
             "continuation": continuation.to(torch.int32).contiguous(),
             "full_logits": full.float().contiguous()}
    for index, state in enumerate(out.hidden_states):
        extra[f"hidden.{index}"] = state[0].float().contiguous()
    for key, value in model.state_dict().items():
        if key == "lm_head.weight":                                 # tied to the embedding; safetensors refuses the alias
            continue
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return out.logits[0].float().contiguous()


def _gemma3_pixel_values(image, checkpoint):
    """The plate through the release's own image processor: `[1, 3, 896, 896]` in -1...1."""
    from PIL import Image
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(checkpoint)
    pil = Image.fromarray((image * 255).astype(np.uint8))
    return processor(images=[pil], return_tensors="pt")["pixel_values"]


def run_gemma3_vision_real(image, checkpoint):
    """The Gemma 3 vision path on the RELEASED weights: the SigLIP tower and the multimodal projector,
    loaded selectively from the multimodal checkpoint (the decoder's 15 GB stays on disk). The plate
    goes through the release's own image processor, so the record carries the reference's pixel
    values and the Swift side measures the network rather than the resize. Records the patch
    embeddings, encoder layer 0, the tower's last hidden state, and the 256 projected soft tokens.
    `--checkpoint` is the release directory.
    """
    import json
    from safetensors import safe_open
    from transformers import Gemma3Config, SiglipVisionConfig, SiglipVisionModel
    from transformers.models.gemma3.modeling_gemma3 import Gemma3MultiModalProjector

    config = json.load(open(os.path.join(checkpoint, "config.json")))
    vision = SiglipVisionModel(SiglipVisionConfig(**config["vision_config"])).eval()
    projector = Gemma3MultiModalProjector(Gemma3Config(**config)).eval()

    vision_state, projector_state = {}, {}
    index = json.load(open(os.path.join(checkpoint, "model.safetensors.index.json")))["weight_map"]
    for shard in sorted(set(index.values())):
        with safe_open(os.path.join(checkpoint, shard), framework="pt") as handle:
            for key in handle.keys():
                # The released file is in the transformers 4.x layout (`vision_tower.vision_model.`);
                # a 5.x-written one nests the same under `model.`.
                name = key[len("model."):] if key.startswith("model.") else key
                if name.startswith("vision_tower.vision_model."):
                    vision_state[name[len("vision_tower.vision_model."):]] = handle.get_tensor(key).float()
                elif name.startswith("multi_modal_projector."):
                    projector_state[name[len("multi_modal_projector."):]] = handle.get_tensor(key).float()
    vision.load_state_dict(vision_state, strict=True)
    projector.load_state_dict(projector_state, strict=True)

    pixel_values = _gemma3_pixel_values(image, checkpoint)
    # transformers 5 flattened SiglipVisionModel (no `vision_model` child); 4.x keeps it.
    tower = getattr(vision, "vision_model", vision)
    with torch.no_grad():
        embeddings = tower.embeddings(pixel_values)
        layer0 = tower.encoder.layers[0](embeddings, attention_mask=None)
        layer0 = layer0[0] if isinstance(layer0, tuple) else layer0
        hidden = vision(pixel_values=pixel_values).last_hidden_state
        projected = projector(hidden)

    globals()["_extra"] = {
        "pixel_values": pixel_values[0].float().contiguous(),
        "vision_embeddings": embeddings[0].float().contiguous(),
        "vision_layer0": layer0[0].float().contiguous(),
        "vision_hidden": hidden[0].float().contiguous(),
    }
    return projected[0].float().contiguous()                        # [256, text hidden]


def run_gemma3_conditional_real(image, checkpoint):
    """The FULL Gemma3ForConditionalGeneration on the released multimodal weights: the plate and a
    question through the processor (the chat template with an image item, expanded to the
    `\n\n<start_of_image>` + 256 soft tokens + `<end_of_image>\n\n` sequence), the vision tower, the
    projector, the splice, and the decoder end to end — with the bidirectional attention among the
    image tokens the processor's `token_type_ids` turn on. `--checkpoint` is the release directory.

    Records the input ids, the token types, the reference's pixel values, the argmax at every
    position, the logits at the last 16 positions (the whole matrix is 290 MB), every hidden state,
    and the greedy continuation.
    """
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(checkpoint)
    model = AutoModelForImageTextToText.from_pretrained(checkpoint, dtype=torch.float32).eval()
    pil = Image.fromarray((image * 255).astype(np.uint8))
    messages = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": "Describe this image in one sentence."}]}]
    # The rendered template already spells `<bos>`; the processor expands `<start_of_image>` into the
    # full image sequence and marks the soft tokens in `token_type_ids`.
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, images=[pil], add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        out = model(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                    token_type_ids=inputs["token_type_ids"], output_hidden_states=True)
        logits = out.logits[0]
        generated = model.generate(**inputs, max_new_tokens=12, do_sample=False)
    continuation = generated[0, inputs["input_ids"].shape[1]:]

    extra = {
        "tokens": inputs["input_ids"][0].to(torch.int32).contiguous(),
        "token_types": inputs["token_type_ids"][0].to(torch.int32).contiguous(),
        "pixel_values": inputs["pixel_values"][0].float().contiguous(),
        "argmax": logits.argmax(dim=-1).to(torch.int32).contiguous(),
        "continuation": continuation.to(torch.int32).contiguous(),
    }
    for index, state in enumerate(out.hidden_states):
        extra[f"hidden.{index}"] = state[0].float().contiguous()
    globals()["_extra"] = extra
    return logits[-16:].float().contiguous()


def run_gemma4_moe(image, checkpoint):
    """The Gemma 4 mixture-of-experts block (the 26B-A4B family) at a tiny random configuration,
    from transformers' own Gemma4ForCausalLM.

    The released mixture does not fit this machine, so the routed branch — the scale-free router norm,
    the learned per-channel and per-expert scales, the softmax and top-k, and the fused-projection
    experts, all summed BESIDE the dense feed-forward — is measured at a size that does, over the full
    Gemma 4 block (per-layer input embeddings, the sandwich norms, dual rotary). Every hidden state is
    recorded so a divergence is located to a layer. The per-layer input vocabulary is shrunk to the
    token vocabulary here (`vocab_size_per_layer_input`) so the record stays small; the released model
    uses the 262144 default. `checkpoint` is unused.
    """
    from transformers import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    config = Gemma4TextConfig(
        hidden_size=64, num_hidden_layers=2, vocab_size=131, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=96,
        hidden_size_per_layer_input=32, vocab_size_per_layer_input=131,
        num_kv_shared_layers=0, sliding_window=64,
        layer_types=["sliding_attention", "full_attention"], rms_norm_eps=1e-6,
        max_position_embeddings=64, tie_word_embeddings=True,
        enable_moe_block=True, num_experts=6, top_k_experts=2, moe_intermediate_size=48,
        hidden_activation="gelu_pytorch_tanh", final_logit_softcapping=None)
    model = _randomized(Gemma4ForCausalLM(config), seed=17)
    tokens = torch.tensor([[3, 17, 42, 99, 7, 61, 12, 5]], dtype=torch.long)
    with torch.no_grad():
        out = model(tokens, output_hidden_states=True)
    extra = {"tokens": tokens[0].to(torch.int32).contiguous()}
    for index, hidden in enumerate(out.hidden_states):
        extra[f"hidden.{index}"] = hidden[0].float().contiguous()
    # `lm_head.weight` is tied to `embed_tokens.weight` (shared storage), which safetensors refuses to
    # save twice; the Swift net ties, so the head is dropped and reconstructed from the embedding.
    for key, value in model.state_dict().items():
        if key == "lm_head.weight":
            continue
        extra[f"w::{key}"] = (value.float() if value.is_floating_point() else value).contiguous()
    globals()["_extra"] = extra
    return out.logits[0].float().contiguous()


def run_gguf_lm(image, checkpoint):
    """End-to-end GGUF language-model logits, from transformers loading the GGUF directly.

    `--checkpoint` is the GGUF FILE. transformers dequantizes the GGUF and runs its own llama/qwen
    decoder, which is the reference the MLX GGUF loader is held to end to end: the tokenizer's ids
    (recorded so the Swift side feeds the identical sequence and its own tokenizer is checked
    separately), the logits over the prompt, and the greedy continuation. The dequantization is the
    canonical one, so a quantized model still has a single correct answer here.
    """
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    directory = os.path.dirname(checkpoint)
    filename = os.path.basename(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(directory, gguf_file=filename)
    model = AutoModelForCausalLM.from_pretrained(directory, gguf_file=filename, dtype=torch.float32).eval()

    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids)
        generated = model.generate(ids, max_new_tokens=10, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
    globals()["_extra"] = {
        "tokens": ids[0].to(torch.int32).contiguous(),
        "continuation": generated[0, ids.shape[1]:].to(torch.int32).contiguous(),
    }
    return out.logits[0].float().contiguous()


def run_gemma4_unified(image, checkpoint):
    """The Gemma 4 unified text decoder (the 12B `gemma4_unified_text`) at a tiny random configuration,
    from transformers' own Gemma4UnifiedForCausalLM.

    A DIFFERENT decoder from the E-series `gemma4_text`: no per-layer input embeddings and no mixture,
    but a scale-free VALUE norm beside the query and key norms, per-layer head widths (a full layer runs
    a 512-wide head where a sliding one runs its own), attention scaling of 1.0, and the sandwich norms
    with a per-layer scalar. Every hidden state is recorded for the isolation harness; the weights are
    saved in the release naming (`model.layers.N.self_attn.v_proj.weight`), and `lm_head` is dropped as
    tied. `checkpoint` is unused.
    """
    import torch
    from transformers import Gemma4UnifiedTextConfig
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedForCausalLM

    config = Gemma4UnifiedTextConfig(
        hidden_size=64, num_hidden_layers=3, vocab_size=131, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=96, sliding_window=64,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"],
        rms_norm_eps=1e-6, max_position_embeddings=64, tie_word_embeddings=True,
        hidden_activation="gelu_pytorch_tanh", final_logit_softcapping=None)
    model = _randomized(Gemma4UnifiedForCausalLM(config), seed=19)
    tokens = torch.tensor([[3, 17, 42, 99, 7, 61, 12, 5]], dtype=torch.long)
    with torch.no_grad():
        out = model(tokens, output_hidden_states=True)
    extra = {"tokens": tokens[0].to(torch.int32).contiguous()}
    for index, hidden in enumerate(out.hidden_states):
        extra[f"hidden.{index}"] = hidden[0].float().contiguous()
    for key, value in model.state_dict().items():
        if key == "lm_head.weight":
            continue
        extra[f"w::{key}"] = (value.float() if value.is_floating_point() else value).contiguous()
    globals()["_extra"] = extra
    return out.logits[0].float().contiguous()


def run_gemma4_vision(image, checkpoint):
    """The Gemma 4 vision encoder at a tiny random configuration, from transformers' own
    Gemma4VisionModel (patch embedder + bidirectional sandwich encoder, before the pooler).

    The vision tower embeds flattened patches through one linear projection, adds a learned 2-D
    position embedding (an x-table and a y-table summed per patch), and runs the same sandwich block
    the text decoder does but bidirectional and with NO rotary — the positions are the learned
    embeddings. `pixel_values` are the flattened patches `[batch, patches, 3·patch²]`, `pixel_position_ids`
    their `(x, y)` grid. The record carries both inputs, the encoder's last hidden state, and the
    weights in the release naming (the projections live under `.linear`, a `Gemma4ClippableLinear`).
    `checkpoint` is unused.
    """
    import torch
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

    config = Gemma4VisionConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        head_dim=8, intermediate_size=48, patch_size=4, position_embedding_size=16,
        pooling_kernel_size=2, rms_norm_eps=1e-6, hidden_activation="gelu_pytorch_tanh",
        use_clipped_linears=False, standardize=True)
    model = _randomized(Gemma4VisionModel(config), seed=23)

    torch.manual_seed(3)
    side = 4
    pixel_values = torch.randn(1, side * side, 3 * config.patch_size ** 2)
    grid = [[x, y] for y in range(side) for x in range(side)]
    position_ids = torch.tensor(grid, dtype=torch.long).unsqueeze(0)
    padding = (position_ids == -1).all(dim=-1)
    with torch.no_grad():
        embeds = model.patch_embedder(pixel_values, position_ids, padding)
        encoded = model.encoder(inputs_embeds=embeds, attention_mask=~padding,
                                pixel_position_ids=position_ids)
        # The full tower: encoder, then the position-based average pooler and the sqrt(hidden) scaling.
        pooled = model(pixel_values, pixel_position_ids=position_ids).last_hidden_state
    extra = {
        "pixel_values": pixel_values[0].float().contiguous(),
        "position_ids": position_ids[0].to(torch.int32).contiguous(),
        "pooled": pooled.float().contiguous(),
    }
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = (value.float() if value.is_floating_point() else value).contiguous()
    globals()["_extra"] = extra
    return encoded.last_hidden_state[0].float().contiguous()


def run_gemma4_vision_real(image, checkpoint):
    """The Gemma 4 vision path on the RELEASED weights: the vision tower and its multimodal embedder,
    loaded selectively from the tri-modal `Gemma4ForConditionalGeneration` checkpoint (the E2B release
    carries a vision tower, an audio tower, and the decoder together).

    `--checkpoint` is the release directory. Random pixel values over a real patch grid are fed to both
    sides, so the tower and embedder are compared on real weights without the resize entering. Records
    the pixel values, the position grid, the pooled soft tokens, and the embedder's projection.
    """
    import json
    import os
    import torch
    from safetensors import safe_open
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel, Gemma4MultimodalEmbedder
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig, Gemma4TextConfig

    config = json.load(open(os.path.join(checkpoint, "config.json")))
    vision_config = Gemma4VisionConfig(**config["vision_config"])
    text_config = Gemma4TextConfig(**config["text_config"])
    vision = Gemma4VisionModel(vision_config).eval()
    embedder = Gemma4MultimodalEmbedder(vision_config, text_config).eval()

    vision_state, embed_state = {}, {}
    with safe_open(os.path.join(checkpoint, "model.safetensors"), framework="pt") as handle:
        for key in handle.keys():
            if key.startswith("model.vision_tower."):
                vision_state[key[len("model.vision_tower."):]] = handle.get_tensor(key).float()
            elif key.startswith("model.embed_vision."):
                embed_state[key[len("model.embed_vision."):]] = handle.get_tensor(key).float()
    vision.load_state_dict(vision_state, strict=True)
    embedder.load_state_dict(embed_state, strict=True)

    side = 6                                                     # 36 patches, 4 soft tokens at pooling 3
    torch.manual_seed(3)
    pixel_values = torch.randn(1, side * side, 3 * vision_config.patch_size ** 2)
    grid = [[x, y] for y in range(side) for x in range(side)]
    position_ids = torch.tensor(grid, dtype=torch.long).unsqueeze(0)
    padding = (position_ids == -1).all(dim=-1)
    seams = {}
    vision.encoder.layers[0].register_forward_hook(
        lambda m, i, o: seams.__setitem__("layer0", (o[0] if isinstance(o, tuple) else o).detach()))
    vision.encoder.layers[0].self_attn.register_forward_hook(
        lambda m, i, o: seams.__setitem__("attn0", (o[0] if isinstance(o, tuple) else o).detach()))
    vision.encoder.layers[0].mlp.register_forward_hook(
        lambda m, i, o: seams.__setitem__("mlp0", o.detach()))
    vision.encoder.layers[0].input_layernorm.register_forward_hook(
        lambda m, i, o: seams.__setitem__("inorm0", o.detach()))
    with torch.no_grad():
        embeds = vision.patch_embedder(pixel_values, position_ids, padding)
        encoded = vision.encoder(inputs_embeds=embeds, attention_mask=~padding,
                                 pixel_position_ids=position_ids).last_hidden_state
        pooled = vision(pixel_values, pixel_position_ids=position_ids).last_hidden_state
        projected = embedder(pooled)

    globals()["_extra"] = {
        "pixel_values": pixel_values[0].float().contiguous(),
        "position_ids": position_ids[0].to(torch.int32).contiguous(),
        "patch_embed": embeds[0].float().contiguous(),
        "inorm0": seams["inorm0"][0].float().contiguous(),
        "attn0": seams["attn0"][0].float().contiguous(),
        "mlp0": seams["mlp0"][0].float().contiguous(),
        "layer0": seams["layer0"][0].float().contiguous(),
        "encoded": encoded[0].float().contiguous(),
        "pooled": pooled.float().contiguous(),
    }
    return projected.float().contiguous()


def run_gemma4_audio_real(image, checkpoint):
    """The Gemma 4 audio path on the RELEASED weights: the audio Conformer and its multimodal embedder,
    loaded selectively from the tri-modal checkpoint. Random mel features are fed to both sides, so the
    tower and the projection into the decoder's space are compared on real weights. `--checkpoint` is
    the release directory.
    """
    import json
    import os
    import torch
    from safetensors import safe_open
    from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel, Gemma4MultimodalEmbedder
    from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig, Gemma4TextConfig

    config = json.load(open(os.path.join(checkpoint, "config.json")))
    audio_config = Gemma4AudioConfig(**config["audio_config"])
    text_config = Gemma4TextConfig(**config["text_config"])
    audio = Gemma4AudioModel(audio_config).eval()
    embedder = Gemma4MultimodalEmbedder(audio_config, text_config).eval()

    audio_state, embed_state = {}, {}
    with safe_open(os.path.join(checkpoint, "model.safetensors"), framework="pt") as handle:
        for key in handle.keys():
            if key.startswith("model.audio_tower."):
                audio_state[key[len("model.audio_tower."):]] = handle.get_tensor(key).float()
            elif key.startswith("model.embed_audio."):
                embed_state[key[len("model.embed_audio."):]] = handle.get_tensor(key).float()
    audio.load_state_dict(audio_state, strict=True)
    embedder.load_state_dict(embed_state, strict=True)

    torch.manual_seed(4)
    features = torch.randn(1, 64, 128)
    mask = torch.ones(1, 64, dtype=torch.float32)
    with torch.no_grad():
        output = audio(features, mask, return_dict=True)
        projected = embedder(output.last_hidden_state)

    globals()["_extra"] = {
        "features": features[0].float().contiguous(),
        "encoded": output.last_hidden_state[0].float().contiguous(),
    }
    return projected[0].float().contiguous()


def run_gemma4_conditional_real(image, checkpoint):
    """The FULL Gemma4ForConditionalGeneration on the released E2B weights: an image and a
    placeholder-carrying prompt through the vision tower, the embedder, the splice, and the decoder,
    end to end. `--checkpoint` is the release directory.

    Random pixel values over a 6x6 grid (36 patches -> 4 soft tokens at pooling 3) fill four image
    placeholder tokens. Records the token ids, the pixel values, the position grid, and the decoder's
    logits over the fused sequence.
    """
    import json
    import os
    import torch
    from transformers import AutoModelForImageTextToText

    config = json.load(open(os.path.join(checkpoint, "config.json")))
    image_token = config["image_token_id"]
    model = AutoModelForImageTextToText.from_pretrained(checkpoint, dtype=torch.float32).eval()

    torch.manual_seed(3)
    side = 6
    pixel_values = torch.randn(1, side * side, 3 * config["vision_config"]["patch_size"] ** 2)
    grid = [[x, y] for y in range(side) for x in range(side)]
    position_ids = torch.tensor(grid, dtype=torch.long).unsqueeze(0)
    input_ids = torch.tensor([[2] + [image_token] * 4 + [1234, 5678, 91011]], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=input_ids, pixel_values=pixel_values,
                       image_position_ids=position_ids).logits[0]

    globals()["_extra"] = {
        "tokens": input_ids[0].to(torch.int32).contiguous(),
        "pixel_values": pixel_values[0].float().contiguous(),
        "position_ids": position_ids[0].to(torch.int32).contiguous(),
    }
    return logits.float().contiguous()


def run_gemma4_embedder(image, checkpoint):
    """The Gemma 4 multimodal embedder — a scale-free RMS norm and a projection into the language
    model's space — from transformers' own Gemma4MultimodalEmbedder at a tiny configuration.

    This is the piece that turns a tower's soft tokens into embeddings the decoder splices in. Records
    the input soft tokens, the projected output, and the weights in the release naming. `checkpoint`
    is unused.
    """
    import torch
    from transformers.models.gemma4 import modeling_gemma4 as G
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig, Gemma4TextConfig

    vision = Gemma4VisionConfig(hidden_size=32, rms_norm_eps=1e-6)
    text = Gemma4TextConfig(hidden_size=48)
    embedder = _randomized(G.Gemma4MultimodalEmbedder(vision, text), seed=31)
    soft = torch.randn(1, 5, 32)
    with torch.no_grad():
        out = embedder(soft)
    extra = {"soft": soft[0].float().contiguous()}
    for key, value in embedder.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return out[0].float().contiguous()


def run_gemma4_mel(image, checkpoint):
    """The Gemma 4 audio front end: raw audio to log-mel features, from transformers' own
    Gemma4AudioFeatureExtractor. `checkpoint` is unused.

    A deterministic waveform is recorded alongside its features so the Swift front end runs the exact
    same samples. This is a pure preprocessing step (framing, Hann window, magnitude FFT, HTK mel bank,
    log), so it matches to floating precision.
    """
    import numpy as np
    from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor

    extractor = Gemma4AudioFeatureExtractor()
    rng = np.random.RandomState(7)
    waveform = (0.3 * np.sin(np.linspace(0, 80, 4000)) + 0.05 * rng.randn(4000)).astype(np.float32)
    mask = np.ones(waveform.shape[0], dtype=np.float32)
    features, _ = extractor._extract_spectrogram(waveform[None, :], mask)

    import torch
    globals()["_extra"] = {"waveform": torch.tensor(waveform, dtype=torch.float32).contiguous()}
    return torch.tensor(np.asarray(features), dtype=torch.float32).contiguous()


def run_gemma4_audio(image, checkpoint):
    """The Gemma 4 audio Conformer at a tiny random configuration, from transformers' own
    Gemma4AudioModel.

    The most involved Gemma 4 tower: a 2-D convolutional subsampler, then Conformer layers — a macaron
    feed-forward, a blocked relative-position attention (Transformer-XL rel-shift, a per-dimension
    softplus scale, a logit softcap), a light depthwise convolution, a second macaron feed-forward, and
    sandwich norms — and an output projection. The record carries the mel input, the post-subsample
    hidden state, the parameter-free relative position encoding, the blocked attention mask, and the
    final output, so the subsampler and the Conformer are each isolated. The blocked mask is fed to the
    Swift side rather than reconstructed there, which isolates the attention arithmetic from the mask
    construction. `checkpoint` is unused.
    """
    import torch
    from transformers.models.gemma4 import modeling_gemma4 as G
    from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig

    config = Gemma4AudioConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4, conv_kernel_size=3,
        attention_chunk_size=4, attention_context_left=3, attention_context_right=0, hidden_act="silu",
        residual_weight=0.5, gradient_clipping=1e10, attention_logit_cap=50.0, rms_norm_eps=1e-6,
        subsampling_conv_channels=[8, 16], output_proj_dims=48)
    model = _randomized(G.Gemma4AudioModel(config), seed=29)
    # `_randomized` also perturbs the clippable-linear clamp buffers; the released model ships them at
    # ±inf (identity), so reset them, matching the port which does not model the clamps.
    for name, buffer in model.named_buffers():
        if name.endswith(("input_min", "output_min")):
            buffer.fill_(-float("inf"))
        elif name.endswith(("input_max", "output_max")):
            buffer.fill_(float("inf"))

    torch.manual_seed(5)
    features = torch.randn(1, 32, config.subsampling_conv_channels[0])
    # Capture the blocked attention mask the model builds by wrapping its own converter.
    captured = {}
    original = model._convert_4d_mask_to_blocked_5d
    def wrapped(mask_4d):
        result = original(mask_4d)
        captured["mask"] = result
        return result
    model._convert_4d_mask_to_blocked_5d = wrapped
    # Hooks to capture the first layer's sub-component outputs, for the isolation harness.
    seams = {}
    handles = []
    handles.append(model.layers[0].feed_forward1.register_forward_hook(
        lambda m, i, o: seams.__setitem__("ff1", o.detach())))
    handles.append(model.layers[0].self_attn.register_forward_hook(
        lambda m, i, o: seams.__setitem__("attn", o[0].detach())))
    handles.append(model.layers[0].lconv1d.register_forward_hook(
        lambda m, i, o: seams.__setitem__("lconv", o.detach())))
    handles.append(model.layers[0].register_forward_hook(
        lambda m, i, o: seams.__setitem__("layer0", o.detach())))
    with torch.no_grad():
        subsampled, _ = model.subsample_conv_projection(features, None)
        position_embeddings = model.rel_pos_enc(subsampled)
        out = model(features)
    for h in handles:
        h.remove()
    mask_5d = captured.get("mask")

    extra = {
        "input_features": features[0].float().contiguous(),
        "subsampled": subsampled[0].float().contiguous(),
        "position_embeddings": position_embeddings.float().contiguous(),
    }
    if mask_5d is not None:
        extra["attention_mask"] = mask_5d[0].to(torch.int32).contiguous()   # [1, num_blocks, chunk, context]
    for name, value in seams.items():
        extra[f"seam_{name}"] = value[0].float().contiguous()
    for key, value in model.state_dict().items():
        # The clippable-linear clamp buffers are identity (±inf); skip them.
        if key.endswith(("input_min", "input_max", "output_min", "output_max")):
            continue
        extra[f"w::{key}"] = (value.float() if value.is_floating_point() else value).contiguous()
    globals()["_extra"] = extra
    return out.last_hidden_state[0].float().clone().contiguous()   # `hidden.N` aliases it otherwise


def run_qwen3_5(image, checkpoint):
    """Qwen3.5 hybrid-decoder logits, from transformers' own Qwen3_5 implementation.

    The family Qwen3.5, Qwen3.6, and Qwen3.8 share: three gated delta-rule recurrence layers then one
    gated full-attention layer. 4B is the smallest release, and the only one that fits here.

    Records the per-layer hidden states as well as the logits, so the isolation harness can locate a
    divergence rather than only detect one.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32).eval()

    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
        logits = out.logits[0]
        generated = model.generate(ids, max_new_tokens=12, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)

    extra = {
        "tokens": ids[0].to(torch.int32).contiguous(),
        "continuation": generated[0, ids.shape[1]:].to(torch.int32).contiguous(),
    }
    for index, state in enumerate(out.hidden_states):
        extra[f"hidden.{index}"] = state[0].float().contiguous()
    globals()["_extra"] = extra
    return logits.float().contiguous()


def _deepseek_v41_serve_gemms(reference, net, args, tokenizer):
    """Make a float net compute its GEMMs the way the release serves them.

    The release builds every `Linear` without an explicit dtype as fp8 and its routed experts as
    fp4, and `linear()` then rounds the INPUT of each to fp8 in blocks of 32 before a narrow GEMM.
    Which layers those are is the reference's decision, so it is read off a probe built with the
    release's own dtypes rather than listed here: the port is measured against the rule, and a rule
    written down on both sides would agree with itself whatever the release does.

    The measured net stays float32. Each weight the probe builds narrow is rounded once to that
    storage format (fp8 in square blocks of 32, fp4 in row blocks of 32, each scale the next power
    of two of the block maximum over the format's range), which is what dequantizing the stored
    weight gives. `linear()` is wrapped so a weight so marked has its input rounded as the release's
    `act_quant` rounds it. A GEMM over dequantized operands with float32 accumulation is the fp8 or
    fp4 GEMM's arithmetic up to the order of the sum.

    `wo_a` is built fp8 and applied through an einsum rather than `linear()`, so its weight is
    rounded and its input is not, which is the release's own behaviour and falls out of this rather
    than being special-cased.
    """
    import dataclasses
    import ml_dtypes
    import numpy as np
    import torch

    probe = reference.Transformer(dataclasses.replace(args, dtype="fp8", expert_dtype="fp4"),
                                  tokenizer)
    narrow = {name: ("fp4" if module.weight.dtype == torch.float4_e2m1fn_x2 else "fp8")
              for name, module in probe.named_modules()
              if isinstance(module, reference.Linear)
              and module.weight.dtype in (torch.float8_e4m3fn, torch.float4_e2m1fn_x2)}
    del probe

    def next_power_of_two(value):
        bits = value.float().contiguous().view(torch.int32)
        exponent = ((bits >> 23) & 0xFF) - 127 + ((bits & ((1 << 23) - 1)) != 0).to(torch.int32)
        return ((exponent + 127) << 23).to(torch.int32).view(torch.float32)

    def stored(weight, kind):
        w = weight.detach().float()
        rows, columns = w.shape
        if kind == "fp8":
            padded = torch.zeros((rows + 31) // 32 * 32, (columns + 31) // 32 * 32)
            padded[:rows, :columns] = w
            blocks = padded.reshape(padded.shape[0] // 32, 32, padded.shape[1] // 32, 32)
            amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-4)
            scale = next_power_of_two(amax / 448.0)
            narrow_values = (blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
            return (narrow_values * scale).reshape(padded.shape)[:rows, :columns]
        blocks = w.reshape(rows, columns // 32, 32)
        amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=6 * 2.0 ** -126)
        scale = next_power_of_two(amax / 6.0)
        values = (blocks / scale).clamp(-6, 6).numpy().astype(ml_dtypes.float4_e2m1fn)
        return (torch.from_numpy(values.astype(np.float32)) * scale).reshape(rows, columns)

    for name, module in net.named_modules():
        kind = narrow.get(name)
        if kind is None:
            continue
        with torch.no_grad():
            module.weight.copy_(stored(module.weight, kind))
        module.weight._served = True

    original_linear = reference.linear
    act_quant = sys.modules["kernel"].act_quant

    def serving_linear(x, weight, bias=None):
        if getattr(weight, "_served", False):
            x = x.clone()
            act_quant(x, reference.fp8_block_size, reference.scale_fmt, reference.scale_dtype, True)
        return original_linear(x, weight, bias)

    reference.linear = serving_linear
    return narrow, original_linear


def _deepseek_v41_bf16(reference, build, seed=11):
    """A module built inside the release's own `set_dtype(torch.bfloat16)`, randomized in place.

    Every parameter takes the dtype the reference's constructor gives it: bf16 by default, float32
    where the constructor says so. Each is randomized in its OWN dtype, because `_randomized` ends by
    casting to float32, which is the one thing a bf16 module must not do.
    """
    import torch

    with reference.set_dtype(torch.bfloat16):
        module = build()
    torch.manual_seed(seed)
    state = module.state_dict()
    for key in sorted(state):
        if state[key].is_floating_point():
            state[key] = (torch.randn(state[key].shape) * 0.05).to(state[key].dtype)
    module.load_state_dict(state)
    return module.eval()


def _deepseek_v41_kernel_shim(quantizes=False):
    """Stand a CPU `kernel` module in front of DeepSeek V4.1's reference implementation.

    With `quantizes`, the two activation quantizers do what the release's tilelang kernels do instead
    of nothing, transcribed line for line from `kernel.py`: a block's absolute maximum, floored, sets
    the scale; with `scale_fmt` the scale is the next power of two, computed as the kernel computes
    it by multiplying by the reciprocal rather than dividing; the value is clamped, cast to the
    narrow type, cast back, and multiplied by the scale. The compressed latent's fp4 takes an E4M3
    scale instead, which the kernel reaches by DIVIDING by 6 and rounding the quotient to e4m3. Both
    casts round to nearest-even, through torch's own float8 and ml_dtypes' float4, which are
    independent of the port that is measured against them.

    `inference/model.py` imports six symbols from `kernel`, whose real bodies are tilelang kernels
    that compile for CUDA. At a float configuration the reachable surface is small:

    - `fp8_gemm` / `fp4_gemm` are reached only through `linear()` for a QUANTIZED weight, and there
      are none here, so they raise. That turns "the float path does not route through them" from an
      assumption into an assertion.
    - `act_quant` / `fp4_act_quant` are called on the sliding-window key-value and the compressed
      latent whatever the weights are. With `inplace=True` each is a fused quantize-then-dequantize
      that writes back in the input's dtype, so it only rounds; leaving the tensor alone makes this
      the UNQUANTIZED model, which is what a float port should be held to.
    - `sparse_attn` and `hc_split_sinkhorn` carry arithmetic. Both are mechanisms already measured
      against transformers' own `deepseek_v4`, which implements each in plain PyTorch, so these are
      transcribed from that verified third party rather than invented here.
    """
    import types
    import torch
    import torch.nn.functional as F

    kernel = types.ModuleType("kernel")

    def gemm(name):
        def stub(*args, **kwargs):
            raise AssertionError(f"{name} reached at a float configuration: only a quantized "
                                 "weight routes through it, and there are none here")
        return stub

    kernel.fp8_gemm = gemm("fp8_gemm")
    kernel.fp4_gemm = gemm("fp4_gemm")

    def next_power_of_two(value):
        # `fast_log2_ceil` then `fast_pow2`: the exponent field, plus one when any mantissa bit is
        # set, so an exact power of two is its own ceiling.
        bits = value.float().contiguous().view(torch.int32)
        exponent = ((bits >> 23) & 0xFF) - 127 + ((bits & ((1 << 23) - 1)) != 0).to(torch.int32)
        return ((exponent + 127) << 23).to(torch.int32).view(torch.float32)

    def blocks(x, block_size):
        width = x.size(-1)
        assert width % block_size == 0, f"{width} does not divide into blocks of {block_size}"
        return x.float().reshape(*x.shape[:-1], width // block_size, block_size)

    def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=None, inplace=False):
        assert inplace, "a non-inplace act_quant feeds a quantized GEMM"
        if not quantizes:
            return None
        z = blocks(x, block_size)
        amax = z.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
        inverse = torch.tensor(1.0 / 448.0, dtype=torch.float32)
        scale = next_power_of_two(amax * inverse) if scale_fmt is not None else amax * inverse
        narrow = (z / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
        x.copy_((narrow * scale).reshape(x.shape).to(x.dtype))
        return x

    def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=None):
        assert inplace, "a non-inplace fp4_act_quant feeds a quantized GEMM"
        if not quantizes:
            return None
        import ml_dtypes
        import numpy as np
        z = blocks(x, block_size)
        amax = z.abs().amax(dim=-1, keepdim=True)
        if scale_dtype == torch.float8_e4m3fn:
            # Training's compressed KV: an all-zero group keeps a nonzero scale. The quotient is
            # rounded to e4m3 as the kernel's `T.Cast(FP8, amax / fp4_max)` does; it is clamped to
            # the format's range first because an overflowing cast is where torch and the kernel
            # could disagree, and no activation here comes near it.
            amax = amax.clamp(min=6 * 2.0 ** -9)
            scale = (amax / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn).float()
        else:
            amax = amax.clamp(min=6 * 2.0 ** -126)
            scale = next_power_of_two(amax * torch.tensor(1.0 / 6.0, dtype=torch.float32))
        clamped = (z / scale).clamp(-6.0, 6.0)
        narrow = torch.from_numpy(clamped.numpy().astype(ml_dtypes.float4_e2m1fn)
                                  .astype(np.float32))
        x.copy_((narrow * scale).reshape(x.shape).to(x.dtype))
        return x

    def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
        hc = hc_mult
        pre_w, post_w, comb_w = mixes.float().split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = hc_base.float().split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = hc_scale.float().unbind(0)
        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
        comb = torch.softmax(logits, dim=-1) + eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        return pre, post, comb

    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        """q [b,s,h,d] against the shared latent kv [b,n,d] at the positions topk_idxs names.

        One latent is both key and value, which is what multi-head latent attention means; -1 marks
        a position the query may not see, and the per-head sink is an extra softmax logit that
        drains mass and contributes no value.
        """
        b, s, h, d = q.shape
        k = topk_idxs.shape[-1]
        valid = topk_idxs >= 0
        gathered = topk_idxs.clamp_min(0).long()
        latents = torch.gather(kv.unsqueeze(1).expand(b, s, kv.shape[1], d), 2,
                               gathered.unsqueeze(-1).expand(b, s, k, d)).float()
        scores = torch.einsum("bshd,bskd->bshk", q.float(), latents) * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
        sink = attn_sink.float().view(1, 1, h, 1).expand(b, s, h, 1)
        probs = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :k]
        return torch.einsum("bshk,bskd->bshd", probs, latents).to(q.dtype)

    kernel.act_quant = act_quant
    kernel.fp4_act_quant = fp4_act_quant
    kernel.hc_split_sinkhorn = hc_split_sinkhorn
    kernel.sparse_attn = sparse_attn
    sys.modules["kernel"] = kernel


def run_deepseek_v4_release(image):
    """DeepSeek V4 Flash's decoder from the release's OWN `inference/model.py`, in float32."""
    return _run_deepseek_v4_release(image, "deepseek-v4")


def run_deepseek_v4_release_bf16(image):
    """The same, built in bf16 as the release runs it. Run under the gemma environment."""
    return _run_deepseek_v4_release(image, "deepseek-v4", bf16=True)


def run_deepseek_v4_pro_release(image):
    """DeepSeek V4 Pro (0813)'s decoder from its release's own `inference/model.py`, in float32."""
    return _run_deepseek_v4_release(image, "deepseek-v4-pro-0813")


def run_deepseek_v4_pro_release_bf16(image):
    """The same, built in bf16 as the release runs it. Run under the gemma environment."""
    return _run_deepseek_v4_release(image, "deepseek-v4-pro-0813", bf16=True)


def _deepseek_v4_release_net(release, bf16=False, dspark=False):
    """DeepSeek V4's decoder from the release's own `inference/model.py`, behind the same CPU
    `kernel` shim V4.1 runs behind (V4 imports the same six symbols with the same signatures).

    `deepseek_v4` measures V4 against transformers at an all-sliding configuration, which never
    reaches a compressor or the indexer. This reaches both: ratio 4 pools overlapping groups and
    owns an indexer that keeps fewer groups than exist, ratio 8 stands in for the release's 128 and
    pools plain groups, layer 0 routes by its token table, and YaRN is on for the compressed layers
    at an original length the sequence exceeds, as the releases configure it. Pro 0813 differs in
    layout (its first layers compress) and routing scale, and moves `hc_head` onto the block; both
    arrangements are recorded. V4 Flash and V4 Pro share one `model.py`, so Flash's record covers
    Pro's first release too.
    """
    import torch
    import torch.nn.functional as F

    _deepseek_v41_kernel_shim()
    # The indexer rotates its query and keys through `fast_hadamard_transform`, a CUDA library.
    # This is the library's own `hadamard_transform_ref` (a Sylvester Hadamard matrix, times the
    # scale) with the CUDA kernel's arithmetic: accumulated in float32, rounded once to the input's
    # dtype. transformers' `deepseek_v4` omits the rotation, so there is no third party to take it
    # from; the definition is the library's.
    import types
    hadamard = types.ModuleType("fast_hadamard_transform")

    def hadamard_transform(x, scale=1.0):
        width = x.size(-1)
        assert width & (width - 1) == 0, f"{width} is not a power of two"
        matrix = torch.ones(1, 1)
        while matrix.size(0) < width:
            matrix = torch.cat([torch.cat([matrix, matrix], 1), torch.cat([matrix, -matrix], 1)], 0)
        return (F.linear(x.float(), matrix) * scale).to(x.dtype)

    hadamard.hadamard_transform = hadamard_transform
    sys.modules["fast_hadamard_transform"] = hadamard
    source = os.path.expanduser(f"~/.inferkit-validation/reference-sources/{release}")
    sys.path.insert(0, source)
    import model as reference
    if not bf16:
        # `rotate_activation` asserts bf16, because the release only ever runs in bf16. A float32
        # record runs the same rotation without the assertion.
        reference.rotate_activation = lambda x: hadamard_transform(x, scale=x.size(-1) ** -0.5)

    pro = "pro" in release
    # 0813's draft stack: two stages over the decoder's own experts, reading the last three layers.
    drafting = dict(n_mtp_layers=2, dspark_block_size=3, dspark_noise_token_id=120,
                    dspark_target_layer_ids=(3, 4, 5), dspark_markov_rank=8, temperature=0) if dspark else {}
    ratios = (8, 8, 4, 8, 4, 0) if pro else (0, 0, 4, 8, 4, 8)
    args = reference.ModelArgs(**drafting,
        max_batch_size=1, max_seq_len=64, dtype="bf16", expert_dtype=None, scale_dtype="fp32",
        scale_fmt=None, vocab_size=128, dim=64, moe_inter_dim=32, n_layers=6, n_hash_layers=1,
        **({} if dspark else {"n_mtp_layers": 0}), n_heads=4, n_routed_experts=8,
        n_shared_experts=1, n_activated_experts=2,
        score_func="sqrtsoftplus", route_scale=2.5 if pro else 1.5, swiglu_limit=10.0,
        q_lora_rank=32, head_dim=32, rope_head_dim=8, o_groups=2, o_lora_rank=16, window_size=8,
        compress_ratios=ratios + ((0, 0) if dspark else ()),
        compress_rope_theta=40000.0, original_seq_len=16, rope_theta=10000.0, rope_factor=4,
        beta_fast=32, beta_slow=1, index_n_heads=4, index_head_dim=32, index_topk=3,
        hc_mult=3, hc_sinkhorn_iters=4, hc_eps=1e-6)
    torch.manual_seed(0)
    if bf16:
        net = _deepseek_v41_bf16(reference, lambda: reference.Transformer(args))
    else:
        net = _randomized(reference.Transformer(args).float())
    torch.manual_seed(5)
    for layer in net.layers:
        if layer.ffn.gate.hash:
            layer.ffn.gate.tid2eid.data = torch.stack(
                [torch.randperm(args.n_routed_experts)[:args.n_activated_experts]
                 for _ in range(args.vocab_size)]).to(torch.int32)
    return reference, net, args


def _run_deepseek_v4_release(image, release, bf16=False):
    """The prefill record for `_deepseek_v4_release_net`'s network: every position's logits, every
    layer's stream, and the seams a disagreement localizes to."""
    import torch
    import torch.nn.functional as F

    reference, net, args = _deepseek_v4_release_net(release, bf16)
    seams = {}

    def capture(name):
        def hook(module, inputs, output):
            seams[name] = output.detach().clone()
        return hook

    for index, layer in enumerate(net.layers):
        layer.attn.register_forward_hook(capture(f"attn.{index}"))
        layer.ffn.register_forward_hook(capture(f"ffn.{index}"))
        if getattr(layer.attn, "indexer", None) is not None:
            layer.attn.indexer.register_forward_hook(capture(f"indexer.{index}"))
    # What the sparse attention reads and returns, per layer in call order: the query after its
    # norm and rotary, the key-value it gathers from, and its output before the de-rotation.
    sparse_calls = []
    shimmed_sparse = reference.sparse_attn

    def recording_sparse(q, kv, attn_sink, topk_idxs, softmax_scale):
        out = shimmed_sparse(q, kv, attn_sink, topk_idxs, softmax_scale)
        sparse_calls.append((q.detach().clone(), kv.detach().clone(), out.detach().clone()))
        return out

    reference.sparse_attn = recording_sparse
    # Each indexer's ranked scores, the receiver of the one `topk` inside `Indexer.forward`.
    index_scores = {}
    pending_score = {}
    original_topk = torch.Tensor.topk

    def recording_topk(self, k, dim=-1, largest=True, sorted=True):
        pending_score["last"] = self.detach().clone()
        return original_topk(self, k, dim=dim, largest=largest, sorted=sorted)

    torch.Tensor.topk = recording_topk
    for index, layer in enumerate(net.layers):
        if getattr(layer.attn, "indexer", None) is not None:
            def take_score(module, inputs, output, index=index):
                if "last" in pending_score:
                    index_scores[index] = pending_score.pop("last")
            layer.attn.indexer.register_forward_hook(take_score)

    tokens = torch.tensor([[5, 9, 2, 31, 7, 44, 2, 18, 60, 3, 12, 25, 7, 19, 40, 2, 99, 71, 8, 64]])
    states = []
    with torch.inference_mode():
        h = net.embed(tokens).unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
        for layer in net.layers:
            states.append(h)
            h = layer(h, 0, tokens)
        # V4 collapses the copies in its head; 0813 moved the same function onto the block.
        collapse = (net.layers[-1].hc_head if hasattr(reference.Block, "hc_head")
                    else net.head.hc_head)
        collapsed = collapse(h, net.hc_head_fn, net.hc_head_scale, net.hc_head_base)
        # `get_logits` keeps only the last position; every one is wanted here.
        logits = F.linear(net.norm(collapsed).float(), net.head.weight)

    reference.sparse_attn = shimmed_sparse
    torch.Tensor.topk = original_topk
    extra = {"tokens": tokens[0].to(torch.int32).contiguous(),
             "collapsed": collapsed[0].float().contiguous()}
    for index, value in index_scores.items():
        extra[f"seam.score.{index}"] = torch.nan_to_num(value[0].float(), neginf=-1e30).contiguous()
    for index, (q, kv, out) in enumerate(sparse_calls):
        extra[f"seam.sparse.q.{index}"] = q[0].float().contiguous()
        extra[f"seam.sparse.kv.{index}"] = kv[0].float().contiguous()
        extra[f"seam.sparse.out.{index}"] = out[0].float().contiguous()
    for name, value in seams.items():
        extra[f"seam.{name}"] = (value.float() if value.is_floating_point()
                                 else value.to(torch.int32))[0].clone().contiguous()
    for index, state in enumerate(states):
        extra[f"hidden.{index}"] = state[0].float().contiguous()
    extra[f"hidden.{len(states)}"] = h[0].float().contiguous()
    for key, value in net.state_dict().items():
        extra[f"w::{key}"] = (value.float() if value.is_floating_point() else value).clone().contiguous()
        if bf16 and value.is_floating_point():
            extra[f"dtype::{key}"] = torch.tensor([1 if value.dtype == torch.float32 else 0],
                                                  dtype=torch.int32)
    globals()["_extra"] = extra
    return logits[0].float().contiguous()


def run_deepseek_v4_release_decode(image):
    """V4 Flash decoding one token at a time through its release's own code, in float32."""
    return _run_deepseek_v4_release_decode(image, "deepseek-v4")


def run_deepseek_v4_release_decode_bf16(image):
    """The same in bf16. Run under the gemma environment."""
    return _run_deepseek_v4_release_decode(image, "deepseek-v4", bf16=True)


def run_deepseek_v4_pro_release_decode(image):
    """V4 Pro (0813) decoding one token at a time through its release's own code, in float32."""
    return _run_deepseek_v4_release_decode(image, "deepseek-v4-pro-0813")


def run_deepseek_v4_pro_release_decode_bf16(image):
    """The same in bf16. Run under the gemma environment."""
    return _run_deepseek_v4_release_decode(image, "deepseek-v4-pro-0813", bf16=True)


def _run_deepseek_v4_release_decode(image, release, bf16=False):
    """A prefill of 11 tokens, then five single-token steps, through the release's own buffers.

    Eleven is odd and past the window of 8, so the ring has wrapped, a ratio-4 compressor parks
    three positions and a ratio-8 one parks three. The steps reach positions 11 to 15: position 11
    closes a ratio-4 group through the overlapping two-window state, and position 15 closes both a
    ratio-4 and a ratio-8 group, so every compressor emits at decode as well as at prefill. The
    indexer's own compressor and keys are exercised the same way. Each call runs the layer loop
    `Transformer.forward` runs, and the head reads the last position, as a step does.
    """
    import torch
    import torch.nn.functional as F

    reference, net, args = _deepseek_v4_release_net(release, bf16)
    index_scores = {}
    pending_score = {}
    original_topk = torch.Tensor.topk

    def recording_topk(self, k, dim=-1, largest=True, sorted=True):
        pending_score["last"] = self.detach().clone()
        return original_topk(self, k, dim=dim, largest=largest, sorted=sorted)

    step_tag = {"tag": "prefill"}
    for index, layer in enumerate(net.layers):
        if getattr(layer.attn, "indexer", None) is not None:
            def take_score(module, inputs, output, index=index):
                if "last" in pending_score:
                    index_scores[f"{step_tag['tag']}.score.{index}"] = pending_score.pop("last")
            layer.attn.indexer.register_forward_hook(take_score)

    def run(ids, start_pos):
        h = net.embed(ids).unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
        for layer in net.layers:
            h = layer(h, start_pos, ids)
        collapse = (net.layers[-1].hc_head if hasattr(reference.Block, "hc_head")
                    else net.head.hc_head)
        collapsed = collapse(h, net.hc_head_fn, net.hc_head_scale, net.hc_head_base)
        return F.linear(net.norm(collapsed)[:, -1].float(), net.head.weight)

    prompt = torch.tensor([[5, 9, 2, 31, 7, 44, 2, 18, 60, 3, 12]])
    extra = {"prompt": prompt[0].to(torch.int32).contiguous()}
    torch.Tensor.topk = recording_topk
    with torch.inference_mode():
        logits = run(prompt, 0)
        extra["prefill.logits"] = logits[0].float().contiguous()
        tokens = [int(logits[0].argmax())]
        for step in range(5):
            step_tag["tag"] = f"step{step}"
            logits = run(torch.tensor([[tokens[-1]]]), prompt.size(1) + step)
            extra[f"step{step}.logits"] = logits[0].float().contiguous()
            tokens.append(int(logits[0].argmax()))
    torch.Tensor.topk = original_topk
    extra["generated"] = torch.tensor(tokens, dtype=torch.int32)
    for name, value in index_scores.items():
        extra[name] = torch.nan_to_num(value[0].float(), neginf=-1e30).contiguous()
    for key, value in net.state_dict().items():
        extra[f"w::{key}"] = (value.float() if value.is_floating_point() else value).clone().contiguous()
    globals()["_extra"] = extra
    return extra["prefill.logits"].clone()


def run_deepseek_v4_pro_dspark(image):
    """V4 Pro 0813's DSpark draft stack from its release's own `inference/model.py`, in float32."""
    return _run_deepseek_v4_pro_dspark(image)


def run_deepseek_v4_pro_dspark_bf16(image):
    """The same in bf16, as the release runs it. Run under the gemma environment."""
    return _run_deepseek_v4_pro_dspark(image, bf16=True)


def _run_deepseek_v4_pro_dspark(image, bf16=False):
    """V4 Pro 0813's draft stack, measured the way V4.1's is, with 0813's four differences in play:
    the main states are the target layers' OUTPUTS rather than the stream entering them, a stage is a
    V4 block (its own read weight per sub-block and V4's query-head norm), the last stage collapses
    the copies through its own learned `hc_head`, and the Markov tables are `markov_w1` and
    `markov_w2`. The stages route over the decoder's experts, which is how 0813's config leaves it.
    `temperature` is zero, so the reference's sampler is an argmax and the walk reproducible.
    """
    import re
    import torch

    reference, net, args = _deepseek_v4_release_net("deepseek-v4-pro-0813", bf16, dspark=True)
    first, last = net.mtp[0], net.mtp[-1]
    width = torch.bfloat16 if bf16 else torch.float32

    torch.manual_seed(2)
    main_hidden = torch.randn(1, args.dspark_block_size,
                              args.dim * len(args.dspark_target_layer_ids)).to(width)
    hidden = torch.randn(1, args.dspark_block_size, args.dim).to(width)
    tokens = torch.tensor([[17, 5, 100]], dtype=torch.long)
    with torch.inference_mode():
        main_state = first.main_norm(first.main_proj(main_hidden))
        bias, embedded = last.markov_head(tokens[0])
        confidence = last.confidence_head(hidden, embedded.unsqueeze(0))

    draft_seams = {}

    def capture(name):
        def hook(module, inputs, output):
            draft_seams[name] = output.detach().clone()
        return hook

    for stage, block in enumerate(net.mtp):
        block.attn.register_forward_hook(capture(f"draft.attn.{stage}"))
        block.register_forward_hook(capture(f"draft.block.{stage}"))
        block.ffn.register_forward_hook(capture(f"draft.ffn.{stage}"))

    # What each draft stage's sparse attention reads and returns; they are the last calls of the step.
    sparse_calls = []
    shimmed_sparse = reference.sparse_attn

    def recording_sparse(q, kv, attn_sink, topk_idxs, softmax_scale):
        out = shimmed_sparse(q, kv, attn_sink, topk_idxs, softmax_scale)
        sparse_calls.append((q.detach().clone(), kv.detach().clone(), topk_idxs.detach().clone(),
                             out.detach().clone()))
        return out

    reference.sparse_attn = recording_sparse
    # Ten tokens, past the window of 8, so each stage's window is a ring that has already wrapped.
    prompt = torch.tensor([[5, 9, 2, 31, 7, 44, 2, 18, 60, 3]])
    length = prompt.size(1)
    with torch.inference_mode():
        committed, _, prompt_states = net(prompt, start_pos=0)
        net.forward_spec(prompt[:, -1], prompt_states, start_pos=0)   # seeds each stage's window
        _, _, step_states = net(committed.view(1, 1), start_pos=length)
        drafted, draft_logits, draft_confidence = net.forward_spec(
            committed, step_states, start_pos=length)

    extra = {
        "main_hidden": main_hidden[0].float().contiguous(),
        "hidden": hidden[0].float().contiguous(),
        "tokens": tokens[0].to(torch.int32).contiguous(),
        "main_state": main_state[0].float().contiguous(),
        "markov_embed": embedded.float().contiguous(),
        "confidence": confidence[0].float().contiguous(),
        "noise_token": torch.tensor([args.dspark_noise_token_id], dtype=torch.int32),
        "block_size": torch.tensor([args.dspark_block_size], dtype=torch.int32),
        "loop.prompt": prompt[0].to(torch.int32).contiguous(),
        "loop.committed": committed.to(torch.int32).contiguous(),
        "loop.main_states": torch.cat([prompt_states, step_states], dim=1)[0].float().contiguous(),
        "loop.drafted": drafted[0].to(torch.int32).contiguous(),
        "loop.logits": draft_logits[0].float().contiguous(),
        "loop.confidence": draft_confidence[0].float().contiguous(),
    }
    reference.sparse_attn = shimmed_sparse
    for stage, (q, kv, idxs, out) in enumerate(sparse_calls[-len(net.mtp):]):
        extra[f"seam.draft.sparse.q.{stage}"] = q[0].float().clone().contiguous()
        extra[f"seam.draft.sparse.kv.{stage}"] = kv[0].float().clone().contiguous()
        extra[f"seam.draft.sparse.idx.{stage}"] = idxs[0].to(torch.int32).clone().contiguous()
        extra[f"seam.draft.sparse.out.{stage}"] = out[0].float().clone().contiguous()
    extra["markov_bias"] = bias.float().clone().contiguous()
    # Cloned one by one: in float32 `.float()` hands back the same buffer, and two names may not share it.
    extra = {name: value.clone() for name, value in extra.items()}
    for name, value in draft_seams.items():
        extra[f"seam.{name}"] = value.detach().float()[0].clone().contiguous()
    for key, value in net.state_dict().items():
        if re.match(r"mtp\.\d+\.(embed|head)\.weight$", key):
            continue   # aliases of the decoder's own embedding and head
        extra[f"w::stack.{key}"] = (value.float() if value.is_floating_point() else value).clone().contiguous()
        if bf16 and value.is_floating_point():
            extra[f"dtype::{key}"] = torch.tensor([1 if value.dtype == torch.float32 else 0],
                                                  dtype=torch.int32)
    globals()["_extra"] = extra
    # Cloned: the harness writes this as `output`, and safetensors refuses two names for one buffer.
    return draft_logits[0].float().clone().contiguous()


def run_deepseek_v41(image):
    """The DeepSeek V4.1 decoder's arithmetic, from the release's own `inference/model.py`."""
    return _run_deepseek_v41(image)


def run_deepseek_v41_quantized(image):
    """The same decoder with the release's own activation round trips switched ON.

    The unquantized record holds the port to the model; this one holds it to what the release's
    `inference/model.py` computes, which rounds the sliding-window key-value to fp8, the compressed
    latent to fp4 with E4M3 scales, and the indexer's keys and queries to fp4, each in place and
    each read by everything downstream. It also keeps the n-gram lookup's cast of its rows to bf16,
    which the unquantized record drops for the same reason. What it does NOT reproduce is running
    in bf16 throughout: the net is float32, as the port is, so this measures the explicit round
    trips and not the dtype the release computes in.

    The index head width is 32 rather than 16 here, because the real kernel asserts that a row
    divides into its blocks and the indexer's fp4 block is 32. Run under the gemma environment,
    which carries ml_dtypes for the fp4 cast.
    """
    return _run_deepseek_v41(image, quantizes=True)


def run_deepseek_v41_bf16(image):
    """The decoder as the release SERVES it: in bf16, with its round trips and its narrow GEMMs.

    The net is built inside the release's own `set_dtype(torch.bfloat16)`, so every parameter takes
    the dtype the reference's constructor gives it, with no fix-up: bf16 by default, and float32 for
    the hyper-connection coefficients, the attention sink, the router bias, the head, and the
    ratio-above-one compressor. That set is exactly what the release's headers store F32 or hold
    float32, entry for entry. Everything the reference computes in float32 it still computes in
    float32, because that is the reference's own code: only the dtype it starts from changes.
    """
    return _run_deepseek_v41(image, quantizes=True, bf16=True)


def run_deepseek_v41_bf16_plain(image):
    """The same bf16 decoder without the round trips or the narrow GEMMs, which isolates the dtype."""
    return _run_deepseek_v41(image, bf16=True)


def _run_deepseek_v41(image, quantizes=False, bf16=False):
    """The DeepSeek V4.1 decoder's arithmetic, from the release's own `inference/model.py`.

    V4.1 is the V4 architecture with a different arrangement, and this measures the arrangement:
    only four layers own a compressor and the rest read one of them, a compressor that pools one
    position per group is a plain projection with no gate, the indexer takes its keys from that
    compressor's latent, and the hyper-connection copies collapse without a learned head. The
    released weights are 510 GB, so the size measured here is the one `ModelArgs` was given defaults
    for, with the layer pattern chosen so every one of those differences is exercised: ratios of 2
    and 1 both appear, the kv sources are fewer than the index sources, a non-source layer sits
    between them, the candidate source is a ratio-1 kv source with one consumer after it (which is
    the released arrangement at layer 20), and two layers carry an n-gram memory.

    The sequence is twice the sliding window, so the window is a real constraint rather than a
    causal mask by another name, and every compressed layer reaches past it.

    transformers carries `deepseek_v4` and no `deepseek_v41`, so the reference here is the release's
    own code, run on the CPU behind `_deepseek_v41_kernel_shim`. The record's weights are saved in
    the release's naming, and the hidden state entering every layer is recorded so a divergence
    localizes to a layer rather than to the stack.
    """
    import torch
    import torch.nn.functional as F

    _deepseek_v41_kernel_shim(quantizes=quantizes)
    source = os.path.expanduser(os.environ.get(
        "IK_DEEPSEEK_V41_SRC", "~/.inferkit-validation/reference-sources/deepseek-v41"))
    sys.path.insert(0, source)
    import model as reference

    # The n-gram table is stored fp8 and its lookup casts the dequantized rows to bf16 on the way
    # out. That is the same storage rounding the kernel shim neutralizes for the window key-value,
    # and at a float configuration it is the only lossy step in the engram path, so it is dropped
    # for the same reason: what is measured is the unquantized model. Dropping it also keeps the
    # lookup in the dtype `wkv` was converted to, which a bf16 result would not match.
    def unrounded_lookup(self, indices):
        values = F.embedding(indices, self.weight)
        scales = F.embedding(indices, self.scale)
        values = values.float().unflatten(-1, (-1, self.block_size)) * scales.float().unsqueeze(-1)
        rows = values.flatten(-2)
        # The release casts the rows to bf16 on the way out. A float32 net brings them back so its
        # float32 `wkv` can read them; a bf16 net keeps them, as the release does.
        if bf16:
            return rows.to(torch.bfloat16)
        return rows.to(torch.bfloat16).float() if quantizes else rows

    original_lookup = reference.ParallelEngramEmbedding.forward
    reference.ParallelEngramEmbedding.forward = unrounded_lookup

    # `build_compressed_token_map` needs a tokenizer only to learn which token ids normalize alike.
    # Ids that all normalize apart make the map the identity, which leaves the compressed vocab size
    # equal to the vocab size and exercises the REAL derivation -- the primes, the per-layer
    # multipliers, and the look-back -- without carrying a 129,280-entry tokenizer into the record.
    class _DistinctBackend:
        def decode(self, ids, skip_special_tokens=False):
            return f"tok{ids[0]}"

        def id_to_token(self, token_id):
            return f"tok{token_id}"

    class _DistinctTokenizer:
        def __init__(self, size):
            self._size = size
            self.backend_tokenizer = _DistinctBackend()

        def __len__(self):
            return self._size

    args = reference.ModelArgs(
        max_batch_size=1, max_seq_len=64, dtype="bf16", expert_dtype=None,
        vocab_size=256, dim=64, moe_inter_dim=32, n_layers=6, n_mtp_layers=0,
        n_heads=4, n_routed_experts=8, n_activated_experts=2,
        q_lora_rank=32, head_dim=32, rope_head_dim=8, o_groups=2, o_lora_rank=16,
        window_size=8, compress_ratios=(0, 0, 2, 2, 1, 1),
        kv_source_layers=(2, 4), index_source_layers=(2, 3, 4, 5),
        index_n_heads=8, index_head_dim=32 if (quantizes or bf16) else 16, index_topk=4,
        candidate_source_layer=4, candidate_topk_blocks=3, candidate_block_size=2,
        engram_layer_ids=(1, 3), engram_num_embeddings=(290, 370),
        engram_max_ngram_size=3, engram_n_heads=2, engram_head_dim=32,
        engram_vocab_size=64, engram_compressed_vocab_size=256, engram_pad_id=2,
        hc_mult=3, hc_sinkhorn_iters=4,
    )
    torch.manual_seed(0)
    if bf16:
        net = _deepseek_v41_bf16(
            reference, lambda: reference.Transformer(args, _DistinctTokenizer(args.vocab_size)))
    else:
        net = reference.Transformer(args, _DistinctTokenizer(args.vocab_size)).float()
        net = _randomized(net)
    served = None
    if quantizes:
        served, unserved_linear = _deepseek_v41_serve_gemms(
            reference, net, args, _DistinctTokenizer(args.vocab_size))
    # The table row counts are not free parameters: each is the sum of that layer's bucket primes,
    # which is how the released 384,006,168 and 384,016,682 are checked too.
    assert tuple(sum(sum(per_size) for per_size in layer) for layer in net.engram_layout.primes) \
        == args.engram_num_embeddings

    # Seams, so a divergence localizes to a mechanism. The hyper-connection coefficients come from a
    # wrapper rather than a hook, because `hc_mixes` is a method on the block and not a submodule.
    seams = {}

    # Cloned AT CAPTURE, not at the end of the run. `_compress_kv` rotates the compressor's output
    # in place after the hook has fired, and `_window_kv` quantizes the key-value in place, so a
    # hook that stores the reference records a tensor that is mutated afterwards. That shows up as a
    # seam whose norm is exactly right and whose direction is not, which is what a rotary does.
    def capture(name, pick=lambda output: output):
        def hook(module, inputs, output):
            seams[name] = pick(output).detach().clone()
        return hook

    layers = net.layers
    def capture_input(name):
        def hook(module, inputs, output):
            seams[name] = inputs[0].detach().clone()
        return hook

    # The compressor's INPUT as well as its output: a mismatch in the pooled latent is otherwise
    # indistinguishable from a mismatch in what the block handed it.
    for layer_id in (2, 4):
        layers[layer_id].attn.compressor.register_forward_hook(capture_input(f"compressor.in.{layer_id}"))
        layers[layer_id].attn.compressor.register_forward_hook(capture(f"compressor.{layer_id}"))
    for layer_id in (2, 3, 4, 5):
        layers[layer_id].attn.indexer.register_forward_hook(capture(f"indexer.{layer_id}"))
    for layer_id in range(args.n_layers):
        layers[layer_id].attn.register_forward_hook(capture(f"attn.{layer_id}"))
    layers[2].attn.register_forward_hook(capture_input("attn.in.2"))
    for layer_id in args.engram_layer_ids:
        layers[layer_id].engram.register_forward_hook(capture(f"engram.{layer_id}"))
        layers[layer_id].engram.register_forward_hook(capture_input(f"engram.in.{layer_id}"))
    layers[0].ffn.register_forward_hook(capture("ffn.0"))
    if bf16:
        # Every layer's mixture, in and out, where the dtype is what is being measured: a bf16
        # difference that is systematic across tokens sits in one op, and these localize it.
        for layer_id in range(args.n_layers):
            layers[layer_id].ffn.register_forward_hook(capture_input(f"ffn.in.{layer_id}"))
            if layer_id:
                layers[layer_id].ffn.register_forward_hook(capture(f"ffn.{layer_id}"))

    # The candidate mask the source publishes, which nothing else records: it is read off the shared
    # runtime right after the layer that writes it, before its consumer masks with it.
    original_select = reference.select_candidate_blocks

    def recording_select(*call):
        result = original_select(*call)
        seams["candidates"] = result.detach().clone()
        return result

    reference.select_candidate_blocks = recording_select

    original_mixes = reference.Block.hc_mixes

    def recording_mixes(self, x, hc_fn, hc_scale, hc_base):
        result = original_mixes(self, x, hc_fn, hc_scale, hc_base)
        part = "attn" if hc_fn is self.hc_attn_fn else "ffn"
        for name, value in zip(("pre", "post", "comb"), result):
            seams[f"hc.{part}.{name}.{self.layer_id}"] = value.detach().clone()
        return result

    reference.Block.hc_mixes = recording_mixes

    # The indexer's ranked scores, when quantizing. fp4 queries against fp4 keys land on few enough
    # values that two positions can tie exactly at the top-k boundary, and a kept position that
    # differs there is a tie broken the other way rather than a defect; only the score says which.
    # `topk` is the only caller inside `Indexer.forward`, so the score is its receiver.
    indexer_scores = {}
    original_topk = torch.Tensor.topk
    if quantizes or bf16:
        pending_score = {}

        def recording_topk(self, k, dim=-1, largest=True, sorted=True):
            pending_score["last"] = self.detach().clone()
            return original_topk(self, k, dim=dim, largest=largest, sorted=sorted)

        torch.Tensor.topk = recording_topk
        for layer_id in (2, 3, 4, 5):
            def take_score(module, inputs, output, layer_id=layer_id):
                if "last" in pending_score:
                    indexer_scores[layer_id] = pending_score.pop("last")
            layers[layer_id].attn.indexer.register_forward_hook(take_score)

    tokens = torch.tensor([[5, 9, 2, 31, 7, 44, 2, 18, 60, 3, 12, 25, 7, 19, 40, 2]])
    states = []
    with torch.inference_mode():
        hashes = net.engram_hash(tokens, 0, None)
        hidden = net.embed(tokens).unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
        pre_mix = reference.make_identity_pre_mix(hidden, args.hc_mult)
        for layer in net.layers:
            if layer.engram is not None:
                hidden = layer.engram(hidden, hashes[:, :, layer.engram.layer_hash_index, :], None)
            states.append(hidden)
            hidden, pre_mix = layer(hidden, 0, pre_mix, None)
        collapsed = net.layers[-1].hc_pre(hidden, pre_mix)
        # `Transformer.forward` keeps only the last position; every one is wanted here.
        logits = net.head(net.norm(collapsed), full_logits=True)

    # The copies of the run above stay within a unit in the last place of one another, because every
    # `post` is near 1 and the stream starts as one copy repeated; which index `comb` sums over is
    # invisible there. This probe hands `hc_post` and `hc_pre` copies that differ, from a private
    # generator, so the rest of the record is unchanged.
    generator = torch.Generator().manual_seed(41)
    probe_block = net.layers[1]
    probe_dtype = hidden.dtype
    with torch.inference_mode():
        probe_residual = torch.randn(1, 4, args.hc_mult, args.dim, generator=generator).to(probe_dtype)
        probe_x = torch.randn(1, 4, args.dim, generator=generator).to(probe_dtype)
        probe_pre, probe_post, probe_comb = original_mixes(
            probe_block, probe_residual, probe_block.hc_attn_fn, probe_block.hc_attn_scale,
            probe_block.hc_attn_base)
        probe_expanded = probe_block.hc_post(probe_x, probe_residual, probe_post, probe_comb)
        probe_reduced = probe_block.hc_pre(probe_expanded, probe_pre)
    probe = {"x": probe_x, "residual": probe_residual, "pre": probe_pre, "post": probe_post,
             "comb": probe_comb, "expanded": probe_expanded, "reduced": probe_reduced}

    torch.Tensor.topk = original_topk
    reference.Block.hc_mixes = original_mixes
    reference.select_candidate_blocks = original_select
    reference.ParallelEngramEmbedding.forward = original_lookup
    if served is not None:
        reference.linear = unserved_linear

    extra = {"tokens": tokens[0].to(torch.int32).contiguous(),
             "collapsed": collapsed[0].float().contiguous(),
             "engram.hashes": hashes[0].to(torch.int32).contiguous(),
             "engram.multipliers": net.engram_hash.multipliers.contiguous(),
             "engram.primes": net.engram_hash.primes.to(torch.int64).contiguous(),
             "engram.offsets": net.engram_hash.offsets.to(torch.int64).contiguous()}
    for name, value in seams.items():
        # A captured input is often a view of a tensor recorded elsewhere, and safetensors refuses
        # aliased storage, so each seam is cloned rather than merely made contiguous.
        tensor = value.detach()
        extra[f"seam.{name}"] = (tensor.float() if tensor.is_floating_point()
                                 else tensor.to(torch.int32))[0].clone().contiguous()
    # Each parameter's dtype as the reference's own constructor assigned it: 1 for float32, 0 for
    # anything narrower. The port's rule for what a bf16 decoder holds float32 is measured against
    # this rather than against a list written down on both sides.
    if bf16:
        for key, value in net.state_dict().items():
            if value.is_floating_point():
                extra[f"dtype::{key}"] = torch.tensor([1 if value.dtype == torch.float32 else 0],
                                                      dtype=torch.int32)
    for name, value in probe.items():
        extra[f"probe.hc.{name}"] = value[0].float().clone().contiguous()
    for layer_id, value in indexer_scores.items():
        extra[f"seam.score.{layer_id}"] = torch.nan_to_num(value[0].float(), neginf=-1e30).contiguous()
    for index, state in enumerate(states):
        extra[f"hidden.{index}"] = state[0].float().contiguous()
    extra[f"hidden.{len(states)}"] = hidden[0].float().contiguous()
    for key, value in net.state_dict().items():
        extra[f"w::{key}"] = (value.float() if value.is_floating_point() else value).contiguous()
    globals()["_extra"] = extra
    return logits[0].float().contiguous()


def run_deepseek_v41_decode_bf16(image):
    """The same decode, in bf16 as the release runs it: every carried buffer is then bf16 too.

    The index heads are 32 wide, as in every bf16 record. Run under the gemma environment.
    """
    return run_deepseek_v41_decode(image, bf16=True)


def run_deepseek_v41_decode(image, bf16=False):
    """DeepSeek V4.1 decoding ONE TOKEN AT A TIME, with the state each step carries.

    Prefill is whole-sequence tensor work; decode is not. The reference keeps five pieces of state
    across steps — the sliding-window ring, the shared compressed cache, the index keys, the
    compressor's parked partial group, and the n-gram id history — and every one of them is a buffer
    indexed by ABSOLUTE position rather than by a chunk. This records all of them after every step,
    so a port is held to each mechanism separately instead of to a logit at the end.

    The prompt is 11 tokens on purpose: odd, so a ratio-2 group parks a partial at the boundary;
    longer than the window of 8, so the ring has already wrapped; and longer than the n-gram
    look-back, so that look-back crosses the prefill/decode split.

    ONE LINE OF THE REFERENCE IS CORRECTED HERE, and a port matching the uncorrected version would
    be matching a bug. `Indexer.forward` publishes `shared_attn.index_k` only when its compressor
    emitted a latent, while `_compress_kv` publishes `shared_attn.compress_kv` unconditionally. On a
    step where a ratio-2 compressor emits nothing, the ratio-2 layers therefore score against
    whatever layer published last — measured: at `start_pos` 12 they score against the ratio-1
    layer's `k_cache` from the previous step, at a different stride. The compressed key-value they
    then read is correct, so the effect is a silently degraded choice of positions on alternate
    steps. The publish is hoisted out of the latent check, which is what `compress_kv` already does.
    """
    import torch

    _deepseek_v41_kernel_shim()
    source = os.path.expanduser(os.environ.get(
        "IK_DEEPSEEK_V41_SRC", "~/.inferkit-validation/reference-sources/deepseek-v41"))
    sys.path.insert(0, source)
    import model as reference

    # The n-gram table's lookup casts its dequantized rows to bf16, which is the storage rounding the
    # kernel shim neutralizes elsewhere; at a float configuration it is also a dtype mismatch against
    # the float `wkv` beside it. Dropped for the same reason and in the same way as in the decoder
    # oracle: what is measured is the unquantized model.
    def unrounded_lookup(self, indices):
        import torch.nn.functional as F
        values = F.embedding(indices, self.weight)
        scales = F.embedding(indices, self.scale)
        values = values.float().unflatten(-1, (-1, self.block_size)) * scales.float().unsqueeze(-1)
        # A bf16 net keeps the release's cast, since its `wkv` reads bf16.
        return values.flatten(-2).to(torch.bfloat16) if bf16 else values.flatten(-2)

    original_lookup = reference.ParallelEngramEmbedding.forward
    reference.ParallelEngramEmbedding.forward = unrounded_lookup
    # `Indexer.forward` computes the combined score and does not return it. It is the only caller of
    # `Tensor.topk` in that function, so the score is captured as topk's own receiver.
    original_topk = torch.Tensor.topk
    pending_score = {}

    def recording_topk(self, k, dim=-1, largest=True, sorted=True):
        pending_score["last"] = self.detach().clone()
        return original_topk(self, k, dim=dim, largest=largest, sorted=sorted)

    original_indexer = reference.Indexer.forward
    chosen, scored = {}, {}

    def publishing_indexer(self, x, qr, latent, start_pos, offset):
        # BEFORE the call, not after. `Indexer.forward` reads `shared_attn.index_k` itself, so an
        # owner that publishes only on its way out still scores its OWN step against whatever the
        # last owner left. Publishing first also covers the emitting case unchanged: the write at
        # `k_cache[...] = k` is in place, and this binds that same buffer.
        if self.owns_k:
            reference.shared_attn.index_k = self.k_cache
        result = original_indexer(self, x, qr, latent, start_pos, offset)
        # Which compressed positions this indexer kept. State can agree while the CHOICE does not,
        # and the choice is what the attention then reads. The SCORES go with it: a choice that
        # differs where the scores agree is a tie, and a choice that differs where they do not is a
        # defect, and nothing short of the scores tells those apart.
        chosen[self.layer_tag] = result.detach().clone()
        if "last" in pending_score:
            scored[self.layer_tag] = pending_score.pop("last")
        return result

    reference.Indexer.forward = publishing_indexer
    torch.Tensor.topk = recording_topk

    class _DistinctBackend:
        def decode(self, ids, skip_special_tokens=False):
            return f"tok{ids[0]}"

        def id_to_token(self, token_id):
            return f"tok{token_id}"

    class _DistinctTokenizer:
        def __init__(self, size):
            self._size = size
            self.backend_tokenizer = _DistinctBackend()

        def __len__(self):
            return self._size

    args = reference.ModelArgs(
        max_batch_size=1, max_seq_len=64, dtype="bf16", expert_dtype=None, temperature=0,
        vocab_size=256, dim=64, moe_inter_dim=32, n_layers=6, n_mtp_layers=0,
        n_heads=4, n_routed_experts=8, n_activated_experts=2,
        q_lora_rank=32, head_dim=32, rope_head_dim=8, o_groups=2, o_lora_rank=16,
        window_size=8, compress_ratios=(0, 0, 2, 2, 1, 1),
        kv_source_layers=(2, 4), index_source_layers=(2, 3, 4, 5),
        index_n_heads=8, index_head_dim=32 if bf16 else 16, index_topk=4,
        candidate_source_layer=4, candidate_topk_blocks=3, candidate_block_size=2,
        engram_layer_ids=(1, 3), engram_num_embeddings=(290, 370),
        engram_max_ngram_size=3, engram_n_heads=2, engram_head_dim=32,
        engram_vocab_size=64, engram_compressed_vocab_size=256, engram_pad_id=2,
        hc_mult=3, hc_sinkhorn_iters=4,
    )
    torch.manual_seed(0)
    if bf16:
        net = _deepseek_v41_bf16(
            reference, lambda: reference.Transformer(args, _DistinctTokenizer(args.vocab_size)))
    else:
        net = reference.Transformer(args, _DistinctTokenizer(args.vocab_size)).float()
        net = _randomized(net)
    for index, layer in enumerate(net.layers):
        if layer.attn.indexer is not None:
            layer.attn.indexer.layer_tag = index

    prompt = torch.tensor([[5, 9, 2, 31, 7, 44, 2, 18, 60, 3, 12]])
    extra = {"prompt": prompt[0].to(torch.int32).contiguous()}

    def record(tag):
        """Every buffer a decode step carries, after the step that wrote them."""
        for index, layer in enumerate(net.layers):
            attn = layer.attn
            extra[f"{tag}.window.{index}"] = attn.window_kv_cache[0].float().clone().contiguous()
            if attn.compressor is not None and attn.compressor.compress_ratio > 1:
                extra[f"{tag}.kv_state.{index}"] = attn.compressor.kv_state[0].float().clone().contiguous()
                extra[f"{tag}.score_state.{index}"] = torch.nan_to_num(
                    attn.compressor.score_state[0].float(), neginf=-1e30).clone().contiguous()
            if attn.is_kv_source:
                extra[f"{tag}.compress.{index}"] = attn.compress_kv_cache[0].float().clone().contiguous()
            if attn.indexer is not None and attn.indexer.owns_k:
                extra[f"{tag}.index_k.{index}"] = attn.indexer.k_cache[0].float().clone().contiguous()
        extra[f"{tag}.engram_ids"] = net.engram_hash.cache[0].to(torch.int32).clone().contiguous()
        for index, picked in chosen.items():
            extra[f"{tag}.chosen.{index}"] = picked[0].to(torch.int32).contiguous()
        for index, value in scored.items():
            extra[f"{tag}.score.{index}"] = torch.nan_to_num(
                value[0].float(), neginf=-1e30).contiguous()
        chosen.clear()
        scored.clear()

    with torch.inference_mode():
        committed, logits, _ = net(prompt, start_pos=0)
        extra["prefill.logits"] = logits[0].float().contiguous()
        record("prefill")
        tokens = [int(committed[0])]
        for step in range(3):
            position = prompt.size(1) + step
            committed, logits, _ = net(committed.view(1, 1), start_pos=position)
            extra[f"step{step}.logits"] = logits[0].float().contiguous()
            record(f"step{step}")
            tokens.append(int(committed[0]))

    reference.Indexer.forward = original_indexer
    torch.Tensor.topk = original_topk
    reference.ParallelEngramEmbedding.forward = original_lookup
    extra["generated"] = torch.tensor(tokens, dtype=torch.int32)
    for key, value in net.state_dict().items():
        extra[f"w::{key}"] = (value.float() if value.is_floating_point() else value).contiguous()
    globals()["_extra"] = extra
    # Cloned: the harness writes the return value as `output`, and safetensors refuses two names for
    # one buffer.
    return extra["prefill.logits"].clone()


def run_deepseek_v41_dspark(image):
    """DeepSeek V4.1's DSpark draft stack, from the release's own `inference/model.py`."""
    return _run_deepseek_v41_dspark(image)


def run_deepseek_v41_dspark_bf16(image):
    """The same draft loop in bf16, as the release runs it. Run under the gemma environment."""
    return _run_deepseek_v41_dspark(image, bf16=True)


def run_deepseek_v41_dspark_quantized(image):
    """The same draft loop with the release's own activation round trips switched ON.

    A draft stage rounds two key-values to fp8 that the decoder does not: the main stack's states run
    through the stage's own `wkv` for its window, and the drafted block's own. The main decoder that
    seeds it rounds its own four as `deepseek_v41_quantized` measures. The index heads are 32 wide for
    the same reason they are there. Run under the gemma environment for ml_dtypes.
    """
    return _run_deepseek_v41_dspark(image, quantizes=True)


def _run_deepseek_v41_dspark(image, quantizes=False, bf16=False):
    """DeepSeek V4.1's DSpark draft stack, from the release's own `inference/model.py`.

    A draft stage is a decoder block that never compresses, routing over its own smaller set of
    experts. Two things are its own, and both are measured here: the attention, which reads the MAIN
    stack's key-value for its sliding window and the whole drafted block for itself with no causal
    mask between the drafts, and the head, which walks the block one position at a time so each
    drafted token biases the next through a Markov embedding.

    The loop is decode-time, so the record carries both halves of it: a prefill that seeds every
    stage's window from the prompt, then one step that drafts `dspark_block_size` tokens after it.
    `temperature` is zero, which makes the reference's own sampler an argmax and the walk
    reproducible. The Markov rank differs from the hidden width on purpose, so a head that confused
    the two would not line up.
    """
    import torch

    _deepseek_v41_kernel_shim(quantizes=quantizes)
    source = os.path.expanduser(os.environ.get(
        "IK_DEEPSEEK_V41_SRC", "~/.inferkit-validation/reference-sources/deepseek-v41"))
    sys.path.insert(0, source)
    import model as reference

    args = reference.ModelArgs(
        max_batch_size=1, max_seq_len=64, dtype="bf16", expert_dtype=None, temperature=0,
        vocab_size=256, dim=64, moe_inter_dim=32, n_layers=6, n_mtp_layers=2,
        n_heads=4, n_routed_experts=8, n_activated_experts=2,
        q_lora_rank=32, head_dim=32, rope_head_dim=8, o_groups=2, o_lora_rank=16,
        window_size=8, compress_ratios=(0, 0, 2, 2, 1, 1, 0, 0),
        kv_source_layers=(2, 4), index_source_layers=(2, 3, 4, 5),
        index_n_heads=8, index_head_dim=32 if (quantizes or bf16) else 16, index_topk=8, hc_mult=3, hc_sinkhorn_iters=4,
        dspark_block_size=3, dspark_noise_token_id=250, dspark_target_layer_ids=(3, 4, 5),
        dspark_markov_rank=8, dspark_n_routed_experts=4, dspark_n_activated_experts=2,
    )
    torch.manual_seed(0)
    if bf16:
        net = _deepseek_v41_bf16(reference, lambda: reference.Transformer(args), seed=7)
    else:
        net = _randomized(reference.Transformer(args).float(), seed=7)
    unserved_linear = None
    if quantizes:
        # The draft stack has no n-gram memory, so the probe needs no tokenizer either.
        _, unserved_linear = _deepseek_v41_serve_gemms(reference, net, args, None)
    first, last = net.mtp[0], net.mtp[-1]

    # The three heads on their own, at inputs of their own, so a disagreement in the loop below
    # localizes to the walk rather than to a head.
    torch.manual_seed(2)
    width = torch.bfloat16 if bf16 else torch.float32
    main_hidden = torch.randn(
        1, args.dspark_block_size, args.dim * len(args.dspark_target_layer_ids)).to(width)
    hidden = torch.randn(1, args.dspark_block_size, args.dim).to(width)
    tokens = torch.tensor([[17, 5, 200]], dtype=torch.long)
    with torch.inference_mode():
        main_state = first.main_norm(first.main_proj(main_hidden))
        bias, embedded = last.markov_head(tokens[0])
        confidence = last.confidence_head(hidden, embedded.unsqueeze(0))

    # The loop. The prompt runs through the MAIN stack first, because what the draft stack reads is
    # the mean over the hyper-connection copies of the stream entering each target layer's attention.
    # `window_size` is 8 and the prompt is 10, so the draft window is a real ring that has already
    # wrapped rather than the whole prefix.
    prompt = torch.tensor([[5, 9, 2, 31, 7, 44, 2, 18, 60, 3]])
    length = prompt.size(1)
    draft_seams = {}

    def capture(name):
        def hook(module, inputs, output):
            draft_seams[name] = output.detach().clone()
        return hook

    for stage, block in enumerate(net.mtp):
        block.attn.register_forward_hook(capture(f"draft.attn.{stage}"))

    with torch.inference_mode():
        committed, _, prompt_states = net(prompt, start_pos=0)
        net.forward_spec(prompt[:, -1], prompt_states, start_pos=0)   # seeds each stage's window
        _, _, step_states = net(committed.view(1, 1), start_pos=length)
        drafted, draft_logits, draft_confidence = net.forward_spec(
            committed, step_states, start_pos=length)

    extra = {
        "main_hidden": main_hidden[0].float().contiguous(),
        "hidden": hidden[0].float().contiguous(),
        "tokens": tokens[0].to(torch.int32).contiguous(),
        "main_state": main_state[0].float().contiguous(),
        "markov_embed": embedded.float().contiguous(),
        "confidence": confidence[0].float().contiguous(),
        "noise_token": torch.tensor([args.dspark_noise_token_id], dtype=torch.int32),
        "block_size": torch.tensor([args.dspark_block_size], dtype=torch.int32),
        # The draft stack reads the main stack at positions 0..length; the prefill produced the
        # first `length` of those and the step produced the last, and their concatenation is what a
        # prefill of the whole committed sequence would give.
        "loop.prompt": prompt[0].to(torch.int32).contiguous(),
        "loop.committed": committed.to(torch.int32).contiguous(),
        "loop.main_states": torch.cat([prompt_states, step_states], dim=1)[0].float().contiguous(),
        "loop.drafted": drafted[0].to(torch.int32).contiguous(),
        "loop.logits": draft_logits[0].float().contiguous(),
        "loop.confidence": draft_confidence[0].float().contiguous(),
    }
    for name, value in draft_seams.items():
        extra[f"seam.{name}"] = value.detach().float()[0].clone().contiguous()
    for key, value in first.main_proj.state_dict().items():
        extra[f"w::main_proj.{key}"] = value.float().clone().contiguous()
    for key, value in first.main_norm.state_dict().items():
        extra[f"w::main_norm.{key}"] = value.float().clone().contiguous()
    for key, value in last.markov_head.state_dict().items():
        extra[f"w::markov.{key}"] = value.float().clone().contiguous()
    for key, value in last.confidence_head.state_dict().items():
        extra[f"w::confidence.{key}"] = value.float().clone().contiguous()
    # The whole stack, in the release's naming. `mtp.<n>.embed` and `mtp.<n>.head` alias the main
    # model's own, so they are skipped rather than written twice.
    import re
    for key, value in net.state_dict().items():
        if re.match(r"mtp\.\d+\.(embed|head)\.weight$", key):
            continue
        # The WHOLE model, decoder included. The draft stack reads the main stack's own states, so a
        # port that only has the `mtp.` weights can be held to the reference's recorded states but
        # not to states it produced itself, which is the half that says the two are connected.
        extra[f"w::stack.{key}"] = (value.float() if value.is_floating_point()
                                    else value).contiguous()
    # Each parameter's dtype as the reference's constructor assigned it, 1 for float32, which is
    # what the port's rule for the draft stack is held to, as the decoder's is.
    if bf16:
        for key, value in net.state_dict().items():
            if value.is_floating_point():
                extra[f"dtype::{key}"] = torch.tensor([1 if value.dtype == torch.float32 else 0],
                                                      dtype=torch.int32)
    if unserved_linear is not None:
        reference.linear = unserved_linear
    globals()["_extra"] = extra
    return bias.float().contiguous()


def run_deepseek_v41_vision_bf16(image):
    """The same tower and aligner in bf16, as the release runs them. Run under the gemma environment."""
    return run_deepseek_v41_vision(image, bf16=True)


def run_deepseek_v41_vision(image, bf16=False):
    """DeepSeek V4.1's image tower and aligner, from the release's own `inference/vision.py`.

    The tower needs none of the substitute kernels the decoder does: `vision.py` imports torch and
    nothing else, and the released tower is the one part of this checkpoint that ships unquantized.
    A grid whose sides are NOT multiples of the downsample ratio is chosen on purpose, so the
    aligner's padding runs; a grid that divides evenly would pass with the padding dropped.
    """
    import torch

    source = os.path.expanduser(os.environ.get(
        "IK_DEEPSEEK_V41_SRC", "~/.inferkit-validation/reference-sources/deepseek-v41"))
    sys.path.insert(0, source)
    _deepseek_v41_kernel_shim()
    import model as reference
    import vision as reference_vision

    args = reference.ModelArgs(
        dim=64, vision_n_layers=3, vision_dim=32, vision_n_heads=4, vision_inter_dim=48,
        vision_patch_size=4, vision_rope_theta=10000.0, vision_downsample_ratio=3,
    )
    torch.manual_seed(0)
    if bf16:
        tower = _deepseek_v41_bf16(reference, lambda: reference_vision.ViT(args), seed=5)
        aligner = _deepseek_v41_bf16(reference, lambda: reference_vision.Aligner(args), seed=6)
    else:
        tower = _randomized(reference_vision.ViT(args).float(), seed=5)
        aligner = _randomized(reference_vision.Aligner(args).float(), seed=6)

    rows, columns = 5, 7                       # neither divides by 3, so the aligner pads both axes
    torch.manual_seed(3)
    patches = torch.randn(rows * columns, 3, args.vision_patch_size, args.vision_patch_size)
    seams = {}
    if bf16:
        patches = patches.to(torch.bfloat16)
        # Every step of every block, where a bf16 rounding can land.
        def capture(name):
            def hook(module, inputs, output):
                seams[name] = output.detach().clone()
            return hook
        tower.patch_embed.register_forward_hook(capture("patch_embed"))
        for index, block in enumerate(tower.blocks):
            for name in ("norm1", "attn", "norm2", "mlp"):
                getattr(block, name).register_forward_hook(capture(f"block{index}.{name}"))
            block.attn.wqkv.register_forward_hook(capture(f"block{index}.wqkv"))
            block.mlp.w1.register_forward_hook(capture(f"block{index}.w1"))
            block.register_forward_hook(capture(f"block{index}"))
        aligner.w1.register_forward_hook(capture("aligner.w1"))
    # `vision.py` calls `F.scaled_dot_product_attention`, whose bf16 arithmetic belongs to the
    # backend rather than to the model: the CPU flash kernel exponentiates through a cubic
    # polynomial (`fexp_u20`) and rounds the softmax numerators to bf16 before the value product,
    # and a CUDA kernel differs again. A bf16 record is therefore taken on torch's own MATH backend,
    # the definition (float32 throughout, one rounding), and the default backend's result is kept
    # beside it so the difference between the two is a measured figure.
    from contextlib import nullcontext
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with torch.inference_mode(), (sdpa_kernel(SDPBackend.MATH) if bf16 else nullcontext()):
        features = tower(patches, rows, columns)
        aligned = aligner(features, rows, columns)
    if bf16:
        kept = dict(seams)
        with torch.inference_mode():
            default_features = tower(patches, rows, columns)
            default_aligned = aligner(default_features, rows, columns)
        seams.clear()
        seams.update(kept)

    extra = {"patches": patches.float().contiguous(),
             "rows": torch.tensor([rows], dtype=torch.int32),
             "columns": torch.tensor([columns], dtype=torch.int32),
             "features": features.float().contiguous()}
    for key, value in tower.state_dict().items():
        extra[f"w::vision.{key}"] = value.float().contiguous()
    for key, value in aligner.state_dict().items():
        extra[f"w::aligner.{key}"] = value.float().contiguous()
    for name, value in seams.items():
        extra[f"seam.{name}"] = value.float().contiguous()
    if bf16:
        extra["default_sdpa.features"] = default_features.float().contiguous()
        extra["default_sdpa.aligned"] = default_aligned.float().contiguous()
        for prefix, module in (("vision", tower), ("aligner", aligner)):
            for key, value in module.state_dict().items():
                extra[f"dtype::{prefix}.{key}"] = torch.tensor(
                    [1 if value.dtype == torch.float32 else 0], dtype=torch.int32)
    globals()["_extra"] = extra
    return aligned.float().contiguous()


def run_deepseek_v41_image(image):
    """DeepSeek V4.1's image PREPROCESSOR, from the release's own `inference/image_processor.py`.

    `deepseek_v41_vision` measures the tower and the aligner starting from patches, which leaves the
    step that produces those patches unmeasured: solving the resize ratio, padding to the patch
    grid, normalizing, and cutting the result into patches. That step decides the grid every later
    shape follows from, so a port that got it wrong would feed a correct tower the wrong picture.

    The input is a deterministic RGB array rather than a file, encoded to PNG only so the release's
    own `load_image` runs its real path. PNG is lossless, so the pixels recorded here are exactly
    the ones the reference resized, and a port reproducing this reads the same pixels without
    needing to agree about PNG decoding.

    Two sizes are recorded. The first is an ordinary picture whose sides are not multiples of the
    patch size, so the pad runs and the token grid is not the naive ratio. The second is wide enough
    to trip `vision_max_wh_ratio`, which takes the other branch: a plain resize with no padding.
    """
    import io

    import numpy as np
    import torch
    from PIL import Image

    source = os.path.expanduser(os.environ.get(
        "IK_DEEPSEEK_V41_SRC", "~/.inferkit-validation/reference-sources/deepseek-v41"))
    sys.path.insert(0, source)
    _deepseek_v41_kernel_shim()
    import model as reference
    import image_processor as processor

    args = reference.ModelArgs(
        dim=64, vision_n_layers=3, vision_dim=32, vision_n_heads=4, vision_inter_dim=48,
        vision_patch_size=4, vision_rope_theta=10000.0, vision_downsample_ratio=3,
    )
    args.vision_max_n_token = 64
    args.vision_min_pixels = 16 * 16
    args.vision_max_wh_ratio = 8

    extra = {"patch_size": torch.tensor([args.vision_patch_size], dtype=torch.int32),
             "downsample_ratio": torch.tensor([args.vision_downsample_ratio], dtype=torch.int32),
             "max_n_token": torch.tensor([args.vision_max_n_token], dtype=torch.int32),
             "min_pixels": torch.tensor([args.vision_min_pixels], dtype=torch.int32),
             "max_wh_ratio": torch.tensor([args.vision_max_wh_ratio], dtype=torch.int32)}

    first = None
    for tag, (width, height) in {"a": (37, 53), "b": (200, 19)}.items():
        rng = np.random.default_rng(7 if tag == "a" else 11)
        pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(pixels, mode="RGB").save(buffer, format="PNG")
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = processor.load_image(
            {"data": buffer.getvalue()}, args)
        types = processor.image_token_types(n_llm_h, n_llm_w)

        extra[f"{tag}.pixels"] = torch.from_numpy(pixels.astype(np.float32)).contiguous()
        extra[f"{tag}.size"] = torch.tensor([width, height], dtype=torch.int32)
        extra[f"{tag}.grid"] = torch.tensor([n_vit_h, n_vit_w, n_llm_h, n_llm_w], dtype=torch.int32)
        extra[f"{tag}.patches"] = patches.float().contiguous()
        extra[f"{tag}.types"] = types.to(torch.int32).contiguous()
        if first is None:
            first = patches.float().contiguous()

    globals()["_extra"] = extra
    return first


def run_qwen4_exp(image):
    """The Qwen4-Exp decoder's arithmetic, from transformers' own Qwen4ExpForCausalLM, at a tiny
    random configuration.

    Qwen3.8-Flash-Next is the released instance at 180B — 360 GB of bfloat16, which no machine here
    holds — so the four mechanisms this architecture adds to the hybrid family are measured at a size
    that runs: hyper-connections carrying the residual stream `hc_count` times over, a per-layer
    embedding over hashed n-grams, the query-sparse-attention indexer that picks which earlier tokens
    a query may see, and a mixture of experts with a shared expert beside it.

    The configuration is chosen so that every one of them bites. Twelve tokens over a compression
    ratio of 2 give six index blocks against a budget of two, so the indexer discards rather than
    admitting everything; the token ids repeat the end-of-sequence id twice, so the n-gram hash has
    to refuse to read across a segment boundary; and the sequence is not a multiple of the ratio, so
    the tail that fills no block is exercised.

    Seam records accompany the logits: the hashed n-gram row indices, the per-layer embedding's
    output, the hyper-connection read and its write shares, the indexer's own mask, and each branch's
    output. A divergence localizes to a mechanism rather than to the stack.

    `IK_QWEN4_INDEXER_HEADS` widens the indexer (default 2). Its scores sum RECTIFIED per-head products,
    so with 2 heads a quarter of the query/block pairs score exactly zero and tie; which of the tied
    blocks `torch.topk` keeps is a partial-sort artifact rather than a rule. At 8 heads a tie needs all
    eight rectified to zero (p = 1/256), so the selection is determined and a match means something.
    """
    import torch
    from transformers import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

    indexer_heads = int(os.environ.get("IK_QWEN4_INDEXER_HEADS", "2"))

    config = Qwen4ExpTextConfig(
        hidden_size=32, num_hidden_layers=4, vocab_size=128,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25, "mrope_section": [2, 1, 1],
                         "mrope_interleaved": True},
        indexer_n_heads=indexer_heads, indexer_kv_heads=1, indexer_head_dim=8,
        indexer_budget=4, indexer_compress_ratio=2,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
        hc_count=3, hc_lowrank=8,
        num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=16, shared_expert_intermediate_size=16,
        ple_layer_ids=[1], ple_embed_dim=32, ple_conv_kernel_size=4,
        ngram_size=3, heads_per_ngram=2, ngram_vocab_size_base=1000,
        make_ngram_vocab_size_divisible_by=8, seed=1234, split_ngram_parts=4,
        output_gate_type="sigmoid", hidden_act="silu",
        eos_token_id=2, bos_token_id=1, tie_word_embeddings=False,
        full_attention_interval=4,
    )
    model = _randomized(Qwen4ExpForCausalLM(config))
    layers = model.model.layers
    seams = {}

    def capture(name, pick=lambda output: output):
        def hook(module, inputs, output):
            seams[name] = pick(output)
        return hook

    def capture_input(name):
        def hook(module, inputs, output):
            seams[name] = inputs[0]
        return hook

    layers[0].ple.ple_embedding.ngram_embedding.register_forward_hook(capture_input("ngram_ids"))
    layers[0].ple.ple_embedding.register_forward_hook(capture("ngram_features"))
    layers[0].ple.register_forward_hook(capture("ple"))
    layers[0].attn_hyper_connection.register_forward_hook(capture("hc_read", lambda o: o[0]))
    layers[0].attn_hyper_connection.register_forward_hook(capture("hc_shares", lambda o: o[2]))
    layers[0].linear_attn.register_forward_hook(capture("linear_attn"))
    layers[0].mlp.register_forward_hook(capture("moe"))
    layers[3].self_attn.indexer.register_forward_hook(capture("indexer_mask"))
    layers[3].self_attn.register_forward_hook(capture("attn", lambda o: o[0]))

    # Two end-of-sequence ids inside the sequence, so the n-gram hash meets a segment boundary.
    tokens = torch.tensor([[5, 9, 2, 31, 7, 44, 2, 18, 60, 3, 12, 25]])
    logits = _tiny_decoder_record(
        model, tokens,
        release_name=lambda key: ("model.language_model." + key[len("model."):]
                                  if key.startswith("model.") else key))
    extra = globals()["_extra"]
    for name, value in seams.items():
        tensor = value.detach()
        extra[f"seam.{name}"] = (tensor.float() if tensor.is_floating_point()
                                 else tensor.to(torch.int32)).contiguous()[0]
    globals()["_extra"] = extra
    return logits


def run_clip_text(image):
    """CLIP ViT-B/32 text embedding, L2-normalized, through transformers.

    The record carries `tokens` (the ids the reference tokenized) so the Swift side embeds exactly the
    same sequence — the port takes ids rather than text, because a byte-level BPE vocabulary is a
    load-time artifact rather than part of the network.
    """
    from transformers import CLIPModel, CLIPTokenizer

    name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(name).eval()
    tokenizer = CLIPTokenizer.from_pretrained(name)
    encoded = tokenizer(["a photograph of a dog running on grass"], padding="max_length",
                        max_length=77, return_tensors="pt")
    with torch.no_grad():
        features = model.get_text_features(**encoded)
    features = features / features.norm(dim=-1, keepdim=True)
    globals()["_extra"] = {"tokens": encoded["input_ids"][0].to(torch.int32).contiguous()}
    return features[0].contiguous()


def run_siggraph17(image, checkpoint):
    """siggraph17 colorization, `[H, W, 3]` sRGB in 0...1, from richzhang's own `siggraph17.py`.

    IK_REF_SRC holds `siggraph17.py` and `base_color.py`. The hint and its mask are left empty, which
    is how the reference colorizes automatically. The record's `ab` carries the network's own output
    before the lightness goes back, so a mismatch says network or Lab conversion rather than both.
    """
    import types
    from skimage import color

    ipython = types.ModuleType("IPython")
    ipython.embed = lambda *args, **kwargs: None
    sys.modules["IPython"] = ipython

    _import_reference(_reference_source(), "siggraph_ref", "base_color")
    module = _import_reference(_reference_source(), "siggraph_ref", "siggraph17")
    model = module.SIGGRAPHGenerator().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)

    lab = color.rgb2lab(image).astype(np.float32)
    lightness = torch.from_numpy(lab[:, :, :1]).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        ab = model(lightness)                                   # already unnormalized by the model
    colorized = color.lab2rgb(np.concatenate([lab[:, :, :1], ab[0].permute(1, 2, 0).numpy()], axis=2))

    globals()["_extra"] = {"lightness": torch.from_numpy(lab[:, :, 0]).contiguous(),
                           "ab": ab[0].permute(1, 2, 0).contiguous()}
    return torch.from_numpy(colorized.astype(np.float32)).contiguous()


def run_bisenetv2(image, checkpoint):
    """BiSeNetV2 class logits, `[H, W, classes]`, from CoinCheung's own `bisenetv2.py`.

    The released `model_final_v2.pth` predates the repository's current `SegmentHead`: it emits
    `classes × upFactor²` channels and pixel-shuffles, where master now emits `classes` and
    interpolates. Everything before the heads is unchanged, so the head is replaced here rather than
    hunting a historical revision — and the substitution is visible instead of buried in a pinned
    commit. The auxiliary heads are training-only and are not built.
    """
    import types
    from torch import nn

    source = os.path.join(_reference_source(), "bisenet")
    module = _import_reference(source, "bisenetv2_ref", "bisenetv2")

    class ShuffleHead(nn.Module):
        """The head the checkpoint was trained with."""

        def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
            super().__init__()
            self.conv = module.ConvBNReLU(in_chan, mid_chan, 3, stride=1)
            self.conv_out = nn.Sequential(
                nn.Conv2d(mid_chan, n_classes * up_factor ** 2, 1, 1, 0, bias=True),
                nn.PixelShuffle(up_factor))

        def forward(self, x):
            return self.conv_out(self.conv(x))

    module.SegmentHead = ShuffleHead
    model = module.BiSeNetV2(n_classes=19, aux_mode="eval")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    model.load_state_dict({k: v for k, v in state.items() if not k.startswith("aux")}, strict=True)
    model.eval()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    deviation = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = ((torch.from_numpy(image).permute(2, 0, 1) - mean) / deviation).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)[0]
    return logits[0].permute(1, 2, 0).contiguous()               # [H, W, classes] to match NHWC


def run_rife_v4(image, checkpoint):
    """RIFE v4 interpolation, `[H, W, 3]` in 0...1.

    IK_REF_SRC holds a `rife4/` directory with `rife_arch.py` — a vendored architecture covering the
    v4 generations, used because the version's own `IFNet.py` ships inside the model zip rather than
    in the repository. The released `rife-flownet-4.13.2` weights match its `4.17` branch: four
    blocks, a `Head_417` encoder, ResConv trunk entries, and a shuffled upsampling convolution. Both
    sides run four scales `[8, 4, 2, 1]` and pad to a multiple of 64.
    """
    import os

    import types

    # The vendored file imports one ComfyUI helper at module scope for device selection; nothing on
    # the forward path uses it, so a stub is enough to import the architecture.
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    management = types.ModuleType("comfy.model_management")
    management.get_torch_device = lambda: torch.device("cpu")
    sys.modules["comfy"], sys.modules["comfy.model_management"] = comfy, management

    sys.path.insert(0, os.path.join(_reference_source(), "rife4"))
    from rife_arch import IFNet
    from safetensors.torch import load_file

    model = IFNet(arch_ver="4.17")
    model.load_state_dict(load_file(checkpoint), strict=True)
    model.eval()

    shift = (4, 6)
    second = np.roll(np.roll(image, shift[0], axis=0), shift[1], axis=1)
    frames = [torch.from_numpy(np.ascontiguousarray(plate)).permute(2, 0, 1)[None]
              for plate in (image, second)]
    with torch.no_grad():
        merged = model(frames[0], frames[1], timestep=0.5, scale_list=[8, 4, 2, 1],
                       training=False, fastmode=True, ensemble=False)
    result = merged if torch.is_tensor(merged) else merged[-1]
    globals()["_extra"] = {"frame1": torch.from_numpy(np.ascontiguousarray(second)).contiguous()}
    return result[0].permute(1, 2, 0).contiguous()


def run_sam2_encoder(image, checkpoint):
    """SAM 2's Hiera image encoder, from facebookresearch's own sources.

    The `sam2` package cannot be installed here (it requires Python >= 3.10), so IK_REF_SRC holds a
    `sam2/` directory with the six files the image encoder needs, which do parse under 3.9;
    `iopath` and `sam2.utils.misc` are stubbed because only the video path and an optional checkpoint
    loader reach them. The record carries the finest two FPN levels plus `output` as the vision
    features the mask decoder reads — three seams, so a trunk mismatch and a neck mismatch are told
    apart.
    """
    import types

    for name in ["iopath", "iopath.common"]:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    file_io = types.ModuleType("iopath.common.file_io")
    file_io.g_pathmgr = None
    sys.modules["iopath.common.file_io"] = file_io
    misc = types.ModuleType("sam2.utils.misc")
    misc.mask_to_box = lambda mask: mask
    sys.modules["sam2.utils.misc"] = misc

    sys.path.insert(0, _reference_source())
    from sam2.modeling.backbones.hieradet import Hiera
    from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
    from sam2.modeling.position_encoding import PositionEmbeddingSine

    # IK_SAM2_VARIANT selects the released size. Each one's numbers come from its own config: the
    # base_plus config sets only the width and heads, so the rest are the Hiera defaults, and large
    # overrides every axis.
    variant = os.environ.get("IK_SAM2_VARIANT", "tiny")
    geometry = {
        "tiny":      dict(embed_dim=96,  num_heads=1, stages=(1, 2, 7, 2),
                          global_att_blocks=(5, 7, 9), window_spec=(8, 4, 14, 7),
                          window_pos_embed_bkg_spatial_size=(7, 7)),
        "small":     dict(embed_dim=96,  num_heads=1, stages=(1, 2, 11, 2),
                          global_att_blocks=(7, 10, 13), window_spec=(8, 4, 14, 7),
                          window_pos_embed_bkg_spatial_size=(7, 7)),
        "base_plus": dict(embed_dim=112, num_heads=2, stages=(2, 3, 16, 3),
                          global_att_blocks=(12, 16, 20), window_spec=(8, 4, 14, 7),
                          window_pos_embed_bkg_spatial_size=(14, 14)),
        "large":     dict(embed_dim=144, num_heads=2, stages=(2, 6, 36, 4),
                          global_att_blocks=(23, 33, 43), window_spec=(8, 4, 16, 8),
                          window_pos_embed_bkg_spatial_size=(7, 7)),
    }[variant]
    channels = {"tiny": [768, 384, 192, 96], "small": [768, 384, 192, 96], "base_plus": [896, 448, 224, 112],
                "large": [1152, 576, 288, 144]}[variant]
    trunk = Hiera(**geometry)
    neck = FpnNeck(position_encoding=PositionEmbeddingSine(num_pos_feats=256), d_model=256,
                   backbone_channel_list=channels, fpn_top_down_levels=[2, 3],
                   fpn_interp_model="nearest")
    encoder = ImageEncoder(trunk=trunk, neck=neck, scalp=1).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    encoder.load_state_dict({k[len("image_encoder."):]: v for k, v in state.items()
                             if k.startswith("image_encoder.")}, strict=True)

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        out = encoder(tensor)
    levels = out["backbone_fpn"]
    globals()["_extra"] = {"level0": levels[0][0].permute(1, 2, 0).contiguous(),
                           "level1": levels[1][0].permute(1, 2, 0).contiguous()}
    return out["vision_features"][0].permute(1, 2, 0).contiguous()      # [64, 64, 256] NHWC


def run_sam2_decoder(image, checkpoint):
    """SAM 2's prompt encoder and mask decoder, from facebookresearch's own sources.

    Runs the encoder first, then a single positive click at the frame's centre, and dumps the raw
    mask logits for all four tokens plus the IoU and object scores. Comparing every token rather than
    the selected mask keeps a token permutation distinguishable from a genuine drift — the diagnostic
    that resolved SAM 1's decoder.
    """
    import types

    for name in ["iopath", "iopath.common"]:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    file_io = types.ModuleType("iopath.common.file_io")
    file_io.g_pathmgr = None
    sys.modules["iopath.common.file_io"] = file_io
    misc = types.ModuleType("sam2.utils.misc")
    misc.mask_to_box = lambda mask: mask
    sys.modules["sam2.utils.misc"] = misc

    sys.path.insert(0, _reference_source())
    from sam2.modeling.backbones.hieradet import Hiera
    from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
    from sam2.modeling.position_encoding import PositionEmbeddingSine
    from sam2.modeling.sam.prompt_encoder import PromptEncoder
    from sam2.modeling.sam.mask_decoder import MaskDecoder
    from sam2.modeling.sam.transformer import TwoWayTransformer

    # IK_SAM2_VARIANT selects the released size. Each one's numbers come from its own config: the
    # base_plus config sets only the width and heads, so the rest are the Hiera defaults, and large
    # overrides every axis.
    variant = os.environ.get("IK_SAM2_VARIANT", "tiny")
    geometry = {
        "tiny":      dict(embed_dim=96,  num_heads=1, stages=(1, 2, 7, 2),
                          global_att_blocks=(5, 7, 9), window_spec=(8, 4, 14, 7),
                          window_pos_embed_bkg_spatial_size=(7, 7)),
        "small":     dict(embed_dim=96,  num_heads=1, stages=(1, 2, 11, 2),
                          global_att_blocks=(7, 10, 13), window_spec=(8, 4, 14, 7),
                          window_pos_embed_bkg_spatial_size=(7, 7)),
        "base_plus": dict(embed_dim=112, num_heads=2, stages=(2, 3, 16, 3),
                          global_att_blocks=(12, 16, 20), window_spec=(8, 4, 14, 7),
                          window_pos_embed_bkg_spatial_size=(14, 14)),
        "large":     dict(embed_dim=144, num_heads=2, stages=(2, 6, 36, 4),
                          global_att_blocks=(23, 33, 43), window_spec=(8, 4, 16, 8),
                          window_pos_embed_bkg_spatial_size=(7, 7)),
    }[variant]
    channels = {"tiny": [768, 384, 192, 96], "small": [768, 384, 192, 96], "base_plus": [896, 448, 224, 112],
                "large": [1152, 576, 288, 144]}[variant]
    trunk = Hiera(**geometry)
    neck = FpnNeck(position_encoding=PositionEmbeddingSine(num_pos_feats=256), d_model=256,
                   backbone_channel_list=channels, fpn_top_down_levels=[2, 3],
                   fpn_interp_model="nearest")
    encoder = ImageEncoder(trunk=trunk, neck=neck, scalp=1).eval()
    prompt = PromptEncoder(embed_dim=256, image_embedding_size=(64, 64),
                           input_image_size=(1024, 1024), mask_in_chans=16).eval()
    decoder = MaskDecoder(num_multimask_outputs=3, transformer_dim=256,
                          transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048,
                                                        num_heads=8),
                          use_high_res_features=True, pred_obj_scores=True,
                          pred_obj_scores_mlp=True, use_multimask_token_for_obj_ptr=True).eval()

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    for module, prefix in [(encoder, "image_encoder."), (prompt, "sam_prompt_encoder."),
                           (decoder, "sam_mask_decoder.")]:
        module.load_state_dict({k[len(prefix):]: v for k, v in state.items()
                                if k.startswith(prefix)}, strict=True)

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        out = encoder(tensor)
        levels = out["backbone_fpn"]
        height, width = image.shape[:2]
        point = torch.tensor([[[width / 2.0, height / 3.0]]], dtype=torch.float32)
        labels = torch.tensor([[1]], dtype=torch.int64)
        sparse, dense = prompt(points=(point, labels), boxes=None, masks=None)
        masks, iou, _, object_score = decoder(
            image_embeddings=out["vision_features"], image_pe=prompt.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=True, repeat_image=False,
            # The reference's base model applies these projections before calling the decoder, even
            # though `conv_s0`/`conv_s1` are the decoder's own parameters.
            high_res_features=[decoder.conv_s0(levels[0]), decoder.conv_s1(levels[1])])
    globals()["_extra"] = {"iou": iou[0].contiguous(),
                           "object_score": object_score[0].contiguous(),
                           "sparse": sparse[0].contiguous()}
    return masks[0].contiguous()                                # [tokens, H, W] raw logits


def run_sam2_memory(image, checkpoint):
    """SAM 2's memory encoder and memory attention, from facebookresearch's own sources.

    These are what make SAM 2 a tracker: the encoder folds a frame's features and its predicted mask
    into a compact memory, and the attention conditions the next frame on it. Both are driven here
    with deterministic tensors rather than a real second frame, so the comparison isolates the two
    modules from the tracking loop that would sequence them. The record carries the memory the encoder
    produced (`memory`) and the attention's output (`output`).
    """
    import types

    for name in ["iopath", "iopath.common"]:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    file_io = types.ModuleType("iopath.common.file_io")
    file_io.g_pathmgr = None
    sys.modules["iopath.common.file_io"] = file_io
    misc = types.ModuleType("sam2.utils.misc")
    misc.mask_to_box = lambda mask: mask
    sys.modules["sam2.utils.misc"] = misc

    sys.path.insert(0, _reference_source())
    from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
    from sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock
    from sam2.modeling.position_encoding import PositionEmbeddingSine
    from sam2.modeling.sam.transformer import RoPEAttention

    encoder = MemoryEncoder(out_dim=64,
                            mask_downsampler=MaskDownSampler(kernel_size=3, stride=2, padding=1),
                            fuser=Fuser(CXBlock(dim=256), num_layers=2),
                            position_encoding=PositionEmbeddingSine(num_pos_feats=64)).eval()
    layer = MemoryAttentionLayer(
        activation="relu", dim_feedforward=2048, dropout=0.1,
        pos_enc_at_attn=False, pos_enc_at_cross_attn_keys=True, pos_enc_at_cross_attn_queries=False,
        self_attention=RoPEAttention(embedding_dim=256, num_heads=1, downsample_rate=1, dropout=0.1),
        cross_attention=RoPEAttention(embedding_dim=256, num_heads=1, downsample_rate=1, dropout=0.1,
                                      kv_in_dim=64, rope_k_repeat=True, feat_sizes=(32, 32)),
        d_model=256)
    attention = MemoryAttention(d_model=256, pos_enc_at_input=True, layer=layer, num_layers=4).eval()

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    for module, prefix in [(encoder, "memory_encoder."), (attention, "memory_attention.")]:
        module.load_state_dict({k[len(prefix):]: v for k, v in state.items()
                                if k.startswith(prefix)}, strict=True)

    generator = np.random.default_rng(31)
    features = torch.from_numpy(generator.standard_normal((1, 256, 64, 64)).astype(np.float32))
    # The tracker upsamples the decoder's low-resolution mask to the full frame before encoding it;
    # the downsampler's total stride of 16 is what lands it back on the feature grid.
    mask = torch.from_numpy(generator.standard_normal((1, 1, 1024, 1024)).astype(np.float32))
    with torch.no_grad():
        encoded = encoder(features, mask)
        memory = encoded["vision_features"]                     # [1, 64, 64, 64]
        # The attention works on flattened tokens, memory first as the tracker feeds it.
        current = torch.from_numpy(generator.standard_normal((1, 256, 64, 64)).astype(np.float32))
        # `batch_first=True` describes what the LAYERS want, so `MemoryAttention` takes its inputs
        # SEQUENCE-first and transposes them itself. Handing it batch-first tensors silently makes the
        # tokens the batch, and every token then attends only to itself.
        current_tokens = current.flatten(2).permute(2, 0, 1)     # [tokens, batch, C]
        memory_tokens = memory.flatten(2).permute(2, 0, 1)
        current_pos = torch.from_numpy(generator.standard_normal(current_tokens.shape).astype(np.float32))
        memory_pos = torch.from_numpy(generator.standard_normal(memory_tokens.shape).astype(np.float32))
        out = attention(curr=current_tokens, memory=memory_tokens,
                        curr_pos=current_pos, memory_pos=memory_pos, num_obj_ptr_tokens=0)

    globals()["_extra"] = {"features": features[0].permute(1, 2, 0).contiguous(),
                           "mask": mask[0].permute(1, 2, 0).contiguous(),
                           "memory": memory[0].permute(1, 2, 0).contiguous(),
                           "current": current_tokens[:, 0].contiguous(),
                           "current_pos": current_pos[:, 0].contiguous(),
                           "memory_pos": memory_pos[:, 0].contiguous()}
    return out[:, 0].contiguous()                               # [tokens, 256]


def run_videosr(image, checkpoint):
    """BasicVSR ×4 super-resolution of a three-frame clip, `[T, 4H, 4W, 3]`, from mmediting's own
    `basicvsr_net.py`.

    IK_REF_SRC holds a `basicvsr/` directory with `basicvsr_net.py` and the real common files it draws
    on (`sr_backbone_utils.py`, `flow_warp.py`, `upsample.py`); only mmcv's `ConvModule` (a plain
    convolution + ReLU at SPyNet's settings) and the registry/logger shells are stubbed. The clip is
    three shifted crops of a larger plate, so SPyNet sees genuine translation; the record's `frames`
    carries all three `[T, H, W, 3]` for the Swift side.
    """
    import os
    import types
    from torch import nn

    class ConvModule(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                     norm_cfg=None, act_cfg=dict(type="ReLU")):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.activate = nn.ReLU() if act_cfg else None

        def forward(self, x):
            x = self.conv(x)
            return self.activate(x) if self.activate else x

    mmcv_cnn = types.ModuleType("mmcv.cnn")
    mmcv_cnn.ConvModule = ConvModule
    mmcv_cnn.constant_init = lambda *args, **kwargs: None
    mmcv_cnn.kaiming_init = lambda *args, **kwargs: None
    mmcv_runner = types.ModuleType("mmcv.runner")
    mmcv_runner.load_checkpoint = lambda *args, **kwargs: None
    parrots = types.ModuleType("mmcv.utils.parrots_wrapper")
    parrots._BatchNorm = nn.BatchNorm2d
    registry = types.ModuleType("mmedit.models.registry")
    registry.BACKBONES = type("Registry", (), {"register_module": staticmethod(lambda: (lambda cls: cls))})()
    utils_module = types.ModuleType("mmedit.utils")
    utils_module.get_root_logger = lambda *args, **kwargs: None
    sys.modules.update({"mmcv": types.ModuleType("mmcv"), "mmcv.cnn": mmcv_cnn,
                        "mmcv.runner": mmcv_runner, "mmcv.utils": types.ModuleType("mmcv.utils"),
                        "mmcv.utils.parrots_wrapper": parrots,
                        "mmedit": types.ModuleType("mmedit"), "mmedit.models": types.ModuleType("mmedit.models"),
                        "mmedit.models.registry": registry, "mmedit.utils": utils_module})

    source = os.path.join(_reference_source(), "basicvsr")
    upsample = _import_reference(source, "mmedit.models.common", "upsample",
                                 siblings=["sr_backbone_utils", "flow_warp"])
    common = sys.modules["mmedit.models.common"]
    common.PixelShufflePack = upsample.PixelShufflePack
    common.ResidualBlockNoBN = sys.modules["mmedit.models.common.sr_backbone_utils"].ResidualBlockNoBN
    common.make_layer = sys.modules["mmedit.models.common.sr_backbone_utils"].make_layer
    common.flow_warp = sys.modules["mmedit.models.common.flow_warp"].flow_warp

    module = _import_reference(source, "basicvsr_ref", "basicvsr_net")
    model = module.BasicVSRNet().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict({name[len("generator."):]: value for name, value in state.items()
                           if name.startswith("generator.")}, strict=True)

    # Three 64×64 crops sliding across an 80×80 plate: real translation for SPyNet to find.
    plate = subject_image(80, 80)
    offsets = [(0, 0), (2, 3), (4, 6)]
    frames = np.stack([plate[dy:dy + 64, dx:dx + 64] for dy, dx in offsets])
    clip = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0)     # [1, T, 3, H, W]
    with torch.no_grad():
        upscaled = model(clip)                                           # [1, T, 3, 4H, 4W]
    globals()["_extra"] = {"frames": torch.from_numpy(frames).contiguous()}
    return upscaled[0].permute(0, 2, 3, 1).contiguous()                  # [T, 4H, 4W, 3]


def run_zero_dce(image, checkpoint):
    """Zero-DCE low-light enhancement, `[H, W, 3]` in 0...1, from the reference `model.py`."""
    module = _import_reference(_reference_source(), "zero_dce_ref", "zerodce")
    model = module.enhance_net_nopool().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)

    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        enhanced = model(tensor)[1]
    return enhanced[0].permute(1, 2, 0).contiguous()


def run_style_transfer(image, checkpoint):
    """Fast style transfer, `[H, W, 3]`, from the reference `transformer_net.py`.

    Johnson's network is trained on a 0...255 scale, so the plate is scaled up on the way in and the
    result divided back down — matching what the port does internally.
    """
    module = _import_reference(_reference_source(), "style_ref", "transformer_net")
    model = module.TransformerNet()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # The released weights carry InstanceNorm running statistics that the current layer no longer keeps.
    for name in [k for k in state if k.endswith(("running_mean", "running_var"))]:
        del state[name]
    model.load_state_dict(state, strict=True)
    model.eval()

    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0) * 255.0
    with torch.no_grad():
        stylized = model(tensor)
    return (stylized[0].permute(1, 2, 0) / 255.0).contiguous()


def run_realesrgan(image, checkpoint):
    """Real-ESRGAN ×4, `[4H, 4W, 3]` in 0...1, from BasicSR's `rrdbnet_arch.py`.

    That file reaches for BasicSR's registry and layer helpers; the registry decorator is a no-op here
    and the two helpers are the reference's own one-liners, so nothing beyond the architecture is
    supplied locally.
    """
    import types
    from torch import nn

    registry = types.ModuleType("basicsr.utils.registry")
    registry.ARCH_REGISTRY = type("Registry", (), {"register": staticmethod(lambda: (lambda cls: cls))})()
    basicsr = types.ModuleType("basicsr")
    utils = types.ModuleType("basicsr.utils")
    sys.modules["basicsr"], sys.modules["basicsr.utils"] = basicsr, utils
    sys.modules["basicsr.utils.registry"] = registry

    arch = types.ModuleType("realesrgan_ref.arch_util")
    arch.default_init_weights = lambda *args, **kwargs: None
    arch.make_layer = lambda block, count, **kwargs: nn.Sequential(*[block(**kwargs) for _ in range(count)])

    def pixel_unshuffle(x, scale):
        b, c, h, w = x.size()
        view = x.view(b, c, h // scale, scale, w // scale, scale)
        return view.permute(0, 1, 3, 5, 2, 4).reshape(b, c * scale * scale, h // scale, w // scale)
    arch.pixel_unshuffle = pixel_unshuffle

    module = _import_reference(_reference_source(), "realesrgan_ref", "rrdbnet_arch",
                               injected={"arch_util": arch})
    # The anime release is a six-block generator; the general one is twenty-three.
    blocks = int(os.environ.get("IK_ESRGAN_BLOCKS", "23"))
    model = module.RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=blocks, num_grow_ch=32)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("params_ema", state.get("params", state)), strict=True)
    model.eval()

    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        upscaled = model(tensor)
    return upscaled[0].permute(1, 2, 0).contiguous()


def run_colorizer(image, checkpoint):
    """Colorization (Zhang et al. ECCV-16), `[H, W, 3]` sRGB in 0...1, from the reference `eccv16.py`.

    Feed a plate at the model's own 256×256 so the reference's input resize is an identity and the only
    resampling left in the comparison is the network's own ×4 upsample of the ab prediction.

    The record's `lightness` carries the L channel the reference used, so the Swift side can score the
    network alone as well as the whole chain.
    """
    import types
    from skimage import color

    # `eccv16.py` imports IPython's debugger at module scope and never calls it.
    ipython = types.ModuleType("IPython")
    ipython.embed = lambda *args, **kwargs: None
    sys.modules["IPython"] = ipython

    _import_reference(_reference_source(), "colorizer_ref", "base_color")
    module = _import_reference(_reference_source(), "colorizer_ref", "eccv16")
    model = module.ECCVGenerator().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)

    lab = color.rgb2lab(image).astype(np.float32)
    lightness = torch.from_numpy(lab[:, :, :1]).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        ab = model(lightness)                                   # [1, 2, H, W], unnormalized
    colorized = color.lab2rgb(np.concatenate([lab[:, :, :1], ab[0].permute(1, 2, 0).numpy()], axis=2))

    globals()["_extra"] = {"lightness": torch.from_numpy(lab[:, :, 0]).contiguous(),
                           "ab": ab[0].permute(1, 2, 0).contiguous()}
    return torch.from_numpy(colorized.astype(np.float32)).contiguous()


def run_deeplab(image, checkpoint):
    """torchvision DeepLabV3-ResNet50 class logits at the head's native stride-8 resolution.

    Comparing logits rather than the label map keeps the check sensitive: argmax hides everything except
    a difference big enough to flip a pixel's winner.
    """
    from torchvision.models.segmentation import deeplabv3_resnet50

    model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=21, aux_loss=True).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    deviation = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = ((torch.from_numpy(image).permute(2, 0, 1) - mean) / deviation).unsqueeze(0)
    with torch.no_grad():
        logits = model.classifier(model.backbone(tensor)["out"])
    return logits[0].permute(1, 2, 0).contiguous()              # [h/8, w/8, classes] to match NHWC


def run_vad(image, checkpoint):
    """NVIDIA NeMo frame-level MarbleNet speech probability per frame, `[frames]`.

    `--checkpoint` is the released `.nemo` archive. The record's `waveform` carries the `[samples]` clip
    the Swift side reads back; both sides run their own mel front end over it, so a preprocessing
    difference surfaces rather than hides.
    """
    import huggingface_hub

    # NeMo imports names the Hub dropped after its pin; they are only used by its model-search helpers,
    # not by the restore path, so a placeholder is enough to let the package import.
    for name in ["ModelFilter", "DatasetFilter"]:
        if not hasattr(huggingface_hub, name):
            setattr(huggingface_hub, name, type(name, (), {}))
    import nemo.collections.asr as nemo_asr

    # The installed NeMo builds a loss with a weight buffer the released archive predates; nothing on
    # the inference path reads it, so the restore is not strict about it.
    model = nemo_asr.models.EncDecFrameClassificationModel.restore_from(checkpoint, strict=False).eval()
    # Dither is training-time noise; zero it so the record is reproducible whichever NeMo gates on.
    model.preprocessor.featurizer.dither = 0.0

    samples = 16000
    time = np.arange(samples, dtype=np.float32) / 16000.0
    generator = np.random.default_rng(13)
    # Half a second of a voiced-like harmonic stack, then noise: a clip with a real boundary in it.
    speech = sum(0.3 / (h + 1) * np.sin(2 * np.pi * 150 * (h + 1) * time) for h in range(6))
    wave = np.where(time < 0.5, speech, 0.05 * generator.standard_normal(samples)).astype(np.float32)

    with torch.no_grad():
        features, lengths = model.preprocessor(input_signal=torch.from_numpy(wave)[None],
                                               length=torch.tensor([samples]))
        logits = model(input_signal=torch.from_numpy(wave)[None],
                       input_signal_length=torch.tensor([samples]))
    # The mel spectrogram is the seam between the front end and the encoder: agreeing here and
    # disagreeing at the output isolates the network, and vice versa.
    globals()["_extra"] = {"waveform": torch.from_numpy(wave).contiguous(),
                           "features": features[0].transpose(0, 1).contiguous()}   # [frames, mels]
    return torch.softmax(logits, dim=-1)[0, :, 1].contiguous()  # [frames]



def run_vad_training(image, checkpoint):
    """The MarbleNet release's own training objective and learning-rate schedule.

    `--checkpoint` is the released `.nemo`, restored as `run_vad` restores it. Objective: a two-second
    clip (voiced for the first second, then noise) with speech labels at NeMo's 40 ms label rate goes
    through `EncDecFrameClassificationModel.forward` in evaluation mode (no dither, SpecAugment, or
    dropout, so the logits are reproducible), `reshape_labels` onto the 20 ms logits, `get_label_masks`,
    and the model's own `loss`. Schedule: the release's `optim.sched` (`PolynomialHoldDecayAnnealing`,
    warm-up 0.05, hold 0.15, power 2, min_lr 1e-8) over a 40-step run on the release's SGD, read back
    for 45 steps so the floor past the run shows.
    """
    import huggingface_hub

    for name in ["ModelFilter", "DatasetFilter"]:
        if not hasattr(huggingface_hub, name):
            setattr(huggingface_hub, name, type(name, (), {}))
    import nemo.collections.asr as nemo_asr
    from nemo.core.optim.lr_scheduler import PolynomialHoldDecayAnnealing

    model = nemo_asr.models.EncDecFrameClassificationModel.restore_from(checkpoint, strict=False).eval()
    model.preprocessor.featurizer.dither = 0.0

    samples = 32000
    time = np.arange(samples, dtype=np.float32) / 16000.0
    generator = np.random.default_rng(29)
    speech = sum(0.3 / (h + 1) * np.sin(2 * np.pi * 140 * (h + 1) * time) for h in range(6))
    wave = np.where(time < 1.0, speech, 0.05 * generator.standard_normal(samples)).astype(np.float32)
    label_frames = samples // 640
    labels = torch.tensor([[1 if (index + 0.5) * 0.04 < 1.0 else 0 for index in range(label_frames)]])

    signal = torch.from_numpy(wave)[None]
    length = torch.tensor([samples])
    with torch.no_grad():
        features, feature_length = model.preprocessor(input_signal=signal, length=length)
        logits = model(input_signal=signal, input_signal_length=length)
        reshaped, reshaped_length = model.reshape_labels(logits, labels, length, torch.tensor([label_frames]))
        masks = model.get_label_masks(reshaped, reshaped_length)
        loss = model.loss(logits=logits, labels=reshaped, loss_mask=masks)

    optim = model.cfg.optim
    parameter = torch.nn.Parameter(torch.zeros(1))
    sgd = torch.optim.SGD([parameter], lr=optim.lr, momentum=optim.momentum, weight_decay=optim.weight_decay)
    run = 40
    scheduler = PolynomialHoldDecayAnnealing(sgd, max_steps=run, warmup_ratio=optim.sched.warmup_ratio,
                                             hold_ratio=optim.sched.hold_ratio, power=optim.sched.power,
                                             min_lr=optim.sched.min_lr)
    rates = []
    for _ in range(run + 5):
        rates.append(sgd.param_groups[0]["lr"] / optim.lr)
        sgd.step()
        scheduler.step()

    globals()["_extra"] = {
        "waveform": signal[0].contiguous(),
        "features": features[0].transpose(0, 1).contiguous(),               # [mel frames, mels]
        "feature_length": feature_length.to(torch.int32).contiguous(),
        "logits": logits[0].contiguous(),                                   # [frames, 2]
        "labels": reshaped[0].to(torch.int32).contiguous(),                 # [frames]
        "mask": masks[0].to(torch.int32).contiguous(),
        "schedule": torch.tensor(rates, dtype=torch.float64),
        "optimizer": torch.tensor([optim.lr, optim.momentum, optim.weight_decay], dtype=torch.float64),
    }
    return loss.reshape(1).contiguous()

def run_silero_vad(image):
    """Silero VAD v6 (snakers4, PyPI `silero_vad` 6.2.1) per-chunk speech probability, `[chunks]`.

    The clip is exactly 32 chunks of 512 samples (1.024 s at 16 kHz) with a boundary at 0.5 s, so no
    padding enters and both sides produce the same chunk count. The record's `waveform` is the clip the
    Swift side reads back; both sides stream 512-sample chunks with a 64-sample look-back and thread the
    LSTM state, so a preprocessing or state-threading difference surfaces rather than hides.

    The reference bundles the model, so this mode needs no `--checkpoint`. Requires: `silero_vad`,
    `torchaudio` (pip install silero-vad torchaudio).
    """
    from silero_vad import load_silero_vad

    model = load_silero_vad()
    model.reset_states()
    rate, chunk, chunks = 16000, 512, 32
    samples = chunks * chunk
    time = np.arange(samples, dtype=np.float32) / rate
    generator = np.random.default_rng(13)
    speech = sum(0.3 / (h + 1) * np.sin(2 * np.pi * 150 * (h + 1) * time) for h in range(6))
    wave = np.where(time < 0.5, speech, 0.05 * generator.standard_normal(samples)).astype(np.float32)

    probabilities = []
    with torch.no_grad():
        for index in range(chunks):
            block = torch.from_numpy(wave[index * chunk:(index + 1) * chunk])[None]   # [1, 512]
            probabilities.append(float(np.asarray(model(block, rate)).reshape(-1)[0]))
    globals()["_extra"] = {"waveform": torch.from_numpy(wave).contiguous()}
    return torch.tensor(probabilities, dtype=torch.float32)                            # [chunks]


def run_dac(image):
    """Descript Audio Codec (44 kHz) round trip: the codebook tokens and the waveform reconstructed from
    them. The record's `waveform` is a clip an exact multiple of the hop long (so no padding enters and
    the frame counts agree); `codes` is `[n_codebooks, frames]` and `reconstruction` is the decode of
    those codes. Comparing codes exercises the encoder + RVQ; decoding the recorded codes isolates the
    decoder from a single code flipping on a codebook near-tie.

    Needs no `--checkpoint`; the `dac` package downloads the released weights. Requires: descript-audio-codec.
    """
    import dac

    model = dac.DAC.load(dac.utils.download(model_type="44khz")).eval()
    hop = model.hop_length
    frames = 87
    samples = frames * hop
    time = np.arange(samples, dtype=np.float32) / model.sample_rate
    generator = np.random.default_rng(19)
    wave = (0.3 * np.sin(2 * np.pi * 220 * time) + 0.2 * np.sin(2 * np.pi * 440 * time)
            + 0.1 * np.sin(2 * np.pi * 880 * time) + 0.02 * generator.standard_normal(samples)).astype(np.float32)
    wave = np.clip(wave, -1, 1)

    with torch.no_grad():
        audio = model.preprocess(torch.from_numpy(wave)[None, None], model.sample_rate)
        _, codes, _, _, _ = model.encode(audio)
        reconstruction = model.decode(model.quantizer.from_codes(codes)[0])

    globals()["_extra"] = {"waveform": torch.from_numpy(wave).contiguous(),
                           "codes": codes[0].to(torch.int32).contiguous()}          # [n_codebooks, frames]
    return reconstruction.reshape(-1)[:samples].contiguous()                         # [samples]


def run_snac(image):
    """SNAC (24 kHz multi-scale codec) round trip: the per-codebook tokens (`codes0`, `codes1`, … at each
    codebook's own temporal rate) and the waveform reconstructed from them. The decoder's noise block is
    DISABLED so the reconstruction is deterministic; its expected contribution is zero, and the Swift side
    decodes with the same noise-off path.

    Needs no `--checkpoint`; `SNAC.from_pretrained` fetches the released weights. Requires: snac.
    """
    import snac.layers as layers
    from snac import SNAC

    layers.NoiseBlock.forward = lambda self, x: x                      # deterministic decode
    # IK_SNAC_MODEL selects the release (24khz speech by default; 32khz / 44khz are the music
    # codecs, with four codebooks and bottleneck attention).
    model = SNAC.from_pretrained(os.environ.get("IK_SNAC_MODEL", "hubertsiuzdak/snac_24khz")).eval()
    hop = int(np.prod(model.encoder_rates)) if hasattr(model, "encoder_rates") else 512
    frames = 24
    samples = frames * hop
    time = np.arange(samples, dtype=np.float32) / model.sampling_rate
    generator = np.random.default_rng(23)
    wave = (0.3 * np.sin(2 * np.pi * 180 * time) + 0.2 * np.sin(2 * np.pi * 360 * time)
            + 0.02 * generator.standard_normal(samples)).astype(np.float32)
    wave = np.clip(wave, -1, 1)

    with torch.no_grad():
        codes = model.encode(torch.from_numpy(wave)[None, None])
        reconstruction = model.decode(codes)

    extra = {"waveform": torch.from_numpy(wave).contiguous()}
    for index, stream in enumerate(codes):
        extra[f"codes{index}"] = stream[0].to(torch.int32).contiguous()
    globals()["_extra"] = extra
    return reconstruction.reshape(-1)[:samples].contiguous()


def run_siglip2(image):
    """SigLIP 2 (base-patch16-224) image-text model: the L2-normalized image embedding, plus the text
    embeddings, token ids, and sigmoid logits for a few captions. The image pixel values are the plate
    normalized to -1…1 (SigLIP's normalization), fed directly so the processor's resize is out of the
    comparison; the plate must be 224×224 (`--size 224`).

    Requires: transformers >= 4.51 (SigLIP 2), which the llm oracle env carries.
    """
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained("google/siglip2-base-patch16-224").eval()
    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
    pixel_values = torch.from_numpy((image * 2 - 1).transpose(2, 0, 1))[None].float()
    texts = ["a photo of a cat", "a photo of two cats", "a city street at night"]
    tokens = processor(text=texts, return_tensors="pt", padding="max_length", max_length=64).input_ids

    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=pixel_values)
        text_features = model.get_text_features(input_ids=tokens)
        # Isolation seams: the embeddings output and the post-layernorm hidden (pre-head).
        vision = model.vision_model
        patch_embeds = vision.embeddings(pixel_values)
        vision_last = vision(pixel_values=pixel_values).last_hidden_state
    image_embeds = image_features / image_features.norm(dim=-1, keepdim=True)
    text_embeds = text_features / text_features.norm(dim=-1, keepdim=True)
    logits = text_embeds @ image_embeds.t() * model.logit_scale.exp() + model.logit_bias

    globals()["_extra"] = {"text_embeds": text_embeds.contiguous(),
                           "tokens": tokens.to(torch.int32).contiguous(),
                           "logits": logits.reshape(-1).contiguous(),
                           "patch_embeds": patch_embeds[0].contiguous(),
                           "vision_last": vision_last[0].contiguous()}
    return image_embeds[0].contiguous()                                    # [embed], L2-normalized


def run_taesd(image):
    """TAESD (tiny SD autoencoder) round trip: the encoder's latent and the decoder's reconstruction.
    Feed a plate whose side is a multiple of 8 (`--size 256`), so the latent is `side/8` square. The
    reference is madebyollin's own `taesd.py` (vendored under the source root) loaded with the released
    `.pth` weights. Requires: huggingface_hub, and `taesd.py` in the source root.
    """
    import os
    import sys
    import tempfile
    import urllib.request

    sys.path.insert(0, os.environ.get("IK_REF_SRC", os.path.expanduser("~/.inferkit-validation/sources")))
    from taesd import TAESD

    def weights(name):
        # The .pth live in the GitHub repo, not the HF repo (which carries the diffusers safetensors).
        path = os.path.join(tempfile.gettempdir(), name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(f"https://github.com/madebyollin/taesd/raw/main/{name}", path)
        return path

    model = TAESD(weights("taesd_encoder.pth"), weights("taesd_decoder.pth")).eval()
    with torch.no_grad():
        latent = model.encoder(torch.from_numpy(image.transpose(2, 0, 1))[None].float())
        reconstruction = model.decoder(latent)
    globals()["_extra"] = {"latent": latent[0].permute(1, 2, 0).contiguous()}       # [h, w, 4] NHWC
    return reconstruction[0].permute(1, 2, 0).contiguous()                          # [H, W, 3]


def run_ltx_vae(image):
    """LTX-Video VAE round trip: the deterministic latent (posterior mean) and the decoded video, plus
    the encoder seams (conv_in, first down block, mid block). Uses a fixed random video, so the plate is
    ignored. The VAE is loaded from `IK_LTX_VAE_DIR` (default `~/.inferkit-validation/raw/ltx-vae`).

    Runs under the `ltx` oracle env (diffusers >= 0.32). Tensors are returned in NDHWC to match the port.
    """
    import os
    from diffusers import AutoencoderKLLTXVideo

    directory = os.environ.get("IK_LTX_VAE_DIR", os.path.expanduser("~/.inferkit-validation/raw/ltx-vae"))
    vae = AutoencoderKLLTXVideo.from_pretrained(directory, torch_dtype=torch.float32).eval()
    generator = np.random.default_rng(11)
    video = (generator.random((1, 3, 9, 64, 64), dtype=np.float32) * 2 - 1)

    seams = {}
    vae.encoder.conv_in.register_forward_hook(lambda m, i, o: seams.__setitem__("enc_conv_in", o.detach()))
    vae.encoder.down_blocks[0].register_forward_hook(lambda m, i, o: seams.__setitem__("enc_down0", o.detach()))
    vae.encoder.mid_block.register_forward_hook(lambda m, i, o: seams.__setitem__("enc_mid", o.detach()))
    with torch.no_grad():
        mean = vae.encode(torch.from_numpy(video)).latent_dist.mean
        decoded = vae.decode(mean).sample

    def ndhwc(t):
        return t[0].permute(1, 2, 3, 0).contiguous()

    globals()["_extra"] = {"input_video": ndhwc(torch.from_numpy(video)), "latent": ndhwc(mean),
                           "enc_conv_in": ndhwc(seams["enc_conv_in"]), "enc_down0": ndhwc(seams["enc_down0"]),
                           "enc_mid": ndhwc(seams["enc_mid"])}
    return ndhwc(decoded)                                                   # [T, H, W, 3]


def run_ltx_transformer(image):
    """LTX-Video DiT velocity prediction from random latent tokens, a random text embedding, and a
    timestep, plus the rope / proj_in / first-block seams. The text embedding is fed directly, so the DiT
    is verified in isolation (no T5). The model loads from `IK_LTX_TF_DIR`
    (default `~/.inferkit-validation/raw/ltx-transformer`). Runs under the `ltx` oracle env.
    """
    import os
    from diffusers import LTXVideoTransformer3DModel

    directory = os.environ.get("IK_LTX_TF_DIR", os.path.expanduser("~/.inferkit-validation/raw/ltx-transformer"))
    model = LTXVideoTransformer3DModel.from_pretrained(directory, torch_dtype=torch.float32).eval()
    generator = torch.Generator().manual_seed(7)
    frames, height, width = 2, 2, 2
    latent = torch.randn(1, frames * height * width, 128, generator=generator)
    text = torch.randn(1, 4, 4096, generator=generator)
    timestep = torch.tensor([500.0])
    scale = (1.0, 1.0, 1.0)

    seams = {}
    model.proj_in.register_forward_hook(lambda m, i, o: seams.__setitem__("proj_in", o.detach()))
    model.transformer_blocks[0].register_forward_hook(lambda m, i, o: seams.__setitem__("block0", o.detach()))
    original_rope = model.rope.forward

    def rope_hook(*a, **k):
        out = original_rope(*a, **k)
        seams["rope_cos"], seams["rope_sin"] = out[0].detach(), out[1].detach()
        return out

    model.rope.forward = rope_hook
    with torch.no_grad():
        output = model(hidden_states=latent, encoder_hidden_states=text, timestep=timestep,
                       encoder_attention_mask=torch.ones(1, 4), num_frames=frames, height=height, width=width,
                       rope_interpolation_scale=scale, return_dict=False)[0]

    globals()["_extra"] = {"latent": latent[0].contiguous(), "text": text[0].contiguous(),
                           "timestep": timestep.contiguous(), "proj_in": seams["proj_in"][0].contiguous(),
                           "block0": seams["block0"][0].contiguous(), "rope_cos": seams["rope_cos"][0].contiguous(),
                           "rope_sin": seams["rope_sin"][0].contiguous(),
                           "grid": torch.tensor([frames, height, width], dtype=torch.int32)}
    return output[0].contiguous()                                          # [S, 128]


def run_ltx_t5(image):
    """LTX-Video T5-XXL text encoder output for a fixed padded token sequence, plus the embedding and
    first-block seams. Loads from `IK_LTX_T5_DIR` (default `~/.inferkit-validation/raw/ltx-t5`). Runs
    under the `ltx` oracle env (transformers with T5EncoderModel).
    """
    import os
    from transformers import T5EncoderModel

    directory = os.environ.get("IK_LTX_T5_DIR", os.path.expanduser("~/.inferkit-validation/raw/ltx-t5"))
    model = T5EncoderModel.from_pretrained(directory, torch_dtype=torch.float32).eval()
    ids = torch.tensor([[3, 19, 2523, 40, 8, 1946, 55, 1, 0, 0, 0, 0, 0, 0, 0, 0]])

    seams = {}
    model.shared.register_forward_hook(lambda m, i, o: seams.__setitem__("embed", o.detach()))
    model.encoder.block[0].register_forward_hook(lambda m, i, o: seams.__setitem__("block0", o[0].detach()))
    with torch.no_grad():
        output = model(input_ids=ids).last_hidden_state

    globals()["_extra"] = {"tokens": ids.to(torch.int32).contiguous(), "embed": seams["embed"][0].contiguous(),
                           "block0": seams["block0"][0].contiguous()}
    return output[0].contiguous()                                          # [S, 4096]


def run_z_image(image):
    """The Z-Image S3-DiT velocity prediction at a tiny random configuration, from diffusers' own
    ZImageTransformer2DModel, plus the timestep, noise-refiner, and first unified-layer seams.

    Single-stream: the image latent and the caption features are concatenated and every layer's
    self-attention runs over the join. The caption features are supplied directly (no Qwen3), so the
    DiT is verified in isolation, as the LTX DiT is. Sequence lengths are deliberately NOT a multiple
    of 32, so the learned pad tokens (`x_pad_token`, `cap_pad_token`) and the (0,0,0) pad positions are
    exercised. Runs under the `ltx` oracle env (diffusers with ZImageTransformer2DModel). `image` unused.
    """
    from diffusers import ZImageTransformer2DModel

    model = ZImageTransformer2DModel(
        all_patch_size=(2,), all_f_patch_size=(1,), in_channels=4, dim=32, n_layers=2,
        n_refiner_layers=1, n_heads=2, n_kv_heads=2, norm_eps=1e-5, qk_norm=True, cap_feat_dim=24,
        rope_theta=256.0, t_scale=1000.0, axes_dims=[4, 6, 6], axes_lens=[1024, 512, 512])
    model = _randomized(model, seed=19)

    generator = torch.Generator().manual_seed(5)
    latent = torch.randn(4, 1, 8, 8, generator=generator)                  # [C, F, H, W] -> 16 tokens
    cap = torch.randn(20, 24, generator=generator)                         # 20 caption tokens
    t = torch.tensor([0.3])

    seams = {}
    model.t_embedder.register_forward_hook(lambda m, i, o: seams.__setitem__("t_emb", o.detach()))
    model.noise_refiner[-1].register_forward_hook(lambda m, i, o: seams.__setitem__("noise_out", o.detach()))
    model.layers[0].register_forward_hook(lambda m, i, o: seams.__setitem__("layer0", o.detach()))
    with torch.no_grad():
        output = model([latent], t, [cap], patch_size=2, f_patch_size=1, return_dict=False)[0][0]

    extra = {"latent": latent.contiguous(), "cap_feats": cap.contiguous(), "timestep": t.contiguous(),
             "t_emb": seams["t_emb"][0].contiguous(), "noise_out": seams["noise_out"][0].contiguous(),
             "layer0": seams["layer0"][0].contiguous()}
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return output.contiguous()                                             # [C, F, H, W]


def run_sana(image):
    """The SANA linear-attention DiT velocity at a tiny random configuration, from diffusers'
    SanaTransformer2DModel, plus the patch-embed, first-block, and embedded-timestep seams.

    SANA replaces softmax self-attention with ReLU LINEAR attention and the MLP with a GLUMBConv
    (gated depthwise convolution over the token grid); each block cross-attends to the caption. The
    caption embedding is supplied directly (no Gemma), so the DiT is verified in isolation. Runs under
    the `ltx` oracle env (diffusers with SanaTransformer2DModel). `image` unused.
    """
    from diffusers import SanaTransformer2DModel

    model = SanaTransformer2DModel(
        in_channels=8, out_channels=8, num_attention_heads=2, attention_head_dim=8, num_layers=2,
        num_cross_attention_heads=2, cross_attention_head_dim=8, cross_attention_dim=16,
        caption_channels=12, mlp_ratio=2.0, patch_size=1, attention_bias=False,
        norm_elementwise_affine=False, norm_eps=1e-6, qk_norm=None, guidance_embeds=False,
        timestep_scale=1.0, sample_size=8)
    model = _randomized(model, seed=23)

    generator = torch.Generator().manual_seed(6)
    latent = torch.randn(1, 8, 4, 4, generator=generator)                  # [B, C, H, W]
    cap = torch.randn(1, 6, 12, generator=generator)                       # [B, Lc, caption_channels]
    t = torch.tensor([0.4])

    seams = {}
    model.patch_embed.register_forward_hook(lambda m, i, o: seams.__setitem__("patch", o.detach()))
    model.time_embed.register_forward_hook(lambda m, i, o: seams.__setitem__("embedded", o[1].detach()))
    model.transformer_blocks[0].register_forward_hook(lambda m, i, o: seams.__setitem__("block0", o.detach()))
    with torch.no_grad():
        output = model(hidden_states=latent, encoder_hidden_states=cap, timestep=t, return_dict=False)[0][0]

    extra = {"latent": latent[0].contiguous(), "cap_feats": cap[0].contiguous(), "timestep": t.contiguous(),
             "patch": seams["patch"][0].contiguous(), "embedded": seams["embedded"][0].contiguous(),
             "block0": seams["block0"][0].contiguous()}
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return output.contiguous()                                             # [C, H, W]


def run_sd3(image):
    """The Stable Diffusion 3 MMDiT velocity at a tiny random configuration, from diffusers'
    SD3Transformer2DModel, plus the patch-embed and first-block seams.

    Dual-stream: the image latent and the text tokens each carry their own projections, feed-forward,
    and adaptive-norm modulation, with attention over the concatenation (JointAttnProcessor2_0). The
    tiny config exercises the SD3.5 additions — RMS query/key norm (`qk_norm='rms_norm'`) and a
    dual-attention first layer (`dual_attention_layers=(0,)`, a second image-only self-attention) —
    and the last block's `context_pre_only` path (the text stream ends after the attention). The
    latent grid (4x4 patches) is smaller than `pos_embed_max_size` (8), so the center-crop of the
    positional table is exercised. The caption/pooled embeddings are supplied directly (no CLIP/T5),
    so the DiT is verified in isolation. Runs under the `ltx` oracle env. `image` unused.
    """
    from diffusers import SD3Transformer2DModel

    model = SD3Transformer2DModel(
        sample_size=16, patch_size=2, in_channels=4, num_layers=2, attention_head_dim=8,
        num_attention_heads=2, joint_attention_dim=24, caption_projection_dim=16,
        pooled_projection_dim=20, out_channels=4, pos_embed_max_size=8,
        dual_attention_layers=(0,), qk_norm="rms_norm")
    model = _randomized(model, seed=27)

    generator = torch.Generator().manual_seed(7)
    latent = torch.randn(2, 4, 8, 8, generator=generator)                  # [B, C, H, W] -> 16 tokens
    encoder = torch.randn(2, 7, 24, generator=generator)                   # [B, L, joint_attention_dim]
    pooled = torch.randn(2, 20, generator=generator)                       # [B, pooled_projection_dim]
    t = torch.tensor([500.0, 500.0])

    seams = {}
    model.pos_embed.register_forward_hook(lambda m, i, o: seams.__setitem__("patch", o.detach()))
    model.transformer_blocks[0].register_forward_hook(lambda m, i, o: seams.__setitem__("block0", o[1].detach()))
    with torch.no_grad():
        output = model(hidden_states=latent, encoder_hidden_states=encoder, pooled_projections=pooled,
                       timestep=t, return_dict=False)[0]

    extra = {"latent": latent.contiguous(), "encoder": encoder.contiguous(), "pooled": pooled.contiguous(),
             "timestep": t.contiguous(), "patch": seams["patch"].contiguous(), "block0": seams["block0"].contiguous()}
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return output.contiguous()                                             # [B, C, H, W]


def run_flux(image):
    """The FLUX.1 transformer velocity at a tiny random configuration, from diffusers'
    FluxTransformer2DModel, plus the double-block and single-block seams.

    Two block kinds: DOUBLE-stream MMDiT joint-attention blocks (image and text streams, attention over
    the concatenation) and SINGLE-stream blocks (the streams concatenated, a parallel attention+MLP over
    the join). Position is an axial rotary over the token ids (text ids all zero; image ids the (0,row,
    col) grid). The guidance-distilled variant (`guidance_embeds=True`) is exercised, so the guidance
    embedding is covered. The text conditioning is supplied directly (no T5/CLIP), so the DiT is verified
    in isolation. Runs under the `ltx` oracle env. `image` unused.
    """
    from diffusers import FluxTransformer2DModel

    model = FluxTransformer2DModel(
        patch_size=1, in_channels=8, num_layers=2, num_single_layers=2, attention_head_dim=6,
        num_attention_heads=2, joint_attention_dim=24, pooled_projection_dim=10, guidance_embeds=True,
        axes_dims_rope=(2, 2, 2))
    model = _randomized(model, seed=29)

    generator = torch.Generator().manual_seed(8)
    lh, lw = 2, 3                                                          # packed latent grid -> 6 tokens
    hidden = torch.randn(2, lh * lw, 8, generator=generator)              # [B, img_seq, in_channels]
    encoder = torch.randn(2, 5, 24, generator=generator)                  # [B, txt_seq, joint_attention_dim]
    pooled = torch.randn(2, 10, generator=generator)                     # [B, pooled_projection_dim]
    t = torch.tensor([0.5, 0.5])
    guidance = torch.tensor([3.5, 3.5])
    img_ids = torch.zeros(lh * lw, 3)
    img_ids[:, 1] = torch.arange(lh).unsqueeze(1).expand(lh, lw).reshape(-1).float()
    img_ids[:, 2] = torch.arange(lw).unsqueeze(0).expand(lh, lw).reshape(-1).float()
    txt_ids = torch.zeros(5, 3)

    seams = {}
    model.transformer_blocks[0].register_forward_hook(lambda m, i, o: seams.__setitem__("double0", o[1].detach()))
    model.single_transformer_blocks[0].register_forward_hook(lambda m, i, o: seams.__setitem__("single0", o[1].detach()))
    with torch.no_grad():
        output = model(hidden_states=hidden, encoder_hidden_states=encoder, pooled_projections=pooled,
                       timestep=t, img_ids=img_ids, txt_ids=txt_ids, guidance=guidance, return_dict=False)[0]

    extra = {"hidden": hidden.contiguous(), "encoder": encoder.contiguous(), "pooled": pooled.contiguous(),
             "timestep": t.contiguous(), "guidance": guidance.contiguous(), "img_ids": img_ids.contiguous(),
             "double0": seams["double0"].contiguous(), "single0": seams["single0"].contiguous()}
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return output.contiguous()                                             # [B, img_seq, out_channels]


def run_flux_real(image, checkpoint):
    """The FLUX.1 [schnell] transformer velocity on the RELEASED weights, at the precision they ship in.

    `--checkpoint` is the release's `transformer/` directory. The 12B transformer at float32 is ~48 GB,
    which this machine cannot hold, so both sides run bfloat16 and the comparison describes the released
    precision rather than the arithmetic's ceiling; `run_flux` is where the arithmetic is measured exactly
    at a tiny configuration. The spatial input is deliberately tiny (a 2×2 packed-latent grid, 4 image
    tokens, 8 text tokens) so only the WEIGHTS are large, not the activations. The text conditioning is
    random rather than the CLIP/T5 encoders' output, which keeps the transformer measured in isolation,
    the way the LTX DiT and Qwen-Image transformer are. schnell carries no guidance embedding.
    """
    import torch
    from diffusers import FluxTransformer2DModel

    model = FluxTransformer2DModel.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).eval()

    generator = torch.Generator().manual_seed(8)
    lh, lw = 2, 2                                                          # packed latent grid -> 4 tokens
    channels = model.config.in_channels                                   # 64 for schnell
    joint = model.config.joint_attention_dim                              # 4096
    pooledDim = model.config.pooled_projection_dim                        # 768
    hidden = torch.randn(1, lh * lw, channels, generator=generator).to(torch.bfloat16)
    encoder = torch.randn(1, 8, joint, generator=generator).to(torch.bfloat16)
    pooled = torch.randn(1, pooledDim, generator=generator).to(torch.bfloat16)
    t = torch.tensor([0.5], dtype=torch.bfloat16)
    img_ids = torch.zeros(lh * lw, 3)
    img_ids[:, 1] = torch.arange(lh).unsqueeze(1).expand(lh, lw).reshape(-1).float()
    img_ids[:, 2] = torch.arange(lw).unsqueeze(0).expand(lh, lw).reshape(-1).float()
    txt_ids = torch.zeros(8, 3)

    with torch.no_grad():
        output = model(hidden_states=hidden, encoder_hidden_states=encoder, pooled_projections=pooled,
                       timestep=t, img_ids=img_ids, txt_ids=txt_ids, guidance=None, return_dict=False)[0]

    globals()["_extra"] = {
        "hidden": hidden.float().contiguous(), "encoder": encoder.float().contiguous(),
        "pooled": pooled.float().contiguous(), "timestep": t.float().contiguous(),
        "img_ids": img_ids.contiguous()}
    return output.float().contiguous()                                     # [B, img_seq, out_channels]


def run_flux_text(image, checkpoint):
    """FLUX.1's text front end on the RELEASED weights: the CLIP-L pooled projection and the T5-XXL
    sequence, from transformers' own CLIPTextModel and T5EncoderModel.

    `--checkpoint` is a FLUX release directory (the diffusers layout: `text_encoder/`, `text_encoder_2/`,
    `tokenizer/`, `tokenizer_2/`). The encoders together are ~10 GB in bfloat16, which fits where the 24 GB
    transformer does not, so this measures the whole text path — tokenization included — that the Swift
    `NFKMLXFlux.encode` reproduces. CLIP pads to 77 and reads the pooled embedding at the end-of-text
    token (a causal model, so padding past it does not change that token); T5 pads to 256 and encodes the
    whole padded sequence with no attention mask, the way diffusers' FluxPipeline does.
    """
    import torch
    from transformers import (CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast)

    prompt = "a photograph of an astronaut riding a horse on the moon"
    clip_tokenizer = CLIPTokenizer.from_pretrained(checkpoint, subfolder="tokenizer")
    clip = CLIPTextModel.from_pretrained(checkpoint, subfolder="text_encoder",
                                         torch_dtype=torch.bfloat16).eval()
    clip_ids = clip_tokenizer(prompt, padding="max_length", max_length=77, truncation=True,
                              return_tensors="pt").input_ids
    with torch.no_grad():
        pooled = clip(clip_ids).pooler_output[0]                           # [768]

    t5_tokenizer = T5TokenizerFast.from_pretrained(checkpoint, subfolder="tokenizer_2")
    t5 = T5EncoderModel.from_pretrained(checkpoint, subfolder="text_encoder_2",
                                        torch_dtype=torch.bfloat16).eval()
    t5_ids = t5_tokenizer(prompt, padding="max_length", max_length=256, truncation=True,
                          return_tensors="pt").input_ids
    with torch.no_grad():
        embeds = t5(t5_ids)[0][0]                                          # [256, 4096]

    globals()["_extra"] = {
        "clip_ids": clip_ids[0].to(torch.int32).contiguous(),
        "t5_ids": t5_ids[0].to(torch.int32).contiguous(),
        "pooled": pooled.float().contiguous(),
        "embeds": embeds.float().contiguous()}
    return pooled.float().contiguous()


def run_sd3_controlnet(image):
    """The SD3 ControlNet (dual-stream) end to end, from diffusers' SD3ControlNetModel plus the base
    SD3Transformer2DModel with the ControlNet residuals injected.

    The ControlNet is a partial MMDiT: it runs the first N JointTransformerBlocks over the noisy latent
    plus a control latent (added through a zero-initialized `pos_embed_input` that carries no positional
    table), and emits one zero-initialized residual per block. The base transformer adds them into its
    own blocks, strided over the residual list by `interval_control`. The tiny base has four layers
    (so blocks 0-2 inject and block 3 is context_pre_only) and the ControlNet two residuals (interval
    2.0), so both residuals and the striding are exercised. Both nets are recorded (`w::` the ControlNet,
    `t::` the base) so the Swift side validates the residuals and the injected output. Runs under the
    `ltx` oracle env. `image` unused.
    """
    from diffusers import SD3ControlNetModel, SD3Transformer2DModel

    base = SD3Transformer2DModel(
        sample_size=16, patch_size=2, in_channels=4, num_layers=4, attention_head_dim=8,
        num_attention_heads=2, joint_attention_dim=24, caption_projection_dim=16,
        pooled_projection_dim=20, out_channels=4, pos_embed_max_size=8, qk_norm="rms_norm")
    control = SD3ControlNetModel(
        sample_size=16, patch_size=2, in_channels=4, num_layers=2, attention_head_dim=8,
        num_attention_heads=2, joint_attention_dim=24, caption_projection_dim=16,
        pooled_projection_dim=20, out_channels=4, pos_embed_max_size=8, qk_norm="rms_norm",
        extra_conditioning_channels=0)
    base = _randomized(base, seed=31)
    control = _randomized(control, seed=32)

    generator = torch.Generator().manual_seed(9)
    latent = torch.randn(2, 4, 8, 8, generator=generator)
    control_cond = torch.randn(2, 4, 8, 8, generator=generator)
    encoder = torch.randn(2, 7, 24, generator=generator)
    pooled = torch.randn(2, 20, generator=generator)
    t = torch.tensor([500.0, 500.0])

    with torch.no_grad():
        residuals = control(hidden_states=latent, controlnet_cond=control_cond, conditioning_scale=0.7,
                            encoder_hidden_states=encoder, pooled_projections=pooled, timestep=t,
                            return_dict=False)[0]
        output = base(hidden_states=latent, encoder_hidden_states=encoder, pooled_projections=pooled,
                      timestep=t, block_controlnet_hidden_states=residuals, return_dict=False)[0]

    extra = {"latent": latent.contiguous(), "control_cond": control_cond.contiguous(),
             "encoder": encoder.contiguous(), "pooled": pooled.contiguous(), "timestep": t.contiguous()}
    for i, r in enumerate(residuals):
        extra[f"residual_{i}"] = r.contiguous()
    for key, value in control.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    for key, value in base.state_dict().items():
        extra[f"t::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return output.contiguous()                                             # [B, C, H, W]


def run_sd3_controlnet_single(image):
    """The Stability SD3.5-large 8B ControlNet (single-stream) residuals at a tiny random configuration,
    from diffusers' SD3ControlNetModel with `joint_attention_dim=None` and `use_pos_embed=False`.

    This variant drops the position embedding and the context embedder and runs single-stream
    SD3SingleTransformerBlocks over the image tokens alone (the base transformer's `pos_embed` supplies
    the 3-D `hidden_states`). `extra_conditioning_channels=1` widens `pos_embed_input` to five channels.
    The residuals are recorded directly (their injection into the base is the same rule the dual-stream
    mode already validates). Runs under the `ltx` oracle env. `image` unused.
    """
    from diffusers import SD3ControlNetModel

    control = SD3ControlNetModel(
        sample_size=16, patch_size=2, in_channels=4, num_layers=2, attention_head_dim=8,
        num_attention_heads=2, joint_attention_dim=None, pooled_projection_dim=20, out_channels=4,
        pos_embed_max_size=8, extra_conditioning_channels=1, use_pos_embed=False)
    control = _randomized(control, seed=34)

    generator = torch.Generator().manual_seed(10)
    hidden = torch.randn(2, 16, 16, generator=generator)                   # already patch-embedded tokens
    control_cond = torch.randn(2, 5, 8, 8, generator=generator)            # in_channels + extra = 5
    pooled = torch.randn(2, 20, generator=generator)
    t = torch.tensor([500.0, 500.0])

    with torch.no_grad():
        residuals = control(hidden_states=hidden, controlnet_cond=control_cond, conditioning_scale=0.8,
                            encoder_hidden_states=None, pooled_projections=pooled, timestep=t,
                            return_dict=False)[0]

    extra = {"hidden": hidden.contiguous(), "control_cond": control_cond.contiguous(),
             "pooled": pooled.contiguous(), "timestep": t.contiguous()}
    for i, r in enumerate(residuals):
        extra[f"residual_{i}"] = r.contiguous()
    for key, value in control.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return torch.cat([r.reshape(-1) for r in residuals])                   # the residuals, flattened


def run_flux_controlnet(image):
    """The FLUX ControlNet end to end, from diffusers' FluxControlNetModel plus the base
    FluxTransformer2DModel with the ControlNet residuals injected.

    The ControlNet runs a few double-stream and single-stream blocks over the packed noisy latent plus a
    packed control latent (added through the zero-initialized `controlnet_x_embedder`), and emits a
    zero-initialized residual per block — a double-block list and a single-block list. The base
    transformer adds each into its own block stacks, strided by the `ceil` interval. The tiny base has
    three double and three single blocks, the ControlNet two of each (interval ceil(3/2)=2), so both
    residuals and the striding are exercised. Both nets are recorded (`w::` the ControlNet, `t::` the
    base). Runs under the `ltx` oracle env. `image` unused.
    """
    from diffusers import FluxControlNetModel, FluxTransformer2DModel

    base = FluxTransformer2DModel(
        patch_size=1, in_channels=8, num_layers=3, num_single_layers=3, attention_head_dim=6,
        num_attention_heads=2, joint_attention_dim=24, pooled_projection_dim=10, guidance_embeds=True,
        axes_dims_rope=(2, 2, 2))
    control = FluxControlNetModel(
        patch_size=1, in_channels=8, num_layers=2, num_single_layers=2, attention_head_dim=6,
        num_attention_heads=2, joint_attention_dim=24, pooled_projection_dim=10, guidance_embeds=True,
        axes_dims_rope=(2, 2, 2))
    base = _randomized(base, seed=35)
    control = _randomized(control, seed=36)

    generator = torch.Generator().manual_seed(11)
    lh, lw = 2, 3
    hidden = torch.randn(2, lh * lw, 8, generator=generator)
    control_cond = torch.randn(2, lh * lw, 8, generator=generator)
    encoder = torch.randn(2, 5, 24, generator=generator)
    pooled = torch.randn(2, 10, generator=generator)
    t = torch.tensor([0.5, 0.5])
    guidance = torch.tensor([3.5, 3.5])
    img_ids = torch.zeros(lh * lw, 3)
    img_ids[:, 1] = torch.arange(lh).unsqueeze(1).expand(lh, lw).reshape(-1).float()
    img_ids[:, 2] = torch.arange(lw).unsqueeze(0).expand(lh, lw).reshape(-1).float()
    txt_ids = torch.zeros(5, 3)

    with torch.no_grad():
        double, single = control(
            hidden_states=hidden, controlnet_cond=control_cond, conditioning_scale=0.6,
            encoder_hidden_states=encoder, pooled_projections=pooled, timestep=t, img_ids=img_ids,
            txt_ids=txt_ids, guidance=guidance, return_dict=False)
        output = base(hidden_states=hidden, encoder_hidden_states=encoder, pooled_projections=pooled,
                      timestep=t, img_ids=img_ids, txt_ids=txt_ids, guidance=guidance,
                      controlnet_block_samples=double, controlnet_single_block_samples=single,
                      return_dict=False)[0]

    extra = {"hidden": hidden.contiguous(), "control_cond": control_cond.contiguous(),
             "encoder": encoder.contiguous(), "pooled": pooled.contiguous(), "timestep": t.contiguous(),
             "guidance": guidance.contiguous(), "img_ids": img_ids.contiguous()}
    for i, r in enumerate(double):
        extra[f"double_{i}"] = r.contiguous()
    for i, r in enumerate(single):
        extra[f"single_{i}"] = r.contiguous()
    for key, value in control.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    for key, value in base.state_dict().items():
        extra[f"t::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return output.contiguous()                                             # [B, img_seq, out_channels]


def run_flux_controlnet_hint(image):
    """A FLUX ControlNet with an `input_hint_block` end to end, from diffusers' FluxControlNetModel plus
    the base FluxTransformer2DModel with the residuals injected.

    Instead of a packed VAE control latent, this shape takes a FULL-RESOLUTION control image and runs it
    through the `ControlNetConditioningEmbedding` pyramid (a conv_in, three stride-2 downsampling stages
    over a (16,16,16,16) channel pyramid, a zero-init conv_out, SiLU between) before the linear embed.
    The control image is 8× the packed latent grid (16×24 → the 2×3 grid = six tokens). Runs under the
    `ltx` oracle env. `image` unused.
    """
    from diffusers import FluxControlNetModel, FluxTransformer2DModel

    base = FluxTransformer2DModel(
        patch_size=1, in_channels=8, num_layers=3, num_single_layers=3, attention_head_dim=6,
        num_attention_heads=2, joint_attention_dim=24, pooled_projection_dim=10, guidance_embeds=True,
        axes_dims_rope=(2, 2, 2))
    control = FluxControlNetModel(
        patch_size=1, in_channels=8, num_layers=2, num_single_layers=2, attention_head_dim=6,
        num_attention_heads=2, joint_attention_dim=24, pooled_projection_dim=10, guidance_embeds=True,
        axes_dims_rope=(2, 2, 2), conditioning_embedding_channels=8)
    base = _randomized(base, seed=37)
    control = _randomized(control, seed=38)

    generator = torch.Generator().manual_seed(12)
    lh, lw = 2, 3
    hidden = torch.randn(2, lh * lw, 8, generator=generator)
    control_image = torch.randn(2, 3, lh * 8, lw * 8, generator=generator)  # full-resolution control image
    encoder = torch.randn(2, 5, 24, generator=generator)
    pooled = torch.randn(2, 10, generator=generator)
    t = torch.tensor([0.5, 0.5])
    guidance = torch.tensor([3.5, 3.5])
    img_ids = torch.zeros(lh * lw, 3)
    img_ids[:, 1] = torch.arange(lh).unsqueeze(1).expand(lh, lw).reshape(-1).float()
    img_ids[:, 2] = torch.arange(lw).unsqueeze(0).expand(lh, lw).reshape(-1).float()
    txt_ids = torch.zeros(5, 3)

    with torch.no_grad():
        double, single = control(
            hidden_states=hidden, controlnet_cond=control_image, conditioning_scale=0.6,
            encoder_hidden_states=encoder, pooled_projections=pooled, timestep=t, img_ids=img_ids,
            txt_ids=txt_ids, guidance=guidance, return_dict=False)
        output = base(hidden_states=hidden, encoder_hidden_states=encoder, pooled_projections=pooled,
                      timestep=t, img_ids=img_ids, txt_ids=txt_ids, guidance=guidance,
                      controlnet_block_samples=double, controlnet_single_block_samples=single,
                      return_dict=False)[0]

    extra = {"hidden": hidden.contiguous(), "control_image": control_image.contiguous(),
             "encoder": encoder.contiguous(), "pooled": pooled.contiguous(), "timestep": t.contiguous(),
             "guidance": guidance.contiguous(), "img_ids": img_ids.contiguous()}
    for i, r in enumerate(double):
        extra[f"double_{i}"] = r.contiguous()
    for i, r in enumerate(single):
        extra[f"single_{i}"] = r.contiguous()
    for key, value in control.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    for key, value in base.state_dict().items():
        extra[f"t::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return output.contiguous()                                             # [B, img_seq, out_channels]


def run_wan(image):
    """The Wan text-to-video DiT velocity at a tiny random configuration, from diffusers'
    WanTransformer3DModel, plus the patch-embed and first-block seams.

    A 3-D sequence transformer over a Conv3d-patchified video latent, with a 3-axis interleaved rotary
    over the (frame, height, width) grid, self + cross attention, and an across-heads RMS q/k norm. The
    text embedding is supplied directly (no umT5), so the DiT is verified in isolation. Runs under the
    `ltx` oracle env (diffusers with WanTransformer3DModel). `image` unused.
    """
    from diffusers import WanTransformer3DModel

    model = WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=16, in_channels=4,
        out_channels=4, text_dim=10, freq_dim=256, ffn_dim=48, num_layers=2, cross_attn_norm=True,
        qk_norm="rms_norm_across_heads", eps=1e-6, image_dim=None, added_kv_proj_dim=None,
        rope_max_seq_len=1024)
    model = _randomized(model, seed=29)

    generator = torch.Generator().manual_seed(8)
    latent = torch.randn(1, 4, 2, 4, 4, generator=generator)               # [B, C, T, H, W]
    text = torch.randn(1, 6, 10, generator=generator)                      # [B, Lc, text_dim]
    t = torch.tensor([0.35])

    seams = {}
    model.patch_embedding.register_forward_hook(lambda m, i, o: seams.__setitem__("patch", o.detach()))
    model.blocks[0].register_forward_hook(lambda m, i, o: seams.__setitem__("block0", o.detach()))
    with torch.no_grad():
        output = model(hidden_states=latent, timestep=t, encoder_hidden_states=text, return_dict=False)[0][0]

    extra = {"latent": latent[0].contiguous(), "text": text[0].contiguous(), "timestep": t.contiguous(),
             "patch": seams["patch"][0].flatten(1).transpose(0, 1).contiguous(),
             "block0": seams["block0"][0].contiguous()}
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return output.contiguous()                                             # [C, F, H, W]


def run_sam3_vision(image):
    """SAM 3's vision encoder (`Sam3VisionModel`, released `facebook/sam3`) on a real plate: the
    32-layer rotary ViT and the FPN neck that reads its one output map at four scales.

    Measured at 504 pixels rather than the released 1008, which exercises both edges the full size
    hides: the pretraining position grid is 24x24 patches and the input is 36x36, so the grid is
    tiled and cropped, and 36 is not a whole number of 24-wide windows, so a windowed layer pads.
    Records the plate, the ViT's output, and the four FPN levels. The vision encoder is loaded on its
    own out of the 1797-tensor release, so the detector and tracker stay off the machine. Runs under
    the `wananimate` oracle env (transformers >= 5.16). `image` is the plate.
    """
    from safetensors.torch import safe_open
    from transformers import AutoConfig
    from transformers.models.sam3.modeling_sam3 import Sam3VisionModel

    directory = os.path.expanduser(os.environ.get("IK_SAM3_DIR", "~/.inferkit-validation/sam3"))
    config = AutoConfig.from_pretrained(directory).detector_config.vision_config
    # The global layers' rotary table is built from the CONFIGURED image size, not the input's, so a
    # plate of another size needs the configuration to say so.
    config.backbone_config.image_size = int(os.environ.get("IK_SAM3_SIZE", 504))
    model = Sam3VisionModel(config)

    prefix = "detector_model.vision_encoder."
    state = {}
    with safe_open(os.path.join(directory, "model.safetensors"), framework="pt") as handle:
        for key in handle.keys():
            if key.startswith(prefix):
                state[key[len(prefix):]] = handle.get_tensor(key).float()
    model.load_state_dict(state, strict=True)
    model = model.eval().float()

    plate = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)              # [1, 3, H, W]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    deviation = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    pixel_values = (plate - mean) / deviation

    with torch.no_grad():
        output = model(pixel_values=pixel_values)

    grid = pixel_values.shape[-1] // config.backbone_config.patch_size
    extra = {"pixel_values": pixel_values[0].permute(1, 2, 0).contiguous(),
             "backbone": output.last_hidden_state[0].reshape(grid, grid, -1).contiguous()}
    for index, level in enumerate(output.fpn_hidden_states):
        extra[f"fpn_{index}"] = level[0].permute(1, 2, 0).contiguous()
    globals()["_extra"] = extra
    return extra["fpn_0"].clone()                                                  # the finest level


def run_sam3_text(image):
    """SAM 3's prompt side (`CLIPTextModelWithProjection` + the detector's projection, released
    `facebook/sam3`): the 24-layer causal CLIP text tower 1024 wide, and the 1024 -> 256 projection
    that carries EVERY token to the detector's width rather than the pooled end-of-text one.

    Records the token ids, the tower's last hidden state, and the projected prompt. The text tower is
    loaded on its own out of the 1797-tensor release. Runs under the `wananimate` oracle env
    (transformers >= 5.16). `image` unused.
    """
    from safetensors.torch import safe_open
    from transformers import AutoConfig, CLIPTextModelWithProjection

    directory = os.path.expanduser(os.environ.get("IK_SAM3_DIR", "~/.inferkit-validation/sam3"))
    detector = AutoConfig.from_pretrained(directory).detector_config
    model = CLIPTextModelWithProjection(detector.text_config)
    projection = torch.nn.Linear(detector.text_config.hidden_size, detector.detr_encoder_config.hidden_size)

    state, projection_state = {}, {}
    with safe_open(os.path.join(directory, "model.safetensors"), framework="pt") as handle:
        for key in handle.keys():
            if key.startswith("detector_model.text_encoder."):
                state[key[len("detector_model.text_encoder."):]] = handle.get_tensor(key).float()
            elif key.startswith("detector_model.text_projection."):
                projection_state[key[len("detector_model.text_projection."):]] = handle.get_tensor(key).float()
    model.load_state_dict(state, strict=True)
    projection.load_state_dict(projection_state, strict=True)
    model, projection = model.eval().float(), projection.eval().float()

    # A short prompt between CLIP's start and end tokens, padded to the trained context.
    ids = torch.full((1, detector.text_config.max_position_embeddings),
                     detector.text_config.pad_token_id, dtype=torch.long)
    prompt = [49406, 2368, 49407]
    ids[0, : len(prompt)] = torch.tensor(prompt)
    attention_mask = torch.zeros_like(ids)
    attention_mask[0, : len(prompt)] = 1

    with torch.no_grad():
        output = model(input_ids=ids, attention_mask=attention_mask, return_dict=True)
        projected = projection(output.last_hidden_state)

    extra = {"input_ids": ids[0].to(torch.int32).contiguous(),
             "attention_mask": attention_mask[0].to(torch.int32).contiguous(),
             "last_hidden_state": output.last_hidden_state[0].contiguous()}
    globals()["_extra"] = extra
    return projected[0].contiguous()                                       # [L, 256]


def run_sam3_detector(image):
    """SAM 3's detector (`Sam3Model`, released `facebook/sam3`) on a real plate and a worded prompt:
    the DETR encoder that fuses one vision level with the prompt, the DETR decoder with its 200
    queries and presence token, the dot-product scoring head, and the mask decoder.

    Records the FPN levels and their position encodings, the projected prompt, and every output, so
    the detector can be driven from recorded inputs and measured on its own as well as end to end.
    Runs at 504 pixels, where the detector reads a 36x36 level and lifts its result back to 144x144.
    Runs under the `wananimate` oracle env (transformers >= 5.16). `image` is the plate.
    """
    from transformers import AutoConfig
    from transformers.models.sam3.modeling_sam3 import Sam3Model

    directory = os.path.expanduser(os.environ.get("IK_SAM3_DIR", "~/.inferkit-validation/sam3"))
    config = AutoConfig.from_pretrained(directory)
    size = int(os.environ.get("IK_SAM3_SIZE", 504))
    config.detector_config.vision_config.backbone_config.image_size = size
    model = Sam3Model.from_pretrained(directory, config=config, dtype=torch.float32).eval()

    plate = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    deviation = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    pixel_values = (plate - mean) / deviation

    text_config = config.detector_config.text_config
    ids = torch.full((1, text_config.max_position_embeddings), text_config.pad_token_id, dtype=torch.long)
    prompt = [49406, 2368, 49407]
    ids[0, : len(prompt)] = torch.tensor(prompt)
    attention_mask = torch.zeros_like(ids)
    attention_mask[0, : len(prompt)] = 1

    with torch.no_grad():
        vision = model.get_vision_features(pixel_values=pixel_values)
        text = model.get_text_features(input_ids=ids, attention_mask=attention_mask, return_dict=True)
        output = model(vision_embeds=vision, text_embeds=text, attention_mask=attention_mask)

    extra = {"pixel_values": pixel_values[0].permute(1, 2, 0).contiguous(),
             "input_ids": ids[0].to(torch.int32).contiguous(),
             "attention_mask": attention_mask[0].to(torch.int32).contiguous(),
             "prompt": text.pooler_output[0].contiguous(),
             "pred_boxes": output.pred_boxes[0].contiguous(),
             "pred_logits": output.pred_logits[0].reshape(-1).contiguous(),
             "presence_logits": output.presence_logits.reshape(-1).contiguous(),
             "semantic_seg": output.semantic_seg[0].permute(1, 2, 0).contiguous()}
    # The detector reads every FPN level but the coarsest.
    for index, (level, position) in enumerate(zip(vision.fpn_hidden_states[:-1],
                                                  vision.fpn_position_encoding[:-1])):
        extra[f"level_{index}"] = level[0].permute(1, 2, 0).contiguous()
        extra[f"position_{index}"] = position[0].permute(1, 2, 0).contiguous()
    globals()["_extra"] = extra
    return output.pred_masks[0].contiguous()                               # [queries, H, W]


def run_sam2_loss(image):
    """SAM 2's training objective (`MultiStepMultiMasksAndIous`, facebookresearch/sam2), scored on
    random predictions and a target, at the weights the reference's own fine-tuning configuration
    sets: mask 20, dice 1, IoU 1, class 1, every IoU supervised, IoU by L1.

    The loss is the reference's own file, not a reimplementation of its paper: `loss_fns.py` is
    executed with `training.trainer` and `training.utils.distributed` stubbed, since neither the
    trainer nor distribution is needed to score one example. Records the tensors it scored and the
    four terms it returns, each before the weighting, so a port can be compared term by term. Runs
    under any oracle env with torch. `image` unused.

    The source is `IK_SAM2_SRC` (default `~/.inferkit-validation/reference-sources/sam2`), curled
    from the repository rather than cloned.
    """
    import sys
    import types

    directory = os.path.expanduser(os.environ.get("IK_SAM2_SRC",
                                                  "~/.inferkit-validation/reference-sources/sam2"))
    trainer = types.ModuleType("training.trainer")
    trainer.CORE_LOSS_KEY = "core_loss"
    distributed = types.ModuleType("training.utils.distributed")
    distributed.get_world_size = lambda: 1
    distributed.is_dist_avail_and_initialized = lambda: False
    package = types.ModuleType("training")
    utilities = types.ModuleType("training.utils")
    sys.modules.update({"training": package, "training.trainer": trainer,
                        "training.utils": utilities, "training.utils.distributed": distributed})

    namespace = {"__name__": "sam2_loss_fns"}
    with open(os.path.join(directory, "loss_fns.py")) as handle:
        exec(compile(handle.read(), os.path.join(directory, "loss_fns.py"), "exec"), namespace)

    objective = namespace["MultiStepMultiMasksAndIous"](
        weight_dict={"loss_mask": 20, "loss_dice": 1, "loss_iou": 1, "loss_class": 1},
        supervise_all_iou=True, iou_use_l1_loss=True, pred_obj_scores=True,
        focal_gamma_obj_score=0.0, focal_alpha_obj_score=-1.0)

    generator = torch.Generator().manual_seed(17)
    # Three multimask slots over a small map, a target with an object in it, and an IoU head that is
    # wrong enough for its term to carry signal.
    masks = torch.randn(1, 3, 24, 24, generator=generator) * 3
    target = torch.zeros(1, 24, 24)
    target[0, 6:18, 8:20] = 1
    ious = torch.rand(1, 3, generator=generator)
    object_score = torch.randn(1, 1, generator=generator)

    outputs = {"multistep_pred_multimasks_high_res": [masks],
               "multistep_pred_ious": [ious],
               "multistep_object_score_logits": [object_score]}
    losses = objective._forward(outputs, target, 1.0)

    extra = {"masks": masks[0].contiguous(), "target": target[0].contiguous(),
             "ious": ious[0].contiguous(), "object_score": object_score.reshape(-1).contiguous(),
             "loss_mask": losses["loss_mask"].detach().reshape(1).contiguous(),
             "loss_dice": losses["loss_dice"].detach().reshape(1).contiguous(),
             "loss_iou": losses["loss_iou"].detach().reshape(1).contiguous(),
             "loss_class": losses["loss_class"].detach().reshape(1).contiguous()}
    globals()["_extra"] = extra
    return losses["core_loss"].detach().reshape(1).contiguous()            # the weighted total

def run_sam3_loss(image):
    """SAM 3's training objective for a text-only fine-tune (facebookresearch/sam3), scored on random
    predictions and a set of target boxes, at the settings its own
    `configs/odinw13/odinw_text_only_train.yaml` sets: a `BinaryHungarianMatcherV2` (class 2, box 5,
    GIoU 2, focal alpha 0.25, gamma 2), then `Boxes` (L1 5, GIoU 2) and `IABCEMdetr` (classification
    20, presence 20, positive weight 5). That configuration turns segmentation off, so there is no
    mask term to score.

    The loss and the matcher are the reference's own files, executed from a tree under `IK_SAM3_SRC`
    (default `~/.inferkit-validation/reference-sources/sam3`) that was curled rather than cloned. Only
    three leaves are stand-ins, none of them arithmetic the loss depends on: the distributed helpers,
    the metric the loss reports and never trains on, and the focal loss's Triton kernel, which needs
    CUDA and is replaced by the eager form the reference's own file falls back to.

    Records the predictions, the targets, the matcher's assignment, and the four terms, so a port can
    be compared assignment first and then term by term. Runs under the `wananimate` oracle env, which
    needs `scipy` for the reference's `linear_sum_assignment`. `image` unused.
    """
    root = os.path.expanduser(os.environ.get("IK_SAM3_SRC",
                                             "~/.inferkit-validation/reference-sources/sam3"))
    sys.path.insert(0, root)
    from sam3.model.box_ops import box_cxcywh_to_xyxy
    from sam3.train.loss.loss_fns import Boxes, IABCEMdetr
    from sam3.train.matcher import BinaryHungarianMatcherV2

    torch.manual_seed(11)
    queries, count = 12, 3
    logits = torch.randn(1, queries, 1)
    boxes = torch.rand(1, queries, 4) * 0.5 + 0.25
    boxes[..., 2:] = boxes[..., 2:] * 0.4 + 0.05
    presence = torch.randn(1, 1)
    target_boxes = torch.rand(count, 4) * 0.5 + 0.25
    target_boxes[:, 2:] = target_boxes[:, 2:] * 0.4 + 0.05

    outputs = {"pred_logits": logits, "pred_boxes": boxes,
               "pred_boxes_xyxy": box_cxcywh_to_xyxy(boxes), "presence_logit_dec": presence}
    targets = {"boxes": target_boxes, "boxes_xyxy": box_cxcywh_to_xyxy(target_boxes),
               "boxes_padded": target_boxes.unsqueeze(0), "num_boxes": torch.tensor([count]),
               "object_ids_padded": torch.arange(count).unsqueeze(0),
               "is_exhaustive": torch.tensor([True])}

    matcher = BinaryHungarianMatcherV2(focal=True, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0,
                                       alpha=0.25, gamma=2, stable=False)
    indices = matcher(outputs, targets)

    box_loss = Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0})
    class_loss = IABCEMdetr(weight_dict={"loss_ce": 20.0, "presence_loss": 20.0}, pos_weight=5.0,
                            alpha=0.25, gamma=2, use_presence=True, pos_focal=False,
                            pad_n_queries=queries, pad_scale_pos=1.0, weak_loss=False)
    boxes_out = box_loss.get_loss(outputs, targets, indices, float(count))
    class_out = class_loss.get_loss(outputs, targets, indices, float(count))
    total = (boxes_out["loss_bbox"] * 5.0 + boxes_out["loss_giou"] * 2.0
             + class_out["loss_ce"] * 20.0 + class_out["presence_loss"] * 20.0)

    extra = {"logits": logits.reshape(1, queries).contiguous(),
             "boxes": boxes[0].contiguous(),
             "presence": presence.reshape(1).contiguous(),
             "targets": target_boxes.contiguous(),
             "matched": indices[1].to(torch.int32).contiguous(),
             "loss_bbox": boxes_out["loss_bbox"].detach().reshape(1).contiguous(),
             "loss_giou": boxes_out["loss_giou"].detach().reshape(1).contiguous(),
             "loss_ce": class_out["loss_ce"].detach().reshape(1).contiguous(),
             "presence_loss": class_out["presence_loss"].detach().reshape(1).contiguous()}
    globals()["_extra"] = extra
    return total.detach().reshape(1).contiguous()                          # the weighted total


def run_sam2_video(image):
    """SAM 2.1's video tracker (`Sam2VideoModel`, released `facebook/sam2.1-hiera-tiny`) over a
    three-frame clip: one click on the first frame, then two tracked frames.

    This is the whole model, not a seam: the Hiera encoder, the prompt encoder and mask decoder, the
    memory encoder that folds each frame's mask into a memory, and the memory attention that reads
    those memories and the object pointers on the frames after. It also exercises what 2.1 adds over
    2.0 — the occlusion spatial embedding and the projected temporal encoding on the object pointers.
    Records the normalized frames, the click, and per frame the mask logits, the object score, the
    object pointer, and the stored memory, so a divergence localizes to a frame and a stage. Runs
    under the `wananimate` oracle env (transformers >= 5.16, which carries SAM 2 video). `image` unused.
    """
    import numpy as np
    from transformers import Sam2VideoModel
    from transformers.models.sam2_video.modeling_sam2_video import Sam2VideoInferenceSession

    model = Sam2VideoModel.from_pretrained("facebook/sam2.1-hiera-tiny", dtype=torch.float32).eval()

    # A moving disc on a blocky background: the tracker needs something to follow, and a plate of
    # noise gives it nothing, which would compare two near-empty masks.
    generator = np.random.default_rng(3)
    frames = []
    for index in range(3):
        base = generator.random((16, 16, 3), dtype=np.float32)
        frame = np.repeat(np.repeat(base, 64, axis=0), 64, axis=1)
        rows, columns = np.mgrid[0:1024, 0:1024]
        disc = ((rows - 512 - 40 * index) ** 2 + (columns - 512) ** 2) < 200 ** 2
        frame[disc] = 0.9
        frames.append(frame)
    pixel_values = torch.tensor(np.stack(frames)).permute(0, 3, 1, 2)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    deviation = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    pixel_values = (pixel_values - mean) / deviation

    session = Sam2VideoInferenceSession(video=pixel_values, video_height=1024, video_width=1024,
                                        dtype=torch.float32)
    session.obj_id_to_idx(1)
    session.add_point_inputs(0, 0, {"point_coords": torch.tensor([[[[512.0, 512.0]]]]),
                                    "point_labels": torch.tensor([[[1]]], dtype=torch.int32)})
    session.obj_with_new_inputs = [1]

    masks = []
    with torch.no_grad():
        for output in model.propagate_in_video_iterator(session, start_frame_idx=0):
            masks.append(output.pred_masks[0, 0])

    extra = {"frames": pixel_values.permute(0, 2, 3, 1).contiguous(),
             "point": torch.tensor([512.0, 512.0])}
    for index in range(3):
        key = "cond_frame_outputs" if index == 0 else "non_cond_frame_outputs"
        stored = session.output_dict_per_obj[0][key][index]
        extra[f"mask_{index}"] = masks[index].contiguous()
        extra[f"object_score_{index}"] = stored["object_score_logits"].reshape(-1).contiguous()
        extra[f"pointer_{index}"] = stored["object_pointer"].reshape(-1).contiguous()
        extra[f"memory_{index}"] = stored["maskmem_features"].float().permute(1, 0, 2).contiguous()
    globals()["_extra"] = extra
    return torch.stack(masks).contiguous()                                 # [3, 256, 256]


def run_wan_animate(image):
    """The Wan 2.2 Animate 2 DiT at a tiny random configuration, from diffusers'
    `WanAnimate2Transformer3DModel`, across BOTH of its passes.

    Animate adds three things to the Wan text-to-video block this package already ports: an image
    cross-attention branch (`add_k_proj`/`add_v_proj`/`norm_added_k`) fed by an `img_emb` projector
    over CLIP embeddings, and an in-context reference mechanism — a `kv_cache_mode="extract"` pass
    over the reference latents that stores every block's pre-rotary key/value, and a
    `kv_cache_mode="cached"` pass where each generation frame attends over the whole video's
    generation buffer plus the reference tokens at its own frame index. Both passes are recorded, as
    are block 0's output in each and the cache's first layer, so a divergence localizes. Runs under
    the `wananimate` oracle env (diffusers 0.40 with `WanAnimate2Transformer3DModel`). `image` unused.
    """
    from diffusers.models.transformers.transformer_wan_animate_2 import (
        WanAnimate2KVCache, WanAnimate2Transformer3DModel)

    model = WanAnimate2Transformer3DModel(
        patch_size=(1, 2, 2), text_len=16, in_dim=8, dim=32, ffn_dim=48, freq_dim=256,
        text_dim=10, out_dim=4, num_heads=2, num_layers=2, cross_attn_norm=True, eps=1e-6,
        use_img_emb=True, refer_offset_t=1, refer_offset_h=0, refer_offset_w=-1, refer_stride=1)
    model = _randomized(model, seed=31)

    generator = torch.Generator().manual_seed(5)
    # The reference stream is two latent frames, the generation stream three; both are 8x8 latents,
    # so each frame patchifies to a 4x4 grid and the generation stream fills the full video buffer
    # the block mask is built over (`origin_len` 4 -> 2 + 1 latent frames, `origin_area` 64x64 -> 16).
    reference_latent = torch.randn(4, 2, 8, 8, generator=generator)
    reference_condition = torch.randn(4, 2, 8, 8, generator=generator)
    latent = torch.randn(4, 3, 8, 8, generator=generator)
    condition = torch.randn(4, 3, 8, 8, generator=generator)
    text = torch.randn(7, 10, generator=generator)
    image_embeds = torch.randn(1, 5, 1280, generator=generator)
    t = torch.tensor([0.35])
    reference_grid = torch.tensor([[2, 4, 4]], dtype=torch.long)

    seams = {}
    model.blocks[0].register_forward_hook(
        lambda m, i, o, s=seams: s.__setitem__(f"block0_{s['mode']}", o.detach()))
    model.img_emb.register_forward_hook(lambda m, i, o: seams.__setitem__("img_emb", o.detach()))

    cache = WanAnimate2KVCache(2)
    seams["mode"] = "extract"
    with torch.no_grad():
        extract = model(hidden_states=[reference_latent], timestep=t, encoder_hidden_states=[text],
                        condition_latents=[reference_condition], kv_cache=cache,
                        kv_cache_mode="extract", seq_len=32, encoder_hidden_states_image=image_embeds,
                        offset_grid_sizes=reference_grid, return_dict=False)[0][0]
    cached_key, cached_value = cache.get(0).get()

    seams["mode"] = "cached"
    with torch.no_grad():
        cached = model(hidden_states=[latent], timestep=t, encoder_hidden_states=[text],
                       condition_latents=[condition], kv_cache=cache, kv_cache_mode="cached",
                       seq_len=48, encoder_hidden_states_image=image_embeds,
                       reference_grid_sizes=reference_grid, origin_len=4, origin_area=[64, 64],
                       return_dict=False)[0][0]

    # A second generation pass as a CHUNK: four frames scattered into a five-frame video buffer, so
    # the packed generation keys carry a zero-filled frame, and the fourth frame's reference slot sits
    # past the two frames the cache holds and is zero-filled as well. Both are unmasked, so those zero
    # keys enter the softmax denominator without contributing a value. Nothing in the full-length pass
    # above exercises either, and a port that skips the dilution matches the full pass exactly.
    chunk_latent = torch.randn(4, 4, 8, 8, generator=generator)
    chunk_condition = torch.randn(4, 4, 8, 8, generator=generator)
    seams["mode"] = "chunk"
    with torch.no_grad():
        chunk = model(hidden_states=[chunk_latent], timestep=t, encoder_hidden_states=[text],
                      condition_latents=[chunk_condition], kv_cache=cache, kv_cache_mode="cached",
                      seq_len=64, encoder_hidden_states_image=image_embeds,
                      reference_grid_sizes=reference_grid, origin_len=12, origin_area=[64, 64],
                      return_dict=False)[0][0]

    extra = {"reference_latent": reference_latent.contiguous(),
             "reference_condition": reference_condition.contiguous(),
             "latent": latent.contiguous(), "condition": condition.contiguous(),
             "text": text.contiguous(), "image_embeds": image_embeds[0].contiguous(),
             "timestep": t.contiguous(), "extract": extract.contiguous(),
             "img_emb": seams["img_emb"][0].contiguous(),
             "cache_key": cached_key[0].contiguous(), "cache_value": cached_value[0].contiguous(),
             "extract_block0": seams["block0_extract"][0].contiguous(),
             "cached_block0": seams["block0_cached"][0].contiguous(),
             "chunk_latent": chunk_latent.contiguous(),
             "chunk_condition": chunk_condition.contiguous(),
             "chunk": chunk.contiguous(),
             "chunk_block0": seams["block0_chunk"][0].contiguous()}
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return cached.contiguous()                                             # [C, F, H, W]


def run_umt5(image):
    """The umT5 text encoder (`UMT5EncoderModel`, Wan's text encoder) at a tiny random configuration,
    from transformers. umT5 differs from plain T5 in giving EVERY layer its own relative-position bias;
    everything else (T5LayerNorm, the gated FFN, the unscaled attention) is shared with T5. Records the
    last hidden state and the weights. Runs under the `llm` oracle env. `image` unused.
    """
    from transformers import UMT5Config, UMT5EncoderModel

    config = UMT5Config(
        d_model=32, num_layers=3, num_heads=2, d_kv=16, d_ff=64, vocab_size=128,
        relative_attention_num_buckets=16, relative_attention_max_distance=32, dropout_rate=0.0)
    model = _randomized(UMT5EncoderModel(config), seed=47)
    tokens = torch.tensor([[3, 17, 42, 99, 7, 61, 12, 5]], dtype=torch.long)
    with torch.no_grad():
        output = model(input_ids=tokens).last_hidden_state[0]

    extra = {"tokens": tokens[0].to(torch.int32).contiguous()}
    for key, value in model.state_dict().items():
        if key == "encoder.embed_tokens.weight":                          # tied to shared.weight
            continue
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return output.contiguous()                                             # [N, d_model]


def run_gemma2(image):
    """The Gemma-2 text decoder (`Gemma2Model`, SANA's text encoder) at a tiny random configuration,
    from transformers. Records the last hidden state and the weights. Exercises the sandwich norms, the
    (1+w) RMS, the attention logit soft-cap, GQA, and the alternating sliding-window / full attention
    (a small window makes the sliding layers differ from the full ones). Runs under the `llm` oracle env
    (transformers with Gemma2Model). `image` unused.
    """
    from transformers import Gemma2Config, Gemma2Model

    config = Gemma2Config(
        hidden_size=32, num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        intermediate_size=64, vocab_size=128, query_pre_attn_scalar=8, attn_logit_softcapping=50.0,
        sliding_window=3, rope_theta=10000.0, hidden_activation="gelu_pytorch_tanh", max_position_embeddings=64)
    model = _randomized(Gemma2Model(config), seed=43)
    print("layer_types:", config.layer_types)
    tokens = torch.tensor([[3, 17, 42, 99, 7, 61, 12, 5, 88, 30]], dtype=torch.long)
    with torch.no_grad():
        output = model(tokens).last_hidden_state[0]

    extra = {"tokens": tokens[0].to(torch.int32).contiguous()}
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return output.contiguous()                                             # [N, hidden]


def run_dpm_solver(image):
    """SANA's released sampler, `DPMSolverMultistepScheduler` in its flow-prediction configuration, run
    over a FIXED sequence of velocities so the scheduler math is verified with no model. Records the
    sigma schedule, the timesteps, and the sample trajectory. Runs under the `ltx` oracle env.
    """
    from diffusers import DPMSolverMultistepScheduler

    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000, algorithm_type="dpmsolver++", solver_order=2,
        prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0,
        final_sigmas_type="zero", solver_type="midpoint", lower_order_final=True)
    steps = 8
    scheduler.set_timesteps(steps)

    generator = torch.Generator().manual_seed(3)
    initial = torch.randn(4, 4, 4, generator=generator)
    velocities = [torch.randn(4, 4, 4, generator=generator) for _ in range(steps)]

    sample = initial.clone()
    trajectory = []
    for i, t in enumerate(scheduler.timesteps):
        sample = scheduler.step(velocities[i], t, sample, return_dict=False)[0]
        trajectory.append(sample.clone())

    extra = {"sigmas": scheduler.sigmas.float().contiguous(), "timesteps": scheduler.timesteps.float().contiguous(),
             "initial": initial.contiguous(), "velocities": torch.stack(velocities).contiguous(),
             "trajectory": torch.stack(trajectory).contiguous()}
    globals()["_extra"] = extra
    return trajectory[-1].contiguous()


def run_unipc(image):
    """Wan's released sampler, `UniPCMultistepScheduler` in its flow-prediction configuration, run over a
    FIXED sequence of velocities so the predictor-corrector math is verified with no model. Records the
    sigma schedule, timesteps, and the sample trajectory. Runs under the `ltx` oracle env.
    """
    from diffusers import UniPCMultistepScheduler

    scheduler = UniPCMultistepScheduler(
        num_train_timesteps=1000, solver_order=2, solver_type="bh2", predict_x0=True,
        prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=5.0, lower_order_final=True)
    steps = 8
    scheduler.set_timesteps(steps)

    generator = torch.Generator().manual_seed(4)
    initial = torch.randn(4, 4, 4, generator=generator)
    velocities = [torch.randn(4, 4, 4, generator=generator) for _ in range(steps)]

    sample = initial.clone()
    trajectory = []
    for i, t in enumerate(scheduler.timesteps):
        sample = scheduler.step(velocities[i], t, sample, return_dict=False)[0]
        trajectory.append(sample.clone())

    extra = {"sigmas": scheduler.sigmas.float().contiguous(), "timesteps": scheduler.timesteps.float().contiguous(),
             "initial": initial.contiguous(), "velocities": torch.stack(velocities).contiguous(),
             "trajectory": torch.stack(trajectory).contiguous()}
    globals()["_extra"] = extra
    return trajectory[-1].contiguous()


def run_wan_vae(image):
    """The Wan 3D causal VAE (`AutoencoderKLWan`, Wan 2.2 residual path) at a tiny random configuration,
    from diffusers' own class. Exercises the stateful feat_cache streaming loop — the encoder consumes
    frames in chunks (1 then 4) and the decoder emits one latent frame at a time, threading the causal
    convolutions' temporal cache — plus the residual AvgDown3D / DupUp3D shortcuts, the temporal
    resample time_convs, and the patchify (2×). Records the encoded latent mean and the decode. Runs
    under the `ltx` oracle env. `image` unused.
    """
    from diffusers import AutoencoderKLWan

    model = AutoencoderKLWan(
        base_dim=8, decoder_base_dim=8, z_dim=4, dim_mult=[2, 2], num_res_blocks=1, attn_scales=[],
        temperal_downsample=[True], is_residual=True, patch_size=2, in_channels=12, out_channels=12)
    model = _randomized(model, seed=41)

    generator = torch.Generator().manual_seed(13)
    video = torch.randn(1, 3, 5, 16, 16, generator=generator)              # [B, C, T, H, W], 5 frames
    seams = {}
    model.decoder.conv_in.register_forward_hook(lambda m, i, o: seams.__setitem__("dec_conv_in", o.detach()))
    model.decoder.mid_block.register_forward_hook(lambda m, i, o: seams.__setitem__("dec_mid", o.detach()))
    model.decoder.up_blocks[0].register_forward_hook(lambda m, i, o: seams.__setitem__("dec_up0", o.detach()))
    with torch.no_grad():
        latent = model.encode(video).latent_dist.mean
        enc_raw = model._encode(video)                                     # pre-quant-split moments
        enc_1frame = model._encode(video[:, :, 0:1])                       # single-chunk encode
        decoded = model.decode(latent).sample
        decoded_f0 = model.decode(latent[:, :, 0:1]).sample                # first latent frame alone (hooks fire here)

    def ndhwc(t):
        return t[0].permute(1, 2, 3, 0).contiguous()                       # [C,T,H,W] -> [T,H,W,C]

    extra = {"video": ndhwc(video), "latent": ndhwc(latent),
             "decoded_f0": ndhwc(decoded_f0), "enc_raw": ndhwc(enc_raw), "enc_1frame": ndhwc(enc_1frame),
             "dec_conv_in": ndhwc(seams["dec_conv_in"]), "dec_mid": ndhwc(seams["dec_mid"]),
             "dec_up0": ndhwc(seams["dec_up0"])}
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return ndhwc(decoded)                                                  # [T, H, W, 3]


def run_wan_vae_21(image):
    """The Wan 2.1 autoencoder (`AutoencoderKLWan`, the NON-residual path) at a tiny random configuration.
    Wan 2.1 differs from 2.2 in a flat down-block list (no AvgDown3D / DupUp3D shortcuts), a halving
    upsampler, no patchify, and 16 latent channels. Records the encoded latent mean and the decode. Runs
    under the `ltx` oracle env. `image` unused.
    """
    from diffusers import AutoencoderKLWan

    model = AutoencoderKLWan(
        base_dim=8, decoder_base_dim=8, z_dim=4, dim_mult=[2, 2], num_res_blocks=1, attn_scales=[],
        temperal_downsample=[True], is_residual=False, patch_size=None, in_channels=3, out_channels=3)
    model = _randomized(model, seed=53)

    generator = torch.Generator().manual_seed(17)
    video = torch.randn(1, 3, 5, 16, 16, generator=generator)
    with torch.no_grad():
        latent = model.encode(video).latent_dist.mean
        decoded = model.decode(latent).sample

    def ndhwc(t):
        return t[0].permute(1, 2, 3, 0).contiguous()

    extra = {"video": ndhwc(video), "latent": ndhwc(latent)}
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return ndhwc(decoded)


def run_dc_ae(image):
    """The Deep-Compression Autoencoder (SANA's VAE) at a tiny random configuration, from diffusers'
    own AutoencoderDC. Deterministic (encode returns one latent). Exercises a ResBlock stage, an
    EfficientViTBlock stage (multiscale ReLU linear attention + GLUMBConv), the Conv downsample /
    interpolate upsample with their channel-average / channel-repeat shortcuts, and the in/out
    shortcuts. Records the latent and the decode. Runs under the `ltx` oracle env. `image` unused.
    """
    from diffusers import AutoencoderDC

    model = AutoencoderDC(
        in_channels=3, latent_channels=8, attention_head_dim=4,
        encoder_block_types=["ResBlock", "EfficientViTBlock"],
        decoder_block_types=["ResBlock", "EfficientViTBlock"],
        encoder_block_out_channels=[16, 32], decoder_block_out_channels=[16, 32],
        encoder_layers_per_block=[1, 1], decoder_layers_per_block=[1, 1],
        encoder_qkv_multiscales=[(), (3,)], decoder_qkv_multiscales=[(), (3,)],
        downsample_block_type="Conv", upsample_block_type="interpolate")
    model = _randomized(model, seed=37)

    generator = torch.Generator().manual_seed(11)
    sample = torch.randn(1, 3, 8, 8, generator=generator)
    with torch.no_grad():
        latent = model.encode(sample).latent
        decoded = model.decode(latent).sample

    extra = {"sample": sample[0].permute(1, 2, 0).contiguous(),           # [H, W, 3] NHWC
             "latent": latent[0].permute(1, 2, 0).contiguous()}            # [h, w, latent] NHWC
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return decoded[0].permute(1, 2, 0).contiguous()                        # [H, W, 3] NHWC


def run_ip_adapter(image):
    """IP-Adapter's two pieces at a tiny random configuration, from diffusers: the ImageProjection (CLIP
    image embedding → image-text tokens) and the decoupled cross-attention (`IPAdapterAttnProcessor2_0`:
    the text cross-attention plus a scaled image-conditioned attention sharing the query). Records both
    seams. Runs under the `ltx` oracle env. `image` unused.
    """
    from diffusers.models.embeddings import ImageProjection
    from diffusers.models.attention_processor import Attention, IPAdapterAttnProcessor2_0

    torch.manual_seed(59)
    # 1. Image projection.
    proj = ImageProjection(image_embed_dim=16, cross_attention_dim=12, num_image_text_embeds=4)
    proj = _randomized(proj, seed=59)
    image_embeds = torch.randn(1, 16)
    with torch.no_grad():
        ip_tokens = proj(image_embeds)                                     # [1, 4, 12]

    # 2. Decoupled cross-attention.
    attn = Attention(query_dim=12, cross_attention_dim=12, heads=2, dim_head=6)
    processor = IPAdapterAttnProcessor2_0(hidden_size=12, cross_attention_dim=12, num_tokens=[4], scale=0.7)
    attn.set_processor(processor)
    attn = _randomized(attn, seed=61)
    hidden = torch.randn(1, 8, 12)
    text = torch.randn(1, 5, 12)
    with torch.no_grad():
        output = attn(hidden, encoder_hidden_states=(text, [ip_tokens]))   # [1, 8, 12]

    extra = {"image_embeds": image_embeds[0].contiguous(), "ip_tokens": ip_tokens[0].contiguous(),
             "hidden": hidden[0].contiguous(), "text": text[0].contiguous()}
    for key, value in proj.state_dict().items():
        extra[f"wp::{key}"] = value.float().contiguous()
    for key, value in attn.state_dict().items():
        extra[f"wa::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return output[0].contiguous()


def run_ip_adapter_unet(image, checkpoint):
    """The IP-Adapter WIRED INTO the real Stable Diffusion 1.5 UNet, from diffusers — the end-to-end
    validation of the threading, the adapter loader, and the per-layer key/value ordering, not just the
    isolated mechanism `ip_adapter` covers.

    `--checkpoint` is the SD 1.5 release directory (its `unet/` is loaded). The adapter file is
    `IK_VAL_IPADAPTER` (the released `ip-adapter_sd15.safetensors`). diffusers loads the adapter into
    the UNet, projecting the image embedding to image tokens through the UNet's own encoder_hid_proj and
    blending a scaled image-conditioned attention at every cross-attention. Random latent, timestep,
    text context, and CLIP image embedding are recorded so the MLX side loads the SAME base UNet and the
    SAME adapter file and must reproduce the output. Runs under the `sd` (diffusers 0.31) oracle env.
    """
    import os
    import torch
    from diffusers import UNet2DConditionModel
    from safetensors.torch import load_file

    unet = UNet2DConditionModel.from_pretrained(
        os.path.join(checkpoint, "unet"), torch_dtype=torch.float32).eval()
    adapter = os.environ.get("IK_VAL_IPADAPTER",
                             os.path.expanduser("~/.inferkit-validation/ip-adapter/ip-adapter_sd15.safetensors"))
    state = load_file(adapter)
    image_proj = {k[len("image_proj."):]: v for k, v in state.items() if k.startswith("image_proj.")}
    ip_layers = {k[len("ip_adapter."):]: v for k, v in state.items() if k.startswith("ip_adapter.")}
    unet._load_ip_adapter_weights([{"image_proj": image_proj, "ip_adapter": ip_layers}])

    scale = 0.7
    for processor in unet.attn_processors.values():
        if hasattr(processor, "scale"):
            processor.scale = [scale]

    torch.manual_seed(7)
    latent = torch.randn(1, 4, 32, 32)
    timestep = torch.tensor(951)
    text = torch.randn(1, 77, 768)
    image_embeds = torch.randn(1, 1, 1024)                         # [batch, num_images, embed]
    with torch.no_grad():
        output = unet(latent, timestep, encoder_hidden_states=text,
                      added_cond_kwargs={"image_embeds": [image_embeds]}).sample

    globals()["_extra"] = {
        "latent": latent[0].permute(1, 2, 0).contiguous(),        # [H, W, 4] NHWC
        "context": text[0].contiguous(),                          # [77, 768]
        "timestep": timestep.reshape(1).float().contiguous(),
        "image_embeds": image_embeds[0, 0].contiguous(),          # [1024]
    }
    return output[0].permute(1, 2, 0).contiguous()                # [H, W, 4] NHWC


def run_dc_ae_real(image):
    """The Deep-Compression Autoencoder on the RELEASED SANA weights, from diffusers' AutoencoderDC. A
    real-weights end-to-end validation: it loads the actual `Sana_600M` VAE, encodes the plate, and
    decodes, so the released config, the sharded/real checkpoint, and the loader are exercised — not just
    the architecture at a tiny config. Loads from `IK_VAL_DCAE` (default `~/.inferkit-validation/raw/
    sana-dcae`). Runs under the `ltx` oracle env.
    """
    import os
    from diffusers import AutoencoderDC

    directory = os.environ.get("IK_VAL_DCAE", os.path.expanduser("~/.inferkit-validation/raw/sana-dcae"))
    model = AutoencoderDC.from_pretrained(directory, torch_dtype=torch.float32).eval()
    sample = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0) * 2 - 1   # [1, 3, H, W] in -1..1
    with torch.no_grad():
        latent = model.encode(sample).latent
        decoded = model.decode(latent).sample

    globals()["_extra"] = {"latent": latent[0].permute(1, 2, 0).contiguous()} # [h, w, 32] NHWC
    return decoded[0].permute(1, 2, 0).contiguous()                          # [H, W, 3] NHWC (input_image is the plate)


def run_flux_vae(image):
    """The Flux autoencoder (the one Z-Image encodes into) at a tiny random configuration, from
    diffusers' own AutoencoderKL. It is the same class Stable Diffusion uses, differing only in the
    16 latent channels and in dropping the quant convolutions (`use_quant_conv=False`), which is the
    path this record exercises. Records the encoded mean latent and the decode. Runs under the `ltx`
    oracle env. `image` unused.
    """
    from diffusers import AutoencoderKL

    model = AutoencoderKL(
        in_channels=3, out_channels=3, block_out_channels=[8, 16], layers_per_block=1,
        latent_channels=4, norm_num_groups=4, use_quant_conv=False, use_post_quant_conv=False,
        mid_block_add_attention=True,
        down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"])
    model = _randomized(model, seed=31)

    generator = torch.Generator().manual_seed(9)
    sample = torch.randn(1, 3, 16, 16, generator=generator)
    with torch.no_grad():
        mean = model.encode(sample).latent_dist.mean
        decoded = model.decode(mean).sample

    extra = {"sample": sample[0].permute(1, 2, 0).contiguous(),           # [H, W, 3] NHWC
             "mean": mean[0].permute(1, 2, 0).contiguous()}                # [h, w, latent] NHWC
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return decoded[0].permute(1, 2, 0).contiguous()                        # [H, W, 3] NHWC


def run_sam_encoder(image, checkpoint):
    """The official SAM image encoder's neck output — the seam between the ViT and the mask decoder.

    Feed a 1024×1024 plate so `ResizeLongestSide` and the square padding are both identities; then the
    only work on either side is normalization + encoder, and a mismatch localizes to the ViT itself.
    """
    from segment_anything import sam_model_registry, SamPredictor

    predictor = SamPredictor(sam_model_registry[os.environ.get("IK_SAM_VARIANT", "vit_b")](checkpoint=checkpoint).eval())
    predictor.set_image((image * 255).astype(np.uint8))
    features = predictor.features[0]                            # [256, 64, 64]
    return features.permute(1, 2, 0).contiguous()               # [64, 64, 256], matching MLX's NHWC


def run_sam(image, checkpoint):
    """The official SAM mask from a single positive point at (w/2, h/3), as probabilities in 0...1."""
    from segment_anything import sam_model_registry, SamPredictor

    predictor = SamPredictor(sam_model_registry[os.environ.get("IK_SAM_VARIANT", "vit_b")](checkpoint=checkpoint).eval())
    predictor.set_image((image * 255).astype(np.uint8))
    height, width = image.shape[:2]
    point = np.array([[width / 2.0, height / 3.0]])
    masks, scores, _ = predictor.predict(point_coords=point, point_labels=np.array([1]),
                                         multimask_output=True, return_logits=True)
    best = int(np.argmax(scores))
    return torch.sigmoid(torch.from_numpy(masks[best]))         # [H, W] probabilities


def run_sam_decoder(image, checkpoint):
    """All four raw mask-token logits and IoU predictions from the official mask decoder.

    Comparing the selected mask conflates three failure modes: token permutation, logit inversion, and a
    genuine decoder divergence. The full `[4, 256, 256]` token set (`output`) plus `scores` `[4]`
    disentangles them — the Swift side prints the pairwise cosine matrix.
    """
    from segment_anything import sam_model_registry, SamPredictor

    sam = sam_model_registry[os.environ.get("IK_SAM_VARIANT", "vit_b")](checkpoint=checkpoint).eval()
    predictor = SamPredictor(sam)
    predictor.set_image((image * 255).astype(np.uint8))
    height, width = image.shape[:2]
    coords = torch.tensor([[[width / 2.0 + 0.5, height / 3.0 + 0.5]]], dtype=torch.float32)
    labels = torch.tensor([[1]], dtype=torch.int64)
    with torch.no_grad():
        sparse, dense = sam.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        masks, scores = sam.mask_decoder.predict_masks(
            image_embeddings=predictor.features, image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense)
    globals()["_extra"] = {"scores": scores[0].contiguous()}
    return masks[0].contiguous()                                # [4, 256, 256] logits


def run_zero_dce_losses(image):
    """The four zero-reference losses Zero-DCE trains against.

    Transcribed from the reference implementation (Li-Chongyi/Zero-DCE, `Zero-DCE_code/Myloss.py`,
    Guo et al. CVPR 2020) with the `.cuda()` calls removed and the training script's settings applied
    (`L_exp(16, 0.6)`). Several lines look like slips and are reproduced exactly anyway: the released
    weights were trained with these, so matching them is what makes a fine-tune behave like the
    published method. Each is marked below.

    This record isolates the loss arithmetic, so the enhanced image and the curve maps are synthesized
    here and written into the record rather than produced by a network. Both sides then score the SAME
    tensors, and any difference can only come from the losses themselves.
    """
    import torch.nn.functional as F

    def l_exp(x, patch_size=16, mean_val=0.6):
        x = torch.mean(x, 1, keepdim=True)
        mean = F.avg_pool2d(x, patch_size)
        # Squared deviation, not absolute.
        return torch.mean(torch.pow(mean - mean_val, 2))

    def l_color(x):
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        d_rg = torch.pow(mr - mg, 2)
        d_rb = torch.pow(mr - mb, 2)
        d_gb = torch.pow(mb - mg, 2)
        # The differences are squared, then squared AGAIN under the square root: the result is the
        # 4th powers, not the squares. Reproduced as written.
        k = torch.pow(torch.pow(d_rg, 2) + torch.pow(d_rb, 2) + torch.pow(d_gb, 2), 0.5)
        return torch.mean(k)

    def l_tv(x):
        batch_size, _, h_x, w_x = x.size()
        count_h = (h_x - 1) * w_x
        count_w = h_x * (w_x - 1)
        # The sums run over channels too, but the counts do not, so the result carries a factor of
        # the channel count on top of the leading 2.
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, : h_x - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, : w_x - 1], 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def l_spa(org, enhance):
        kernels = [
            [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],                 # left
            [[0, 0, 0], [0, 1, -1], [0, 0, 0]],                 # right
            [[0, -1, 0], [0, 1, 0], [0, 0, 0]],                 # up
            [[0, 0, 0], [0, 1, 0], [0, -1, 0]],                 # down
        ]
        pool = torch.nn.AvgPool2d(4)
        org_pool = pool(torch.mean(org, 1, keepdim=True))
        enhance_pool = pool(torch.mean(enhance, 1, keepdim=True))
        # padding=1 means the border regions difference against zero, so the four directions are not
        # redundant there even though squaring makes the interior pairs symmetric.
        total = 0
        for kernel in kernels:
            weight = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
            d_org = F.conv2d(org_pool, weight, padding=1)
            d_enhance = F.conv2d(enhance_pool, weight, padding=1)
            total = total + torch.pow(d_org - d_enhance, 2)
        return torch.mean(total)

    original = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)     # [1, 3, H, W]
    enhanced = torch.clamp(original * 1.8 + 0.05, 0.0, 1.0)

    generator = np.random.default_rng(11)
    height, width = image.shape[0], image.shape[1]
    blocks = generator.random(((height + 3) // 4, (width + 3) // 4, 24), dtype=np.float32)
    maps = np.repeat(np.repeat(blocks, 4, axis=0), 4, axis=1)[:height, :width, :] * 2 - 1
    maps = np.ascontiguousarray(maps)
    curve_maps = torch.from_numpy(maps).permute(2, 0, 1).unsqueeze(0)    # [1, 24, H, W]

    globals()["_extra"] = {
        "enhanced": enhanced[0].permute(1, 2, 0).contiguous(),           # [H, W, 3]
        "curve_maps": curve_maps[0].permute(1, 2, 0).contiguous(),       # [H, W, 24]
    }
    return torch.stack([
        l_exp(enhanced),
        l_color(enhanced),
        l_tv(curve_maps),
        l_spa(original, enhanced),
    ]).contiguous()


def run_segformer_loss(image):
    """SegFormer's semantic-segmentation training loss.

    Transcribed from `SegformerForSemanticSegmentation.forward` in transformers: the logits are
    bilinearly upsampled to the label resolution (`align_corners=False`) and scored with
    `CrossEntropyLoss`. Downsampling the labels instead would be cheaper and would throw away the thin
    structures segmentation is judged on.

    As with `zero_dce_losses`, this isolates the loss: the logits and labels are synthesized here and
    written into the record, so both sides score identical tensors and the network is factored out.
    """
    import torch.nn.functional as F

    height, width = image.shape[0], image.shape[1]
    classes = 5
    generator = np.random.default_rng(23)
    # Logits at the decode head's own quarter resolution, as the model produces them.
    logits = generator.normal(size=(1, classes, height // 4, width // 4)).astype(np.float32)
    labels = generator.integers(0, classes, size=(1, height, width)).astype(np.int64)

    logits_tensor = torch.from_numpy(logits)
    labels_tensor = torch.from_numpy(labels)
    upsampled = F.interpolate(logits_tensor, size=(height, width), mode="bilinear", align_corners=False)
    loss = torch.nn.CrossEntropyLoss()(upsampled, labels_tensor)

    globals()["_extra"] = {
        "logits": logits_tensor[0].permute(1, 2, 0).contiguous(),        # [h/4, w/4, classes] NHWC
        "labels": labels_tensor[0].to(torch.int32).contiguous(),         # [H, W]
    }
    return loss.reshape(1).contiguous()


def run_fastspeech2(image, checkpoint):
    """The FastSpeech2 conformer acoustic model, from transformers' own implementation, on the
    released espnet/fastspeech2_conformer LJSpeech weights.

    `--checkpoint` is the release DIRECTORY (config.json + pytorch_model.bin). The input is a fixed
    phoneme-id sequence — the ids come from the release's own vocabulary, so no grapheme-to-phoneme
    dependency enters the oracle. The record carries every seam: the encoder output (where a conformer
    mistake shows), the predicted durations (which gate everything downstream — one frame off and the
    mel comparison is meaningless), pitch and energy, and `output` as the post-postnet mel.
    """
    from transformers import FastSpeech2ConformerModel

    model = FastSpeech2ConformerModel.from_pretrained(checkpoint).eval()
    ids = torch.tensor([[13, 5, 30, 22, 17, 41, 9, 25, 33, 4]])
    with torch.no_grad():
        out = model(input_ids=ids, return_dict=True)

    globals()["_extra"] = {
        "tokens": ids[0].to(torch.int32).contiguous(),
        "encoder_hidden": out.encoder_last_hidden_state[0].float().contiguous(),
        "durations": out.duration_outputs[0].to(torch.int32).contiguous(),
        "pitch": out.pitch_outputs[0].float().contiguous(),
        "energy": out.energy_outputs[0].float().contiguous(),
    }
    return out.spectrogram[0].float().contiguous()


def run_hifigan(image, checkpoint):
    """The HiFi-GAN generator (jik876's own models.py) on the released UNIVERSAL_V1 weights.

    `--checkpoint` is the released `g_*` file; the config beside it is the UNIVERSAL_V1 geometry,
    which is also this port's default. The input is a deterministic synthetic mel — the vocoder is a
    pure function of it, so nothing about speech is assumed — recorded beside the waveform it
    produces. The reference fuses its weight norm (`remove_weight_norm`) before running, which is the
    arithmetic the converter bakes into the safetensors.
    """
    import numpy as np, types, json, os

    source = os.path.join(_reference_source(), "hifigan")
    sys.path.insert(0, source)
    # utils.py imports matplotlib for a plotting helper the generator never calls.
    for name in ["matplotlib", "matplotlib.pylab"]:
        module = types.ModuleType(name)
        module.use = lambda *a, **k: None
        sys.modules.setdefault(name, module)
    from models import Generator
    from env import AttrDict

    config_path = os.path.join(os.path.dirname(checkpoint), "hifigan_universal_v1_config.json")
    with open(config_path) as f:
        h = AttrDict(json.load(f))
    generator = Generator(h)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    generator.load_state_dict(state["generator"], strict=True)
    generator.eval()
    generator.remove_weight_norm()

    frames = 60
    grid_m, grid_t = np.meshgrid(np.arange(h.num_mels), np.arange(frames), indexing="ij")
    mel = (np.sin(grid_m * 0.31 + grid_t * 0.17) * 1.5 - 2.0).astype(np.float32)   # [80, frames]

    with torch.no_grad():
        wave = generator(torch.from_numpy(mel).unsqueeze(0))[0, 0]

    globals()["_extra"] = {"mel": torch.from_numpy(mel).contiguous()}
    return wave.float().contiguous()


def run_music_vocoder(image, checkpoint):
    """The MiniMax Music 3 Flow-VAE decoder (diffusers' own MiniMaxMusic3Vocoder) on the released
    vocoder component.

    `--checkpoint` is the release's `vocoder/` DIRECTORY (config.json + the weight-normed float32
    safetensors); diffusers applies the weight norm itself, which is the arithmetic the Swift loader
    fuses at load. The input is a deterministic standard-normal latent `[1, 128, T]` — the vocoder is
    a pure function of it, so nothing about music is assumed — recorded beside the stereo waveform it
    produces. Runs under the `music` oracle environment (diffusers >= 0.40.0)."""
    from diffusers import MiniMaxMusic3Vocoder

    vocoder = MiniMaxMusic3Vocoder.from_pretrained(checkpoint).eval()
    generator = np.random.default_rng(11)
    latents = torch.from_numpy(generator.standard_normal((1, 128, 64)).astype(np.float32))
    with torch.no_grad():
        wave = vocoder(latents)[0]

    globals()["_extra"] = {"latents": latents[0].contiguous()}
    return wave.float().contiguous()


def run_music_depth(image, checkpoint):
    """The MiniMax Music 3 RVQ depth decoder (diffusers' own MiniMaxMusic3RVQDepthDecoder) on the
    released component.

    `--checkpoint` is the release's `rvq_depth_decoder/` DIRECTORY. The record covers every parameter
    family: the transformer forward on a deterministic depth sequence (position embedding, the four
    causal blocks, the final norm), all seven codebook heads on the last step, the shared projection,
    and the offset-packed residual embedding table. The release ships bf16; both sides run float32.
    Runs under the `music` oracle environment (diffusers >= 0.40.0)."""
    from diffusers import MiniMaxMusic3RVQDepthDecoder

    # `IK_MUSIC_DEPTH_DTYPE=bfloat16` runs the release's own bf16 with every block probed piece by piece
    # (`enc.<layer>.<submodule>`); `bfloat16-inputs` keeps float32 arithmetic on the same bf16 inputs.
    mode = os.environ.get("IK_MUSIC_DEPTH_DTYPE")
    dtype = torch.bfloat16 if mode == "bfloat16" else torch.float32
    decoder = MiniMaxMusic3RVQDepthDecoder.from_pretrained(checkpoint, torch_dtype=dtype).eval()
    generator = np.random.default_rng(13)
    inputs = torch.from_numpy(generator.standard_normal((2, 8, 4096)).astype(np.float32))
    projection_input = torch.from_numpy(generator.standard_normal((2, 4096)).astype(np.float32))
    ids = torch.from_numpy(generator.integers(0, 1024 * 7, size=(2, 7)))
    probes = {}
    if mode:
        inputs = inputs.to(torch.bfloat16).to(dtype)
        projection_input = projection_input.to(torch.bfloat16).to(dtype)
        _probe_encoder_layers(decoder.layers, probes)
    with torch.no_grad():
        hidden = decoder(inputs)
        head_logits = torch.stack([head(hidden[:, -1]) for head in decoder.audio_heads])
        projected = decoder.projection(projection_input)
        embedded = decoder.audio_embeddings(ids)

    globals()["_extra"] = {
        "inputs_embeds": inputs.contiguous(),
        "head_logits": head_logits.float().contiguous(),
        "projection_input": projection_input.contiguous(),
        "projected": projected.float().contiguous(),
        "embedding_ids": ids.to(torch.int32).contiguous(),
        "embedded": embedded.float().contiguous(),
        **probes,
    }
    return hidden.float().contiguous()


def run_music_condition(image, checkpoint):
    """The MiniMax Music 3 condition encoder (diffusers' own MiniMaxMusic3ConditionEncoder) on the
    released component.

    `--checkpoint` is the release's `condition_encoder/` DIRECTORY. The input is a deterministic
    stand-in for the fused per-frame hidden states `[1, frames, 8 * 4096]`; the output is the
    latent-aligned conditioning, which pins the learned softmax blend, the projection, and the exact
    nearest-neighbor resample from 13 frames to int(13 * 44100/24000 * 960/512) = 44 latents.
    Runs under the `music` oracle environment (diffusers >= 0.40.0)."""
    from diffusers import MiniMaxMusic3ConditionEncoder

    encoder = MiniMaxMusic3ConditionEncoder.from_pretrained(checkpoint, torch_dtype=torch.float32).eval()
    generator = np.random.default_rng(17)
    hidden = torch.from_numpy(generator.standard_normal((1, 13, 8 * 4096)).astype(np.float32))
    with torch.no_grad():
        condition = encoder(hidden)

    globals()["_extra"] = {"hidden_states": hidden[0].contiguous()}
    return condition[0].float().contiguous()


def run_music_dit(image, checkpoint):
    """The MiniMax Music 3 flow-matching DiT (diffusers' own MiniMaxMusic3Transformer1DModel) on the
    released component, plus the sigma schedule from diffusers' own FlowMatchEulerDiscreteScheduler.

    `--checkpoint` is the RELEASE ROOT (the transformer loads from its `transformer/` subfolder and
    the scheduler config from `scheduler/`). Velocities are recorded at three timesteps and for the
    zero-condition unconditional branch, because the CFG path runs both; the sigma record pins the
    `invert_sigmas` schedule the pipeline drives the sampler with. The component ships float32.
    Runs under the `music` oracle environment (diffusers >= 0.40.0)."""
    import os
    from diffusers import FlowMatchEulerDiscreteScheduler, MiniMaxMusic3Transformer1DModel

    model = MiniMaxMusic3Transformer1DModel.from_pretrained(
        os.path.join(checkpoint, "transformer"), torch_dtype=torch.float32).eval()
    generator = np.random.default_rng(19)
    length = 24
    latents = torch.from_numpy(generator.standard_normal((1, 128, length)).astype(np.float32))
    condition = torch.from_numpy((generator.standard_normal((1, length, 2048)) * 0.5).astype(np.float32))

    extra = {"latents": latents[0].contiguous(), "condition": condition[0].contiguous()}
    with torch.no_grad():
        for name, t in (("t0", 0.0), ("tmid", 0.5), ("tlate", 1.0 - 1.0 / 30.0)):
            velocity = model(latents, torch.tensor([t]), condition, return_dict=False)[0]
            extra[f"velocity_{name}"] = velocity[0].float().contiguous()
        unconditional = model(latents, torch.tensor([0.5]), torch.zeros_like(condition), return_dict=False)[0]
        extra["velocity_unconditional"] = unconditional[0].float().contiguous()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(checkpoint, "scheduler"))
    steps = 30
    scheduler.set_timesteps(sigmas=np.linspace(1.0, 1.0 / steps, steps))
    extra["sigmas"] = scheduler.sigmas.float().contiguous()
    extra["timesteps"] = scheduler.timesteps.float().contiguous()

    globals()["_extra"] = extra
    # `output` must not alias an extra: safetensors refuses tensors that share memory.
    return extra["velocity_tmid"].clone()


def run_music_ar(image, checkpoint):
    """The MiniMax Music 3 autoregressive stage: the released Qwen3-8B language model and the depth
    decoder driven by the Diffusers pipeline's OWN helper functions (`_sample_top_k`,
    `_generate_depth_codes`, `_embed_audio_frame`, the prompt cleaners), teacher-forceably.

    `--checkpoint` is the RELEASE ROOT. The record carries the assembled prompt pair, every sampled
    code per loop iteration (so the Swift side replays the same choices and the comparison measures
    the networks rather than two random streams), the prompt prefill's last hidden state, the first
    step's raw and guided logits, and the fused per-frame hidden states that condition synthesis.
    Both sides run bf16 — the model does not fit this machine at float32.
    Runs under the `music` oracle environment (diffusers >= 0.40.0, transformers >= 5)."""
    import os

    from diffusers import MiniMaxMusic3RVQDepthDecoder
    from diffusers.modular_pipelines.minimax_music3 import encoders as ref
    from transformers import Qwen2Tokenizer, Qwen3ForCausalLM

    lm = Qwen3ForCausalLM.from_pretrained(
        os.path.join(checkpoint, "language_model"), torch_dtype=torch.bfloat16).eval()
    depth = MiniMaxMusic3RVQDepthDecoder.from_pretrained(
        os.path.join(checkpoint, "rvq_depth_decoder"), torch_dtype=torch.bfloat16).eval()
    tokenizer = Qwen2Tokenizer.from_pretrained(os.path.join(checkpoint, "tokenizer"))

    # Hoisted out of the f-string: a backslash inside an f-string expression parses only from
    # Python 3.12, and the LLM oracle environment is 3.9, so this file has to stay parseable there.
    lyrics = "[verse]\\nHello world"
    text = (
        f"{ref._IM_START}{ref._CAPTION_START}{ref._clean_caption('Dreamy synth-pop, female vocals')}"
        f"{ref._CAPTION_END}{ref._LYRICS_START}{ref._normalize_lyrics(lyrics)}"
        f"{ref._LYRICS_END}{ref._IM_END}{ref._AUDIO_START}"
    )
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    unconditional = input_ids.clone()
    unconditional[:, 1:-2] = ref._AUDIO_CFG_TOKEN_ID
    text_ids = torch.cat((input_ids, unconditional), dim=0)

    generator = torch.Generator().manual_seed(11)
    max_frames = 3

    with torch.no_grad():
        text_embeds = lm.model.embed_tokens(text_ids)
        output = lm.model(inputs_embeds=text_embeds, use_cache=True)
        past = output.past_key_values
        last_hidden = output.last_hidden_state[:, -1]
        prefill_hidden = last_hidden.float().clone()

        vocab_mask = torch.ones(lm.config.vocab_size, dtype=torch.bool)
        vocab_mask[ref._AUDIO_CODE_OFFSET : ref._AUDIO_CODE_OFFSET + ref._SEMANTIC_VOCAB_SIZE] = False
        vocab_mask[ref._AUDIO_END_TOKEN_ID] = False

        first_logits = None
        first_guided = None
        frame_hiddens = []
        frame_codes_list = []
        for frame_index in range(max_frames + 1):
            raw = lm.lm_head(last_hidden).float()
            logits = raw.masked_fill(vocab_mask, -float("inf"))
            conditional, uncond_row = logits[0:1], logits[1:2]
            guided = uncond_row + (conditional - uncond_row) * ref._AR_CFG_SCALE
            threshold = torch.topk(conditional, ref._AR_CFG_TOP_K, dim=-1).values[..., -1, None]
            guided = guided.masked_fill(conditional < threshold, -float("inf"))
            guided = guided.masked_fill(vocab_mask.unsqueeze(0), -float("inf"))
            if first_logits is None:
                first_logits = raw.clone()
                first_guided = guided.clone()
            sampled = ref._sample_top_k(guided, generator)
            if int(sampled.item()) == ref._AUDIO_END_TOKEN_ID:
                break
            semantic_code = sampled - ref._AUDIO_CODE_OFFSET
            frame_codes, depth_hidden = ref._generate_depth_codes(
                lm, depth, last_hidden, semantic_code.repeat(2), generator)
            frame_codes_list.append(frame_codes[0].tolist())
            if frame_index > 0:
                frame_hiddens.append(torch.cat((last_hidden[:1], depth_hidden), dim=-1).float())
                if len(frame_hiddens) >= max_frames:
                    break
            feedback = ref._embed_audio_frame(lm, depth, frame_codes)
            output = lm.model(inputs_embeds=feedback, past_key_values=past, use_cache=True)
            past = output.past_key_values
            last_hidden = output.last_hidden_state[:, -1]

    stacked = torch.stack(frame_hiddens, dim=1)[0]
    globals()["_extra"] = {
        "text_ids": text_ids.to(torch.int32).contiguous(),
        "codes": torch.tensor(frame_codes_list, dtype=torch.int32),
        "prefill_hidden": prefill_hidden.contiguous(),
        "first_logits": first_logits.contiguous(),
        "first_guided": torch.nan_to_num(first_guided, neginf=-1e9)[0].contiguous(),
        "frame_hiddens": stacked.contiguous(),
    }
    return stacked.clone()


# The music prompt cases, mirrored verbatim by NFKMLXMusic3Tests: markdown stripping, the
# <|tag value|> rewrite, structure-tag normalization (text on a tag line is dropped, inline tags
# split onto their own lines, uppercase tags lowercase), multi-byte text, and whitespace forms.
MUSIC_PROMPTS = [
    ("Dreamy synth-pop, female vocals", "[verse]\nHello world"),
    ("# Epic Rock\n- **loud** guitars\n* driving *rhythm*\n---\n• four    spaces",
     "[Verse] ignored text\nFirst line [Chorus] second"),
    ("<|bpm 128|> J-ポップ \U0001f3b5 vocals", "[intro]\nこんにちは ^ 世界"),
    ("  jazz,  with\n\n\nswing!  ", "[verse]\nline one  \n[bridge]\nend"),
]


def run_music_tokenizer(image, checkpoint):
    """The MiniMax Music 3 prompt contract: the Diffusers pipeline's OWN cleaners
    (`_clean_caption`, `_normalize_lyrics`), the special-token template, the release tokenizer, and
    the CFG-row substitution, over MUSIC_PROMPTS.

    `--checkpoint` is the RELEASE ROOT (the tokenizer loads from `tokenizer/`). Each case records
    the `[2, L]` conditional/unconditional id pair; the Swift side rebuilds the same prompts through
    the core byte-level BPE and must match token for token — even whitespace-level changes to the
    assembled prompt change the generated audio.
    Runs under the `music` oracle environment (diffusers >= 0.40.0, transformers >= 5)."""
    import os

    from diffusers.modular_pipelines.minimax_music3 import encoders as ref
    from transformers import Qwen2Tokenizer

    tokenizer = Qwen2Tokenizer.from_pretrained(os.path.join(checkpoint, "tokenizer"))
    extra = {}
    for index, (caption, lyrics) in enumerate(MUSIC_PROMPTS):
        text = (
            f"{ref._IM_START}{ref._CAPTION_START}{ref._clean_caption(caption)}{ref._CAPTION_END}"
            f"{ref._LYRICS_START}{ref._normalize_lyrics(lyrics)}{ref._LYRICS_END}"
            f"{ref._IM_END}{ref._AUDIO_START}"
        )
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        unconditional = input_ids.clone()
        unconditional[:, 1:-2] = ref._AUDIO_CFG_TOKEN_ID
        extra[f"case{index}_ids"] = torch.cat((input_ids, unconditional), dim=0).to(torch.int32)
    extra["case_count"] = torch.tensor([len(MUSIC_PROMPTS)], dtype=torch.int32)
    globals()["_extra"] = extra
    return extra["case0_ids"][0].clone()


def run_deepseek_v41_vl_router_bf16(image):
    """The same router and mixture in bf16, as the release runs them. Run under the gemma environment."""
    return run_deepseek_v41_vl_router(image, bf16=True)


def run_deepseek_v41_vl_router(image, bf16=False):
    """DeepSeek V4.1's router where a token sits inside an IMAGE SPAN.

    The correction bias steers which experts a token selects, and a release with a vision tower
    carries a second one for image spans (`noaux_tc_for_vl` in training). A port that routes those
    tokens with the text bias picks different experts for them and is wrong only where an image is
    present, which a text-only parity run cannot see. `vision_n_layers` is 1 here purely because
    that is what makes the reference build `bias_vl` at all.
    """
    import torch

    _deepseek_v41_kernel_shim()
    source = os.path.expanduser(os.environ.get(
        "IK_DEEPSEEK_V41_SRC", "~/.inferkit-validation/reference-sources/deepseek-v41"))
    sys.path.insert(0, source)
    import model as reference

    args = reference.ModelArgs(
        max_batch_size=1, max_seq_len=32, dtype="bf16", expert_dtype=None,
        vocab_size=128, dim=32, moe_inter_dim=16, n_layers=1, n_mtp_layers=0,
        n_heads=2, n_routed_experts=8, n_activated_experts=2,
        q_lora_rank=16, head_dim=16, rope_head_dim=4, o_groups=1, o_lora_rank=8,
        window_size=8, compress_ratios=(0,), kv_source_layers=(), index_source_layers=(),
        index_n_heads=2, index_head_dim=8, index_topk=4, hc_mult=2, hc_sinkhorn_iters=2,
        vision_n_layers=1, vision_dim=16, vision_n_heads=2, vision_inter_dim=32,
    )
    torch.manual_seed(0)
    if bf16:
        net = _deepseek_v41_bf16(reference, lambda: reference.Transformer(args), seed=5)
    else:
        net = _randomized(reference.Transformer(args).float(), seed=5)
    gate = net.layers[0].ffn.gate
    assert gate.bias_vl is not None, "a vision-enabled release carries the second bias"

    torch.manual_seed(3)
    hidden = torch.randn(9, args.dim)
    if bf16:
        hidden = hidden.to(torch.bfloat16)
    mask = torch.zeros(9, dtype=torch.bool)
    mask[3:7] = True                      # one image span, text either side
    with torch.inference_mode():
        text_weights, text_indices = gate(hidden, None)
        vl_weights, vl_indices = gate(hidden, mask)
        lifted = net.layers[0].ffn(hidden.unsqueeze(0), mask.unsqueeze(0))

    moved = int((text_indices != vl_indices).any(dim=-1).sum())
    assert moved > 0, "the image bias has to move at least one token's experts to be worth testing"
    extra = {
        "hidden": hidden.float().contiguous(),
        "image_mask": mask.to(torch.int32).contiguous(),
        "text_indices": text_indices.to(torch.int32).contiguous(),
        "vl_indices": vl_indices.to(torch.int32).contiguous(),
        # Cloned because the harness writes the RETURN value as `output`, and this is that same
        # tensor: safetensors refuses two names for one buffer.
        "vl_weights": vl_weights.float().clone().contiguous(),
        "ffn_vl": lifted[0].float().contiguous(),
        "tokens_rerouted": torch.tensor([moved], dtype=torch.int32),
    }
    # Cloned, not merely made contiguous: `_randomized` can leave entries of this state dict sharing
    # storage, and safetensors refuses to write an alias.
    for key, value in net.layers[0].ffn.state_dict().items():
        extra[f"w::{key}"] = (value.float() if value.is_floating_point()
                              else value).clone().contiguous()
    globals()["_extra"] = extra
    return vl_weights.float().contiguous()


def run_deepseek_v41_tokens(image, checkpoint):
    """The collapsed id space DeepSeek V4.1's n-gram memory hashes over, from the release's own code.

    `--checkpoint` is the release's `tokenizer.json`. `engram.build_compressed_token_map` decides
    which token ids share a hash bucket, and the SIZE it returns is what every hash multiplier is
    derived from, so a port that collapses differently hashes every n-gram to a different row of a
    384-million-row table. The whole lookup is recorded rather than the size alone: two derivations
    can agree on how many buckets there are and still disagree about which ids share one.
    """
    import torch
    from tokenizers import Tokenizer

    source = os.path.expanduser(os.environ.get(
        "IK_DEEPSEEK_V41_SRC", "~/.inferkit-validation/reference-sources/deepseek-v41"))
    sys.path.insert(0, source)
    from engram import build_compressed_token_map

    backend = Tokenizer.from_file(checkpoint)

    # `build_compressed_token_map` reaches for `len(tokenizer)` and `tokenizer.backend_tokenizer`,
    # which is the transformers wrapper's shape rather than the Rust tokenizer's.
    class _Wrapper:
        def __init__(self, backend, size):
            self.backend_tokenizer = backend
            self._size = size

        def __len__(self):
            return self._size

    size = backend.get_vocab_size(with_added_tokens=True)
    lookup, collapsed = build_compressed_token_map(_Wrapper(backend, size))
    globals()["_extra"] = {
        "lookup": torch.tensor(lookup, dtype=torch.int32).contiguous(),
        "collapsed_size": torch.tensor([collapsed], dtype=torch.int32),
        "vocab_size": torch.tensor([size], dtype=torch.int32),
    }
    return torch.tensor([float(collapsed)])


def run_deepseek_v4(image, checkpoint):
    """The DeepSeek V4 decoder's arithmetic, from transformers' own implementation, at a tiny
    configuration with every layer sliding attention.

    The released weights cannot run on any machine here, and DeepSeek's own `inference/model.py`
    imports GPU-only tilelang kernels — but transformers main now carries a complete plain-PyTorch
    implementation (compressor, indexer, hyper-connections, both routers), which is a genuine
    third-party oracle for the arithmetic at a size that fits.

    Every layer is `sliding_attention` and the sequence is shorter than the window, so the reference's
    attention degenerates to exactly the dense-with-sink path the port computes and no compressed KV
    entry exists (only a CLOSED window emits one). That measures the MLA projections, the per-head
    query norm, the trailing interleaved rope, the sink softmax, the output de-rotation, the grouped
    output projection, both routers, the clamped SwiGLU experts, and the hyper-connections.

    The record's weights are saved in the RELEASE naming (transformers' own rename table inverted,
    the fused per-expert tensors split back), so the Swift module — whose keys are the release's —
    loads them strictly with no new loading code. `checkpoint` is unused; pass anything.
    """
    from transformers import DeepseekV4Config, DeepseekV4ForCausalLM

    config = DeepseekV4Config(
        hidden_size=64, num_hidden_layers=4, vocab_size=128,
        num_attention_heads=4, head_dim=16, qk_rope_head_dim=4,
        q_lora_rank=16, o_lora_rank=16, o_groups=2, sliding_window=16,
        n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2,
        moe_intermediate_size=32, intermediate_size=32, num_hash_layers=1,
        index_n_heads=4, index_head_dim=8, index_topk=4,
        hc_mult=4, first_k_dense_replace=0, max_position_embeddings=64,
        layer_types=["sliding_attention"] * 4,
        rms_norm_eps=1e-6, routed_scaling_factor=1.5, swiglu_limit=10.0,
        scoring_func="sqrtsoftplus", rope_theta=10000.0,
        hc_sinkhorn_iters=20, hc_eps=1e-6)

    torch.manual_seed(11)
    model = DeepseekV4ForCausalLM(config).eval().float()
    # `nn.Parameter(torch.empty(...))` fields depend on _init_weights; randomize EVERYTHING
    # explicitly so the record does not depend on transformers' init policy of the week.
    state = model.state_dict()
    for key in sorted(state):
        if key.endswith("tid2eid"):
            state[key] = torch.randint(0, config.n_routed_experts,
                                       state[key].shape, dtype=state[key].dtype)
        elif not state[key].is_floating_point():
            continue
        else:
            state[key] = torch.randn(state[key].shape) * 0.05
    model.load_state_dict(state)

    tokens = torch.tensor([[3, 17, 42, 99, 7, 61]], dtype=torch.long)
    with torch.no_grad():
        out = model(tokens, output_hidden_states=True)

    # transformers names -> the release's, which are the Swift module's own keys.
    import re
    def release_name(key):
        rules = [
            (r"^model\.embed_tokens\.weight$", "embed.weight"),
            (r"^lm_head\.weight$", "head.weight"),
            (r"^model\.hc_head\.hc_fn$", "hc_head_fn"),
            (r"^model\.hc_head\.hc_base$", "hc_head_base"),
            (r"^model\.hc_head\.hc_scale$", "hc_head_scale"),
            (r"^model\.norm\.weight$", "norm.weight"),
            (r"^model\.", ""),
        ]
        for pattern, replacement in rules:
            key = re.sub(pattern, replacement, key)
        key = key.replace(".self_attn.", ".attn.")
        key = key.replace(".mlp.", ".ffn.")
        key = key.replace(".attn_hc.fn", ".hc_attn_fn")
        key = key.replace(".attn_hc.base", ".hc_attn_base")
        key = key.replace(".attn_hc.scale", ".hc_attn_scale")
        key = key.replace(".ffn_hc.fn", ".hc_ffn_fn")
        key = key.replace(".ffn_hc.base", ".hc_ffn_base")
        key = key.replace(".ffn_hc.scale", ".hc_ffn_scale")
        key = key.replace(".input_layernorm.", ".attn_norm.")
        key = key.replace(".post_attention_layernorm.", ".ffn_norm.")
        key = key.replace(".sinks", ".attn_sink")
        key = key.replace(".q_a_proj.", ".wq_a.")
        key = key.replace(".q_a_norm.", ".q_norm.")
        key = key.replace(".q_b_proj.", ".wq_b.")
        key = key.replace(".kv_proj.", ".wkv.")
        key = key.replace(".kv_norm.", ".norm.")
        key = key.replace(".o_a_proj.", ".wo_a.")
        key = key.replace(".o_b_proj.", ".wo_b.")
        key = key.replace(".gate.e_score_correction_bias", ".gate.bias")
        key = key.replace(".shared_experts.gate_proj.", ".shared_experts.w1.")
        key = key.replace(".shared_experts.down_proj.", ".shared_experts.w2.")
        key = key.replace(".shared_experts.up_proj.", ".shared_experts.w3.")
        return key

    extra = {"tokens": tokens[0].to(torch.int32).contiguous()}
    for index, hidden in enumerate(out.hidden_states):
        extra[f"hidden.{index}"] = hidden[0].float().contiguous()      # [S, hc_mult, D]
    for key, value in model.state_dict().items():
        name = release_name(key)
        if name.endswith(".experts.gate_up_proj"):
            base = name[: -len("gate_up_proj")]
            gate, up = value.chunk(2, dim=1)                # [E, 2I, H] -> two [E, I, H]
            for expert in range(value.shape[0]):
                extra[f"w::{base}{expert}.w1.weight"] = gate[expert].contiguous()
                extra[f"w::{base}{expert}.w3.weight"] = up[expert].contiguous()
        elif name.endswith(".experts.down_proj"):
            base = name[: -len("down_proj")]
            for expert in range(value.shape[0]):
                extra[f"w::{base}{expert}.w2.weight"] = value[expert].contiguous()
        else:
            extra[f"w::{name}"] = (value.float() if value.is_floating_point()
                                   else value).contiguous()
    globals()["_extra"] = extra
    return out.logits[0].float().contiguous()


def run_deepseek_v41_quant(image, checkpoint):
    """The dequantization of real DeepSeek V4.1 weights, in BOTH of its fp8 blockings.

    `--checkpoint` is the release's resolve base (`https://huggingface.co/<repo>/resolve/main`).
    Only the bytes of the tensors read here are fetched, through HTTP range requests, so the record
    costs a few megabytes against a 510 GB release. The shard each tensor lives in comes from the
    stored index (`IK_INDEX_DEEPSEEK_V41`), because the two are in different shards.

    V4.1 blocks its fp8 weights at 32, not V4's 128, and it uses two DIFFERENT blockings that a
    single rule cannot cover. An attention weight is blocked SQUARELY: `[512, 5120]` carries a
    `[16, 160]` scale. The n-gram table is blocked ROW-WISE: `[384006168, 256]` carries a
    `[384006168, 8]` scale, one scale per 32 columns of each row and no sharing down the rows. A
    decoder that repeats the scale along both axes reads the wrong scale for every row of the second.
    """
    import json, os, struct, subprocess
    import torch

    index_path = os.path.expanduser(os.environ.get(
        "IK_INDEX_DEEPSEEK_V41",
        "~/.inferkit-validation/shapes/deepseek-v4.1-flash/model.safetensors.index.json"))
    weight_map = json.load(open(index_path))["weight_map"]

    def fetch(url, rng):
        done = subprocess.run(["curl", "-sSL", "--fail", "-m", "600", "-A", "InferKit/0.1",
                               "-H", f"Range: bytes={rng}", url], capture_output=True)
        if done.returncode != 0:
            raise SystemExit(f"range request failed: {done.stderr.decode()[:200]}")
        return done.stdout

    headers = {}

    def header(shard):
        if shard not in headers:
            url = f"{checkpoint.rstrip('/')}/{shard}"
            length = struct.unpack("<Q", fetch(url, "0-7"))[0]
            headers[shard] = (url, json.loads(fetch(url, f"8-{8 + length - 1}")), 8 + length)
        return headers[shard]

    def whole(key):
        url, head, base = header(weight_map[key])
        entry = head[key]
        start, end = entry["data_offsets"]
        raw = fetch(url, f"{base + start}-{base + end - 1}")
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(*entry["shape"]), entry

    def rows(key, first, count):
        """`count` whole rows of a 2-D uint8 tensor, without fetching the other 98 GB of it."""
        url, head, base = header(weight_map[key])
        entry = head[key]
        start, _ = entry["data_offsets"]
        width = entry["shape"][1]
        begin = base + start + first * width
        raw = fetch(url, f"{begin}-{begin + count * width - 1}")
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(count, width)

    extra = {}

    # Square 32: an attention weight, whole.
    weight, meta = whole("layers.0.attn.wkv.weight")
    scale, scale_meta = whole("layers.0.attn.wkv.scale")
    values = weight.view(torch.float8_e4m3fn).float()
    scales = scale.view(torch.float8_e8m0fnu).float()
    spread = scales.repeat_interleave(32, 0).repeat_interleave(32, 1)
    spread = spread[: values.shape[0], : values.shape[1]]
    extra["square_bytes"] = weight.clone()
    extra["square_scale_bytes"] = scale.clone()
    extra["square_expected"] = (values * spread).contiguous()
    extra["square_shape"] = torch.tensor(meta["shape"], dtype=torch.int32)

    # Row-wise 32: eight rows of the n-gram table, and eight more from deep inside it so a large
    # offset is exercised rather than only the first bytes of the shard.
    first, deep, count = 0, 384_006_000, 8
    for label, start in (("near", first), ("deep", deep)):
        table = rows("layers.1.engram.embed.weight", start, count)
        table_scale = rows("layers.1.engram.embed.scale", start, count)
        values = table.view(torch.float8_e4m3fn).float()
        scales = table_scale.view(torch.float8_e8m0fnu).float()
        spread = scales.repeat_interleave(32, 1)[:, : values.shape[1]]
        extra[f"rowwise_{label}_bytes"] = table.clone()
        extra[f"rowwise_{label}_scale_bytes"] = table_scale.clone()
        extra[f"rowwise_{label}_expected"] = (values * spread).contiguous()
    extra["rowwise_shape"] = torch.tensor([count, 256], dtype=torch.int32)

    globals()["_extra"] = extra
    return extra["square_expected"][:4, :8].clone().contiguous()


def run_deepseek_quant(image, checkpoint):
    """The dequantization of a real DeepSeek V4 weight, both formats the release stores.

    `--checkpoint` is a safetensors SHARD URL of the release. Only the two tensors this reads are
    fetched, through HTTP range requests, so the record costs a few megabytes rather than the
    checkpoint's hundreds of gigabytes.

    torch decodes the fp8 side: it has `float8_e4m3fn` and `float8_e8m0fnu` on the CPU, so the
    expectation is the reference's own arithmetic. It has NO CPU kernel for `float4_e2m1fn_x2`, so the
    4-bit side decodes through `ml_dtypes`, whose `float4_e2m1fn` is the same format from a different
    vendor. Needs the Python 3.12 interpreter that `oracle_environments` records for gemma; the vision
    oracles' torch predates both dtypes.
    """
    import json, struct, subprocess
    import numpy as np, ml_dtypes

    def fetch(rng):
        done = subprocess.run(["curl", "-sSL", "--fail", "-m", "600", "-A", "InferKit/0.1",
                               "-H", f"Range: bytes={rng}", checkpoint], capture_output=True)
        if done.returncode != 0:
            raise SystemExit(f"range request failed: {done.stderr.decode()[:200]}")
        return done.stdout

    length = struct.unpack("<Q", fetch("0-7"))[0]
    header = json.loads(fetch(f"8-{8 + length - 1}"))
    base = 8 + length

    def tensor(key):
        entry = header[key]
        start, end = entry["data_offsets"]
        return fetch(f"{base + start}-{base + end - 1}"), entry

    extra = {}

    raw, meta = tensor("layers.0.attn.wkv.weight")
    scale_raw, scale_meta = tensor("layers.0.attn.wkv.scale")
    weight = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(*meta["shape"])
    scale = torch.frombuffer(bytearray(scale_raw), dtype=torch.uint8).view(*scale_meta["shape"])
    values = weight.view(torch.float8_e4m3fn).float()
    scales = scale.view(torch.float8_e8m0fnu).float()
    spread = scales.repeat_interleave(128, 0).repeat_interleave(128, 1)[: values.shape[0], : values.shape[1]]
    extra["fp8_bytes"] = weight.clone()
    extra["fp8_scale_bytes"] = scale.clone()
    fp8_expected = (values * spread).contiguous()

    raw, meta = tensor("layers.0.ffn.experts.0.w1.weight")
    scale_raw, scale_meta = tensor("layers.0.ffn.experts.0.w1.scale")
    packed = np.frombuffer(raw, dtype=np.uint8).reshape(*meta["shape"])
    scale = np.frombuffer(scale_raw, dtype=np.uint8).reshape(*scale_meta["shape"])
    # A byte holds the earlier value in its low nibble.
    low = (packed & 0x0F).view(ml_dtypes.float4_e2m1fn).astype(np.float32)
    high = (packed >> 4).view(ml_dtypes.float4_e2m1fn).astype(np.float32)
    values = np.stack([low, high], axis=-1).reshape(packed.shape[0], packed.shape[1] * 2)
    scales = scale.view(ml_dtypes.float8_e8m0fnu).astype(np.float32)
    spread = np.repeat(scales, 32, axis=1)[:, : values.shape[1]]
    extra["fp4_bytes"] = torch.from_numpy(packed.copy())
    extra["fp4_scale_bytes"] = torch.from_numpy(scale.copy())
    extra["fp4_expected"] = torch.from_numpy(np.ascontiguousarray(values * spread))

    globals()["_extra"] = extra
    return fp8_expected



def run_gpt_oss_quant(image, checkpoint):
    """The MXFP4 decode of a real gpt-oss expert tensor, from transformers' own converter.

    `--checkpoint` is the safetensors SHARD URL holding layer 0's experts. Only the first 64 rows of
    the first expert's `gate_up_proj_blocks` and their scales are fetched, through HTTP range requests
    on the safetensors header and data, so the record costs about 100 KB rather than the shard's
    5 GB. The expectation is `transformers.integrations.mxfp4.convert_moe_packed_tensors`, the code
    the library itself dequantizes the release with: a byte holds the earlier value in its low nibble,
    the sixteen e2m1 values are looked up, and each group of 32 is scaled by two to the power of its
    e8m0 byte less 127.
    """
    import json, struct, subprocess
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors

    def fetch(rng):
        done = subprocess.run(["curl", "-sSL", "--fail", "-m", "600", "-A", "InferKit/0.1",
                               "-H", f"Range: bytes={rng}", checkpoint], capture_output=True)
        if done.returncode != 0:
            raise SystemExit(f"range request failed: {done.stderr.decode()[:200]}")
        return done.stdout
    length = struct.unpack("<Q", fetch("0-7"))[0]
    header = json.loads(fetch(f"8-{8 + length - 1}"))
    base = 8 + length
    rows = 64
    blocks_meta = header["model.layers.0.mlp.experts.gate_up_proj_blocks"]
    scales_meta = header["model.layers.0.mlp.experts.gate_up_proj_scales"]
    _, out_rows, groups, bytes_per_group = blocks_meta["shape"]
    assert scales_meta["shape"][1:] == [out_rows, groups] and blocks_meta["dtype"] == "U8"
    block_bytes = rows * groups * bytes_per_group
    scale_bytes = rows * groups
    blocks_start = base + blocks_meta["data_offsets"][0]
    scales_start = base + scales_meta["data_offsets"][0]
    blocks = torch.frombuffer(bytearray(fetch(f"{blocks_start}-{blocks_start + block_bytes - 1}")),
                              dtype=torch.uint8).view(1, rows, groups, bytes_per_group)
    scales = torch.frombuffer(bytearray(fetch(f"{scales_start}-{scales_start + scale_bytes - 1}")),
                              dtype=torch.uint8).view(1, rows, groups)
    # The converter returns the x @ W layout [experts, in, out]; the row-major [out, in] is what the
    # packed bytes describe, and what the Swift side compares.
    expected = convert_moe_packed_tensors(blocks, scales, dtype=torch.float32).transpose(1, 2)[0].contiguous()
    globals()["_extra"] = {"mxfp4_bytes": blocks[0].clone(), "mxfp4_scale_bytes": scales[0].clone(),
                           "geometry": torch.tensor([out_rows, groups, bytes_per_group], dtype=torch.int32)}
    return expected


def run_cmgan(image, checkpoint):
    """CMGAN (ruizhecao96/CMGAN, MIT) through the released `TSCNet` generator, seam by seam, on one of
    the repository's own noisy VCTK-DEMAND clips.

    `--checkpoint` is a directory holding the released `ckpt` (a plain state dict) and the noisy clip
    `p232_052_noisy.wav`; `IK_CMGAN_SRC` is the cloned `src/` directory (`models/`, `utils.py`). The
    record reproduces `evaluation.enhance_one_track`: the RMS normalization `c`, the circular pad to a
    multiple of the hop, a 400-point periodic-Hamming STFT at hop 100 (center, reflect), the 0.3 power
    compression, the generator, decompression, the inverse STFT, `/ c`, and the trim to the input
    length. Seams: the compressed spectrum, the dense encoder, every TSCB output, the mask, the
    complex residual, and the final real / imaginary parts.
    """
    import sys
    import torchaudio
    sys.path.insert(0, os.environ["IK_CMGAN_SRC"])
    from models import generator
    from utils import power_compress, power_uncompress

    n_fft, hop = 400, 100
    model = generator.TSCNet(num_channel=64, num_features=n_fft // 2 + 1)
    model.load_state_dict(torch.load(os.path.join(checkpoint, "ckpt"), map_location="cpu"))
    model.eval()
    waveform, rate = torchaudio.load(os.path.join(checkpoint, "p232_052_noisy.wav"))
    assert rate == 16000, rate
    noisy = waveform[:1]
    with torch.no_grad():
        c = torch.sqrt(noisy.size(-1) / torch.sum(noisy ** 2.0, dim=-1))
        scaled = noisy * c
        length = scaled.size(-1)
        padded_len = int(np.ceil(length / hop)) * hop
        padded = torch.cat([scaled, scaled[:, : padded_len - length]], dim=-1)
        window = torch.hamming_window(n_fft)
        spec = torch.view_as_real(torch.stft(padded, n_fft, hop, window=window, onesided=True,
                                             return_complex=True))
        compressed = power_compress(spec).permute(0, 1, 3, 2)          # [1, 2, T, F]
        mag = torch.sqrt(compressed[:, 0] ** 2 + compressed[:, 1] ** 2).unsqueeze(1)
        phase = torch.angle(torch.complex(compressed[:, 0], compressed[:, 1])).unsqueeze(1)
        out_1 = model.dense_encoder(torch.cat([mag, compressed], dim=1))
        out_2 = model.TSCB_1(out_1)
        out_3 = model.TSCB_2(out_2)
        out_4 = model.TSCB_3(out_3)
        out_5 = model.TSCB_4(out_4)
        mask = model.mask_decoder(out_5)
        complex_out = model.complex_decoder(out_5)
        est_real, est_imag = model(compressed)
        uncompressed = power_uncompress(est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)).squeeze(1)
        est_audio = torch.istft(torch.view_as_complex(uncompressed.contiguous()), n_fft, hop, window=window,
                                onesided=True)
        est_audio = (est_audio / c).flatten()[:length]
    extra = {"waveform": noisy[0].contiguous(), "scale": c.reshape(1).contiguous(),
             "padded": padded[0].contiguous(), "compressed": compressed[0].contiguous(),
             "encoder": out_1[0].contiguous(), "tscb1": out_2[0].contiguous(), "tscb2": out_3[0].contiguous(),
             "tscb3": out_4[0].contiguous(), "tscb4": out_5[0].contiguous(), "mask": mask[0].contiguous(),
             "complex": complex_out[0].contiguous(), "final_real": est_real[0].contiguous(),
             "final_imag": est_imag[0].contiguous()}
    globals()["_extra"] = extra
    return est_audio.contiguous()


def run_frcrn(image, checkpoint):
    """FRCRN SE 16K (modelscope/ClearerVoice-Studio, Apache-2.0) through the released `DCCRN`
    (two complex UNets over a conv-STFT), seam by seam.

    `--checkpoint` is the released `last_best_checkpoint.pt`; `IK_FRCRN_SRC` is the ClearerVoice source
    dir holding `models/frcrn_se/`; `IK_FRCRN_CLIP` a 16 kHz WAV (defaults to the CMGAN noisy sample).
    The clip is zero-padded the way `decode_one_audio_frcrn_se_16k` pads (a 1 s window, a 0.75 s
    stride), then `model.inference` runs it whole. Seams (batch dropped, `[C, D, T, 2]`): the conv-STFT
    spectrum, encoder 0 before and after its squeeze-excite, the bottleneck FSMN, decoder 0, the first
    UNet's output, the combined mask, the masked spectrum, and the waveform.
    """
    import sys
    import torchaudio
    sys.path.insert(0, os.environ["IK_FRCRN_SRC"])
    from models.frcrn_se.frcrn import DCCRN

    model = DCCRN(complex=True, model_complexity=45, model_depth=14, log_amp=False, padding_mode="zeros",
                  win_len=640, win_inc=320, fft_len=640, win_type="hanning")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    clip = os.environ.get("IK_FRCRN_CLIP", os.path.expanduser("~/.inferkit-validation/cmgan/p232_052_noisy.wav"))
    waveform, rate = torchaudio.load(clip)
    assert rate == 16000, rate
    noisy = waveform[:1]
    t = noisy.shape[1]
    window, stride = 16000, 12000
    if t < window:
        pad = window - t
    elif t < window + stride:
        pad = window + stride - t
    else:
        pad = (t - (t - window) // stride * stride) if (t - window) % stride != 0 else 0
    padded = torch.cat([noisy, torch.zeros(1, pad)], dim=1)

    seams = {}
    def hook(name):
        def fn(_m, _i, o):
            seams[name] = o
        return fn
    unet = model.unet
    handles = [unet.encoder0.register_forward_hook(hook("encoder0")),
               unet.se_layer_enc0.register_forward_hook(hook("se0")),
               unet.fsmn.register_forward_hook(hook("bottleneck")),
               unet.decoder0.register_forward_hook(hook("decoder0")),
               unet.linear.register_forward_hook(hook("unet1"))]
    with torch.no_grad():
        cmp_spec = model.stft(padded).unsqueeze(1)
        cmp_spec = torch.cat([cmp_spec[:, :, :model.feat_dim, :], cmp_spec[:, :, model.feat_dim:, :]], 1)
        cmp_spec = torch.transpose(torch.unsqueeze(cmp_spec, 4), 1, 4)     # [1, 1, D, T, 2]
        unet1_out = model.unet(cmp_spec)
        mask = torch.tanh(model.unet2(unet1_out)) + torch.tanh(unet1_out)
        est_spec, est_wav, _ = model.apply_mask(cmp_spec, mask)
        reference = model.inference(padded)
    for h in handles:
        h.remove()
    assert torch.allclose(reference, est_wav[0]), "the staged path must reproduce inference()"
    extra = {"waveform": noisy[0].contiguous(), "padded": padded[0].contiguous(),
             "spec": cmp_spec[0].contiguous(), "unet1": unet1_out[0].contiguous(), "mask": mask[0].contiguous(),
             "est_spec": est_spec[0].contiguous()}
    for name, value in seams.items():
        extra[name] = value[0].contiguous()
    globals()["_extra"] = extra
    return reference.contiguous()


def run_mossformer2_sr(image, checkpoint):
    """MossFormer2 SR 48K (modelscope/ClearerVoice-Studio, Apache-2.0) speech super-resolution, seam by
    seam: the HiFi-GAN log-mel, the mel-to-mel MossFormer2 backbone, the Snake HiFi-GAN generator, and
    the scipy `bandwidth_sub` post-process of `decode_one_audio_mossformer2_sr_48k`.

    `--checkpoint` is the release DIRECTORY holding `last_best_checkpoint_m.pt` and
    `last_best_checkpoint_g.pt`; `IK_MOSSFORMER2_SR_SRC` is the ClearerVoice source dir (with
    `models/mossformer2_sr/`, `dataloader/meldataset.py`, `bandwidth_sub.py`, and
    `MossFormer2_SR_48K.json`). The input is a 16 kHz clip (`IK_MOSSFORMER2_SR_CLIP`, default the CMGAN
    clean sample) resampled to 48 kHz by torchaudio and recorded, so both sides read one waveform.
    """
    import json
    import sys
    import torchaudio
    src = os.environ["IK_MOSSFORMER2_SR_SRC"]
    sys.path.insert(0, src)
    from models.mossformer2_sr.generator import Mossformer, Generator
    from models.mossformer2_sr.env import AttrDict
    from dataloader.meldataset import mel_spectrogram
    from bandwidth_sub import bandwidth_sub, detect_bandwidth

    with open(os.path.join(src, "MossFormer2_SR_48K.json")) as f:
        h = AttrDict(json.load(f))
    model_m = Mossformer()
    model_m.load_state_dict(torch.load(os.path.join(checkpoint, "last_best_checkpoint_m.pt"),
                                       map_location="cpu", weights_only=False)["mossformer"], strict=True)
    model_g = Generator(h)
    model_g.load_state_dict(torch.load(os.path.join(checkpoint, "last_best_checkpoint_g.pt"),
                                       map_location="cpu", weights_only=False)["generator"], strict=True)
    model_m.eval()
    model_g.eval()
    clip = os.environ.get("IK_MOSSFORMER2_SR_CLIP", os.path.expanduser("~/.inferkit-validation/cmgan/p232_052_clean.wav"))
    waveform, rate = torchaudio.load(clip)
    audio = torchaudio.functional.resample(waveform[:1], rate, 48000)[0].contiguous()
    with torch.no_grad():
        mel = mel_spectrogram(audio.unsqueeze(0), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
        restored_mel = model_m(mel)
        generated = model_g(restored_mel).squeeze()
    inputs = audio.numpy()
    f_low, f_high = detect_bandwidth(inputs, 48000)
    output = bandwidth_sub(inputs, generated.numpy())
    extra = {"waveform": audio, "mel": mel[0].contiguous(), "restored_mel": restored_mel[0].contiguous(),
             "generated": generated.contiguous(), "f_high": torch.tensor([float(f_high)])}
    globals()["_extra"] = extra
    return torch.from_numpy(np.asarray(output, dtype=np.float32)).contiguous()


def _load_nuwave2(checkpoint):
    """The released `Diffusion` wrapper and its hparams from the official Lightning checkpoint, with
    `pytorch_lightning` stubbed (the pickled callbacks are inert) and `torch.istft` taught to take the
    real view the repository hands it. `IK_NUWAVE2_SRC` is the cloned repository."""
    import sys
    import types
    from omegaconf import OmegaConf
    src = os.environ["IK_NUWAVE2_SRC"]
    sys.path.insert(0, src)

    class _Stub:
        def __init__(self, *a, **k): pass
        def __setstate__(self, state): pass
    def _stub_attribute(attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _Stub
    for name in ["pytorch_lightning", "pytorch_lightning.callbacks", "pytorch_lightning.callbacks.model_checkpoint",
                 "pytorch_lightning.callbacks.early_stopping", "pytorch_lightning.trainer", "pytorch_lightning.utilities"]:
        module = types.ModuleType(name)
        module.__getattr__ = _stub_attribute
        sys.modules[name] = module
    from diffusion import Diffusion

    # The repository predates torch 2.x: its istft is handed the real (…, 2) view torch.stft returned,
    # which a current torch refuses; the complex view is the same numbers.
    original_istft = torch.istft
    def istft_accepting_real(spec, *args, **kwargs):
        if not spec.is_complex():
            spec = torch.view_as_complex(spec.contiguous())
        return original_istft(spec, *args, **kwargs)
    torch.istft = istft_accepting_real

    hparams = OmegaConf.load(os.path.join(src, "hparameter.yaml"))
    diffusion = Diffusion(hparams)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    diffusion.load_state_dict({k[len("model."):]: v for k, v in state.items()}, strict=True)
    diffusion.eval()
    return diffusion, hparams


def run_nuwave2(image, checkpoint):
    """NU-Wave 2 (maum-ai/nuwave2, BSD-3) diffusion bandwidth extension through the released
    `Diffusion` wrapper, seam by seam and step by step.

    `--checkpoint` is the official Lightning checkpoint (`nuwave2_official.ckpt`); `IK_NUWAVE2_SRC` is
    the cloned repository (`model.py`, `diffusion.py`, `hparameter.yaml`); `IK_NUWAVE2_CLIP` a 16 kHz
    WAV (default the CMGAN clean sample). The input follows `inference.py`'s non-ground-truth path: the
    clip peak-normalized, `resample_poly` to 48 kHz, trimmed to a multiple of the hop, the band one-hot
    over the first `int(hi · 513)` bins. The eight-step schedule runs from a seeded standard-normal start
    (recorded), and the record carries the start, every step's signal, and step 0's diffusion
    embedding, first residual block, and noise prediction.
    """
    import librosa
    from scipy.signal import resample_poly
    diffusion, hparams = _load_nuwave2(checkpoint)

    clip = os.environ.get("IK_NUWAVE2_CLIP", os.path.expanduser("~/.inferkit-validation/cmgan/p232_052_clean.wav"))
    wav, sr = librosa.load(clip, sr=None, mono=True)
    wav = wav / np.max(np.abs(wav))
    hop = hparams.audio.hop_length
    wav_l = resample_poly(wav, hparams.audio.sampling_rate, sr)
    wav_l = wav_l[: len(wav_l) - len(wav_l) % hop]
    fft_size = hparams.audio.filter_length // 2 + 1
    hi = (sr // 2) / (0.5 * hparams.audio.sampling_rate)
    band = torch.zeros(fft_size, dtype=torch.int64)
    band[: int(hi * fft_size)] = 1
    wav_l = torch.from_numpy(wav_l.copy()).float().unsqueeze(0)
    band = band.unsqueeze(0)
    schedule = eval(hparams.dpm.infer_schedule)

    seams = {}
    def hook(name):
        def fn(_m, _i, o):
            if name not in seams:
                seams[name] = o
        return fn
    net = diffusion.model
    handles = [net.diffusion_embedding.register_forward_hook(hook("emb0")),
               net.residual_layers[0].register_forward_hook(hook("layer0"))]
    torch.manual_seed(0)
    signal = torch.randn(wav_l.shape)
    extra = {"waveform_low": wav_l[0].contiguous(), "band": band[0].contiguous(), "noise": signal[0].contiguous(),
             "source": torch.from_numpy(wav.astype(np.float32)), "source_rate": torch.tensor([float(sr)])}
    with torch.no_grad():
        first_level = (hparams.logsnr.logsnr_max - schedule[0]) / (hparams.logsnr.logsnr_max - hparams.logsnr.logsnr_min)
        extra["eps0"] = net(signal, wav_l, band, first_level * torch.ones(1))[0].contiguous()
        for i, logsnr_t in enumerate(schedule):
            logsnr_s = torch.tensor(hparams.logsnr.logsnr_max) if i == len(schedule) - 1 else schedule[i + 1]
            signal, _ = diffusion.denoise_ddim(signal, wav_l, band, logsnr_t * torch.ones(1), logsnr_s * torch.ones(1))
            extra["step_%d" % i] = signal[0].contiguous()
        recon = torch.clamp(signal, min=-1, max=1 - torch.finfo(torch.float16).eps)
    for h in handles:
        h.remove()
    extra["emb0"] = seams["emb0"][0].contiguous()
    extra["layer0_x"] = seams["layer0"][0][0].contiguous()
    extra["layer0_skip"] = seams["layer0"][1][0].contiguous()
    globals()["_extra"] = extra
    return recon[0].contiguous()


def run_apollo(image, checkpoint):
    """Apollo (JusperLee/Apollo, CC-BY-SA-4.0) music restoration through the released `Apollo` model,
    seam by seam, on channel 0 of the repository's own sample clip (`asserts/input_wav.wav`, 44.1 kHz).

    `--checkpoint` is the Hugging Face `pytorch_model.bin`; `IK_APOLLO_SRC` is the cloned repository
    (its `look2hear` package). Records the 2-second mono input, the band features, the first and last
    BSNet outputs, band 0's bottleneck and head, and the restored waveform.
    """
    import sys
    import soundfile as sf
    src = os.environ["IK_APOLLO_SRC"]
    sys.path.insert(0, src)
    import look2hear.models

    model = look2hear.models.BaseModel.from_pretrain(checkpoint, sr=44100, win=20, feature_dim=256, layer=6).eval()
    clip = os.environ.get("IK_APOLLO_CLIP", os.path.expanduser("~/.inferkit-validation/apollo/input_wav.wav"))
    audio, rate = sf.read(clip, dtype="float32", always_2d=True)
    assert rate == 44100, rate
    mono = torch.from_numpy(np.ascontiguousarray(audio[: 2 * rate, 0])).reshape(1, 1, -1)

    seams = {}
    def hook(name):
        def fn(_m, _i, o):
            seams[name] = o
        return fn
    handles = [model.BN[0].register_forward_hook(hook("bn0")), model.net[0].register_forward_hook(hook("net0")),
               model.net.register_forward_hook(hook("net_final")), model.output[0].register_forward_hook(hook("head0"))]
    with torch.no_grad():
        features = model.feature_extractor(mono)
        output = model(mono)
    for h in handles:
        h.remove()
    extra = {"waveform": mono[0, 0].contiguous(), "features": features[0].contiguous(),
             "bn0": seams["bn0"][0].contiguous(), "net0": seams["net0"][0].contiguous(),
             "net_final": seams["net_final"][0].contiguous(), "head0": seams["head0"][0].contiguous()}
    globals()["_extra"] = extra
    return output[0, 0].contiguous()


def run_metricgan(image, checkpoint):
    """MetricGAN+ (speechbrain/metricgan-plus-voicebank) through speechbrain's own
    SpectralMaskEnhancement, seam by seam, on the noisy clip the model card ships.

    `--checkpoint` is the release DIRECTORY (hyperparams.yaml + enhance_model.ckpt + example.wav). The
    record carries the 16 kHz waveform both sides read, the log1p-magnitude features (a 512-point
    Hamming STFT at hop 256, zero-padded and centered), the BLSTM mask `1.2 * sigmoid(slope * x)`,
    and the enhanced waveform (expm1 of the masked features under the noisy phase, inverse STFT,
    peak-normalized). The generator's state dict rides along under the release's own names.
    """
    import torchaudio
    from speechbrain.inference.enhancement import SpectralMaskEnhancement

    enhancer = SpectralMaskEnhancement.from_hparams(source=checkpoint, savedir=checkpoint,
                                                    run_opts={"device": "cpu"})
    waveform, rate = torchaudio.load(os.path.join(checkpoint, "example.wav"))
    assert rate == 16000, rate
    noisy = waveform[:1, : 16000 * 3]                       # mono, first three seconds
    with torch.no_grad():
        features = enhancer.compute_features(noisy)
        mask = enhancer.mods.enhance_model(features, lengths=torch.tensor([1.0]))
        enhanced = enhancer.enhance_batch(noisy, lengths=torch.tensor([1.0]))
    extra = {"waveform": noisy[0].contiguous(), "features": features[0].contiguous(),
             "mask": mask[0].contiguous()}
    for key, value in enhancer.mods.enhance_model.state_dict().items():
        extra["w::" + key] = value.float().contiguous()
    globals()["_extra"] = extra
    return enhanced[0].contiguous()


def run_rope_scaling(image):
    """RoPE frequency scaling: the inverse frequencies and attention factor transformers computes.

    Weight-free and architecture-free — scaling is a function of the rotary geometry and the config's
    `rope_scaling` block alone, so this isolates it completely. `ROPE_INIT_FUNCTIONS` is the same
    dispatch every transformers decoder uses, so matching it here matches every model that declares
    one of these.

    Each case writes the parameters it used alongside its result, so the Swift side reads the
    configuration from the record rather than repeating literals that could drift apart from it.
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    class Config:
        def __init__(self, dim, base, max_positions, scaling):
            self.rope_theta = base
            self.head_dim = dim
            self.hidden_size = dim
            self.num_attention_heads = 1
            self.max_position_embeddings = max_positions
            self.rope_scaling = scaling
            self.partial_rotary_factor = 1.0
            # transformers 5.x reads `rope_parameters` (the scaling block plus `rope_theta`) and calls
            # `standardize_rope_params`; older releases read `rope_scaling`. Both are served.
            self.rope_parameters = dict(scaling, rope_theta=base)

        def standardize_rope_params(self):
            pass

    # dim, base, max_positions, scaling
    cases = [
        # DeepSeek V4 Pro's shape: a long extension over a short original window.
        (64, 10000.0, 4096, {"rope_type": "yarn", "factor": 40.0,
                             "original_max_position_embeddings": 4096}),
        # A modest extension, and the beta defaults left implicit.
        (128, 10000.0, 32768, {"rope_type": "yarn", "factor": 4.0,
                               "original_max_position_embeddings": 8192}),
        # Non-default ramp boundaries, which move `low`/`high` and so the whole blend.
        (128, 500000.0, 131072, {"rope_type": "yarn", "factor": 8.0,
                                 "original_max_position_embeddings": 16384,
                                 "beta_fast": 16, "beta_slow": 2}),
        # An explicit attention factor overrides the derived one.
        (64, 10000.0, 8192, {"rope_type": "yarn", "factor": 16.0,
                             "original_max_position_embeddings": 2048,
                             "attention_factor": 1.25}),
        # Linear, which is position scaling rather than a frequency blend.
        (64, 10000.0, 8192, {"rope_type": "linear", "factor": 4.0}),
        (64, 500000.0, 131072, {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}),
        (128, 500000.0, 131072, {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
                                 "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}),
    ]

    kinds = {"yarn": 1.0, "linear": 0.0, "llama3": 2.0}
    _extra = globals().setdefault("_extra", {})
    _extra.clear()
    outputs = []
    for index, (dim, base, max_positions, scaling) in enumerate(cases):
        config = Config(dim, base, max_positions, scaling)
        initializer = ROPE_INIT_FUNCTIONS[scaling["rope_type"]]
        inv_freq, attention_factor = initializer(config, torch.device("cpu"))
        _extra[f"case{index}_inv_freq"] = inv_freq.float().contiguous()
        _extra[f"case{index}_attention_factor"] = torch.tensor([float(attention_factor)])
        _extra[f"case{index}_params"] = torch.tensor([
            float(dim), float(base), float(max_positions),
            float(scaling["factor"]),
            float(scaling.get("original_max_position_embeddings", max_positions)),
            float(scaling.get("beta_fast", 32)),
            float(scaling.get("beta_slow", 1)),
            kinds[scaling["rope_type"]],
            # -1 marks "the config declares none", so the Swift side exercises the derivation.
            float(scaling.get("attention_factor", -1.0)),
            float(scaling.get("low_freq_factor", 1.0)),
            float(scaling.get("high_freq_factor", 4.0)),
        ])
        outputs.append(inv_freq.float())

    # LongRoPE (Phi-3, Phi-4): two per-pair factor tables chosen by sequence length, over a partial
    # rotary, with the attention factor derived from the extended-to-original window ratio. The tables
    # are ramps so the per-pair multiply is exercised; Phi-4-mini's own short table is all ones.
    class LongRoPEConfig(Config):
        def __init__(self, scaling):
            super().__init__(128, 10000.0, 131072, scaling)
            self.partial_rotary_factor = 0.75
            self.original_max_position_embeddings = 4096

    pairs = 48
    short = [1.0 + 0.02 * i for i in range(pairs)]
    long = [1.0 + 0.5 * i for i in range(pairs)]
    index = len(cases)
    for sequence_length, override in [(None, {}), (8192, {}), (None, {"attention_factor": 1.1})]:
        scaling = dict({"rope_type": "longrope", "short_factor": short, "long_factor": long}, **override)
        config = LongRoPEConfig(scaling)
        inv_freq, attention_factor = ROPE_INIT_FUNCTIONS["longrope"](config, torch.device("cpu"), sequence_length)
        _extra[f"case{index}_inv_freq"] = inv_freq.float().contiguous()
        _extra[f"case{index}_attention_factor"] = torch.tensor([float(attention_factor)])
        _extra[f"case{index}_params"] = torch.tensor([
            96.0, 10000.0, 131072.0, 131072.0 / 4096.0, 4096.0, 32.0, 1.0, 3.0,
            float(override.get("attention_factor", -1.0)), 1.0, 4.0])
        _extra[f"case{index}_short_factor"] = torch.tensor(short, dtype=torch.float32)
        _extra[f"case{index}_long_factor"] = torch.tensor(long, dtype=torch.float32)
        _extra[f"case{index}_sequence_length"] = torch.tensor([sequence_length or 0], dtype=torch.int32)
        outputs.append(inv_freq.float())
        index += 1
    _extra["case_count"] = torch.tensor([index], dtype=torch.int32)
    return torch.cat(outputs)


def run_rtdetr(image):
    """RT-DETR object detection end to end at a tiny random configuration, from transformers' own
    RTDetrForObjectDetection: the ResNet-D backbone (deep 3-conv stem, avgpool-in-shortcut bottleneck),
    the hybrid encoder (an AIFI transformer on the deepest level + a CSP-RepVGG FPN/PAN), query
    selection over generated anchors, and the deformable-attention decoder with iterative box
    refinement. `anchor_image_size=None`, so anchors are generated per forward; `with_box_refine=True`,
    so class/box heads are cloned per decoder layer. Records the backbone features, the encoder's PAN
    outputs, the query-selection seams, and the per-layer decoder logits/boxes so a divergence localizes.
    License-clean (Apache-2.0). Runs under the `llm` oracle env (transformers). `image` unused.
    """
    from transformers import RTDetrConfig, RTDetrForObjectDetection
    from transformers.models.rt_detr.configuration_rt_detr_resnet import RTDetrResNetConfig

    backbone = RTDetrResNetConfig(
        embedding_size=16, hidden_sizes=[16, 32, 64, 128], depths=[1, 1, 1, 1],
        layer_type="bottleneck", downsample_in_bottleneck=False, downsample_in_first_stage=False,
        out_features=["stage2", "stage3", "stage4"], num_channels=3)
    config = RTDetrConfig(
        backbone_config=backbone, use_timm_backbone=False, backbone=None,
        encoder_in_channels=[32, 64, 128], feat_strides=[8, 16, 32], encoder_hidden_dim=32,
        encoder_ffn_dim=48, num_attention_heads=2, encoder_layers=1, encode_proj_layers=[2],
        d_model=32, decoder_attention_heads=2, decoder_ffn_dim=48, decoder_layers=2,
        decoder_n_points=4, num_feature_levels=3, decoder_in_channels=[32, 32, 32],
        num_queries=10, num_denoising=0, learn_initial_query=False, anchor_image_size=None,
        with_box_refine=True, num_labels=4, hidden_expansion=1.0)
    # Randomize the trainable parameters only, NOT the BatchNorm buffers: a randomized `running_var`
    # can go negative, and `rsqrt(var + eps)` is then NaN in both the reference and the port. Left at
    # their init (mean 0, var 1), the frozen BatchNorms stay physical and deterministic.
    model = RTDetrForObjectDetection(config)
    torch.manual_seed(29)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.05)
    model = model.eval().float()

    generator = torch.Generator().manual_seed(7)
    pixels = torch.randn(1, 3, 64, 64, generator=generator)

    feats = {}
    model.model.backbone.register_forward_hook(
        lambda m, i, o: feats.__setitem__("bb", [f.detach() for f, _ in o]))
    with torch.no_grad():
        out = model(pixels, output_hidden_states=False, return_dict=True)
        topk_ind = torch.topk(out.enc_outputs_class.max(-1).values, config.num_queries, dim=1).indices

    extra = {
        "pixels": pixels[0].permute(1, 2, 0).contiguous(),                 # [H, W, 3] NHWC
        "pred_boxes": out.pred_boxes[0].clone().contiguous(),              # [Q, 4]
        "enc_class": out.enc_outputs_class[0].contiguous(),               # [S, num_labels]
        "enc_coord": out.enc_outputs_coord_logits[0].contiguous(),         # [S, 4]
        "init_ref": out.init_reference_points[0].contiguous(),             # [Q, 4]
        "topk_ind": topk_ind[0].to(torch.int32).contiguous(),              # [Q]
        "inter_logits": out.intermediate_logits[0].clone().contiguous(),   # [layers, Q, num_labels]
        "inter_ref": out.intermediate_reference_points[0].clone().contiguous(),  # [layers, Q, 4]
        "enc_last": out.encoder_last_hidden_state[-1][0].permute(1, 2, 0).contiguous(),  # last PAN, NHWC
    }
    for level, feature in enumerate(feats["bb"]):
        extra[f"bb.{level}"] = feature[0].permute(1, 2, 0).contiguous()     # backbone stage NHWC
    for key, value in model.state_dict().items():
        if not key.startswith("model.") or key.endswith("num_batches_tracked"):
            continue
        extra[f"w::{key}"] = value.float().contiguous() if value.is_floating_point() else value.contiguous()
    globals()["_extra"] = extra
    return out.logits[0].clone().contiguous()                              # [Q, num_labels]


def run_table_transformer_loss(image, checkpoint):
    """Table Transformer's training objective on a released model (`checkpoint`), from the reference's
    own code: microsoft/table-transformer's vendored DETR (`IK_TABLE_TRANSFORMER_SRC`, the files the
    manifest pins), `HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)` and `SetCriterion` with
    `eos_coef` 0.4 and the losses `labels`, `boxes`, and `cardinality`, weighted 1, 5, and 2 for
    `loss_ce`, `loss_bbox`, and `loss_giou` (`structure_config.json`, `detection_config.json`; their
    `aux_loss` is false, so only the last decoder layer is scored).

    The release's own forward (transformers, on the pixels the parity mode feeds it) supplies the
    outputs; three fixed targets are scored against them. Records the pixels, the logits and boxes, the
    targets, each loss term, the matched query of each target (`matched_queries`), transformers' own
    `labels=` loss (`hf_loss`), and the weighted total (the output).
    """
    import ast
    import sys

    import torch.nn.functional as F
    from transformers import TableTransformerForObjectDetection, DetrImageProcessor
    from PIL import Image

    model = TableTransformerForObjectDetection.from_pretrained(checkpoint, torch_dtype=torch.float32).eval()
    processor = DetrImageProcessor.from_pretrained(checkpoint)
    pil = Image.fromarray((image * 255).astype("uint8"))
    size = processor.size
    if "longest_edge" in size and "shortest_edge" not in size:
        edge = size["longest_edge"]
        pixel_values = processor(images=pil, size={"max_height": edge, "max_width": edge},
                                 return_tensors="pt")["pixel_values"]
    else:
        pixel_values = processor(images=pil, return_tensors="pt")["pixel_values"]
    classes = model.config.num_labels
    target_classes = torch.tensor([0, 1 % classes, (classes - 1)], dtype=torch.int64)
    target_boxes = torch.tensor([[0.5, 0.5, 0.8, 0.7], [0.3, 0.25, 0.4, 0.1], [0.62, 0.7, 0.3, 0.2]])
    with torch.no_grad():
        out = model(pixel_values, return_dict=True)
        hf = model(pixel_values, labels=[{"class_labels": target_classes, "boxes": target_boxes}], return_dict=True)

    root = os.path.expanduser(os.environ.get("IK_TABLE_TRANSFORMER_SRC", "~/.inferkit-validation/sources/table-transformer"))
    sys.path.insert(0, os.path.join(root, "detr"))
    from util import box_ops
    from util.misc import accuracy, get_world_size, is_dist_avail_and_initialized
    from models.matcher import HungarianMatcher
    source = open(os.path.join(root, "detr/models/detr.py")).read()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "SetCriterion")
    namespace = {"torch": torch, "F": F, "nn": torch.nn, "box_ops": box_ops, "accuracy": accuracy,
                 "get_world_size": get_world_size, "is_dist_avail_and_initialized": is_dist_avail_and_initialized}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "detr.py", "exec"), namespace)

    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    weights = {"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2}
    criterion = namespace["SetCriterion"](classes, matcher=matcher, weight_dict=weights, eos_coef=0.4,
                                          losses=["labels", "boxes", "cardinality"])
    outputs = {"pred_logits": out.logits, "pred_boxes": out.pred_boxes}
    targets = [{"labels": target_classes, "boxes": target_boxes}]
    with torch.no_grad():
        losses = criterion(outputs, targets)
        rows, columns = matcher(outputs, targets)[0]
    total = sum(losses[k] * w for k, w in weights.items())
    matched = torch.empty(len(target_classes), dtype=torch.int32)
    matched[columns] = rows.to(torch.int32)

    globals()["_extra"] = {
        "pixels": pixel_values[0].permute(1, 2, 0).contiguous(),
        "logits": out.logits[0].contiguous(),
        "pred_boxes": out.pred_boxes[0].contiguous(),
        "target_classes": target_classes.to(torch.int32),
        "target_boxes": target_boxes.contiguous(),
        "matched_queries": matched,
        "loss_ce": losses["loss_ce"].reshape(1).contiguous(),
        "loss_bbox": losses["loss_bbox"].reshape(1).contiguous(),
        "loss_giou": losses["loss_giou"].reshape(1).contiguous(),
        "hf_loss": hf.loss.reshape(1).contiguous(),
    }
    return total.reshape(1).contiguous()

def run_table_transformer(image, checkpoint):
    """Table Transformer (`TableTransformerForObjectDetection`, microsoft/table-transformer-*) on the
    RELEASED weights, from transformers' own model and its DETR image processor. A vanilla DETR: a timm
    ResNet-18 backbone with frozen batch norm, a normalized 2D sine position embedding, a 1x1 input
    projection to d_model, a PRE-NORM transformer encoder and decoder (the layer norm precedes each
    sub-block, and a final layer norm follows each stack, which is Table Transformer's one difference
    from post-norm DETR), and the class / box heads over the decoder queries. Records the preprocessed
    pixels so the port runs on the identical input, the last backbone feature map, the encoder and
    decoder outputs, the predicted boxes, and the post-processed detections. `checkpoint` is the local
    release directory. Runs under the `llm` oracle env (transformers, needs Pillow). `image` is the table.
    """
    from transformers import TableTransformerForObjectDetection, DetrImageProcessor
    from PIL import Image

    model = TableTransformerForObjectDetection.from_pretrained(checkpoint, torch_dtype=torch.float32).eval()
    processor = DetrImageProcessor.from_pretrained(checkpoint)
    pil = Image.fromarray((image * 255).astype("uint8"))
    size = processor.size
    if "longest_edge" in size and "shortest_edge" not in size:
        # The v1.1 releases declare `longest_edge` alone, which transformers 4.57's resize rejects; the
        # processor's own max_height / max_width path bounds the longer edge the way that size intends.
        edge = size["longest_edge"]
        pixel_values = processor(images=pil, size={"max_height": edge, "max_width": edge},
                                 return_tensors="pt")["pixel_values"]
    else:
        pixel_values = processor(images=pil, return_tensors="pt")["pixel_values"]    # [1, 3, H, W]

    seams = {}
    handle = model.model.backbone.conv_encoder.register_forward_hook(
        lambda m, i, o: seams.__setitem__("bb", o[-1][0][0].detach()))
    with torch.no_grad():
        out = model(pixel_values, return_dict=True)
    handle.remove()

    target_sizes = torch.tensor([pil.size[::-1]])
    results = processor.post_process_object_detection(out, threshold=0.6, target_sizes=target_sizes)[0]

    extra = {
        "pixels": pixel_values[0].permute(1, 2, 0).contiguous(),                 # [H, W, 3] NHWC
        "backbone": seams["bb"].permute(1, 2, 0).contiguous(),                   # [H/32, W/32, 512] NHWC
        "enc_last": out.encoder_last_hidden_state[0].contiguous(),               # [HW, 256]
        "dec_last": out.last_hidden_state[0].contiguous(),                       # [num_queries, 256]
        "pred_boxes": out.pred_boxes[0].clone().contiguous(),                    # [num_queries, 4] cxcywh
        "scores": results["scores"].contiguous(),                               # [detections]
        "labels": results["labels"].to(torch.int32).contiguous(),               # [detections]
        "boxes": results["boxes"].contiguous(),                                 # [detections, 4] xyxy px
    }
    globals()["_extra"] = extra
    return out.logits[0].clone().contiguous()                                    # [num_queries, num_labels + 1]


def run_vjepa2_probe(image):
    """V-JEPA 2's frozen-encoder probe recipe, from the reference's own code (facebookresearch/vjepa2
    at the commit the manifest pins, under `IK_VJEPA2_SRC`).

    Three records the Swift recipe is held to:
      - the loss: `torch.nn.CrossEntropyLoss()` as `run_one_epoch` builds it, on seeded logits `[4, 10]`
        and labels, which both sides score (the output);
      - the schedule: `WarmupCosineLRSchedule` from `evals/video_classification_frozen/eval.py`, taken
        from the file and stepped before each update as `run_one_epoch` steps it (`lr_default` with no
        warm-up and a cosine to zero, the recipe's default; `lr_warmup` with a warm-up and a floor);
      - the initialization: `AttentiveClassifier(depth=4)` built by `src/models/attentive_pooler.py`,
        each tensor's standard deviation keyed by the port's parameter name (`std/<name>`).
    """
    import ast
    import math
    import sys

    root = os.path.expanduser(os.environ.get("IK_VJEPA2_SRC", "~/.inferkit-validation/sources/vjepa2"))
    sys.path.insert(0, root)
    from src.models.attentive_pooler import AttentiveClassifier

    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(4, 10, generator=generator)
    labels = torch.randint(0, 10, (4,), generator=generator)
    loss = torch.nn.CrossEntropyLoss()(logits, labels)

    source = open(os.path.join(root, "evals/video_classification_frozen/eval.py")).read()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WarmupCosineLRSchedule")
    namespace = {"math": math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "eval.py", "exec"), namespace)
    schedule_class = namespace["WarmupCosineLRSchedule"]

    class _Optimizer:
        def __init__(self, group):
            self.param_groups = [group]

    def rates(steps, warmup, start, ref, final):
        optimizer = _Optimizer({"mc_warmup_steps": warmup, "mc_start_lr": start, "mc_ref_lr": ref,
                                "mc_final_lr": final})
        schedule = schedule_class(optimizer, T_max=steps)
        values = []
        for _ in range(steps):
            schedule.step()
            values.append(optimizer.param_groups[0]["lr"])
        return torch.tensor(values, dtype=torch.float32)

    torch.manual_seed(0)
    width, depth = 512, 4
    classifier = AttentiveClassifier(embed_dim=width, num_heads=8, depth=depth, num_classes=16)
    pooler = classifier.pooler
    stds = {"query_tokens": pooler.query_tokens}
    for index, block in enumerate(pooler.blocks):
        prefix = f"self_attention_layers.{index}."
        qkv = block.attn.qkv.weight.detach()
        for part, name in enumerate(["q_proj", "k_proj", "v_proj"]):
            stds[prefix + f"self_attn.{name}.weight"] = qkv[part * width:(part + 1) * width]
        stds[prefix + "self_attn.out_proj.weight"] = block.attn.proj.weight
        stds[prefix + "mlp.fc1.weight"] = block.mlp.fc1.weight
        stds[prefix + "mlp.fc2.weight"] = block.mlp.fc2.weight
    cross = pooler.cross_attention_block
    stds["cross_attention_layer.cross_attn.q_proj.weight"] = cross.xattn.q.weight
    kv = cross.xattn.kv.weight.detach()
    stds["cross_attention_layer.cross_attn.k_proj.weight"] = kv[:width]
    stds["cross_attention_layer.cross_attn.v_proj.weight"] = kv[width:]
    stds["cross_attention_layer.mlp.fc1.weight"] = cross.mlp.fc1.weight
    stds["cross_attention_layer.mlp.fc2.weight"] = cross.mlp.fc2.weight

    extra = {"logits": logits.contiguous(), "labels": labels.to(torch.int32).contiguous(),
             "lr_default": rates(12, 0, 5e-3, 5e-3, 0.0),
             "lr_warmup": rates(12, 3, 1e-3, 5e-3, 1e-4),
             "width": torch.tensor([width, depth], dtype=torch.int32)}
    for name, tensor in stds.items():
        extra["std/" + name] = tensor.detach().float().std().reshape(1).contiguous()
    globals()["_extra"] = extra
    return loss.reshape(1).contiguous()

def run_vjepa2(image, checkpoint):
    """V-JEPA 2 vision encoder (`VJEPA2Model.get_vision_features`, facebook/vjepa2-*) on the RELEASED
    weights, from transformers' own model. A ViT-L: a 3D-convolution tubelet patch embedding
    (tubelet x patch x patch, no class token and no learned position table), 24 pre-norm blocks whose
    attention rotates queries and keys with a 3D rotary embedding (temporal / height / width, each
    2 * floor(floor(head_dim / 3) / 2) channels; the remainder unrotated), and a final layer norm. The
    predictor and pooler heads in the checkpoint belong to pretraining and classification and are not
    exercised (`skip_predictor=True`). A small deterministic clip (8 frames) is used so the token count
    stays light while all three rotary axes vary; the port consumes the identical clip in `[T, H, W, C]`.
    Records the patch embedding and a middle block's hidden state for localization; the returned `output`
    is the final normed token feature sequence. `checkpoint` is the local release directory. Runs under
    the `llm` oracle env (transformers). `image` is unused (this is a video model).
    """
    from transformers import VJEPA2Model, VJEPA2ForVideoClassification, AutoConfig

    # A classification release (`VJEPA2ForVideoClassification`) wraps the encoder under `vjepa2` and adds
    # the attentive pooler and classifier; its record also carries the pooler output and the logits.
    classifies = "VJEPA2ForVideoClassification" in (AutoConfig.from_pretrained(checkpoint).architectures or [])
    if classifies:
        classifier = VJEPA2ForVideoClassification.from_pretrained(checkpoint, torch_dtype=torch.float32).eval()
        model = classifier.vjepa2
    else:
        model = VJEPA2Model.from_pretrained(checkpoint, torch_dtype=torch.float32).eval()
    size = model.config.crop_size
    rng = np.random.default_rng(0)
    clip = rng.standard_normal((8, size, size, 3)).astype(np.float32)             # normalized clip [T, H, W, C]
    pixel_values_videos = torch.from_numpy(clip).permute(0, 3, 1, 2).unsqueeze(0).contiguous()  # [1, T, C, H, W]

    with torch.no_grad():
        out = model(pixel_values_videos, skip_predictor=True, output_hidden_states=True)
        hidden = out.hidden_states                                                # (num_layers + 1) x [1, N, hidden]
        extra = {
            "clip": torch.from_numpy(clip).contiguous(),                         # [T, H, W, 3]
            "patch": hidden[0][0].contiguous(),                                  # [N, hidden] patch embedding
            "mid": hidden[12][0].contiguous(),                                   # [N, hidden] after block 11
        }
        if classifies:
            pooled = classifier.pooler(out.last_hidden_state)                     # [1, hidden]
            extra["pooled"] = pooled[0].contiguous()
            extra["logits"] = classifier.classifier(pooled)[0].contiguous()      # [labels]
    globals()["_extra"] = extra
    return out.last_hidden_state[0].contiguous()                                 # [N, hidden] final normed


def run_sa2va_loss(image, checkpoint):
    """Sa2VA's fine-tuning objective on a released InternVL model (`checkpoint`), from the authors' own
    training code (bytedance/Sa2VA at the commit the manifest pins, under `IK_SA2VA_SRC`):
    `Sa2VAModel.forward` with `sa2va_finetune.py`'s settings. The loss is the language model's shifted
    cross-entropy (`InternVLMLLM._compute_loss`) plus 2.0 x the sigmoid cross-entropy and 0.5 x the naive
    dice (eps 1) of the mask, both on 12,544 points `sample_points` draws (uncertainty-weighted, oversample
    3, importance 0.75; the vendored mmdet `point_sample`, `binary_cross_entropy`, and `dice_loss`).

    The mask path is `SAM2TrainRunner.get_sam2_embeddings` and `inject_language_embd` over the training
    extension's `_forward_sam_heads` (the release's inference copy suppresses by object score and takes
    no language embedding), bound onto the release's own SAM 2 modules. `check_obj_number` first fixes
    the sample at five objects, repeating the one `[SEG]` and its mask. The example is the `run_sa2va`
    plate and prompt with the answer `Sure, [SEG].` and the template's end, the prompt labeled -100, and
    the plate's bright disk as the mask. The draws `torch.rand` makes inside the sampler are recorded
    (`point_candidates`, `point_random`) with the coordinates they select (`points`), so the port can
    score the same points. Records the inputs, the low-resolution mask logits, each term, the language
    loss recomputed in float64 (`llm_loss_f64`), the learning rate mmengine's own `LinearLR` warm-up and
    `CosineAnnealingLR` give over 40- and 200-iteration runs (`lr_40`, `lr_200`, the configuration's 4e-5,
    `start_factor` 1e-5, and 5% warm-up), and the total (the output).
    """
    import ast
    import sys
    import types
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    from PIL import Image

    root = os.path.expanduser(os.environ.get("IK_SA2VA_SRC", "~/.inferkit-validation/sources/sa2va"))

    def extract(path, names, namespace, cls=None):
        tree = ast.parse(open(os.path.join(root, path)).read())
        body = tree.body
        if cls is not None:
            body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == cls).body
        nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert len(nodes) == len(names), (path, names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
        return namespace

    utils = {"__name__": "mmdet_utils"}
    exec(compile(open(os.path.join(root, "third_parts/mmdet/models/losses/utils.py")).read(), "utils.py", "exec"), utils)
    sampling = {"__name__": "mmdet_point_sample"}
    exec(compile(open(os.path.join(root, "third_parts/mmdet/models/utils/point_sample.py")).read(), "point_sample.py", "exec"), sampling)
    losses = {"torch": torch, "F": F, "weight_reduce_loss": utils["weight_reduce_loss"]}
    extract("third_parts/mmdet/models/losses/dice_loss.py", ["dice_loss"], losses)
    extract("third_parts/mmdet/models/losses/cross_entropy_loss.py", ["binary_cross_entropy", "_expand_onehot_labels"], losses)

    model = AutoModel.from_pretrained(checkpoint, torch_dtype=torch.float32, trust_remote_code=True,
                                      low_cpu_mem_usage=True).eval()
    tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, use_fast=True)
    model.preparing_for_generation(tokenizer=tok, torch_dtype=torch.float32)
    model.torch_dtype = torch.float32
    IMG_CTX, SEG = model.img_context_token_id, model.seg_token_idx
    mod = sys.modules[type(model).__module__]

    size = 448
    plate = np.zeros((size, size, 3), dtype=np.uint8)
    plate[: size // 2, : size // 2] = (40, 60, 90)
    plate[: size // 2, size // 2:] = (90, 40, 60)
    plate[size // 2:, : size // 2] = (60, 90, 40)
    plate[size // 2:, size // 2:] = (30, 30, 30)
    yy, xx = np.mgrid[0:size, 0:size]
    disk = ((xx - size * 0.62) ** 2 + (yy - size * 0.40) ** 2) < (size * 0.16) ** 2
    plate[disk] = (230, 210, 120)
    pil = Image.fromarray(plate, "RGB")

    images = mod.dynamic_preprocess(pil, 1, model.max_dynamic_patch, model.image_size, model.use_thumbnail)
    pixel_values = torch.stack([model.transformer(im) for im in images]).to(torch.float32)
    num_image_tokens = pixel_values.shape[0] * model.patch_token
    g_np = model.extra_image_processor.apply_image(plate)
    g_pixel = torch.from_numpy(g_np).permute(2, 0, 1).contiguous().to(torch.float32)
    g_pixel = torch.stack([model.grounding_encoder.preprocess_image(g_pixel)]).to(torch.float32)

    text = "<image>Please segment the bright object.".replace(
        "<image>", f"{model.IMG_START_TOKEN}{model.IMG_CONTEXT_TOKEN * num_image_tokens}{model.IMG_END_TOKEN}")
    prompt = tok.encode(model.template["INSTRUCTION"].format(input=text, round=1, bot_name=model.bot_name))
    answer = tok.encode("Sure, [SEG]." + model.template["SUFFIX"], add_special_tokens=False)
    input_ids = torch.tensor([prompt + answer])
    labels = torch.tensor([[-100] * len(prompt) + answer])
    gt_masks = torch.from_numpy(disk.astype(np.uint8))[None]                           # [1, 448, 448]

    llm = types.SimpleNamespace(model=model)
    compute = extract("projects/sa2va/models/mllm/internvl.py", ["_compute_loss"],
                      {"torch": torch, "CrossEntropyLoss": torch.nn.CrossEntropyLoss}, cls="InternVLMLLM")

    sam = model.grounding_encoder.sam2_model
    heads = {"torch": torch, "F": F, "NO_OBJ_SCORE": -1024.0}
    extract("projects/sa2va/models/extension/sam2_base.py", ["_forward_sam_heads"], heads, cls="SAM2Base")
    sam._forward_sam_heads = types.MethodType(heads["_forward_sam_heads"], sam)
    runner_ns = {"torch": torch}
    extract("projects/sa2va/models/sam2_train.py", ["get_sam2_embeddings", "inject_language_embd"], runner_ns,
            cls="SAM2TrainRunner")
    runner = types.SimpleNamespace(sam2_model=sam, hidden_dim=sam.hidden_dim)

    draws = []
    real_rand = torch.rand

    def recording_rand(*args, **kwargs):
        value = real_rand(*args, **kwargs)
        draws.append(value.clone())
        return value

    sampler_ns = {"torch": torch, "F": F, "point_sample": sampling["point_sample"],
                  "get_uncertain_point_coords_with_randomness": sampling["get_uncertain_point_coords_with_randomness"]}
    extract("projects/sa2va/models/sa2va.py", ["sample_points", "check_obj_number"], sampler_ns, cls="Sa2VAModel")
    sampler = types.SimpleNamespace(num_points=12544, oversample_ratio=3.0, importance_sample_ratio=0.75)

    with torch.no_grad():
        vit_embeds = model.extract_feature(pixel_values)
        embeds = model.language_model.get_input_embeddings()(input_ids).clone()
        selected = input_ids == IMG_CTX
        embeds[selected] = vit_embeds.reshape(-1, embeds.shape[-1])
        output = model.language_model(inputs_embeds=embeds, output_hidden_states=True, return_dict=True)
        llm_loss = compute["_compute_loss"](llm, output.logits, labels)
        llm_loss_f64 = F.cross_entropy(output.logits[0, :-1].double(), labels[0, 1:])
        hidden = model.text_hidden_fcs(output.hidden_states[-1])
        # The reference fixes every sample at five objects before the mask path: fewer are repeated,
        # more are subsampled.
        found, masks_found = sampler_ns["check_obj_number"](sampler, [hidden[input_ids == SEG]], [gt_masks])
        embeddings, gt_masks = found[0], masks_found[0]                               # [5, 256], [5, H, W]
        states = runner_ns["get_sam2_embeddings"](runner, g_pixel, expand_size=embeddings.shape[0])
        pred_masks = runner_ns["inject_language_embd"](runner, states, embeddings[:, None],
                                                       nf_nobj=(1, embeddings.shape[0])).flatten(0, 1)
        target = F.interpolate(gt_masks[None].float(), size=pred_masks.shape[-2:], mode="nearest")[0]

        sampling["torch"].rand = recording_rand
        torch.manual_seed(0)
        try:
            sampled_pred, sampled_gt = sampler_ns["sample_points"](sampler, pred_masks, target)
        finally:
            sampling["torch"].rand = real_rand
        dice = 0.5 * losses["dice_loss"](sampled_pred.sigmoid(), sampled_gt, weight=None, eps=1.0,
                                         reduction="mean", naive_dice=True, avg_factor=len(target) + 1e-4)
        bce = 2.0 * losses["binary_cross_entropy"](sampled_pred.reshape(-1), sampled_gt.reshape(-1),
                                                    reduction="mean",
                                                    avg_factor=pred_masks.shape[0] * sampled_pred.shape[1] + 1e-4)
    def mmengine_rates(steps):
        # mmengine's own LinearParamScheduler and CosineAnnealingParamScheduler (the file the manifest
        # pins), built from the configuration's epoch-based settings for a one-epoch run of `steps`
        # iterations and stepped after each iteration, as ParamSchedulerHook steps them.
        stubs = {name: types.ModuleType(name) for name in ["mmengine", "mmengine.logging", "mmengine.optim", "mmengine.registry"]}
        stubs["mmengine.logging"].print_log = lambda *args, **kwargs: None
        stubs["mmengine.optim"].BaseOptimWrapper = type("BaseOptimWrapper", (), {})
        stubs["mmengine.registry"].PARAM_SCHEDULERS = types.SimpleNamespace(
            register_module=lambda *args, **kwargs: (lambda cls: cls))
        saved = {name: sys.modules.get(name) for name in stubs}
        sys.modules.update(stubs)
        try:
            namespace = {"__name__": "mmengine_param_scheduler"}
            exec(compile(open(os.path.join(root, "mmengine/param_scheduler.py")).read(), "param_scheduler.py", "exec"),
                 namespace)
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=4e-5)
        schedulers = [
            namespace["LinearParamScheduler"].build_iter_from_epoch(
                optimizer, param_name="lr", start_factor=1e-5, by_epoch=True, begin=0, end=0.05, epoch_length=steps),
            namespace["CosineAnnealingParamScheduler"].build_iter_from_epoch(
                optimizer, param_name="lr", eta_min=0.0, by_epoch=True, begin=0.05, end=1, epoch_length=steps),
        ]
        rates = []
        for _ in range(steps):
            rates.append(optimizer.param_groups[0]["lr"])
            for scheduler in schedulers:
                scheduler.step()
        return torch.tensor(rates, dtype=torch.float64).to(torch.float32)

    candidates, random_points = draws[0], draws[1]
    # The coordinates the sampler selected, rebuilt from its draws the way it builds them.
    logits = sampling["point_sample"](pred_masks[:, None], candidates)
    k = int(0.75 * 12544)
    top = torch.topk(-logits.abs()[:, 0], k=k, dim=1)[1]
    points = torch.cat([torch.gather(candidates, 1, top[..., None].expand(-1, -1, 2)), random_points], dim=1)

    globals()["_extra"] = {
        "pixel_values": pixel_values.contiguous(),
        "g_pixel_values": g_pixel.contiguous(),
        "input_ids": input_ids[0].to(torch.int32).contiguous(),
        "labels": labels[0].to(torch.int32).contiguous(),
        "gt_masks": gt_masks.to(torch.int32).contiguous(),
        "seg_embedding": embeddings.contiguous(),
        "pred_masks": pred_masks.contiguous(),
        "point_candidates": candidates.contiguous(),
        "point_random": random_points.contiguous(),
        "points": points.contiguous(),
        "llm_loss": llm_loss.reshape(1).contiguous(),
        "llm_loss_f64": llm_loss_f64.float().reshape(1).contiguous(),
        "loss_mask": bce.reshape(1).contiguous(),
        "loss_dice": dice.reshape(1).contiguous(),
        "lr_40": mmengine_rates(40),
        "lr_200": mmengine_rates(200),
    }
    return (llm_loss + bce + dice).reshape(1).contiguous()

def run_sa2va(image, checkpoint):
    """Sa2VA-4B (ByteDance/Sa2VA-4B, Apache) on the RELEASED weights, from the repo's own custom code
    (`AutoModel.from_pretrained(..., trust_remote_code=True)`). A segmentation VLM: an InternViT-300M
    image encoder (a pre-norm ViT with per-channel LayerScale, a class token, a learned position table,
    qkv bias, no query/key normalization, and no final layer norm), a pixel-shuffle + 2-layer MLP
    projector (`mlp1`, downsample 0.5, ps v2), a Qwen2.5-3B decoder, a `[SEG]` bridge (`text_hidden_fcs`,
    Linear -> ReLU -> Linear, 2048 -> 256), and a SAM 2 Hiera-Large grounding encoder driven by the
    decoder's hidden state at each `[SEG]` position. The prompt asks the model to segment a bright object
    in a deterministic quadrant image; the model answers "Sure, it is [SEG]." and the branch produces the
    object's mask. Records seam by seam: the InternViT last hidden state, the projected vision tokens, the
    fused decoder embeddings, the `[SEG]` embedding after the bridge, the conditioned SAM feature, and the
    three multimask logits with their IoUs; the returned `output` is the best mask's low-resolution logits
    [1, 1, 256, 256]. The SAM branch is run through the first-frame glue directly (the released video
    predictor hardcodes CUDA), which is the conditioning-frame path a single image takes. `checkpoint` is
    the local release directory; `image` is unused (a deterministic plate is synthesized). Runs under the
    `llm` oracle env (transformers + peft + timm), on CPU in float32.
    """
    import sys
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer

    def make_image(size=448):
        a = np.zeros((size, size, 3), dtype=np.uint8)
        a[: size // 2, : size // 2] = (40, 60, 90)
        a[: size // 2, size // 2:] = (90, 40, 60)
        a[size // 2:, : size // 2] = (60, 90, 40)
        a[size // 2:, size // 2:] = (30, 30, 30)
        yy, xx = np.mgrid[0:size, 0:size]
        disk = ((xx - size * 0.62) ** 2 + (yy - size * 0.40) ** 2) < (size * 0.16) ** 2
        a[disk] = (230, 210, 120)
        return Image.fromarray(a, "RGB")

    model = AutoModel.from_pretrained(checkpoint, torch_dtype=torch.float32, trust_remote_code=True,
                                      low_cpu_mem_usage=True).eval()
    tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, use_fast=True)
    model.preparing_for_generation(tokenizer=tok, torch_dtype=torch.float32)
    model.torch_dtype = torch.float32
    IMG_CTX, SEG = model.img_context_token_id, model.seg_token_idx
    mod = sys.modules[type(model).__module__]

    plate = make_image(448)
    images = mod.dynamic_preprocess(plate, 1, model.max_dynamic_patch, model.image_size, model.use_thumbnail)
    pixel_values = torch.stack([model.transformer(im) for im in images]).to(torch.float32)
    num_image_tokens = pixel_values.shape[0] * model.patch_token

    g_np = model.extra_image_processor.apply_image(np.array(plate))
    g_pixel = torch.from_numpy(g_np).permute(2, 0, 1).contiguous().to(torch.float32)
    g_pixel = torch.stack([model.grounding_encoder.preprocess_image(g_pixel)]).to(torch.float32)

    text = "<image>Please segment the bright object.".replace(
        "<image>", f"{model.IMG_START_TOKEN}{model.IMG_CONTEXT_TOKEN * num_image_tokens}{model.IMG_END_TOKEN}")
    input_text = model.template["INSTRUCTION"].format(input=text, round=1, bot_name=model.bot_name)
    ids = torch.tensor(tok.encode(input_text)).unsqueeze(0)
    attn = torch.ones_like(ids, dtype=torch.bool)

    with torch.no_grad():
        vit_last = model.vision_model(pixel_values=pixel_values, output_hidden_states=False,
                                      return_dict=True).last_hidden_state
        vit_embeds = model.extract_feature(pixel_values)
        input_embeds = model.language_model.get_input_embeddings()(ids).clone()
        B, N, Cn = input_embeds.shape
        flat = input_embeds.reshape(B * N, Cn)
        flat[(ids.reshape(B * N) == IMG_CTX)] = vit_embeds.reshape(-1, Cn).to(flat.dtype)
        fused = flat.reshape(B, N, Cn)

        gen = model.generate(pixel_values=pixel_values, input_ids=ids, attention_mask=attn,
                             generation_config=model.gen_config, output_hidden_states=True,
                             return_dict_in_generate=True, bos_token_id=tok.bos_token_id,
                             stopping_criteria=model.stop_criteria, max_new_tokens=40)
        seq = gen.sequences[0]
        last_hidden = torch.cat([item[-1][0] for item in gen.hidden_states], dim=0)
        seg_hidden = mod.get_seg_hidden_states(last_hidden, seq[:-1], seg_id=SEG)
        all_seg = model.text_hidden_fcs(seg_hidden)

        sam = model.grounding_encoder.sam2_model
        backbone_out = sam.forward_image(g_pixel)
        _, vision_feats, _, feat_sizes = sam._prepare_backbone_features(backbone_out)
        Hf, Wf = feat_sizes[-1]
        conditioned = (vision_feats[-1] + sam.no_mem_embed).permute(1, 2, 0).view(1, sam.hidden_dim, Hf, Wf)
        high_res = [x.permute(1, 2, 0).view(1, x.size(2), *s)
                    for x, s in zip(vision_feats[:-1], feat_sizes[:-1])]
        sam_out = sam._forward_sam_heads(backbone_features=conditioned, point_inputs=None, mask_inputs=None,
                                         high_res_features=high_res, multimask_output=True,
                                         language_embd=all_seg[0].unsqueeze(0).unsqueeze(0))
        low_res_multi, _, ious, low_res_best, _, _, _ = sam_out

    globals()["_extra"] = {
        "pixel_values": pixel_values.contiguous(),
        "g_pixel_values": g_pixel.contiguous(),
        "input_ids": ids.to(torch.int32).contiguous(),
        "sequence": seq.to(torch.int32).contiguous(),
        "vit_last_hidden": vit_last.contiguous(),
        "vit_embeds": vit_embeds.contiguous(),
        "fused": fused.contiguous(),
        "seg_embedding": all_seg.contiguous(),
        "sam_conditioned": conditioned.contiguous(),
        "low_res_multi": low_res_multi.contiguous(),
        "ious": ious.contiguous(),
        "low_res_best": low_res_best.contiguous(),
    }
    return low_res_best.clone().contiguous()



def run_sa2va_teacher(image, checkpoint):
    """Sa2VA with the answer teacher-forced rather than generated, for a release cut to its first
    decoder layers (`truncate.py`), whose shortened decoder cannot be expected to answer with `[SEG]`.
    Everything up to the decoder is `run_sa2va`'s: the same plate, prompt, tiling, and remote code. The
    prompt is followed by the fixed answer "Sure, it is [SEG]." and the whole sequence runs through the
    decoder once. Records the InternViT last hidden state, the projected vision tokens, the fused
    embeddings, the decoder's final normalized hidden states over the whole sequence (`dec_last`), the
    last position's logits, the `[SEG]` embedding after the bridge, and the SAM 2 mask it drives
    (`low_res_best`, also the returned `output`). Runs under the `llm` oracle env, float32 on the CPU."""
    import sys
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer

    size = 448
    a = np.zeros((size, size, 3), dtype=np.uint8)
    a[: size // 2, : size // 2] = (40, 60, 90)
    a[: size // 2, size // 2:] = (90, 40, 60)
    a[size // 2:, : size // 2] = (60, 90, 40)
    a[size // 2:, size // 2:] = (30, 30, 30)
    yy, xx = np.mgrid[0:size, 0:size]
    a[((xx - size * 0.62) ** 2 + (yy - size * 0.40) ** 2) < (size * 0.16) ** 2] = (230, 210, 120)
    plate = Image.fromarray(a, "RGB")

    model = AutoModel.from_pretrained(checkpoint, torch_dtype=torch.float32, trust_remote_code=True,
                                      low_cpu_mem_usage=True).eval()
    tok = _release_tokenizer(checkpoint, use_fast=True)
    model.preparing_for_generation(tokenizer=tok, torch_dtype=torch.float32)
    IMG_CTX, SEG = model.img_context_token_id, model.seg_token_idx
    mod = sys.modules[type(model).__module__]

    images = mod.dynamic_preprocess(plate, 1, model.max_dynamic_patch, model.image_size, model.use_thumbnail)
    pixel_values = torch.stack([model.transformer(im) for im in images]).to(torch.float32)
    num_image_tokens = pixel_values.shape[0] * model.patch_token
    g_np = model.extra_image_processor.apply_image(np.array(plate))
    g_pixel = torch.from_numpy(g_np).permute(2, 0, 1).contiguous().to(torch.float32)
    g_pixel = torch.stack([model.grounding_encoder.preprocess_image(g_pixel)]).to(torch.float32)

    text = "<image>Please segment the bright object.".replace(
        "<image>", f"{model.IMG_START_TOKEN}{model.IMG_CONTEXT_TOKEN * num_image_tokens}{model.IMG_END_TOKEN}")
    prompt_text = model.template["INSTRUCTION"].format(input=text, round=1, bot_name=model.bot_name)
    prompt_ids = tok.encode(prompt_text)
    answer_ids = tok.encode("Sure, it is [SEG].", add_special_tokens=False)
    ids = torch.tensor(prompt_ids + answer_ids).unsqueeze(0)

    with torch.no_grad():
        vit_last = model.vision_model(pixel_values=pixel_values, output_hidden_states=False,
                                      return_dict=True).last_hidden_state
        vit_embeds = model.extract_feature(pixel_values)
        embeds = model.language_model.get_input_embeddings()(ids).clone()
        B, N, Cn = embeds.shape
        flat = embeds.reshape(B * N, Cn)
        flat[(ids.reshape(B * N) == IMG_CTX)] = vit_embeds.reshape(-1, Cn).to(flat.dtype)
        fused = flat.reshape(B, N, Cn)
        out = model.language_model(inputs_embeds=fused, output_hidden_states=True, return_dict=True)
        dec_last = out.hidden_states[-1][0]
        position = int((ids[0] == SEG).nonzero()[0])
        seg_embedding = model.text_hidden_fcs(dec_last[position].unsqueeze(0))

        sam = model.grounding_encoder.sam2_model
        backbone_out = sam.forward_image(g_pixel)
        _, vision_feats, _, feat_sizes = sam._prepare_backbone_features(backbone_out)
        Hf, Wf = feat_sizes[-1]
        conditioned = (vision_feats[-1] + sam.no_mem_embed).permute(1, 2, 0).view(1, sam.hidden_dim, Hf, Wf)
        high_res = [x.permute(1, 2, 0).view(1, x.size(2), *s) for x, s in zip(vision_feats[:-1], feat_sizes[:-1])]
        sam_out = sam._forward_sam_heads(backbone_features=conditioned, point_inputs=None, mask_inputs=None,
                                         high_res_features=high_res, multimask_output=True,
                                         language_embd=seg_embedding.unsqueeze(0))
        low_res_best = sam_out[3]

    globals()["_extra"] = {
        "pixel_values": pixel_values.contiguous(),
        "g_pixel_values": g_pixel.contiguous(),
        "input_ids": ids.to(torch.int32).contiguous(),
        "vit_last_hidden": vit_last.contiguous(),
        "vit_embeds": vit_embeds.contiguous(),
        "fused": fused.contiguous(),
        "dec_last": dec_last.contiguous(),
        "last_logits": out.logits[0, -1].contiguous(),
        "seg_embedding": seg_embedding.contiguous(),
        "low_res_best": low_res_best.contiguous(),
        # The tokenizer both ways: the instruction text the ids encode, and the reference's decode of the
        # whole sequence and of the answer alone, as UTF-8.
        "prompt_utf8": torch.tensor(list(prompt_text.encode("utf-8")), dtype=torch.int32),
        "decoded_utf8": torch.tensor(list(tok.decode(ids[0]).encode("utf-8")), dtype=torch.int32),
        "answer_decoded_utf8": torch.tensor(list(tok.decode(answer_ids).encode("utf-8")), dtype=torch.int32),
        "prompt_length": torch.tensor([len(prompt_ids)], dtype=torch.int32),
    }
    return low_res_best.clone().contiguous()



def _stage_remote_code(checkpoint):
    """Copies every `.py` of a local release into the dynamic-module folder transformers imports it from.
    transformers copies the entry file and the modules it imports directly, and a release whose imports
    nest deeper (Sa2VA-Qwen3-VL-4B-SAM3: `sam3.py` imports `sam3pkg_*`, which import each other) fails at
    the second level. The copies place on the CPU what the release places on `"cuda"` by name
    (SAM 3's position-encoding precompute and decoder coordinate cache are built there in `__init__`);
    a device placement changes no arithmetic, and the release directory is left untouched. SAM 3's fused
    `addmm_act` casts the ViT MLP's first projection to bfloat16, as the CUDA runtime's bfloat16 autocast
    around the whole grounding call would; the staged copy keeps the input's dtype, so the float32 oracle
    holds the same float32 bar as every other seam (on the CPU its GELU is the exact erf form). transformers
    re-copies only the entry file and its direct imports, which none of this touches."""
    from transformers.dynamic_module_utils import (HF_MODULES_CACHE, TRANSFORMERS_DYNAMIC_MODULE_NAME,
                                                   _sanitize_module_name, create_dynamic_module)
    submodule = os.path.join(TRANSFORMERS_DYNAMIC_MODULE_NAME,
                             _sanitize_module_name(os.path.basename(os.path.normpath(checkpoint))))
    create_dynamic_module(submodule)
    target = os.path.join(HF_MODULES_CACHE, submodule)
    for name in sorted(os.listdir(checkpoint)):
        if name.endswith(".py"):
            with open(os.path.join(checkpoint, name), encoding="utf-8") as source:
                code = source.read()
            code = code.replace('device="cuda"', 'device="cpu"').replace('torch.device("cuda")', 'torch.device("cpu")')
            if name == "sam3pkg_perflib_fused.py":
                code = code.replace(".to(torch.bfloat16)", ".to(mat1.dtype)")
            with open(os.path.join(target, name), "w", encoding="utf-8") as staged:
                staged.write(code)


def run_sa2va_qwen(image, checkpoint):
    """Sa2VA on Qwen3-VL or Qwen2.5-VL (ByteDance/Sa2VA-Qwen3-VL-*, -Qwen2_5-VL-*, Apache) on the RELEASED
    weights, from the repo's own `Sa2VAChatModelQwen` (trust_remote_code): transformers' VLM under `model.`, the `[SEG]`
    bridge, and SAM 2 grounding. `predict_forward` is followed step by step rather than called, since
    it imports `qwen_vl_utils` only to lay out its messages; that module is stubbed. The same plate and
    request as `run_sa2va`, formatted by the release's chat template with the reference's pixel bounds
    (512·28² to 2048·28²). Records the processed patches and grid, the input ids, the merged vision
    tokens and deepstack, the greedy generation, the decoder's final hidden states over the prompt and
    the generation, the `[SEG]` embedding, the grounding image, and the mask (`low_res_best`, also the
    returned `output`). Runs under the `llm` oracle env, float32 on the CPU."""
    import sys
    import types
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    sys.modules.setdefault("qwen_vl_utils", types.SimpleNamespace(process_vision_info=None))
    _stage_remote_code(checkpoint)
    size = 448
    a = np.zeros((size, size, 3), dtype=np.uint8)
    a[: size // 2, : size // 2] = (40, 60, 90)
    a[: size // 2, size // 2:] = (90, 40, 60)
    a[size // 2:, : size // 2] = (60, 90, 40)
    a[size // 2:, size // 2:] = (30, 30, 30)
    yy, xx = np.mgrid[0:size, 0:size]
    a[((xx - size * 0.62) ** 2 + (yy - size * 0.40) ** 2) < (size * 0.16) ** 2] = (230, 210, 120)
    plate = Image.fromarray(a, "RGB")

    model = AutoModel.from_pretrained(checkpoint, torch_dtype=torch.float32, trust_remote_code=True,
                                      low_cpu_mem_usage=True).eval()
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    seg = processor.tokenizer.convert_tokens_to_ids("[SEG]")
    messages = [{"role": "user", "content": [{"type": "image", "image": plate},
                                             {"type": "text", "text": "Please segment the bright object."}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[plate], padding=True, return_tensors="pt",
                       min_pixels=model.min_pixels, max_pixels=model.max_pixels)
    g = torch.from_numpy(model.extra_image_processor.apply_image(np.array(plate))).permute(2, 0, 1).contiguous()
    g_pixel = torch.stack([model.grounding_encoder.preprocess_image(g.to(torch.float32))])

    with torch.no_grad():
        qwen = model.model
        grid = inputs["image_grid_thw"]
        vision_out = qwen.model.visual(inputs["pixel_values"], grid_thw=grid)
        # Qwen3-VL returns the merged tokens and its deepstack; Qwen2.5-VL returns the tokens alone.
        merged, deepstack = (vision_out[0], vision_out[1]) if isinstance(vision_out, tuple) else (vision_out, [])
        prompt_length = inputs["input_ids"].shape[1]
        if os.environ.get("SA2VA_TEACHER"):
            # A release cut to its first decoder layers cannot be expected to answer with [SEG]; the fixed
            # answer is teacher-forced instead, as `run_sa2va_teacher` does for the InternVL releases.
            generated = torch.tensor(processor.tokenizer.encode("Sure, it is [SEG].", add_special_tokens=False))
            full = torch.cat([inputs["input_ids"][0], generated]).unsqueeze(0)
        else:
            gen = qwen.generate(**inputs, max_new_tokens=40, do_sample=False,
                                output_hidden_states=True, return_dict_in_generate=True)
            generated = gen.sequences[0, prompt_length:]
            full = gen.sequences[:, :]
        full_out = qwen(input_ids=full, pixel_values=inputs["pixel_values"], image_grid_thw=grid,
                        output_hidden_states=True, return_dict=True)
        dec_last = full_out.hidden_states[-1][0]
        position = int((full[0] == seg).nonzero()[0])
        seg_embedding = model.text_hidden_fcs(dec_last[position].unsqueeze(0))

        sam = model.grounding_encoder.sam2_model
        backbone_out = sam.forward_image(g_pixel)
        _, vision_feats, _, feat_sizes = sam._prepare_backbone_features(backbone_out)
        Hf, Wf = feat_sizes[-1]
        conditioned = (vision_feats[-1] + sam.no_mem_embed).permute(1, 2, 0).view(1, sam.hidden_dim, Hf, Wf)
        high_res = [x.permute(1, 2, 0).view(1, x.size(2), *s) for x, s in zip(vision_feats[:-1], feat_sizes[:-1])]
        low_res_best = sam._forward_sam_heads(backbone_features=conditioned, point_inputs=None, mask_inputs=None,
                                              high_res_features=high_res, multimask_output=True,
                                              language_embd=seg_embedding.unsqueeze(0))[3]
    print("generated:", processor.batch_decode([generated], skip_special_tokens=False)[0])
    extra = {
        "pixel_values": inputs["pixel_values"].contiguous(),
        "image_grid_thw": grid.to(torch.int32).contiguous(),
        "input_ids": inputs["input_ids"][0].to(torch.int32).contiguous(),
        "vision_merged": merged.contiguous(),
        "generated": generated.to(torch.int32).contiguous(),
        "dec_last": dec_last.contiguous(),
        "seg_embedding": seg_embedding.contiguous(),
        "g_pixel_values": g_pixel.contiguous(),
        "low_res_best": low_res_best.contiguous(),
    }
    for index, feature in enumerate(deepstack):
        extra[f"deepstack_{index}"] = feature.contiguous()
    globals()["_extra"] = extra
    return low_res_best.clone().contiguous()



def run_sa2va_processor(image, checkpoint):
    """Sa2VA's image preprocessing from the release's own code, with no weights: the model's parameters
    are built on the meta device (`init_empty_weights`), so `__init__` sets the processing constants
    and objects (InternVL's `dynamic_preprocess` and `transformer`, the Qwen-VL pixel bounds, LLaVA's
    `transformer`, the grounding `DirectResize` and `preprocess_image`) without reading a tensor. The picture is a
    640×360 gradient with a disk, so every resize runs, where the 448 plate resizes nothing on the
    InternVL path. Records `input_rgb`, the family's understanding pixels (`tile_pixel_values`,
    `pixel_values` with `image_grid_thw`, or `llava_pixel_values`), and the grounding image
    `g_pixel_values`. Runs under the `llm` oracle env on the CPU in seconds."""
    import sys
    import types
    from PIL import Image
    from transformers import AutoConfig, AutoModel, AutoProcessor

    sys.modules.setdefault("qwen_vl_utils", types.SimpleNamespace(process_vision_info=None))
    width, height = 640, 360
    yy, xx = np.mgrid[0:height, 0:width]
    a = np.stack([xx * 255 // (width - 1), yy * 255 // (height - 1), ((xx + yy) * 7) % 256], axis=-1).astype(np.uint8)
    a[((xx - width * 0.3) ** 2 + (yy - height * 0.6) ** 2) < (height * 0.2) ** 2] = (230, 210, 120)
    picture = Image.fromarray(a, "RGB")

    from accelerate import init_empty_weights

    _stage_remote_code(checkpoint)
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    # Parameters only go to the meta device: InternViT's constructor reads `torch.linspace(...).item()`.
    with init_empty_weights():
        model = AutoModel.from_config(config, trust_remote_code=True)
    architecture = type(model).__name__
    extra = {"input_rgb": torch.from_numpy(a.astype(np.int32)).contiguous()}
    with torch.no_grad():
        if architecture == "Sa2VAChatModel":
            # InternVL sets its processing objects in the reference's own `preparing_for_generation`.
            from transformers import AutoTokenizer
            model.preparing_for_generation(AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True))
            mod = sys.modules[type(model).__module__]
            tiles = mod.dynamic_preprocess(picture, 1, model.max_dynamic_patch, model.image_size, model.use_thumbnail)
            extra["tile_pixel_values"] = torch.stack([model.transformer(t) for t in tiles]).to(torch.float32).contiguous()
        elif architecture == "Sa2VAChatModelQwen":
            processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
            messages = [{"role": "user", "content": [{"type": "image", "image": picture},
                                                     {"type": "text", "text": "Please segment the bright object."}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[picture], padding=True, return_tensors="pt",
                               min_pixels=model.min_pixels, max_pixels=model.max_pixels)
            extra["pixel_values"] = inputs["pixel_values"].to(torch.float32).contiguous()
            extra["image_grid_thw"] = inputs["image_grid_thw"].to(torch.int32).contiguous()
        elif architecture == "Sa2VAChatModelLlava":
            extra["llava_pixel_values"] = model.transformer(picture).unsqueeze(0).to(torch.float32).contiguous()
        else:
            raise ValueError(f"{architecture} is not a Sa2VA family this mode knows")
        g = torch.from_numpy(model.extra_image_processor.apply_image(a)).permute(2, 0, 1).contiguous()
        g_pixel = torch.stack([model.grounding_encoder.preprocess_image(g.to(torch.float32))]).contiguous()
    extra["g_pixel_values"] = g_pixel
    globals()["_extra"] = extra
    return g_pixel.clone()


def run_sa2va_llava_teacher(image, checkpoint):
    """Sa2VA on LLaVA-1.5 (ByteDance/Sa2VA-LLaVA-1.5-7B, Apache), from the repo's own
    `Sa2VAChatModelLlava` (trust_remote_code), with the answer teacher-forced: the 7B decoder does not fit
    this machine at float32, so the release is cut to its first decoder layers (`truncate.py`). The same
    plate and request as `run_sa2va`, the image through the release's own transform (PIL bicubic to 336,
    CLIP normalization), the prompt through its Vicuna template with 576 `<image>` tokens, followed by the
    fixed answer "Sure, it is [SEG].". Records the pixels, the CLIP tower's second-to-last hidden state,
    the projected features, the fused embeddings, the decoder's final hidden states over the sequence,
    the last logits, the `[SEG]` embedding, the grounding image, and the mask (`low_res_best`, also the
    returned `output`), plus the prompt and the reference's decodes for the tokenizer. Runs under the
    `llm` oracle env, float32 on the CPU."""
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer

    size = 448
    a = np.zeros((size, size, 3), dtype=np.uint8)
    a[: size // 2, : size // 2] = (40, 60, 90)
    a[: size // 2, size // 2:] = (90, 40, 60)
    a[size // 2:, : size // 2] = (60, 90, 40)
    a[size // 2:, size // 2:] = (30, 30, 30)
    yy, xx = np.mgrid[0:size, 0:size]
    a[((xx - size * 0.62) ** 2 + (yy - size * 0.40) ** 2) < (size * 0.16) ** 2] = (230, 210, 120)
    plate = Image.fromarray(a, "RGB")

    model = AutoModel.from_pretrained(checkpoint, torch_dtype=torch.float32, trust_remote_code=True,
                                      low_cpu_mem_usage=True).eval()
    tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    seg = tok.convert_tokens_to_ids("[SEG]")
    image_token = model.model.config.image_token_index
    pixel_values = model.transformer(plate).unsqueeze(0).to(torch.float32)
    text = "<image>Please segment the bright object.".replace(
        "<image>", model.IMG_CONTEXT_TOKEN * model.patch_token + "\n")
    prompt_text = model.template["INSTRUCTION"].format(input=text, round=1)
    prompt_ids = tok.encode(prompt_text)
    answer_ids = tok.encode("Sure, it is [SEG].", add_special_tokens=False)
    ids = torch.tensor(prompt_ids + answer_ids).unsqueeze(0)
    g = torch.from_numpy(model.extra_image_processor.apply_image(np.array(plate))).permute(2, 0, 1).contiguous()
    g_pixel = torch.stack([model.grounding_encoder.preprocess_image(g.to(torch.float32))])

    with torch.no_grad():
        llava = model.model
        vision_hidden = llava.model.vision_tower(pixel_values, output_hidden_states=True).hidden_states[-2]
        features = llava.model.get_image_features(pixel_values=pixel_values, vision_feature_layer=-2,
                                                  vision_feature_select_strategy="default")
        features = features[0] if isinstance(features, (list, tuple)) else features
        embeds = llava.model.get_input_embeddings()(ids).clone()
        flat = embeds.reshape(-1, embeds.shape[-1])
        flat[ids.reshape(-1) == image_token] = features.reshape(-1, embeds.shape[-1]).to(flat.dtype)
        fused = flat.reshape(embeds.shape)
        out = llava(input_ids=ids, pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        dec_last = out.hidden_states[-1][0]
        position = int((ids[0] == seg).nonzero()[0])
        seg_embedding = model.text_hidden_fcs(dec_last[position].unsqueeze(0))

        sam = model.grounding_encoder.sam2_model
        backbone_out = sam.forward_image(g_pixel)
        _, vision_feats, _, feat_sizes = sam._prepare_backbone_features(backbone_out)
        Hf, Wf = feat_sizes[-1]
        conditioned = (vision_feats[-1] + sam.no_mem_embed).permute(1, 2, 0).view(1, sam.hidden_dim, Hf, Wf)
        high_res = [x.permute(1, 2, 0).view(1, x.size(2), *s) for x, s in zip(vision_feats[:-1], feat_sizes[:-1])]
        low_res_best = sam._forward_sam_heads(backbone_features=conditioned, point_inputs=None, mask_inputs=None,
                                              high_res_features=high_res, multimask_output=True,
                                              language_embd=seg_embedding.unsqueeze(0))[3]

    globals()["_extra"] = {
        "pixel_values": pixel_values[0].permute(1, 2, 0).contiguous(),
        "input_rgb": torch.from_numpy(np.asarray(plate)).to(torch.int32).contiguous(),
        "input_ids": ids.to(torch.int32).contiguous(),
        "vision_hidden": vision_hidden.contiguous(),
        "features": features.contiguous(),
        "fused": fused.contiguous(),
        "dec_last": dec_last.contiguous(),
        "last_logits": out.logits[0, -1].contiguous(),
        "seg_embedding": seg_embedding.contiguous(),
        "g_pixel_values": g_pixel.contiguous(),
        "low_res_best": low_res_best.contiguous(),
        "prompt_utf8": torch.tensor(list(prompt_text.encode("utf-8")), dtype=torch.int32),
        "answer_decoded_utf8": torch.tensor(list(tok.decode(answer_ids).encode("utf-8")), dtype=torch.int32),
        "prompt_length": torch.tensor([len(prompt_ids)], dtype=torch.int32),
    }
    return low_res_best.clone().contiguous()



def run_qwen25vl_vision_tiny(image):
    """Qwen2.5-VL's vision tower (transformers' `Qwen2_5_VisionTransformerPretrainedModel`) at a tiny
    seeded configuration: 4 blocks 64 wide over 4 heads, the SiLU-gated feed-forward 96 wide, a 48-wide
    output, windows of 112 pixels, and full attention at blocks 1 and 3, over a 20 x 28 patch grid (a
    window grid that does not divide it, so the edge windows are partial). Records every parameter
    under its checkpoint name, the patches, the window order (`_window_index`), the window boundaries
    (`_cu_window`), the rotary angles in raster order (`_rotary`), and the merged output in raster order
    (the returned `output`). `image` is unused."""
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

    torch.manual_seed(0)
    config = Qwen2_5_VLVisionConfig(depth=4, hidden_size=64, num_heads=4, intermediate_size=96, out_hidden_size=48,
                                    patch_size=14, temporal_patch_size=2, spatial_merge_size=2, window_size=112,
                                    fullatt_block_indexes=[1, 3], hidden_act="silu")
    config._attn_implementation = "eager"
    model = Qwen2_5_VisionTransformerPretrainedModel(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.05)
    grid = torch.tensor([[1, 20, 28]])
    patches = torch.randn(20 * 28, 3 * 2 * 14 * 14)
    with torch.no_grad():
        output = model(patches, grid_thw=grid)
    window_index, cu_window = model.get_window_index(grid)
    extra = {key: value.contiguous() for key, value in model.state_dict().items()}
    extra.update({"_patches": patches, "_window_index": window_index.to(torch.int32),
                  "_cu_window": torch.tensor(cu_window, dtype=torch.int32), "_rotary": model.rot_pos_emb(grid)})
    globals()["_extra"] = extra
    return output.contiguous()



def run_internvit_qknorm_tiny(image, checkpoint):
    """InternViT with InternViT-6B's switches (`norm_type` rms_norm, `qk_normalization` on, no qkv bias)
    at a tiny seeded configuration, from the release's own `modeling_intern_vit.py` (`checkpoint` is any
    Sa2VA directory that carries it): 2 layers 64 wide over 4 heads, a 28-pixel image of 14-pixel
    patches. Records every parameter under its checkpoint name, the channels-last pixels (`_pixels`),
    and the last hidden state (the returned `output`)."""
    import importlib
    import types
    # The release's modules import one another relatively; a synthetic package lets them, and lets the
    # optional flash-attention import fall back as the code intends when `flash_attn` is absent.
    package = types.ModuleType("sa2va_remote")
    package.__path__ = [checkpoint]
    sys.modules["sa2va_remote"] = package
    Config = importlib.import_module("sa2va_remote.configuration_intern_vit").InternVisionConfig
    Model = importlib.import_module("sa2va_remote.modeling_intern_vit").InternVisionModel
    torch.manual_seed(0)
    config = Config(num_channels=3, patch_size=14, image_size=28, hidden_size=64, num_attention_heads=4,
                    intermediate_size=96, num_hidden_layers=2, qkv_bias=False, qk_normalization=True,
                    norm_type="rms_norm", layer_norm_eps=1e-6, use_flash_attn=False, drop_path_rate=0.0)
    model = Model(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.05)
    pixels = torch.randn(1, 3, 28, 28)
    with torch.no_grad():
        output = model(pixel_values=pixels, output_hidden_states=False, return_dict=True).last_hidden_state
    extra = {key: value.contiguous() for key, value in model.state_dict().items()}
    extra["_pixels"] = pixels.permute(0, 2, 3, 1).contiguous()
    globals()["_extra"] = extra
    return output[0].contiguous()



def run_llava_tiny(image):
    """transformers' `LlavaForConditionalGeneration` at a tiny seeded configuration (a CLIP vision tower 2
    layers 32 wide over a 28-pixel image of 14-pixel patches, read at `vision_feature_layer` -2 without
    its class token; a Llama decoder 2 layers 48 wide), the LLaVA-1.5 layout Sa2VA-LLaVA wraps. Records
    every parameter under its checkpoint name, the channels-last pixels (`_pixels`), the input ids with
    four image tokens (`_ids`), the tower's second-to-last hidden state (`_vision_hidden`), the projected
    features (`_features`), and the decoder's final hidden states (the returned `output`). `image` is
    unused."""
    from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration

    torch.manual_seed(0)
    vision = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                              image_size=28, patch_size=14, projection_dim=32)
    text = LlamaConfig(hidden_size=48, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
                       num_key_value_heads=4, vocab_size=64, max_position_embeddings=64,
                       architectures=["LlamaForCausalLM"])
    config = LlavaConfig(vision_config=vision, text_config=text, image_token_index=60,
                         vision_feature_layer=-2, vision_feature_select_strategy="default",
                         projector_hidden_act="gelu", multimodal_projector_bias=True)
    config._attn_implementation = "eager"
    model = LlavaForConditionalGeneration(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.05)
    pixels = torch.randn(1, 3, 28, 28)
    ids = torch.tensor([[1, 5, 60, 60, 60, 60, 7, 9]])
    with torch.no_grad():
        vision_hidden = model.model.vision_tower(pixels, output_hidden_states=True).hidden_states[-2]
        features = model.model.get_image_features(pixel_values=pixels, vision_feature_layer=-2,
                                                  vision_feature_select_strategy="default")
        features = features[0] if isinstance(features, (list, tuple)) else features
        output = model(input_ids=ids, pixel_values=pixels, output_hidden_states=True).hidden_states[-1][0]
    extra = {key: value.contiguous() for key, value in model.state_dict().items()}
    extra.update({"_pixels": pixels.permute(0, 2, 3, 1).contiguous(), "_ids": ids[0].to(torch.int32),
                  "_vision_hidden": vision_hidden[0].contiguous(), "_features": features.contiguous()})
    globals()["_extra"] = extra
    return output.contiguous()



def run_internlm2_tiny(image, checkpoint):
    """InternLM2 (`InternLM2ForCausalLM`) at a tiny seeded configuration, from the release's own
    `modeling_internlm2.py` (`checkpoint` is a Sa2VA InternLM2 directory that carries it): 2 layers 64
    wide, 8 query heads over 2 key-value heads (so the fused `wqkv` groups 4 queries with each key and
    value), a 32-token vocabulary. Records every parameter under its checkpoint name, the input ids
    (`_ids`), and the final hidden states (the returned `output`); the logits are `_logits`."""
    import importlib
    import types
    package = types.ModuleType("sa2va_remote")
    package.__path__ = [checkpoint]
    sys.modules["sa2va_remote"] = package
    Config = importlib.import_module("sa2va_remote.configuration_internlm2").InternLM2Config
    Model = importlib.import_module("sa2va_remote.modeling_internlm2").InternLM2ForCausalLM
    torch.manual_seed(0)
    config = Config(vocab_size=32, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=8,
                    num_key_value_heads=2, rms_norm_eps=1e-5, rope_theta=1_000_000, bias=False,
                    attn_implementation="eager", rope_scaling={"type": "dynamic", "factor": 2.0},
                    max_position_embeddings=64)
    config._attn_implementation = "eager"
    model = Model(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.05)
    ids = torch.tensor([[1, 5, 9, 3, 17, 22, 8, 2, 30, 11]])
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, return_dict=True)
    extra = {key: value.contiguous() for key, value in model.state_dict().items()}
    extra.update({"_ids": ids[0].to(torch.int32), "_logits": out.logits[0].contiguous()})
    globals()["_extra"] = extra
    return out.hidden_states[-1][0].contiguous()



def _release_tokenizer(checkpoint, **kwargs):
    """The release's tokenizer. transformers 4.57's `AutoTokenizer` hands back a bool for an `auto_map`
    that names a slow class and no fast one (the InternLM2 Sa2VA releases'), so that class is then
    loaded from the release's own module directly."""
    import importlib
    import json
    import types
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, **kwargs)
    if not isinstance(tok, bool):
        return tok
    config = json.load(open(os.path.join(checkpoint, "tokenizer_config.json")))
    module_name, class_name = config["auto_map"]["AutoTokenizer"][0].rsplit(".", 1)
    package = types.ModuleType("sa2va_remote")
    package.__path__ = [checkpoint]
    sys.modules["sa2va_remote"] = package
    return getattr(importlib.import_module("sa2va_remote." + module_name), class_name).from_pretrained(checkpoint)


INTERNLM2_TOKENIZER_CASES = [
    "<|im_start|>user\n<img><IMG_CONTEXT><IMG_CONTEXT></img>\nPlease segment the bright object.<|im_end|>\n<|im_start|>assistant\n",
    "Sure, it is [SEG].",
    "Sure, [SEG] and [SEG].<|im_end|>",
    "  leading spaces, café, 数字 123",
    "<p>left</p> [SEG]",
]


def run_internlm2_tokenizer(image, checkpoint):
    """The InternLM2 releases' slow `InternLM2Tokenizer` (their only one) on INTERNLM2_TOKENIZER_CASES,
    from the repo's own tokenization code (`checkpoint` is a Sa2VA InternLM2 directory): for case i,
    `ids_<i>` (with the `<s>` the tokenizer adds) and `decoded_<i>` (the decode of those ids, UTF-8).
    Returns the first case's ids."""
    tok = _release_tokenizer(checkpoint)
    extra = {}
    for index, text in enumerate(INTERNLM2_TOKENIZER_CASES):
        ids = tok.encode(text)
        extra[f"ids_{index}"] = torch.tensor(ids, dtype=torch.int32)
        extra[f"decoded_{index}"] = torch.tensor(list(tok.decode(ids).encode("utf-8")), dtype=torch.int32)
        print(index, ids[:12], repr(tok.decode(ids))[:80])
    globals()["_extra"] = extra
    return extra["ids_0"].float()


_BF16_POOL = []


def _allow_bfloat16_average_pool():
    """CPU torch has no bfloat16 `avg_pool3d`, which the shipped video graphs call. Registers one for this
    process: the window mean accumulated in float32 and rounded once, as the CUDA kernel accumulates.
    The graphs only pool non-overlapping windows without padding."""
    if _BF16_POOL:
        return
    library = torch.library.Library("aten", "IMPL")

    def average_pool(x, kernel_size, stride=(), padding=0, ceil_mode=False, count_include_pad=True,
                     divisor_override=None):
        k = list(kernel_size)
        if list(stride or k) != k or any(padding if isinstance(padding, (list, tuple)) else [padding]):
            raise NotImplementedError("only non-overlapping unpadded windows are needed")
        b, c, t, h, w = x.shape
        t, h, w = t // k[0] * k[0], h // k[1] * k[1], w // k[2] * k[2]
        y = x.float()[:, :, :t, :h, :w].reshape(b, c, t // k[0], k[0], h // k[1], k[1], w // k[2], k[2])
        return y.mean((3, 5, 7)).to(x.dtype)

    library.impl("avg_pool3d", average_pool, "CPU")
    _BF16_POOL.append(library)


def run_cosmos_tokenizer(image, checkpoint):
    """Cosmos Tokenizer (nvidia/Cosmos-0.1-Tokenizer-*, NVIDIA Open Model License) on the RELEASED
    weights, from NVIDIA's own modules in `cosmos_predict1.tokenizer` (vendored under
    `IK_COSMOS_TOKENIZER_SRC`, default `~/.inferkit-validation/cosmos-tokenizer-src`). The older
    `cosmos_tokenizer` package does not build the releases: it creates every hybrid resampling
    convolution, where the releases omit the ones a level does not use. `checkpoint` is a release
    directory named for its variant (`CI8x8`, `DV8x16x16`, ...) holding `encoder.jit` and `decoder.jit`,
    or the path of that directory's single `autoencoder.jit`, whose weights differ from the pair's for
    DI8x8, DV4x8x8, DV8x8x8, and DV8x16x16 (the other six are bit-identical).
    The release's config.json is empty, so the geometry comes from the name (compression) and the
    release's tensors (patch size, widths, each half's level count); the network is rebuilt, every
    parameter loaded from the TorchScript state dicts, and run in float32 on the CPU. The stored
    derived constants are not loaded: the Haar taps are kept at bfloat16 in the releases (0.70703125),
    and the modules compute 1/sqrt(2). The shipped TorchScript graphs are also run on the same input,
    at float32 where their traced constants allow and otherwise at bfloat16 (CPU torch gains a bf16
    `avg_pool3d` for that), which checks the rebuilt geometry against the release's own graph. Image
    variants take `image` (resized by --size) in [-1, 1]; video variants take a 9-frame clip panning
    across it. Records channels-last seams: the wavelet-patched input, the encoder after conv_in / the
    middle block / the output, the latent (continuous) or the FSQ input, codes, and indices (discrete),
    the decoder after conv_in / the middle block / conv_out; `output` is the reconstruction. Runs under
    the `llm` oracle env.
    """
    import re
    import time

    sys.path.insert(0, os.environ.get("IK_COSMOS_TOKENIZER_SRC",
                                      os.path.expanduser("~/.inferkit-validation/cosmos-tokenizer-src")))
    from cosmos_predict1.tokenizer.networks import TokenizerModels, configs
    from cosmos_predict1.tokenizer.modules import Decoder3DType, DecoderType

    # A release directory reads its encoder.jit and decoder.jit; a path to a release's single
    # autoencoder.jit reads that file instead, whose weights differ from the pair's in four releases.
    path = os.path.normpath(checkpoint)
    combined = path.endswith(".jit")
    variant = os.path.basename(os.path.dirname(path) if combined else path)
    match = re.fullmatch(r"([CD])([IV])(\d+)x(\d+)(?:x(\d+))?", variant)
    if match is None:
        raise SystemExit(f"cannot read a Cosmos tokenizer variant from {variant!r}")
    discrete, video = match.group(1) == "D", match.group(2) == "V"
    kind = match.group(1) + match.group(2)

    if combined:
        autoencoder_jit = torch.jit.load(path, map_location="cpu").eval()
        encoder_state = decoder_state = autoencoder_jit.state_dict()
        shipped_graphs = [autoencoder_jit]
    else:
        encoder_jit = torch.jit.load(os.path.join(path, "encoder.jit"), map_location="cpu").eval()
        decoder_jit = torch.jit.load(os.path.join(path, "decoder.jit"), map_location="cpu").eval()
        encoder_state, decoder_state = encoder_jit.state_dict(), decoder_jit.state_dict()
        shipped_graphs = [encoder_jit, decoder_jit]

    config = dict({"CI": configs.continuous_image, "DI": configs.discrete_image,
                   "CV": configs.continuous_video, "DV": configs.discrete_video}[kind])
    if video:
        config.update(temporal_compression=int(match.group(3)), spatial_compression=int(match.group(4)))
        conv_in = encoder_state["encoder.conv_in.0.conv3d.weight"]
        patch = round((conv_in.shape[1] / 3) ** (1 / 3))
        quant = encoder_state["quant_conv.conv3d.weight"]
    else:
        config.update(spatial_compression=int(match.group(3)))
        conv_in = encoder_state["encoder.conv_in.weight"]
        patch = round((conv_in.shape[1] / 3) ** 0.5)
        quant = encoder_state["quant_conv.weight"]
    channels = conv_in.shape[0]

    def multipliers(state, prefix):
        # Each level's width, read from its first block's first convolution, so a release trained at
        # fewer levels than the package default (CV4x8x8's two) builds as it was trained.
        found, level = [], 0
        while True:
            key = next((k for k in state if k.startswith(f"{prefix}.{level}.block.0.conv1.")
                        and k.endswith("weight")), None)
            if key is None:
                return found
            found.append(state[key].shape[0] // channels)
            level += 1

    encoder_mult = multipliers(encoder_state, "encoder.down")
    decoder_mult = multipliers(decoder_state, "decoder.up")
    config.update(patch_size=patch, channels=channels, z_channels=quant.shape[1], channels_mult=encoder_mult)
    if discrete:
        config.update(embedding_dim=quant.shape[0])
    else:
        config.update(latent_channels=quant.shape[0])
    print(f"{variant}: patch {patch}, channels {conv_in.shape[0]}, z {quant.shape[1]}, "
          f"latent {quant.shape[0]}, stored {conv_in.dtype}")

    print(f"{variant}: encoder levels {encoder_mult}, decoder levels {decoder_mult}")
    model = TokenizerModels[kind].value(**config).eval()
    if decoder_mult != encoder_mult:
        decoder_type = Decoder3DType.FACTORIZED.value if video else DecoderType.Default.value
        model.decoder = decoder_type(**{**config, "channels_mult": decoder_mult,
                                        "z_channels": config["z_channels"]}).eval()
    if discrete:
        model.quantizer.dtype = torch.float32
    encoder, decoder = model.encoder_jit(), model.decoder_jit()
    # The releases also store derived constants: the Haar taps (at bfloat16, 0.70703125 rather than
    # 1/sqrt(2)) and the FSQ tables. Loading them would run float32 arithmetic on bf16-rounded taps, so
    # the modules keep the constants they compute; every parameter must load.
    derived = ("wavelets", "_arange", "patch_size_buffer", "_levels", "_basis", "implicit_codebook")
    for half, state in ((encoder, encoder_state), (decoder, decoder_state)):
        own = half.state_dict()
        parameters = {k: v.float() for k, v in state.items()
                      if not k.endswith(derived) and (not combined or k in own)}
        result = half.load_state_dict(parameters, strict=False)
        missing = [k for k in result.missing_keys if not k.endswith(derived)]
        if missing or result.unexpected_keys:
            raise SystemExit(f"{variant}: missing {missing}, unexpected {result.unexpected_keys}")

    frame = torch.from_numpy(np.ascontiguousarray(image)).float() * 2 - 1           # [H, W, 3] in [-1, 1]
    if video:
        # Panning across the photo gives the clip real motion, so the causal temporal path carries signal.
        frames = [torch.roll(frame, shifts=(2 * t, 3 * t), dims=(0, 1)) for t in range(9)]
        clip = torch.stack(frames)                                                   # [T, H, W, 3]
        pixels = clip.permute(3, 0, 1, 2).unsqueeze(0).contiguous()                  # [1, 3, T, H, W]
    else:
        clip = frame
        pixels = frame.permute(2, 0, 1).unsqueeze(0).contiguous()                    # [1, 3, H, W]

    def last(x):
        # channels-last: [B, C, H, W] -> [B, H, W, C]; [B, C, T, H, W] -> [B, T, H, W, C]
        return x.permute(0, *range(2, x.ndim), 1).contiguous().float()

    seams = {}

    def keep(name):
        def hook(module, inputs, output):
            seams[name] = last(output.detach().clone())
        return hook

    net_encoder, net_decoder = model.encoder, model.decoder
    handles = [
        (net_encoder.patcher3d if video else net_encoder.patcher).register_forward_hook(keep("patched")),
        net_encoder.conv_in.register_forward_hook(keep("encoder_in")),
        net_encoder.mid.block_2.register_forward_hook(keep("encoder_mid")),
        net_decoder.conv_in.register_forward_hook(keep("decoder_in")),
        net_decoder.mid.block_2.register_forward_hook(keep("decoder_mid")),
        net_decoder.conv_out.register_forward_hook(keep("decoder_out")),
    ]
    started = time.time()
    with torch.no_grad():
        encoded_raw = net_encoder(pixels)
        seams["encoder_out"] = last(encoded_raw)
        encoded = encoder(pixels)
        if discrete:
            # The latent before rounding: token parity is only defined away from a rounding boundary.
            seams["quantizer_input"] = last(model.quant_conv(encoded_raw))
            indices, codes = encoded[0], encoded[1]
            seams["codes"] = last(codes)
            seams["indices"] = indices.to(torch.int32).contiguous()
            reconstruction = decoder(indices)
        else:
            latent = encoded[0]
            seams["latent"] = last(latent)
            reconstruction = decoder(latent)
    for handle in handles:
        handle.remove()
    print(f"float32 forward {time.time() - started:.1f}s")

    def cosine(a, b):
        a, b = a.flatten().double(), b.flatten().double()
        return float(a @ b / (a.norm() * b.norm()))

    # The shipped graphs, lifted to float32, on the identical input: the TorchScript IR NVIDIA traced is
    # the release's own definition of the network, so agreement here proves the rebuilt geometry.
    # The image graphs carry bfloat16 constants in their traced IR and run only at that precision; they
    # then agree to the bf16 floor rather than to float32 rounding.
    shipped_dtype = torch.float32
    for graph in shipped_graphs:
        graph.float()
    try:
        with torch.no_grad():
            shipped_graphs[0](pixels)
    except RuntimeError:
        shipped_dtype = conv_in.dtype
        _allow_bfloat16_average_pool()
        for graph in shipped_graphs:
            graph.to(shipped_dtype)
    print(f"shipped graph runs at {shipped_dtype}")
    started = time.time()
    with torch.no_grad():
        shipped = shipped_graphs[0](pixels.to(shipped_dtype))
        if combined:
            shipped_reconstruction = shipped[0] if isinstance(shipped, (tuple, list)) else shipped
        elif discrete:
            shipped_indices = shipped[0]
            agree = (shipped_indices.to(torch.int64) == indices.to(torch.int64)).double().mean().item()
            shipped_reconstruction = decoder_jit(indices)
            print(f"shipped graph: index agreement {agree:.5f}")
            seams["shipped_indices"] = shipped_indices.to(torch.int32).contiguous()
        else:
            shipped_latent = shipped[0] if isinstance(shipped, (tuple, list)) else shipped
            print(f"shipped graph: latent cosine {cosine(shipped_latent.float(), latent):.8f}")
            seams["shipped_latent"] = last(shipped_latent.float())
            shipped_reconstruction = decoder_jit(latent.to(shipped_dtype))
        print(f"shipped graph: reconstruction cosine "
              f"{cosine(shipped_reconstruction.float(), reconstruction):.8f} ({time.time() - started:.1f}s)")
    seams["shipped_reconstruction"] = last(shipped_reconstruction.float())

    seams["clip"] = clip.contiguous()
    globals()["_extra"] = seams
    return last(reconstruction)[0]


def run_cosmos_tokenizer_loss(image, checkpoint):
    """The Cosmos Tokenizer post-training objective, from NVIDIA's own `ColorLoss` and `PerceptualLoss`
    (`cosmos_predict1/tokenizer/training/losses`, vendored beside the network code under
    `IK_COSMOS_TOKENIZER_SRC`), scored on identical tensors: a deterministic image batch and a clip,
    each a target and a perturbed reconstruction in [-1, 1]. `checkpoint` is the VGG-16 ImageNet
    weights file the port loads (`timm/vgg16.tv_in1k` `model.safetensors`), loaded into torchvision's
    `vgg16` in place of its download, so both sides read one file. The LPIPS linear heads that
    `LPIPS.__init__` downloads are never read by `PerceptualLoss.forward` and are skipped. Two helper
    modules the losses import (`utils.lazy_config`, `utils.distributed`) are stubbed; neither carries
    arithmetic. Records the color, perceptual, and Gram terms unweighted (Gram switched on, which
    post-training leaves off) for the image batch and the clip; `output` is the image batch's color
    term. Runs under the `llm` oracle env (torchvision).
    """
    import types

    sys.path.insert(0, os.environ.get("IK_COSMOS_TOKENIZER_SRC",
                                      os.path.expanduser("~/.inferkit-validation/cosmos-tokenizer-src")))
    for name, attributes in (("cosmos_predict1.utils", {}),
                             ("cosmos_predict1.utils.lazy_config", {"instantiate": lambda x: x}),
                             ("cosmos_predict1.utils.distributed", {"is_rank0": lambda: True})):
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        sys.modules[name] = module
    from safetensors.torch import load_file
    import torchvision
    from cosmos_predict1.tokenizer.training.losses import lpips as lpips_module
    from cosmos_predict1.tokenizer.training.losses.continuous import ColorLoss, PerceptualLoss

    weights = load_file(checkpoint)
    torchvision_vgg16 = torchvision.models.vgg16

    def vgg16(pretrained=False, **kwargs):
        model = torchvision_vgg16(weights=None)
        model.features.load_state_dict({k[len("features."):]: v.float() for k, v in weights.items()
                                        if k.startswith("features.")})
        return model

    lpips_module.models.vgg16 = vgg16
    lpips_module.LPIPS.load_from_pretrained = lambda self, name="vgg_lpips": None

    class Config:
        pass

    color_config = Config()
    color_config.boundaries, color_config.values = [0], [1.0]
    perceptual_config = Config()
    perceptual_config.__dict__.update(
        lpips_boundaries=[0], lpips_values=[1.0], layer_weights=[1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10.0 / 1.5],
        gram_enabled=True, gram_boundaries=[0], gram_values=[1.0], corr_enabled=False, corr_boundaries=[0],
        corr_values=[0.0], checkpoint_activations=False)
    color, perceptual = ColorLoss(color_config), PerceptualLoss(perceptual_config).eval()

    rng = np.random.default_rng(7)
    extra = {}
    for label, shape in (("image", (2, 3, 48, 64)), ("clip", (1, 3, 3, 48, 64))):
        target = torch.from_numpy(np.clip(rng.standard_normal(shape) * 0.5, -1, 1).astype(np.float32))
        reconstruction = torch.from_numpy(
            np.clip(target.numpy() + rng.standard_normal(shape).astype(np.float32) * 0.2, -1, 1))
        inputs = {"INPUT": target, "loss_mask": torch.ones_like(target)}
        outputs = {"reconstructions": reconstruction}
        with torch.no_grad():
            terms = {**color(inputs, outputs, 0), **perceptual(inputs, outputs, 0)}
        channels_last = (0, 2, 3, 1) if len(shape) == 4 else (0, 2, 3, 4, 1)
        extra[f"{label}_target"] = target.permute(*channels_last).contiguous()
        extra[f"{label}_reconstruction"] = reconstruction.permute(*channels_last).contiguous()
        for term in ("color", "lpips", "gram"):
            extra[f"{label}_{term}"] = terms[term].mean().reshape(1)
        print(label, {k: float(v) for k, v in extra.items() if k.startswith(label) and v.numel() == 1})
    globals()["_extra"] = extra
    return extra["image_color"].clone()


def run_kokoro(image, checkpoint):
    """Kokoro-82M (StyleTTS2 / iSTFTNet) text-to-speech, from the vendored `kokoro_vendor.KModel` on the
    RELEASED weights — a real-weights end-to-end validation captured seam by seam: the PL-BERT (Albert)
    text encoder, the bert_encoder projection, the duration predictor (DurationEncoder + LSTM + duration
    proj), the F0/energy predictor, the TextEncoder, the alignment-expanded asr, and the iSTFTNet decoder
    audio. The phonemes are supplied directly (misaki is NOT installed), so the model is verified without
    the phonemizer. The sine source's phase noise and additive noise are zeroed so the decoder is
    deterministic and comparable. `checkpoint` is the local Kokoro release directory. Runs under the
    `llm` oracle env (torch, transformers, scipy; the model files are vendored, no spacy).
    """
    import sys
    sys.path.insert(0, os.path.expanduser("~/.inferkit-validation"))
    import kokoro_vendor.istftnet as istftnet
    from kokoro_vendor.model import KModel

    # Make the sine source deterministic: zero the initial phase noise and the additive noise, so the
    # excitation is a pure function of F0 and the whole decoder is reproducible cross-framework.
    original_rand, original_randn = torch.rand, torch.randn_like
    istftnet.torch.rand = lambda *a, **k: torch.zeros(*a, **{key: v for key, v in k.items() if key != "device"})
    istftnet.torch.randn_like = lambda x, *a, **k: torch.zeros_like(x)

    model = KModel(config=os.path.join(checkpoint, "config.json"),
                   model=os.path.join(checkpoint, "kokoro-v1_0.pth")).eval()
    voice = torch.load(os.path.join(checkpoint, "voices/af_heart.pt"), weights_only=True)
    phonemes = "həlˈoʊ wˈɜːld"
    ref_s = voice[len(phonemes) - 1]

    with torch.no_grad():
        input_ids = [i for i in (model.vocab.get(p) for p in phonemes) if i is not None]
        input_ids = torch.LongTensor([[0, *input_ids, 0]])
        input_lengths = torch.LongTensor([input_ids.shape[-1]])
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))
        bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        lstm_out, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(lstm_out)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1]), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]))
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        t_en = model.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        decoder_seams = {}
        model.decoder.encode.register_forward_hook(
            lambda m, i, o: decoder_seams.__setitem__("encode", o.detach()))
        model.decoder.generator.register_forward_pre_hook(
            lambda m, args: decoder_seams.__setitem__("gen_in", args[0].detach()))
        model.decoder.generator.conv_post.register_forward_hook(
            lambda m, i, o: decoder_seams.__setitem__("conv_post", o.detach()))
        model.decoder.generator.m_source.register_forward_hook(
            lambda m, i, o: decoder_seams.__setitem__("har_source", o[0].detach()))
        audio = model.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()

    istftnet.torch.rand, istftnet.torch.randn_like = original_rand, original_randn

    extra = {
        "ids": input_ids[0].to(torch.int32).contiguous(),
        "ref_s": ref_s[0].contiguous(),                                    # [256]
        "bert": bert_dur[0].contiguous(),                                  # [T, 768]
        "d_en": d_en[0].contiguous(),                                      # [512, T]
        "d": d[0].contiguous(),                                            # [T, 640]
        "duration": duration[0].contiguous(),                             # [T] pre-round
        "pred_dur": pred_dur.to(torch.int32).contiguous(),                 # [T]
        "f0": F0_pred[0].contiguous(),                                     # [frames]
        "n": N_pred[0].contiguous(),                                       # [frames]
        "t_en": t_en[0].contiguous(),                                      # [512, T]
        "asr": asr[0].contiguous(),                                        # [512, frames]
        "dec_encode": decoder_seams["encode"][0].contiguous(),            # [1024, frames]
        "dec_gen_in": decoder_seams["gen_in"][0].contiguous(),            # [512, frames·2]
        "dec_conv_post": decoder_seams["conv_post"][0].contiguous(),      # [22, T_stft]
        "dec_har_source": decoder_seams["har_source"][0].contiguous(),    # [L, 1] the sine excitation
    }
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous() if value.is_floating_point() else value.contiguous()
    globals()["_extra"] = extra
    return audio.contiguous()                                              # [samples]


def run_rtdetr_real(image, checkpoint):
    """RT-DETR on the RELEASED `PekingU/rtdetr_r50vd` weights, from transformers' own
    RTDetrForObjectDetection and its image processor — a real-weights end-to-end validation (the r50vd
    ResNet-50-vd geometry, the actual checkpoint, and the loader, including the stage-1 stride-1 shortcut
    the tiny config does not exercise). `checkpoint` is the local release directory. Records the
    preprocessed pixel values so the port runs on the identical input, plus the reference's query
    selection. Runs under the `llm` oracle env (transformers, needs Pillow).
    """
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
    from PIL import Image

    model = RTDetrForObjectDetection.from_pretrained(checkpoint, torch_dtype=torch.float32).eval()
    processor = RTDetrImageProcessor.from_pretrained(checkpoint)
    pil = Image.fromarray((image * 255).astype("uint8"))
    pixel_values = processor(images=pil, return_tensors="pt")["pixel_values"]   # [1, 3, 640, 640]
    with torch.no_grad():
        out = model(pixel_values, return_dict=True)
        topk = torch.topk(out.enc_outputs_class.max(-1).values, model.config.num_queries, dim=1).indices

    globals()["_extra"] = {
        "pixels": pixel_values[0].permute(1, 2, 0).contiguous(),                # [640, 640, 3] NHWC
        "pred_boxes": out.pred_boxes[0].clone().contiguous(),                   # [300, 4]
        "topk_ind": topk[0].to(torch.int32).contiguous(),                       # [300]
    }
    return out.logits[0].clone().contiguous()                                   # [300, 80]


def run_rtdetr_v2(image):
    """RT-DETRv2 at a tiny random configuration, exercising what v2 adds to RT-DETR.

    v2's only architectural change is the deformable decoder's sampling: `decoder_offset_scale`
    replaces RT-DETR's fixed 0.5, and `decoder_method` offers `discrete` (the nearest cell, clamped)
    beside `default` (bilinear). Every released v2 configuration states the RT-DETR values, so this
    mode deliberately sets neither: `discrete` with an offset scale of 0.35, which is the only way the
    new code is measured rather than merely present. `rtdetr_v2_real` covers the released settings.

    The BatchNorm buffers stay physical for the reason `run_rtdetr` documents: the shared `_randomized`
    helper would randomize `running_var` negative and `rsqrt` would be NaN on both sides.
    """
    from transformers import RTDetrV2Config, RTDetrV2ForObjectDetection
    from transformers.models.rt_detr.configuration_rt_detr_resnet import RTDetrResNetConfig

    backbone = RTDetrResNetConfig(
        embedding_size=16, hidden_sizes=[16, 32, 64, 128], depths=[1, 1, 1, 1],
        layer_type="bottleneck", downsample_in_bottleneck=False, downsample_in_first_stage=False,
        out_features=["stage2", "stage3", "stage4"], num_channels=3)
    config = RTDetrV2Config(
        backbone_config=backbone, use_timm_backbone=False, backbone=None,
        encoder_in_channels=[32, 64, 128], feat_strides=[8, 16, 32], encoder_hidden_dim=32,
        encoder_ffn_dim=48, num_attention_heads=2, encoder_layers=1, encode_proj_layers=[2],
        d_model=32, decoder_attention_heads=2, decoder_ffn_dim=48, decoder_layers=2,
        decoder_n_points=4, num_feature_levels=3, decoder_in_channels=[32, 32, 32],
        num_queries=10, num_denoising=0, learn_initial_query=False, anchor_image_size=None,
        with_box_refine=True, num_labels=4, hidden_expansion=1.0,
        decoder_method="discrete", decoder_offset_scale=0.35, decoder_n_levels=3)

    model = RTDetrV2ForObjectDetection(config)
    torch.manual_seed(29)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.05)
    model = model.eval().float()

    generator = torch.Generator().manual_seed(7)
    pixels = torch.randn(1, 3, 64, 64, generator=generator)

    extra = {"pixels": pixels[0].permute(1, 2, 0).contiguous()}
    features = []
    handle = model.model.backbone.register_forward_hook(lambda m, i, o: features.append([f for f, _ in o]))
    with torch.no_grad():
        out = model(pixels, return_dict=True)
        topk = torch.topk(out.enc_outputs_class.max(-1).values, config.num_queries, dim=1).indices
    handle.remove()
    for index, feature in enumerate(features[0]):
        extra[f"bb.{index}"] = feature[0].permute(1, 2, 0).contiguous()
    extra["pred_boxes"] = out.pred_boxes[0].clone().contiguous()
    extra["enc_class"] = out.enc_outputs_class[0].clone().contiguous()
    extra["enc_coord"] = out.enc_outputs_coord_logits[0].clone().contiguous()
    extra["topk_ind"] = topk[0].to(torch.int32).contiguous()
    for key, value in model.state_dict().items():
        if key.startswith("model.") and not key.endswith("num_batches_tracked"):
            extra[f"w::{key}"] = value.contiguous()

    globals()["_extra"] = extra
    return out.logits[0].clone().contiguous()


def run_rtdetr_v2_real(image, checkpoint):
    """RT-DETRv2 on a RELEASED `PekingU/rtdetr_v2_*` checkpoint, from transformers' own
    RTDetrV2ForObjectDetection and its image processor. `checkpoint` is the local release directory.
    One mode serves all four sizes; the geometry comes from the release's own config. Runs under the
    `llm` oracle env (transformers, needs Pillow).
    """
    from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
    from PIL import Image

    model = RTDetrV2ForObjectDetection.from_pretrained(checkpoint, torch_dtype=torch.float32).eval()
    processor = RTDetrImageProcessor.from_pretrained(checkpoint)
    pil = Image.fromarray((image * 255).astype("uint8"))
    pixel_values = processor(images=pil, return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        out = model(pixel_values, return_dict=True)
        topk = torch.topk(out.enc_outputs_class.max(-1).values, model.config.num_queries, dim=1).indices

    globals()["_extra"] = {
        "pixels": pixel_values[0].permute(1, 2, 0).contiguous(),
        "pred_boxes": out.pred_boxes[0].clone().contiguous(),
        "topk_ind": topk[0].to(torch.int32).contiguous(),
    }
    return out.logits[0].clone().contiguous()


def run_vitpose(image, checkpoint):
    """ViTPose on a RELEASED `usyd-community/vitpose-*` checkpoint, from transformers' own
    VitPoseForPoseEstimation. `checkpoint` is the local release directory; one mode serves the simple
    and classic decoders, since the geometry comes from the release's own config.

    The image is resized to the trained crop and normalized here rather than through the release's
    image processor, because that processor warps a person's BOX through an affine transform and a
    whole-image caller has no box — the Swift backend resizes for the same reason. Records the
    prepared pixels so the port runs on the identical input, the backbone feature map, the heatmaps,
    and the DARK-refined keypoints the processor's own decode produces. Runs under the `llm` oracle
    env (transformers, needs Pillow).
    """
    import numpy as np
    from transformers import VitPoseForPoseEstimation
    from transformers.models.vitpose.image_processing_vitpose import (
        get_keypoint_predictions,
        post_dark_unbiased_data_processing,
    )

    model = VitPoseForPoseEstimation.from_pretrained(checkpoint, dtype=torch.float32).eval()
    height = model.config.backbone_config.image_size[0]
    width = model.config.backbone_config.image_size[1]
    patch = model.config.backbone_config.patch_size[0]

    pixels = torch.from_numpy(image).permute(2, 0, 1)[None].float()
    pixels = torch.nn.functional.interpolate(pixels, size=(height, width), mode="bilinear",
                                             align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    prepared = (pixels - mean) / std

    features = []
    handle = model.backbone.register_forward_hook(lambda m, i, o: features.append(o.feature_maps[-1]))
    with torch.no_grad():
        out = model(prepared, return_dict=True)
    handle.remove()

    heatmaps = out.heatmaps
    coords, scores = get_keypoint_predictions(heatmaps.numpy())
    refined = post_dark_unbiased_data_processing(coords.copy(), heatmaps.numpy(), kernel=11)

    globals()["_extra"] = {
        "pixels": prepared[0].permute(1, 2, 0).contiguous(),                 # [H, W, 3] NHWC
        # The backbone returns the token sequence; the model permutes and reshapes it to a feature map
        # before the head, which is what the port holds in NHWC.
        "features": features[0].reshape(1, height // patch, width // patch, -1)[0].contiguous(),
        "coords": torch.from_numpy(np.ascontiguousarray(coords[0])).float(), # [K, 2] integer peaks
        "refined": torch.from_numpy(np.ascontiguousarray(refined[0])).float(),
        "scores": torch.from_numpy(np.ascontiguousarray(scores[0])).float(),
    }
    return heatmaps[0].permute(1, 2, 0).contiguous()                         # [h, w, K] NHWC


def run_ddcolor(image, checkpoint):
    """DDColor on a RELEASED checkpoint, from the authors' own `DDColor` architecture.

    The reference is not in transformers; IK_DDCOLOR_SRC holds the cloned repository's `basicsr/`
    (its `ddcolor_arch.py` plus `ddcolor_arch_utils/`). The model is built at the released
    ConvNeXt-L geometry and loaded strictly, so a key with no counterpart fails here.

    The image is fed as the gray three-channel image the pipeline builds from the lightness alone, at
    the model's own 512, and the recorded seams are the four hooked encoder features, the three U-Net
    stage outputs, the pixel embedding, the color attention maps, and the two chroma channels. The
    model normalizes with ImageNet statistics inside its own forward, so `pixels` is recorded before
    that. Runs under the `llm` oracle env (torch; no transformers).
    """
    import numpy as np
    from skimage import color

    source = os.environ.get("IK_DDCOLOR_SRC")
    if not source:
        raise SystemExit("set IK_DDCOLOR_SRC to the cloned DDColor repository's directory")
    sys.path.insert(0, source)
    # `ddcolor_arch` imports basicsr's registry, which pulls the whole training package; a stub with
    # the one decorator it uses keeps the import to the architecture itself.
    import types
    registry = types.ModuleType("basicsr.utils.registry")
    registry.ARCH_REGISTRY = types.SimpleNamespace(register=lambda: (lambda cls: cls))
    utils = types.ModuleType("basicsr.utils")
    utils.registry = registry
    sys.modules.setdefault("basicsr.utils", utils)
    sys.modules.setdefault("basicsr.utils.registry", registry)
    from basicsr.archs.ddcolor_arch import DDColor

    size = image.shape[0]
    model = DDColor(encoder_name="convnext-l", decoder_name="MultiScaleColorDecoder",
                    input_size=[512, 512], num_output_channels=2, last_norm="Spectral",
                    do_normalize=False, num_queries=100, num_scales=3, dec_layers=9)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["params"], strict=False)
    model.eval()

    # The pipeline's own input: the lightness alone taken back through Lab with no chroma.
    lab = color.rgb2lab(image)
    lightness = lab[:, :, :1]
    gray = color.lab2rgb(np.concatenate([lightness, np.zeros_like(lab[:, :, 1:])], axis=2))
    tensor = torch.from_numpy(gray).permute(2, 0, 1)[None].float()
    tensor = torch.nn.functional.interpolate(tensor, size=(512, 512), mode="bilinear",
                                             align_corners=False)

    extra = {"pixels": tensor[0].permute(1, 2, 0).contiguous(),
             "lightness": torch.from_numpy(lightness[:, :, 0]).float().contiguous()}
    with torch.no_grad():
        normalized = model.normalize(tensor)
        model.encoder(normalized)
        for index, hook in enumerate(model.encoder.hooks):
            extra[f"hook{index}"] = hook.feature[0].permute(1, 2, 0).contiguous()

        decoder = model.decoder
        out0 = decoder.layers[0](decoder.hooks[-1].feature)
        out1 = decoder.layers[1](out0)
        out2 = decoder.layers[2](out1)
        out3 = decoder.last_shuf(out2)
        for name, value in [("out0", out0), ("out1", out1), ("out2", out2), ("pixels_embed", out3)]:
            extra[name] = value[0].permute(1, 2, 0).contiguous()
        maps = decoder.color_decoder([out0, out1, out2], out3)
        extra["color_maps"] = maps[0].permute(1, 2, 0).contiguous()
        chroma = model.refine_net(torch.cat([maps, normalized], dim=1))

    globals()["_extra"] = extra
    return chroma[0].permute(1, 2, 0).contiguous()                       # [512, 512, 2]


def run_realesrgan_compact(image, checkpoint):
    """Real-ESRGAN's later COMPACT generator (`SRVGGNetCompact`) on a released checkpoint.

    The `realesr-general-x4v3` and `realesr-animevideov3` releases run a VGG-style compact network
    rather than the RRDBNet the earlier releases use, so they need their own mode. The body's
    convolution count is derived from the checkpoint itself, because the two releases differ only in
    that. Runs under the `llm` oracle env (torch alone; the architecture is inlined here rather than
    pulled from basicsr).
    """
    import torch.nn as nn

    class SRVGGNetCompact(nn.Module):
        def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4):
            super().__init__()
            self.upscale = upscale
            self.body = nn.ModuleList()
            self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
            for _ in range(num_conv):
                self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
                self.body.append(nn.PReLU(num_parameters=num_feat))
            self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
            self.upsampler = nn.PixelShuffle(upscale)

        def forward(self, x):
            out = x
            for layer in self.body:
                out = layer(out)
            out = self.upsampler(out)
            return out + torch.nn.functional.interpolate(x, scale_factor=self.upscale, mode="nearest")

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("params", state.get("params_ema", state))
    # Two tensors per convolution and one per activation, plus the first convolution and activation
    # and the last convolution: (len - 5) / 3 body convolutions.
    convolutions = (len(state) - 5) // 3
    model = SRVGGNetCompact(num_conv=convolutions).eval()
    model.load_state_dict(state, strict=True)

    tensor = torch.from_numpy(image).permute(2, 0, 1)[None].float()
    with torch.no_grad():
        out = model(tensor)

    globals()["_extra"] = {"num_conv": torch.tensor(convolutions, dtype=torch.int32)}
    return out[0].permute(1, 2, 0).contiguous()


def run_zero_dce_plus(image, checkpoint):
    """Zero-DCE++ (`enhance_net_nopool`) on the authors' released weights.

    The successor to Zero-DCE: depthwise-separable convolutions, ONE shared three-channel curve map
    rather than eight, and a curve estimator that runs at a reduced resolution and lifts its map back.
    The architecture is inlined here (the reference file is a single short module). The record carries
    the curve map beside the enhanced image, because the map is where a wrong upsample or a wrong
    iteration count shows first. Runs under the `llm` oracle env (torch alone).
    """
    import torch.nn as nn
    import torch.nn.functional as F

    class CSDN_Tem(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.depth_conv = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch)
            self.point_conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

        def forward(self, x):
            return self.point_conv(self.depth_conv(x))

    class EnhanceNet(nn.Module):
        def __init__(self, scale_factor):
            super().__init__()
            self.relu = nn.ReLU(inplace=True)
            self.scale_factor = scale_factor
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
            f = 32
            self.e_conv1 = CSDN_Tem(3, f)
            self.e_conv2 = CSDN_Tem(f, f)
            self.e_conv3 = CSDN_Tem(f, f)
            self.e_conv4 = CSDN_Tem(f, f)
            self.e_conv5 = CSDN_Tem(f * 2, f)
            self.e_conv6 = CSDN_Tem(f * 2, f)
            self.e_conv7 = CSDN_Tem(f * 2, 3)

        def enhance(self, x, x_r):
            for _ in range(8):
                x = x + x_r * (torch.pow(x, 2) - x)
            return x

        def forward(self, x):
            down = x if self.scale_factor == 1 else F.interpolate(
                x, scale_factor=1 / self.scale_factor, mode="bilinear")
            x1 = self.relu(self.e_conv1(down))
            x2 = self.relu(self.e_conv2(x1))
            x3 = self.relu(self.e_conv3(x2))
            x4 = self.relu(self.e_conv4(x3))
            x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
            x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
            x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
            if self.scale_factor != 1:
                x_r = self.upsample(x_r)
            return self.enhance(x, x_r), x_r

    scale = int(os.environ.get("IK_ZERO_DCE_PLUS_SCALE", "12"))
    model = EnhanceNet(scale).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)

    tensor = torch.from_numpy(image).permute(2, 0, 1)[None].float()
    with torch.no_grad():
        enhanced, curve = model(tensor)

    globals()["_extra"] = {
        "curve": curve[0].permute(1, 2, 0).contiguous(),
        "scale_factor": torch.tensor(scale, dtype=torch.int32),
    }
    return enhanced[0].permute(1, 2, 0).contiguous()


def run_rf_detr(image):
    """RF-DETR object detection end to end at a tiny random configuration, from transformers' own
    RfDetrForObjectDetection (transformers 5.16): the WINDOWED DINOv2 backbone (each block partitions
    the patch grid into num_windows^2 local windows with a replicated CLS; a global-attention block
    unpartitions to one sequence per image before attending and re-partitions after; the selected
    stages are layernormed, the CLS dropped, unpartitioned, and reshaped to feature maps), the C2f +
    RepVGG scale projector (concat the stage maps -> C2FLayer -> channels-first LayerNorm), two-stage
    query selection (enc_output Linear + enc_output_norm LayerNorm, the enc_out_class/bbox heads over
    every token, top-k by the class max), the MIXED queries (the learned reference_point_embed refined
    by the top-k coords via refine_bboxes in DIRECT box space — NOT sigmoid space like RT-DETR — plus
    the learned query_feat), and the LW-DETR decoder (self-attention with the query position added to
    q AND k and the value read WITHOUT position; deformable cross-attention over the single feature
    level; NO iterative box refine — the reference points are constant and one box refinement happens
    at the end in the ForObjectDetection head). Group-DETR collapses to one group at inference.

    Recorded seam by seam so a divergence localizes: the backbone feature maps (`bb.N`), the projector
    output (`proj`), the first-stage class head over ALL tokens (`enc_class_full`, pre-mask), the
    reference's top-k selection (`topk_ind`, float-tie-sensitive so the port's decoder is verified over
    it), the mixed init reference points (`init_ref`), the gathered first-stage class/coord
    (`enc_class_topk` / `enc_coord_topk`), the decoder last hidden state (`dec_last`), and the final
    logits/pred_boxes. Weights as `w::model.*` / `w::class_embed.*` / `w::bbox_embed.*` (the top-level
    heads are the REAL final heads here, not tied duplicates as in RT-DETR). Runs under the `rfdetr`
    oracle env (transformers 5.16.1). `image` unused.
    """
    from transformers import RfDetrConfig
    from transformers.models.rf_detr.configuration_rf_detr import RfDetrDinov2Config
    from transformers.models.rf_detr.modeling_rf_detr import RfDetrForObjectDetection

    # A shrunk RF-DETR that carries every structural form: 4 backbone layers with a global-attention
    # block (layer 2) among the windowed ones, two selected stages at one resolution (DINOv2 does not
    # downsample), a C2f projector, group-DETR (2 groups, only group 0 used at inference), and a
    # two-layer LW-DETR decoder over a single 4x4 feature level (16 tokens > 10 queries).
    backbone = RfDetrDinov2Config(
        hidden_size=32, num_hidden_layers=4, num_attention_heads=2, mlp_ratio=4,
        patch_size=14, image_size=56, num_windows=2, use_swiglu_ffn=False,
        layerscale_value=1.0, layer_norm_eps=1e-6, use_mask_token=True,
        out_features=["stage2", "stage4"], apply_layernorm=True, reshape_hidden_states=True)
    config = RfDetrConfig(
        backbone_config=backbone, d_model=32, num_labels=4, num_queries=10, group_detr=2,
        decoder_layers=2, decoder_self_attention_heads=2, decoder_cross_attention_heads=4,
        decoder_n_points=4, num_feature_levels=1, decoder_ffn_dim=48,
        decoder_activation_function="relu", hidden_expansion=0.5, c2f_num_blocks=2,
        activation_function="silu", disable_custom_kernels=True, layer_norm_eps=1e-5)

    model = RfDetrForObjectDetection(config)
    torch.manual_seed(29)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.05)
    model = model.eval().float()

    generator = torch.Generator().manual_seed(7)
    pixels = torch.randn(1, 3, 56, 56, generator=generator)

    seams = {}
    model.model.backbone.backbone.register_forward_hook(
        lambda m, i, o: seams.__setitem__("bb", [f.detach() for f in o.feature_maps]))
    model.model.backbone.register_forward_hook(
        lambda m, i, o: seams.__setitem__("proj", o[0].detach()))
    cls_calls = []
    model.model.enc_out_class_embed[0].register_forward_hook(lambda m, i, o: cls_calls.append(o.detach()))

    # The top-k index selection is float-tie-sensitive (torch.topk vs MLX argSort can break a sub-ulp
    # tie differently), so capture the reference's own indices for the port to verify its decoder over.
    original_topk = torch.topk
    topk_captured = []
    def spy(*a, **k):
        result = original_topk(*a, **k)
        topk_captured.append(result.indices.detach())
        return result
    torch.topk = spy
    with torch.no_grad():
        out = model(pixels, return_dict=True)
    torch.topk = original_topk

    extra = {
        "pixels": pixels[0].permute(1, 2, 0).contiguous(),                       # [H, W, 3] NHWC
        "proj": seams["proj"][0].permute(1, 2, 0).contiguous(),                  # [Hp, Wp, d_model] NHWC
        "enc_class_full": cls_calls[0][0].contiguous(),                          # [S, num_labels] pre-mask
        "topk_ind": topk_captured[0][0].to(torch.int32).contiguous(),            # [Q]
        "init_ref": out.init_reference_points[0].contiguous(),                   # [Q, 4] direct box space
        "enc_coord_topk": out.enc_outputs_coord_logits[0].contiguous(),          # [Q, 4]
        "enc_class_topk": out.enc_outputs_class[0].contiguous(),                 # [Q, num_labels]
        "dec_last": out.last_hidden_state[0].contiguous(),                       # [Q, d_model]
        "pred_boxes": out.pred_boxes[0].contiguous(),                            # [Q, 4]
    }
    for level, feature in enumerate(seams["bb"]):
        extra[f"bb.{level}"] = feature[0].permute(1, 2, 0).contiguous()          # backbone feature map NHWC
    for key, value in model.state_dict().items():
        if key.endswith("num_batches_tracked"):
            continue
        extra[f"w::{key}"] = value.float().contiguous() if value.is_floating_point() else value.contiguous()
    globals()["_extra"] = extra
    return out.logits[0].clone().contiguous()                                    # [Q, num_labels]


def run_rf_detr_real(image, checkpoint):
    """RF-DETR on the RELEASED `Roboflow/rf-detr-base` weights (transformers 5.16), from
    RfDetrForObjectDetection and its image processor — a real-weights end-to-end validation exercising
    the DINOv2-small windowed backbone (hidden 384, 12 layers, num_windows 4, global attention at the
    out-index layers), the bicubic-antialias position-embedding interpolation to the processor's
    resolution (the tiny config sidesteps it, so this is where that seam is measured), the projector,
    query selection, and the decoder on the actual checkpoint. `checkpoint` is the local release
    directory. Records the preprocessed pixel values (so the port runs on the identical input), the
    projector output, the mixed init reference points, the decoder last hidden state, the reference's
    top-k selection, and the final logits/pred_boxes. Runs under the `rfdetr` oracle env (transformers
    5.16.1, needs Pillow).
    """
    from transformers import RfDetrForObjectDetection, AutoImageProcessor
    from PIL import Image

    model = RfDetrForObjectDetection.from_pretrained(checkpoint, torch_dtype=torch.float32).eval()
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    pil = Image.fromarray((image * 255).astype("uint8"))
    pixel_values = processor(images=pil, return_tensors="pt")["pixel_values"]    # [1, 3, S, S]

    seams = {}
    model.model.backbone.backbone.register_forward_hook(
        lambda m, i, o: seams.__setitem__("bb", [f.detach() for f in o.feature_maps]))
    model.model.backbone.register_forward_hook(lambda m, i, o: seams.__setitem__("proj", o[0].detach()))
    original_topk = torch.topk
    topk_captured = []
    def spy(*a, **k):
        result = original_topk(*a, **k)
        topk_captured.append(result.indices.detach())
        return result
    torch.topk = spy
    with torch.no_grad():
        out = model(pixel_values, return_dict=True)
    torch.topk = original_topk

    extra = {
        "pixels": pixel_values[0].permute(1, 2, 0).contiguous(),                 # [S, S, 3] NHWC
        "proj": seams["proj"][0].permute(1, 2, 0).contiguous(),                  # [Hp, Wp, d_model] NHWC
        "init_ref": out.init_reference_points[0].contiguous(),                   # [Q, 4]
        "dec_last": out.last_hidden_state[0].contiguous(),                       # [Q, d_model]
        "topk_ind": topk_captured[0][0].to(torch.int32).contiguous(),            # [Q]
        "pred_boxes": out.pred_boxes[0].contiguous(),                            # [Q, 4]
    }
    for level, feature in enumerate(seams["bb"]):
        extra[f"bb.{level}"] = feature[0].permute(1, 2, 0).contiguous()          # backbone feature map NHWC
    # No weight dump: the port loads the RELEASED file directly through NFKMLXRFDetr.loadWeights, which
    # converts the original Roboflow naming (backbone./transformer./refpoint_embed, the fused self_attn
    # in_proj) to the module names on device — the same conversion from_pretrained does here.
    globals()["_extra"] = extra
    return out.logits[0].clone().contiguous()                                    # [Q, num_labels]


def run_parakeet(image, checkpoint):
    """NVIDIA Parakeet-TDT (0.6B v2) speech recognition, from NeMo's own EncDecRNNTBPEModel on the
    RELEASED `.nemo` archive (`--checkpoint`): the FastConformer encoder (dw-striding 8x subsampling,
    24 rel-pos conformer layers, no biases) and the token-and-duration transducer (an LSTM prediction
    net, the joint, greedy TDT decoding over durations [0..4]). The clip is the validation speech WAV
    (`IK_VAL_AUDIO`, 16 kHz), so the transcription is a real measurement. Recorded seam by seam: the
    normalized mel features, the pre-encode output, the first conformer layer, the encoder output, the
    joint logits for frame 0 at the blank/SOS state, and the greedy tokens, text, and frame timestamps.
    Dither is zeroed (training-time noise). Runs under the `nemo` oracle env (nemo_toolkit[asr]).
    """
    import wave as wavemodule
    import huggingface_hub
    for name in ["ModelFilter", "DatasetFilter"]:
        if not hasattr(huggingface_hub, name):
            setattr(huggingface_hub, name, type(name, (), {}))
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(checkpoint, strict=False).eval()
    model.preprocessor.featurizer.dither = 0.0

    path = os.environ.get("IK_VAL_AUDIO", os.path.expanduser("~/.inferkit-validation/inputs/speech.wav"))
    with wavemodule.open(path) as handle:
        assert handle.getframerate() == 16000 and handle.getnchannels() == 1 and handle.getsampwidth() == 2
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    wave = (pcm.astype(np.float32) / 32768.0)
    signal = torch.from_numpy(wave)[None]
    length = torch.tensor([wave.shape[0]])

    seams = {}
    model.encoder.pre_encode.register_forward_hook(lambda m, i, o: seams.__setitem__("pre", o[0].detach()))
    model.encoder.layers[0].register_forward_hook(lambda m, i, o: seams.__setitem__("layer0", o.detach()))
    with torch.no_grad():
        features, feature_length = model.preprocessor(input_signal=signal, length=length)
        encoded, encoded_length = model.encoder(audio_signal=features, length=feature_length)  # [1, D, T]
        g, _ = model.decoder.predict(None, None, add_sos=False, batch_size=1)            # SOS state
        f = encoded.transpose(1, 2)[:, 0:1, :]
        joint0 = model.joint.joint(f, g)                                                 # [1, 1, 1, V+1+durations]
        hypotheses = model.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded, encoded_lengths=encoded_length, return_hypotheses=True)
    if isinstance(hypotheses, tuple):
        hypotheses = hypotheses[0]
    hypothesis = hypotheses[0]
    tokens = list(hypothesis.y_sequence.tolist() if hasattr(hypothesis.y_sequence, "tolist") else hypothesis.y_sequence)
    text = model.tokenizer.ids_to_text(tokens)
    timestamps = list(hypothesis.timestamp.tolist() if hasattr(hypothesis.timestamp, "tolist") else hypothesis.timestamp)
    print(f"parakeet transcription: {text!r} ({len(tokens)} tokens, {int(encoded_length[0])} frames)")

    extra = {
        "waveform": torch.from_numpy(wave).contiguous(),
        "features": features[0].transpose(0, 1).contiguous(),               # [frames, mels] normalized
        "pre": seams["pre"][0].contiguous(),                                 # [T', d_model]
        "layer0": seams["layer0"][0].contiguous(),                           # [T', d_model]
        "encoded": encoded[0].transpose(0, 1).contiguous(),                  # [T', d_model]
        "joint0": joint0[0, 0, 0].contiguous(),                              # [V + 1 + durations]
        "tokens": torch.tensor(tokens, dtype=torch.int32),
        "timestamps": torch.tensor(timestamps, dtype=torch.int32),
        "text": torch.tensor(list(text.encode("utf-8")), dtype=torch.int32),
    }
    for key, value in model.state_dict().items():
        if key.endswith("num_batches_tracked"):
            continue
        extra[f"w::{key}"] = value.float().contiguous() if value.is_floating_point() else value.contiguous()
    globals()["_extra"] = extra
    return encoded[0].transpose(0, 1).clone().contiguous()


def run_canary(image, checkpoint):
    """NVIDIA Canary-1B-v2 speech recognition/translation, from NeMo's own EncDecMultiTaskModel on the
    RELEASED `.nemo` archive (`--checkpoint`): the FastConformer encoder (dw-striding 8x subsampling, 32
    rel-pos conformer layers, WITH biases) and an attention encoder-decoder — a Transformer decoder with
    self-attention, cross-attention into the encoder frames, and a ReLU feed-forward, generating the
    transcription from a task prompt of control tokens. The clip is the validation speech WAV, so the
    transcription is a real measurement. Recorded seam by seam: the normalized mel features, the
    pre-encode output, the first conformer layer, the encoder output, the decoder's first-step logits at
    the transcription prompt, and the greedy tokens and text. Dither is zeroed. Runs under the `nemo`
    oracle env.
    """
    import wave as wavemodule
    import huggingface_hub
    for name in ["ModelFilter", "DatasetFilter"]:
        if not hasattr(huggingface_hub, name):
            setattr(huggingface_hub, name, type(name, (), {}))
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.EncDecMultiTaskModel.restore_from(checkpoint, strict=False).eval()
    model.preprocessor.featurizer.dither = 0.0

    path = os.environ.get("IK_VAL_AUDIO", os.path.expanduser("~/.inferkit-validation/inputs/speech.wav"))
    with wavemodule.open(path) as handle:
        assert handle.getframerate() == 16000 and handle.getnchannels() == 1 and handle.getsampwidth() == 2
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    wave = (pcm.astype(np.float32) / 32768.0)
    signal = torch.from_numpy(wave)[None]
    length = torch.tensor([wave.shape[0]])

    seams = {}
    model.encoder.pre_encode.register_forward_hook(lambda m, i, o: seams.__setitem__("pre", o[0].detach()))
    model.encoder.layers[0].register_forward_hook(lambda m, i, o: seams.__setitem__("layer0", o.detach()))

    # The transcription prompt (English ASR, punctuation on) is the release chat template's ASR turn:
    # <|startofcontext|><|startoftranscript|><|emo:undefined|><|en|><|en|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>
    prompt = [7, 4, 16, 64, 64, 5, 9, 11, 13]
    eos = 3
    with torch.no_grad():
        features, feature_length = model.preprocessor(input_signal=signal, length=length)
        encoded, encoded_length = model.encoder(audio_signal=features, length=feature_length)  # [1, D, T]
        enc_states = encoded.transpose(1, 2)                                                    # [1, T, D]
        enc_mask = torch.ones(enc_states.shape[:2], dtype=enc_states.dtype)
        ids = list(prompt)
        first_logits = None
        for _ in range(512):
            inp = torch.tensor([ids], dtype=torch.long)
            dec_mask = torch.ones_like(inp, dtype=enc_states.dtype)
            hidden = model.transf_decoder(input_ids=inp, decoder_mask=dec_mask,
                                          encoder_embeddings=enc_states, encoder_mask=enc_mask)
            step = model.log_softmax.mlp(hidden[:, -1])                                          # [1, V] raw logits
            if first_logits is None:
                first_logits = step[0].detach()
            nxt = int(step[0].argmax().item())
            if nxt == eos:
                break
            ids.append(nxt)
    generated = ids[len(prompt):]
    text = model.tokenizer.ids_to_text(generated)
    print(f"canary transcription: {text!r} ({len(generated)} tokens)")

    extra = {
        "waveform": torch.from_numpy(wave).contiguous(),
        "features": features[0].transpose(0, 1).contiguous(),               # [frames, mels] normalized
        "pre": seams["pre"][0].contiguous(),                                 # [T', d_model]
        "layer0": seams["layer0"][0].contiguous(),                           # [T', d_model]
        "encoded": encoded[0].transpose(0, 1).contiguous(),                  # [T', d_model]
        "prompt": torch.tensor(prompt, dtype=torch.int32),
        "logits0": first_logits.contiguous(),                               # [V] at the prompt's last position
        "tokens": torch.tensor(generated, dtype=torch.int32),
        "text": torch.tensor(list(text.encode("utf-8")), dtype=torch.int32),
    }
    for key, value in model.state_dict().items():
        if key.endswith("num_batches_tracked"):
            continue
        # `log_softmax.mlp.layer0.weight` is tied to the token embedding (shared storage), which
        # safetensors refuses to save; clone so both land as independent tensors.
        cloned = value.float().contiguous().clone() if value.is_floating_point() else value.contiguous().clone()
        extra[f"w::{key}"] = cloned
    globals()["_extra"] = extra
    return encoded[0].transpose(0, 1).clone().contiguous()


def run_phi4mm(image, checkpoint):
    """Phi-4-multimodal (`Phi4MMForCausalLM`, Microsoft) on the released weights, float32, eager
    attention, in every input mode the release serves. `--checkpoint` is the release directory; the
    reference is the release's own remote code (`trust_remote_code`), written for transformers 4.46.1,
    so it runs in its own oracle environment (torch 2.6, transformers 4.46.1, peft 0.13.2, torchvision
    0.21, numpy<2). Inputs are the validation clip and photo.

    Recorded: the text mode's prefill logits and greedy continuation; the speech mode's Conformer
    encoder output, speech-head projection, first-token logits, and transcription; the vision mode's
    SigLIP penultimate features, HD-projected image embeddings, first-token logits, and caption; the same
    for a 900x500 picture that pads into a 2x3 crop grid; the vision-with-speech mode's vision-head audio
    projection, logits, and answer; the tokenizer's ids for a plain text segment; the processor's raw
    inputs (samples, decoded bytes) so the Swift preprocessors are measured on identical values; and the
    feature extractor at 44.1, 48, 8, and 11.025 kHz, one clip per branch of its sample-rate handling.
    `image` unused.
    """
    import math
    import soundfile
    from PIL import Image
    from scipy.signal import resample_poly
    from transformers import AutoModelForCausalLM, AutoProcessor

    root = os.path.expanduser("~/.inferkit-validation/inputs")
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, trust_remote_code=True, torch_dtype=torch.float32,
        attn_implementation="eager", _attn_implementation="eager").eval()
    embed = model.model.embed_tokens_extend
    speech, rate = soundfile.read(os.path.join(root, "speech.wav"))
    photo = Image.open(os.path.join(root, "photo.jpg")).convert("RGB")
    padded = photo.resize((900, 500), Image.BICUBIC)
    extra = {}

    def utf8(text):
        return torch.tensor(list(text.encode("utf-8")), dtype=torch.int32)

    def run(prefix, prompt, new_tokens, hooks, images=None, audios=None):
        """One mode: prefill logits and greedy continuation, with `hooks` capturing seams."""
        captured, handles = {}, []
        for name, module, pick in hooks:
            handles.append(module.register_forward_hook(
                lambda m, i, o, name=name, pick=pick: captured.__setitem__(name, pick(o).detach())))
        inputs = processor(text=prompt, images=images, audios=audios, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits[0]
            generated = model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False)
        for handle in handles:
            handle.remove()
        continuation = generated[0, inputs["input_ids"].shape[1]:]
        extra[f"{prefix}_tokens"] = inputs["input_ids"][0].to(torch.int32).contiguous()
        extra[f"{prefix}_logits_last"] = logits[-1].float().contiguous()
        extra[f"{prefix}_continuation"] = continuation.to(torch.int32).contiguous()
        extra[f"{prefix}_text"] = utf8(tokenizer.decode(continuation, skip_special_tokens=True))
        return inputs, captured, logits

    # Text: no adapter.
    ids = tokenizer("<|user|>What is the capital of France?<|end|><|assistant|>", return_tensors="pt").input_ids
    with torch.no_grad():
        text_logits = model(input_ids=ids, input_mode=torch.tensor([0]), use_cache=False).logits[0]
        generated = model.generate(input_ids=ids, input_mode=torch.tensor([0]), max_new_tokens=12, do_sample=False)
    extra["text_tokens"] = ids[0].to(torch.int32).contiguous()
    extra["text_logits"] = text_logits.float().contiguous()
    extra["text_continuation"] = generated[0, ids.shape[1]:].to(torch.int32).contiguous()

    # Speech: the speech adapter and the speech projector head.
    first = lambda o: (o[0] if isinstance(o, tuple) else o)
    inputs, seams, _ = run("speech", "<|user|><|audio_1|>Transcribe the audio clip into text.<|end|><|assistant|>", 24,
                           [("encoder", embed.audio_embed.encoder, first),
                            ("proj", embed.audio_embed.audio_projection["speech"], lambda o: o)],
                           audios=[(speech, rate)])
    extra["speech_input_audio"] = inputs["input_audio_embeds"][0].float().contiguous()
    extra["speech_audio_embed_sizes"] = inputs["audio_embed_sizes"].to(torch.int32).contiguous()
    extra["speech_audio_encoder"] = seams["encoder"][0].float().contiguous()
    extra["speech_audio_proj"] = seams["proj"][0].float().contiguous()
    extra["speech_waveform"] = torch.tensor(np.asarray(speech, dtype=np.float32)).contiguous()

    # Vision, for the photo and for a padded multi-crop picture: the vision adapter.
    penultimate = lambda o: o.hidden_states[-2]
    for prefix, picture, new_tokens in [("vision", photo, 24), ("pad", padded, 16)]:
        inputs, seams, _ = run(prefix, "<|user|><|image_1|>Describe the image in one sentence.<|end|><|assistant|>",
                               new_tokens, [("siglip", embed.image_embed.img_processor, penultimate),
                                            ("proj", embed.image_embed.img_projection, lambda o: o)],
                               images=[picture])
        extra[f"{prefix}_input_image"] = inputs["input_image_embeds"][0].float().contiguous()
        extra[f"{prefix}_image_sizes"] = inputs["image_sizes"].to(torch.int32).contiguous()
        extra[f"{prefix}_image_attn_mask"] = inputs["image_attention_mask"][0].to(torch.int32).contiguous()
        extra[f"{prefix}_siglip_m2"] = seams["siglip"].float().reshape(-1, seams["siglip"].shape[-1]).contiguous()
        extra[f"{prefix}_img_proj"] = seams["proj"].float().reshape(-1, seams["proj"].shape[-1]).contiguous()
        extra[f"{prefix}_rgb"] = torch.tensor(np.asarray(picture, dtype=np.uint8).astype(np.int32)).contiguous()

    # Vision with speech: the vision adapter, with the audio through the projector's vision head.
    _, seams, _ = run("vs", "<|user|><|image_1|><|audio_1|><|end|><|assistant|>", 24,
                      [("proj", embed.audio_embed.audio_projection["vision"], lambda o: o)],
                      images=[photo], audios=[(speech, rate)])
    extra["vs_audio_proj"] = seams["proj"][0].float().contiguous()

    extra["tok_text_ids"] = torch.tensor(
        tokenizer("Describe the image in one sentence.", add_special_tokens=False).input_ids, dtype=torch.int32)

    # The feature extractor at one rate per branch of its rate handling, on 16-bit samples.
    extractor = processor.audio_processor
    for clip_rate in [44100, 48000, 8000, 11025]:
        g = math.gcd(clip_rate, 16000)
        clip = resample_poly(speech, clip_rate // g, 16000 // g)
        clip = np.round(np.clip(clip, -1, 32767 / 32768) * 32768) / 32768
        features = extractor._extract_features(clip, clip_rate)
        extra[f"rate{clip_rate}_waveform"] = torch.tensor(clip.astype(np.float32))
        extra[f"rate{clip_rate}_features"] = torch.tensor(features.astype(np.float32))
        extra[f"rate{clip_rate}_tokens"] = torch.tensor([extractor._compute_audio_embed_size(len(features))],
                                                         dtype=torch.int32)

    globals()["_extra"] = extra
    # safetensors refuses two names over one buffer, and `text_logits` above is this same tensor.
    return text_logits.float().contiguous().clone()


def run_phi4mm_bf16(image, checkpoint):
    """Phi-4-multimodal (the release's remote code, eager, in the `phi4mm` oracle environment) at bf16,
    for the rounding-placement check. `IK_PHI4MM_DTYPE=bfloat16` loads the model at bf16;
    `bfloat16-inputs` keeps float32 arithmetic on the same inputs rounded to bf16, the floor for that
    record. Recorded, prefill only: the text mode's logits with decoder layers 0, 1, and 31 probed piece
    by piece; the speech mode's Conformer output, speech-head projection, and logits, the Conformer
    layers probed; the vision mode's SigLIP penultimate features, projection, and logits, the SigLIP
    layers probed. `image` unused.
    """
    import soundfile
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    mode = os.environ.get("IK_PHI4MM_DTYPE", "bfloat16")
    dtype = torch.bfloat16 if mode == "bfloat16" else torch.float32
    root = os.path.expanduser("~/.inferkit-validation/inputs")
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="eager", _attn_implementation="eager").eval()
    embed = model.model.embed_tokens_extend
    speech, rate = soundfile.read(os.path.join(root, "speech.wav"))
    photo = Image.open(os.path.join(root, "photo.jpg")).convert("RGB")
    extra = {}

    def rounded(inputs):
        for key in ("input_image_embeds", "input_audio_embeds"):
            if key in inputs and inputs[key] is not None and inputs[key].is_floating_point():
                inputs[key] = inputs[key].to(torch.bfloat16).to(dtype)
        return inputs

    decoder_probes = {}
    restore = _probe_hooks(model, [0, 1, 31], decoder_probes)
    ids = tokenizer("<|user|>What is the capital of France?<|end|><|assistant|>", return_tensors="pt").input_ids
    with torch.no_grad():
        extra["text_logits"] = model(input_ids=ids, input_mode=torch.tensor([0]), use_cache=False).logits[0].float()
    restore()
    extra.update({f"dec.{key}": value for key, value in decoder_probes.items()})
    extra["text_tokens"] = ids[0].to(torch.int32).contiguous()

    def run(prefix, prompt, layers, hooks, **media):
        probes, captured = {}, {}
        _probe_encoder_layers(layers, probes)
        for name, module, pick in hooks:
            def capture(m, i, o, name=name, pick=pick):
                captured.setdefault(name, pick(o).detach().float())   # a hook's return replaces the output
            module.register_forward_hook(capture)
        inputs = rounded(processor(text=prompt, return_tensors="pt", **media))
        with torch.no_grad():
            extra[f"{prefix}_logits"] = model(**inputs).logits[0].float()
        extra[f"{prefix}_tokens"] = inputs["input_ids"][0].to(torch.int32).contiguous()
        extra.update({f"{prefix}.{key}": value for key, value in probes.items()})
        return inputs, captured

    first = lambda o: (o[0] if isinstance(o, tuple) else o)
    inputs, seams = run("speech", "<|user|><|audio_1|>Transcribe the audio clip into text.<|end|><|assistant|>",
                        embed.audio_embed.encoder.encoders,
                        [("encoder", embed.audio_embed.encoder, first),
                         ("proj", embed.audio_embed.audio_projection["speech"], lambda o: o)],
                        audios=[(speech, rate)])
    extra["speech_input_audio"] = inputs["input_audio_embeds"][0].float().contiguous()
    extra["speech_audio_encoder"] = seams["encoder"][0].contiguous()
    extra["speech_audio_proj"] = seams["proj"][0].contiguous()

    inputs, seams = run("vision", "<|user|><|image_1|>Describe the image in one sentence.<|end|><|assistant|>",
                        embed.image_embed.img_processor.encoder.layers,
                        [("siglip", embed.image_embed.img_processor, lambda o: o.hidden_states[-2]),
                         ("proj", embed.image_embed.img_projection, lambda o: o)],
                        images=[photo])
    extra["vision_input_image"] = inputs["input_image_embeds"][0].float().contiguous()
    extra["vision_siglip_m2"] = seams["siglip"].reshape(-1, seams["siglip"].shape[-1]).contiguous()
    extra["vision_img_proj"] = seams["proj"].reshape(-1, seams["proj"].shape[-1]).contiguous()

    globals()["_extra"] = {key: value.contiguous() for key, value in extra.items()}
    return extra["text_logits"].contiguous().clone()


def run_phi4mm_conversation(image, checkpoint):
    """Phi-4-multimodal (the release's remote code, float32, eager, in the `phi4mm` oracle environment)
    on the request shapes beyond one turn with one picture and one clip. `--checkpoint` is the release
    directory.

    Recorded, per case, the processor's prompt ids (`<case>_tokens`), the first answer token's logits
    (`<case>_logits_last`), and the greedy continuation (`<case>_continuation`, `<case>_text`):
    - `chat`: the release's chat template over a system turn, a finished exchange, and a second user
      turn, with two pictures and two clips. A turn that opens with whitespace shows the template's
      markers stripping it (`rstrip`). With a picture present, the clips go through the projector's
      vision head (`chat_audio_proj`, `[clips, frames, width]`, padded to the longest).
    - `clips`: two clips in speech mode, one past the encoder's 500-frame window (45 s) and one short,
      so the encoder runs batched under its padding mask and unfolds into windows at once, the short
      clip's second window wholly padding (`clips_audio_encoder`, `clips_audio_embed_sizes`).
    - `long`: the 45 s clip alone, which unfolds without a mask (`long_audio_encoder`).
    The raw inputs ride along: `short_waveform` (the validation clip), `long_waveform` (it tiled to
    45 s), and the pictures (`photo_rgb`, `wide_rgb`). `image` unused.
    """
    import soundfile
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    root = os.path.expanduser("~/.inferkit-validation/inputs")
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, trust_remote_code=True, torch_dtype=torch.float32,
        attn_implementation="eager", _attn_implementation="eager").eval()
    audio_embed = model.model.embed_tokens_extend.audio_embed
    speech, rate = soundfile.read(os.path.join(root, "speech.wav"))
    speech = np.asarray(speech, dtype=np.float32)
    long_speech = np.tile(speech, 13)                         # 45.1 s, 564 encoder frames
    photo = Image.open(os.path.join(root, "photo.jpg")).convert("RGB")
    wide = photo.resize((900, 500), Image.BICUBIC)
    extra = {}

    def utf8(text):
        return torch.tensor(list(text.encode("utf-8")), dtype=torch.int32)

    def run(prefix, prompt, new_tokens, hooks, images=None, audios=None):
        captured, handles = {}, []

        def capture(name, pick):
            # A forward hook that returns a value replaces the module's output, so this one returns None.
            def hook(module, inputs, output):
                if name not in captured:
                    captured[name] = pick(output).detach()
            return hook

        for name, module, pick in hooks:
            handles.append(module.register_forward_hook(capture(name, pick)))
        inputs = processor(text=prompt, images=images, audios=audios, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits[0]
            generated = model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False)
        for handle in handles:
            handle.remove()
        continuation = generated[0, inputs["input_ids"].shape[1]:]
        extra[f"{prefix}_tokens"] = inputs["input_ids"][0].to(torch.int32).contiguous()
        extra[f"{prefix}_logits_last"] = logits[-1].float().contiguous()
        extra[f"{prefix}_continuation"] = continuation.to(torch.int32).contiguous()
        extra[f"{prefix}_text"] = utf8(tokenizer.decode(continuation, skip_special_tokens=True))
        extra[f"{prefix}_audio_embed_sizes"] = inputs["audio_embed_sizes"].to(torch.int32).contiguous() \
            if audios else torch.zeros(0, dtype=torch.int32)
        print(f"phi4mm {prefix}: {inputs['input_ids'].shape[1]} ids -> {tokenizer.decode(continuation)!r}")
        return captured

    first = lambda o: (o[0] if isinstance(o, tuple) else o)
    messages = [
        {"role": "system", "content": "You answer in one short sentence."},
        {"role": "user", "content": "<|image_1|><|image_2|>How many pictures are there?"},
        {"role": "assistant", "content": "There are two pictures."},
        {"role": "user", "content": "\n <|audio_1|><|audio_2|>What does the second clip say?"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    extra["chat_prompt"] = utf8(prompt)
    seams = run("chat", prompt, 24, [("proj", audio_embed.audio_projection["vision"], lambda o: o)],
                images=[photo, wide], audios=[(long_speech, rate), (speech, rate)])
    extra["chat_audio_proj"] = seams["proj"].float().contiguous()

    seams = run("clips", "<|user|><|audio_1|><|audio_2|>Transcribe the audio clip into text.<|end|><|assistant|>", 16,
                [("encoder", audio_embed.encoder, first)], audios=[(long_speech, rate), (speech, rate)])
    extra["clips_audio_encoder"] = seams["encoder"].float().contiguous()

    seams = run("long", "<|user|><|audio_1|>Transcribe the audio clip into text.<|end|><|assistant|>", 16,
                [("encoder", audio_embed.encoder, first)], audios=[(long_speech, rate)])
    extra["long_audio_encoder"] = seams["encoder"][0].float().contiguous()

    extra["short_waveform"] = torch.tensor(speech).contiguous()
    extra["long_waveform"] = torch.tensor(long_speech).contiguous()
    extra["photo_rgb"] = torch.tensor(np.asarray(photo, dtype=np.uint8).astype(np.int32)).contiguous()
    extra["wide_rgb"] = torch.tensor(np.asarray(wide, dtype=np.uint8).astype(np.int32)).contiguous()
    globals()["_extra"] = extra
    return extra["chat_logits_last"].clone()


def _chatterbox_reference_audio():
    """The validation clip prepared the way `ChatterboxTTS.prepare_conditionals` prepares a voice
    prompt: `librosa.load(sr=24000)` (a 16 kHz file resampled up), then `librosa.resample` back down to
    16 kHz. Returns (wav24, wav16) as float32 numpy arrays."""
    import librosa
    path = os.environ.get("IK_VAL_AUDIO", os.path.expanduser("~/.inferkit-validation/inputs/speech.wav"))
    wav24, _ = librosa.load(path, sr=24000)
    wav16 = librosa.resample(wav24, orig_sr=24000, target_sr=16000)
    return wav24.astype(np.float32), wav16.astype(np.float32)


def run_chatterbox_voice(image, checkpoint):
    """Chatterbox stage 1 + 2 (ResembleAI, `chatterbox-tts`): the VoiceEncoder speaker embedding and
    the S3 speech tokenizer, on the RELEASED weights (`--checkpoint` is the release directory holding
    `ve.safetensors` and `s3gen.safetensors`) and the validation clip, prepared exactly as
    `prepare_conditionals` does. Recorded: `wav16` (the 16 kHz prompt both networks read); the voice
    encoder's librosa-trimmed input (`ve_wav`), unscaled 40-band power mel (`ve_mel`, [T, 40]), the
    per-partial embeddings (`ve_partials`, [N, 256]) and the utterance embedding (`ve_embed`); the S3
    tokenizer's log-mel on the six-second crop (`s3_mel`, [128, T]), the encoder output (`s3_hidden`,
    [T', 1280]) and the FSQ codes truncated to the 150-token conditioning prompt (`s3_codes`), and the
    codes of the whole 16 kHz prompt as `embed_ref` tokenizes it (`s3_ref_codes`, from torchaudio's
    24 kHz to 16 kHz resample, recorded as `ref_wav16`). Weights as `w::ve.*` and `w::tokenizer.*`.
    Runs under the `chatterbox` oracle env.
    """
    import librosa
    import torchaudio
    from safetensors.torch import load_file
    from chatterbox.models.voice_encoder import VoiceEncoder
    from chatterbox.models.voice_encoder.melspec import melspectrogram
    from chatterbox.models.s3tokenizer import S3Tokenizer, S3_SR
    from chatterbox.models.s3gen.const import S3GEN_SR

    directory = checkpoint
    ve = VoiceEncoder()
    ve.load_state_dict(load_file(os.path.join(directory, "ve.safetensors")))
    ve.eval()
    s3gen_state = load_file(os.path.join(directory, "s3gen.safetensors"))
    tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
    tokenizer_state = {k[len("tokenizer."):]: v for k, v in s3gen_state.items() if k.startswith("tokenizer.")}
    missing, unexpected = tokenizer.load_state_dict(tokenizer_state, strict=False)
    assert not unexpected and set(missing) <= {"_mel_filters", "window"}, (missing, unexpected)
    tokenizer.eval()

    wav24, wav16 = _chatterbox_reference_audio()

    # Voice encoder, as embeds_from_wavs: trim at 20 dB, unscaled power mel, partials at rate 1.3.
    ve_wav = librosa.effects.trim(wav16, top_db=20)[0]
    ve_mel = melspectrogram(ve_wav, ve.hp).T                                  # [T, 40]
    from chatterbox.models.voice_encoder.voice_encoder import get_frame_step, get_num_wins
    frame_step = get_frame_step(0.5, 1.3, ve.hp)
    n_partials, target_len = get_num_wins(len(ve_mel), frame_step, 0.8, ve.hp)
    mel_t = torch.from_numpy(np.ascontiguousarray(ve_mel)).float()
    if target_len > mel_t.shape[0]:
        mel_t = torch.cat([mel_t, torch.zeros(target_len - mel_t.shape[0], ve.hp.num_mels)], dim=0)
    partials = torch.stack([mel_t[i * frame_step: i * frame_step + ve.hp.ve_partial_frames] for i in range(n_partials)])
    with torch.inference_mode():
        partial_embeds = ve(partials)                                          # [N, 256]
        ve_embed = torch.from_numpy(ve.embeds_from_wavs([wav16], sample_rate=S3_SR))  # [1, 256]
    assert ve_embed.shape == (1, 256)
    print(f"chatterbox voice encoder: trimmed {len(ve_wav)} of {len(wav16)} samples, {len(ve_mel)} mel frames, "
          f"{n_partials} partials at step {frame_step}")

    # S3 tokenizer on the six-second crop (the T3 conditioning prompt), seam by seam.
    crop = torch.from_numpy(wav16[: 6 * S3_SR])[None]
    with torch.inference_mode():
        s3_mel = tokenizer.log_mel_spectrogram(crop)                           # [1, 128, T]
        s3_mel = s3_mel[..., :150 * 4]
        mel_len = torch.tensor([s3_mel.shape[-1]])
        hidden, code_len = tokenizer.encoder(s3_mel, mel_len)                  # [1, T', 1280]
        codes = tokenizer.quantizer.encode(hidden)                             # [1, T']
        cond_codes, cond_len = tokenizer.forward([wav16[: 6 * S3_SR]], max_len=150)
    assert torch.equal(codes[0, :cond_len[0]], cond_codes[0, :cond_len[0]])

    # The whole prompt as `S3Gen.embed_ref` tokenizes it: torchaudio's resample of the 24 kHz wav.
    ref_wav16 = torchaudio.transforms.Resample(S3GEN_SR, S3_SR)(torch.from_numpy(wav24)[None])
    with torch.inference_mode():
        ref_codes, ref_len = tokenizer(ref_wav16.float())
    print(f"chatterbox s3 tokenizer: {s3_mel.shape[-1]} mel frames -> {int(code_len[0])} codes; "
          f"ref {ref_wav16.shape[-1]} samples -> {int(ref_len[0])} codes; first codes {codes[0, :8].tolist()}")

    extra = {
        "wav24": torch.from_numpy(wav24).contiguous(),
        "wav16": torch.from_numpy(wav16).contiguous(),
        "ve_wav": torch.from_numpy(np.ascontiguousarray(ve_wav)).contiguous(),
        "ve_mel": torch.from_numpy(np.ascontiguousarray(ve_mel)).float().contiguous(),
        "ve_partials": partial_embeds.contiguous(),
        "ve_embed": ve_embed[0].contiguous(),
        "s3_mel": s3_mel[0].contiguous(),
        "s3_hidden": hidden[0].contiguous(),
        "s3_codes": codes[0, :cond_len[0]].to(torch.int32).contiguous(),
        "ref_wav16": ref_wav16[0].contiguous(),
        "s3_ref_codes": ref_codes[0, :ref_len[0]].to(torch.int32).contiguous(),
    }
    for key, value in ve.state_dict().items():
        extra[f"w::ve.{key}"] = value.float().contiguous()
    for key, value in tokenizer_state.items():
        extra[f"w::tokenizer.{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return ve_embed[0].clone().contiguous()


def run_chatterbox_t3(image, checkpoint):
    """Chatterbox stage 3: T3, the text-to-speech-token model (a Llama 520M with llama3 rope scaling
    over learned position embeddings, conditioned on the VoiceEncoder speaker embedding, a 32-token
    Perceiver resample of the prompt's speech codes, and an emotion scalar), on the RELEASED
    `t3_cfg.safetensors`. The conditioning is rebuilt from the validation clip exactly as
    `prepare_conditionals` builds it. Recorded: the text tokens with SOT/EOT (`text_tokens`), the
    speaker embedding and prompt codes, the conditioning sequence (`cond_emb`, [34, 1024]), the CFG pair
    of input embeddings (`embeds`, [2, L, 1024], row 1 with the text embedding zeroed), the reference's
    own sampled speech tokens under a fixed seed (`speech_tokens`), the teacher-forced speech logits over
    that sequence for both CFG rows (`tf_logits`, [2, n, 8194]), and the first step's fully processed
    logits (`step0_processed`: CFG 0.5, repetition penalty 1.2, temperature 0.8, min-p 0.05, top-p 1.0).
    The 2 GB of weights are not copied into the record; the Swift side loads the release itself. Runs
    under the `chatterbox` oracle env.
    """
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from transformers.generation.logits_process import (RepetitionPenaltyLogitsProcessor, MinPLogitsWarper,
                                                        TopPLogitsWarper)
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    from chatterbox.models.tokenizers import EnTokenizer
    from chatterbox.models.voice_encoder import VoiceEncoder
    from chatterbox.models.s3tokenizer import S3Tokenizer, S3_SR
    from chatterbox.tts import punc_norm

    directory = checkpoint
    t3 = T3()
    t3.load_state_dict(load_file(os.path.join(directory, "t3_cfg.safetensors")))
    t3.eval()
    ve = VoiceEncoder()
    ve.load_state_dict(load_file(os.path.join(directory, "ve.safetensors")))
    ve.eval()
    s3gen_state = load_file(os.path.join(directory, "s3gen.safetensors"))
    tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
    tokenizer.load_state_dict({k[len("tokenizer."):]: v for k, v in s3gen_state.items() if k.startswith("tokenizer.")},
                              strict=False)
    tokenizer.eval()
    del s3gen_state
    text_tokenizer = EnTokenizer(os.path.join(directory, "tokenizer.json"))

    wav24, wav16 = _chatterbox_reference_audio()
    with torch.inference_mode():
        ve_embed = torch.from_numpy(ve.embeds_from_wavs([wav16], sample_rate=S3_SR)).mean(axis=0, keepdim=True)
        cond_tokens, _ = tokenizer.forward([wav16[: 6 * S3_SR]], max_len=t3.hp.speech_cond_prompt_len)
    cond_tokens = torch.atleast_2d(cond_tokens)
    exaggeration = 0.5
    t3_cond = T3Cond(speaker_emb=ve_embed, cond_prompt_speech_tokens=cond_tokens,
                     emotion_adv=exaggeration * torch.ones(1, 1, 1))

    raw_text = "The quick brown fox jumps over the lazy dog."
    text = punc_norm(raw_text)
    text_tokens = text_tokenizer.text_to_tokens(text)
    text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
    text_tokens = F.pad(text_tokens, (1, 0), value=t3.hp.start_text_token)
    text_tokens = F.pad(text_tokens, (0, 1), value=t3.hp.stop_text_token).long()

    cfg_weight, temperature, repetition_penalty, min_p, top_p = 0.5, 0.8, 1.2, 0.05, 1.0
    sos = t3.hp.start_speech_token
    with torch.inference_mode():
        cond_emb = t3.prepare_conditioning(t3_cond)                                         # [1, 34, 1024]
        embeds, len_cond = t3.prepare_input_embeds(
            t3_cond=t3_cond, text_tokens=text_tokens,
            speech_tokens=sos * torch.ones_like(text_tokens[:, :1]), cfg_weight=cfg_weight)  # [2, L0, 1024]
        torch.manual_seed(0)
        speech_tokens = t3.inference(t3_cond=t3_cond, text_tokens=text_tokens, max_new_tokens=1000,
                                     temperature=temperature, cfg_weight=cfg_weight,
                                     repetition_penalty=repetition_penalty, min_p=min_p, top_p=top_p)
        gen = speech_tokens[0].long()                                                        # [n]
        n = gen.numel()
        bos_embed = t3.speech_emb(torch.tensor([[sos]])) + t3.speech_pos_emb.get_fixed_embedding(0)
        gen_embed = t3.speech_emb(gen[None, :-1]) + t3.speech_pos_emb.get_fixed_embedding(torch.arange(1, n))
        tf_embeds = torch.cat([embeds, torch.cat([bos_embed, bos_embed]), torch.cat([gen_embed, gen_embed])], dim=1)
        out = t3.tfmr(inputs_embeds=tf_embeds, output_hidden_states=True, return_dict=True)
        logits = t3.speech_head(out.hidden_states[-1])                                       # [2, L, 8194]
        start = len_cond + text_tokens.size(1) + 1                                           # the second BOS
        tf_logits = logits[:, start: start + n]                                               # predicts gen[0..n-1]
        step0 = tf_logits[:, 0]
        cfg = step0[0:1] + cfg_weight * (step0[0:1] - step0[1:2])
        ids = torch.tensor([[sos]])
        processed = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)(ids, cfg) / temperature
        processed = MinPLogitsWarper(min_p=min_p)(ids, processed)
        processed = TopPLogitsWarper(top_p=top_p)(ids, processed)
    assert bool(torch.isfinite(processed[0, gen[0]])), "the sampled first token is admissible under the processors"
    print(f"chatterbox t3: text {text!r} -> {text_tokens.size(1)} tokens; cond {cond_emb.shape[1]}; "
          f"{n} speech tokens sampled (last {int(gen[-1])}), embeds {tuple(embeds.shape)}")

    extra = {
        "text_tokens": text_tokens[0].to(torch.int32).contiguous(),
        "speaker_emb": ve_embed[0].contiguous(),
        "cond_tokens": cond_tokens[0].to(torch.int32).contiguous(),
        "emotion_adv": torch.tensor([exaggeration]),
        "cond_emb": cond_emb[0].contiguous(),
        "embeds": embeds.contiguous(),
        "speech_tokens": gen.to(torch.int32).contiguous(),
        "tf_logits": tf_logits.contiguous(),
        "step0_processed": processed[0].contiguous(),
        "text": torch.tensor(list(raw_text.encode("utf-8")), dtype=torch.int32),
    }
    globals()["_extra"] = extra
    return tf_logits[0, 0].clone().contiguous()


def run_chatterbox_s3gen(image, checkpoint):
    """Chatterbox stages 4 and 5: S3Gen, the speech-code-to-waveform decoder, on the RELEASED
    `s3gen.safetensors` and the validation clip as the voice prompt (`embed_ref` on its first ten
    seconds at 24 kHz). Stage 4 is the CAMPPlus x-vector over a Kaldi fbank, the 24 kHz prompt mel, the
    UpsampleConformerEncoder over the prompt-plus-target codes, and the causal conditional flow matching
    (ten Euler steps on a cosine schedule, classifier-free guidance 0.7) through the ConditionalDecoder
    U-Net; stage 5 is the HiFT vocoder (ConvRNNF0Predictor, harmonic sine source, iSTFT head). The
    target codes are the T3 record's sampled speech tokens when that record exists (the real pipeline's
    input), else the prompt's own codes. Two random sources are pinned so the seams are exact: the flow's
    initial noise is drawn once and RECORDED (`flow_noise`, the Swift side takes it as an input), and the
    sine source's random harmonic phases and additive noise are zeroed (the reference's `SineGen` draws
    them fresh per call; the port draws its own in the consumer path). Recorded seam by seam: `fbank`,
    `xvector_head`, `xvector`, `prompt_mel`, `token_emb`, `encoder_h`, `mu`, `spk80`, `t_span`,
    `estimator0` (the first Euler step's velocity, both guidance rows), `mel_full`, `mel`, `f0`,
    `source`, and `wav` (with the reference's 40 ms leading fade). Runs under the `chatterbox` oracle env.
    """
    import torch.nn.functional as F
    import torchaudio
    import torchaudio.compliance.kaldi as Kaldi
    from safetensors.torch import load_file, load_file as _lf
    from chatterbox.models.s3gen import S3Gen, S3GEN_SR
    from chatterbox.models.s3tokenizer import S3_SR
    import chatterbox.models.s3gen.hifigan as hifigan_module

    directory = checkpoint
    s3gen = S3Gen()
    s3gen.load_state_dict(load_file(os.path.join(directory, "s3gen.safetensors")), strict=False)
    s3gen.eval()

    wav24, wav16 = _chatterbox_reference_audio()
    ref24 = torch.from_numpy(wav24[: 10 * S3GEN_SR])[None]
    ref16 = torchaudio.transforms.Resample(S3GEN_SR, S3_SR)(ref24)

    seams = {}
    def record(name):
        def hook(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            seams.setdefault(name, []).append(value.detach().clone())
        return hook
    hooks = [
        s3gen.speaker_encoder.head.register_forward_hook(record("xvector_head")),
        s3gen.flow.input_embedding.register_forward_hook(record("token_emb")),
        s3gen.flow.encoder.register_forward_hook(record("encoder_h")),
        s3gen.flow.encoder_proj.register_forward_hook(record("mu")),
        s3gen.flow.spk_embed_affine_layer.register_forward_hook(record("spk80")),
        s3gen.flow.decoder.register_forward_hook(record("mel_full")),
        s3gen.mel2wav.f0_predictor.register_forward_hook(record("f0")),
        s3gen.mel2wav.m_source.register_forward_hook(record("source")),
    ]

    # `solve_euler` calls `estimator.forward` directly, which bypasses forward hooks; wrap the method.
    estimator = s3gen.flow.decoder.estimator
    original_estimator_forward = estimator.forward
    def recording_estimator_forward(*args, **kwargs):
        output = original_estimator_forward(*args, **kwargs)
        seams.setdefault("estimator", []).append(output.detach().clone())
        return output
    estimator.forward = recording_estimator_forward

    t3_record = os.path.expanduser("~/.inferkit-validation/records/chatterbox_t3.safetensors")
    with torch.inference_mode():
        fbank = Kaldi.fbank(ref16, num_mel_bins=80)
        fbank = fbank - fbank.mean(dim=0, keepdim=True)
        ref_dict = s3gen.embed_ref(ref24, S3GEN_SR)
        if os.path.exists(t3_record):
            tokens = _lf(t3_record)["speech_tokens"].long()
            tokens = tokens[tokens < 6561]
            source = "t3 record"
        else:
            tokens = ref_dict["prompt_token"][0].long()
            source = "prompt codes"
        n_prompt, n_tokens = ref_dict["prompt_token"].shape[1], tokens.numel()
        total_mel = 2 * (n_prompt + n_tokens)
        torch.manual_seed(0)
        noise = torch.randn(1, 80, total_mel)

    class ZeroUniform:
        def __init__(self, low, high): pass
        def sample(self, sample_shape):
            return torch.zeros(sample_shape)
    original_uniform, original_randn_like = hifigan_module.Uniform, torch.randn_like
    def pinned_randn_like(tensor, **kwargs):
        if tuple(tensor.shape) == tuple(noise.shape):
            return noise.to(tensor.dtype)
        return torch.zeros_like(tensor)
    hifigan_module.Uniform = ZeroUniform
    torch.randn_like = pinned_randn_like
    try:
        with torch.inference_mode():
            wav, _ = s3gen.inference(speech_tokens=tokens, ref_dict=ref_dict)
    finally:
        hifigan_module.Uniform = original_uniform
        torch.randn_like = original_randn_like
        for hook in hooks:
            hook.remove()
        estimator.forward = original_estimator_forward

    t_span = torch.linspace(0, 1, 11)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    mel_full = seams["mel_full"][0]                         # [1, 80, total]
    mel = mel_full[:, :, ref_dict["prompt_feat"].shape[1]:]
    print(f"chatterbox s3gen: prompt {n_prompt} codes / {ref_dict['prompt_feat'].shape[1]} mel frames, "
          f"{n_tokens} target codes ({source}) -> mel {tuple(mel.shape)}, wav {tuple(wav.shape)}; "
          f"{len(seams['estimator'])} estimator calls")
    assert len(seams["estimator"]) == 10

    extra = {
        "ref_wav24": ref24[0].contiguous(),
        "ref_wav16": ref16[0].contiguous(),
        "fbank": fbank.contiguous(),                                          # [T, 80]
        "xvector_head": seams["xvector_head"][0][0].contiguous(),             # [C·F/8, T']
        "xvector": ref_dict["embedding"][0].contiguous(),                     # [192]
        "prompt_mel": ref_dict["prompt_feat"][0].contiguous(),                # [T2, 80]
        "prompt_tokens": ref_dict["prompt_token"][0].to(torch.int32).contiguous(),
        "speech_tokens": tokens.to(torch.int32).contiguous(),
        "token_emb": seams["token_emb"][0][0].contiguous(),                   # [T_all, 512]
        "encoder_h": seams["encoder_h"][0][0].contiguous(),                   # [2·T_all, 512]
        "mu": seams["mu"][0][0].contiguous(),                                 # [2·T_all, 80]
        "spk80": seams["spk80"][0][0].contiguous(),                           # [80]
        "flow_noise": noise[0].contiguous(),                                  # [80, total]
        "t_span": t_span.contiguous(),
        "estimator0": seams["estimator"][0].contiguous(),                     # [2, 80, total]
        "mel_full": mel_full[0].contiguous(),                                 # [80, total]
        "mel": mel[0].contiguous(),                                           # [80, 2·n_tokens]
        "f0": seams["f0"][0][0].contiguous(),                                 # [2·n_tokens]
        "source": seams["source"][0][0, :, 0].contiguous(),                   # [samples]
        "wav": wav[0].contiguous(),                                           # [samples]
    }
    globals()["_extra"] = extra
    return wav[0].clone().contiguous()


def run_dcn(image):
    """DeformableConv2d core (modulated deform-conv v2) vs torchvision.ops.deform_conv2d — the one novel
    op BiRefNet's ASPPDeformable needs, de-risked in isolation before the tower is built. For each kernel
    in {1, 3, 7} at stride 1, dilation 1, padding k//2: random x, offset, mask (in [0, 4] via
    2*sigmoid), weight, and bias are fed to torchvision's kernel, and the port reproduces the same
    bilinear gather at the offset locations + tap-weighted sum. Every tensor is recorded NHWC except the
    weight, which is kept in PyTorch [OC, C, kH, kW] layout (the port transposes it on load, as every
    other conv here does). Needs torchvision; the plate is ignored.
    """
    from torchvision.ops import deform_conv2d
    g = torch.Generator().manual_seed(0)
    C, OC, H, W = 6, 5, 12, 10
    x = torch.randn(1, C, H, W, generator=g)
    extra = {"dcn_x": x[0].permute(1, 2, 0).contiguous()}                         # [H, W, C]
    primary = None
    for k in (1, 3, 7):
        pad = k // 2
        offset = torch.randn(1, 2 * k * k, H, W, generator=g) * 0.7               # exercise fractional + OOB
        mask = 2.0 * torch.sigmoid(torch.randn(1, k * k, H, W, generator=g))      # the modulator, in [0, 4]
        weight = torch.randn(OC, C, k, k, generator=g)
        bias = torch.randn(OC, generator=g)
        out = deform_conv2d(x, offset, weight, bias, stride=(1, 1), padding=(pad, pad), mask=mask)
        extra[f"dcn{k}_offset"] = offset[0].permute(1, 2, 0).contiguous()         # [H, W, 2·k·k], (Δy,Δx) per tap
        extra[f"dcn{k}_mask"] = mask[0].permute(1, 2, 0).contiguous()             # [H, W, k·k]
        extra[f"dcn{k}_weight"] = weight.contiguous()                             # [OC, C, kH, kW] PyTorch layout
        extra[f"dcn{k}_bias"] = bias.contiguous()                                 # [OC]
        extra[f"dcn{k}_out"] = out[0].permute(1, 2, 0).contiguous()              # [H, W, OC]
        if k == 7:
            primary = out[0].permute(1, 2, 0).contiguous()
    globals()["_extra"] = extra
    return primary


def run_birefnet_backbone(image, checkpoint):
    """BiRefNet's Swin-v1-L backbone (`swin_v1_l`) on the RELEASED `bb.*` weights, against the model's
    own reference (the vendored `swin_v1.py` under `IK_REF_SRC/birefnet`, with a stub `config` that
    disables SDPA so the explicit-softmax path the MLX port reproduces is what runs). `checkpoint` is
    the release `model.safetensors`. Records the ImageNet-normalized pixels both sides read and the four
    stage feature maps (NHWC). The released weights are fp16; both sides upcast to float32 so the
    comparison measures the port rather than half precision. Feed `--size 256 --plate subject`.
    """
    import sys
    from safetensors.torch import load_file
    ref = os.environ.get("IK_REF_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "refsrc"))
    sys.path.insert(0, os.path.join(ref, "birefnet"))
    from swin_v1 import swin_v1_l

    state = load_file(checkpoint)
    backbone_state = {key[len("bb."):]: value.float() for key, value in state.items() if key.startswith("bb.")}
    model = swin_v1_l().float()
    model.eval()   # SwinTransformer overrides train() and returns None, so eval() cannot be chained
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    print(f"birefnet backbone: loaded {len(backbone_state)} bb tensors, missing {len(missing)}, unexpected {len(unexpected)}")

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    pixels = (torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() - mean) / std    # [1, 3, H, W]
    with torch.no_grad():
        feats = model(pixels)

    extra = {"pixels": pixels[0].permute(1, 2, 0).contiguous()}                              # [H, W, 3] NHWC
    for level, feature in enumerate(feats):
        extra[f"bb.{level}"] = feature[0].permute(1, 2, 0).contiguous()                       # [H_i, W_i, C_i] NHWC
    globals()["_extra"] = extra
    return feats[3][0].permute(1, 2, 0).contiguous()


def run_birefnet_neck(image, checkpoint):
    """BiRefNet's neck on the RELEASED weights, from the full model's own forward_enc + squeeze_module.
    mul_scl_ipt='cat' runs the backbone at full and half resolution and concatenates the features
    (doubling channels), cxt concatenates x1/x2/x3 interpolated to x4 with x4 (-> 5760 ch), and the
    squeeze BasicDecBlk reduces that to 3072 — and that BasicDecBlk contains an ASPPDeformable, so this
    exercises the deformable conv on real weights. Records the normalized pixels, the doubled stage
    features x1/x2/x3 (for the decoder step), the cxt-concatenated x4 (pre-squeeze), and the squeezed x4.
    The reference is the vendored HF birefnet.py (`IK_REF_SRC/birefnet_hf`); fp16 release, float32 both
    sides. `checkpoint` is model.safetensors. Feed `--size 256 --plate subject`.
    """
    import sys
    from safetensors.torch import load_file
    ref = os.environ.get("IK_REF_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "refsrc"))
    sys.path.insert(0, ref)
    from birefnet_hf.birefnet import BiRefNet

    model = BiRefNet(bb_pretrained=False)
    model.float()
    model.eval()   # BiRefNet/SwinTransformer override train() and return None, so eval() is a statement
    model.load_state_dict({key: value.float() for key, value in load_file(checkpoint).items()}, strict=True)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    pixels = (torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() - mean) / std
    with torch.no_grad():
        (x1, x2, x3, x4), _ = model.forward_enc(pixels)
        squeezed = model.squeeze_module(x4)

    globals()["_extra"] = {
        "pixels": pixels[0].permute(1, 2, 0).contiguous(),
        "x1": x1[0].permute(1, 2, 0).contiguous(),                     # doubled stage features (NHWC)
        "x2": x2[0].permute(1, 2, 0).contiguous(),
        "x3": x3[0].permute(1, 2, 0).contiguous(),
        "x4_cxt": x4[0].permute(1, 2, 0).contiguous(),                 # cxt-concatenated x4, pre-squeeze (5760)
        "x4_squeezed": squeezed[0].permute(1, 2, 0).contiguous(),      # squeezed x4 (3072)
    }
    return squeezed[0].permute(1, 2, 0).contiguous()


def run_birefnet_decode(image, checkpoint):
    """BiRefNet's decoder on the RELEASED weights, from the full model's own Decoder.forward at eval.
    The four decoder stages (BasicDecBlk with ASPPDeformable) with BasicLatBlk lateral skips, the
    dec_ipt multi-scale image-patch injection (image2patches -> SimpleConvs), the gdt attention gating
    (gdt_convs -> gdt_convs_attn -> sigmoid -> p*attn, which runs at EVAL — only gdt pred/label and the
    ms-supervision heads are training-only), the bilinear upsampling, and conv_out1 -> the 1-channel
    logit. Records the normalized pixels and the four encoder inputs (x1/x2/x3/x4_squeezed) so the port
    runs on identical inputs, the four pre-gate decoder-block outputs (for localization), and the final
    logit. fp16 release, float32 both sides. `checkpoint` is model.safetensors. Feed `--size 256`.
    """
    import sys
    from safetensors.torch import load_file
    ref = os.environ.get("IK_REF_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "refsrc"))
    sys.path.insert(0, ref)
    from birefnet_hf.birefnet import BiRefNet

    model = BiRefNet(bb_pretrained=False)
    model.float()
    model.eval()
    model.load_state_dict({key: value.float() for key, value in load_file(checkpoint).items()}, strict=True)

    seams = {}
    for name in ["decoder_block4", "decoder_block3", "decoder_block2", "decoder_block1"]:
        getattr(model.decoder, name).register_forward_hook(
            lambda module, inputs, output, key=name: seams.__setitem__(key, output.detach()))

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    pixels = (torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() - mean) / std
    with torch.no_grad():
        (x1, x2, x3, x4), _ = model.forward_enc(pixels)
        squeezed = model.squeeze_module(x4)
        outs = model.decoder([pixels, x1, x2, x3, squeezed])
    logit = outs[-1]

    globals()["_extra"] = {
        "pixels": pixels[0].permute(1, 2, 0).contiguous(),
        "x1": x1[0].permute(1, 2, 0).contiguous(),
        "x2": x2[0].permute(1, 2, 0).contiguous(),
        "x3": x3[0].permute(1, 2, 0).contiguous(),
        "x4_squeezed": squeezed[0].permute(1, 2, 0).contiguous(),
        "p4": seams["decoder_block4"][0].permute(1, 2, 0).contiguous(),
        "p3": seams["decoder_block3"][0].permute(1, 2, 0).contiguous(),
        "p2": seams["decoder_block2"][0].permute(1, 2, 0).contiguous(),
        "p1": seams["decoder_block1"][0].permute(1, 2, 0).contiguous(),
    }
    return logit[0].permute(1, 2, 0).contiguous()                 # [H, W, 1] full-resolution logit


def run_mpsenet(image, checkpoint):
    """MP-SENet (yxlu-0102/MP-SENet) speech enhancement of a deterministic noisy clip, seam by seam.

    Prefers the `MPSENet` pip package (JacobLinCool, needs Python >= 3.10), which wraps the released
    generator and its config; falls back to the cloned repository at IK_MPSENET_SRC (its `models/model.py`
    holds `MPNet`), which runs under Python 3.9. The config JSON is `IK_MPSENET_CONFIG` (default the repo's
    `config.json`). `--checkpoint` is the released `g_best` .pth or a `from_pretrained` id.

    Records the compressed magnitude and phase the network reads, the dense-encoder output, every
    TS-conformer block output, the denoised magnitude and phase, and the reconstructed waveform, so the
    Swift parity test locates the first divergence rather than guessing.
    """
    import json
    import os
    import types

    torch.manual_seed(0)
    try:
        from MPSENet import MPSENet as _MPSENet
        wrapper = _MPSENet.from_pretrained(checkpoint)
        model, h = wrapper.model.eval(), wrapper.h
    except Exception:
        source = os.environ.get("IK_MPSENET_SRC", ".")
        sys.path.insert(0, source)
        from models.model import MPNet                            # the generator lives in models/model.py
        config = os.environ.get("IK_MPSENET_CONFIG", os.path.join(source, "config.json"))
        h = types.SimpleNamespace(**json.load(open(config)))
        model = MPNet(h).eval()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("generator", state), strict=True)

    n_fft, hop, win, cf = h.n_fft, h.hop_size, h.win_size, h.compress_factor
    samples = 16000
    t = np.arange(samples, dtype=np.float32) / 16000.0
    gen = np.random.default_rng(9)
    speech = sum(0.3 / (k + 1) * np.sin(2 * np.pi * 140 * (k + 1) * t) for k in range(5))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    wave = (speech * envelope + 0.05 * gen.standard_normal(samples)).astype(np.float32)
    y = torch.from_numpy(wave).unsqueeze(0)

    hann = torch.hann_window(win)
    spec = torch.stft(y, n_fft, hop, win, hann, center=True, pad_mode="reflect", return_complex=True)
    mag, pha = torch.abs(spec), torch.angle(spec)
    mag_c = torch.pow(mag, cf)

    seams = {}
    def hook(name):
        def fn(_m, _i, o):
            seams[name] = o
        return fn
    blocks = model.TSTransformer                                # the released core (not the unused conformer)
    handles = [model.dense_encoder.register_forward_hook(hook("encoder"))]
    for idx, block in enumerate(blocks):
        handles.append(block.register_forward_hook(hook(f"ts{idx}")))
    with torch.no_grad():
        amp_g, pha_g, _com = model(mag_c, pha)
    for handle in handles:
        handle.remove()

    mag_d = torch.pow(amp_g, 1.0 / cf)
    com = torch.complex(mag_d * torch.cos(pha_g), mag_d * torch.sin(pha_g))
    wav = torch.istft(com, n_fft, hop, win, hann, center=True)

    extra = {"waveform": y[0].contiguous(), "noisy_mag": mag_c[0].contiguous(),
             "noisy_pha": pha[0].contiguous(), "encoder": seams["encoder"][0].contiguous(),
             "denoised_mag": amp_g[0].contiguous(), "denoised_pha": pha_g[0].contiguous()}
    for idx in range(len(blocks)):
        extra[f"ts{idx}"] = seams[f"ts{idx}"][0].contiguous()
    globals()["_extra"] = extra
    return wav[0].contiguous()                                  # [samples]


def run_gtcrn(image, checkpoint):
    """GTCRN (Xiaobin-Rong/gtcrn) speech enhancement of a deterministic noisy clip, seam by seam.

    Set IK_GTCRN_SRC to the directory holding `gtcrn.py`. `--checkpoint` is the released state dict
    (e.g. `checkpoints/model_trained_on_dns3.tar`). Records the encoder bottleneck, each DPGRNN block,
    the decoder mask, and the reconstructed waveform.
    """
    import os

    sys.path.insert(0, os.environ.get("IK_GTCRN_SRC", "."))
    from gtcrn import GTCRN

    model = GTCRN().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(state, strict=True)

    n_fft, hop = 512, 256
    samples = 16000
    t = np.arange(samples, dtype=np.float32) / 16000.0
    gen = np.random.default_rng(9)
    speech = sum(0.3 / (k + 1) * np.sin(2 * np.pi * 140 * (k + 1) * t) for k in range(5))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    wave = (speech * envelope + 0.05 * gen.standard_normal(samples)).astype(np.float32)
    y = torch.from_numpy(wave).unsqueeze(0)

    window = torch.hann_window(n_fft).pow(0.5)                              # infer.py uses the sqrt-Hann
    spec = torch.stft(y, n_fft, hop, n_fft, window, center=True, return_complex=True)
    spec_ri = torch.stack([spec.real, spec.imag], dim=-1)                   # [B, F, T, 2]

    seams = {}
    def hook(name):
        def fn(_m, _i, o):
            seams[name] = o[0] if isinstance(o, tuple) else o
        return fn
    handles = [model.encoder.register_forward_hook(hook("encoder")),
               model.dpgrnn1.register_forward_hook(hook("dpgrnn1")),
               model.dpgrnn1.intra_rnn.register_forward_hook(hook("d1_intra_rnn")),
               model.dpgrnn1.intra_fc.register_forward_hook(hook("d1_intra_fc")),
               model.dpgrnn1.intra_ln.register_forward_hook(hook("d1_intra_ln")),
               model.dpgrnn1.inter_ln.register_forward_hook(hook("d1_inter_ln")),
               model.dpgrnn2.register_forward_hook(hook("dpgrnn2")),
               model.decoder.register_forward_hook(hook("decoder"))]
    for idx, block in enumerate(model.encoder.en_convs):
        handles.append(block.register_forward_hook(hook(f"en{idx}")))
    for idx, block in enumerate(model.decoder.de_convs):
        handles.append(block.register_forward_hook(hook(f"de{idx}")))
    with torch.no_grad():
        out_spec = model(spec_ri)                                          # [B, F, T, 2]
    for handle in handles:
        handle.remove()

    enhanced = torch.complex(out_spec[..., 0], out_spec[..., 1])
    wav = torch.istft(enhanced, n_fft, hop, n_fft, window, center=True)

    globals()["_extra"] = {
        "waveform": y[0].contiguous(),
        "in_spec": spec_ri[0].contiguous(),                                # [F, T, 2] the net input
        "encoder": seams["encoder"].contiguous(),
        "dpgrnn1": seams["dpgrnn1"].contiguous(),
        "d1_intra_rnn": seams["d1_intra_rnn"].contiguous(),
        "d1_intra_fc": seams["d1_intra_fc"].contiguous(),
        "d1_intra_ln": seams["d1_intra_ln"].contiguous(),
        "d1_inter_ln": seams["d1_inter_ln"].contiguous(),
        "dpgrnn2": seams["dpgrnn2"].contiguous(),
        "decoder": seams["decoder"].contiguous(),
        "out_spec": out_spec[0].contiguous(),
    }
    for idx in range(len(model.encoder.en_convs)):
        globals()["_extra"][f"en{idx}"] = seams[f"en{idx}"].clone().contiguous()   # en4 aliases `encoder`
    for idx in range(len(model.decoder.de_convs)):
        globals()["_extra"][f"de{idx}"] = seams[f"de{idx}"].clone().contiguous()   # de4 aliases `decoder`
    return wav[0].contiguous()                                             # [samples]



def run_gtcrn_loss(image):
    """GTCRN's training objective, the repo's own `HybridLoss` (`loss.py`), on identical spectrograms.

    Set IK_GTCRN_SRC to the directory holding `loss.py`. The target is the spectrogram of a
    deterministic voiced clip and the prediction the spectrogram of that clip with noise added, both
    through the sqrt-Hann STFT `infer.py` uses, so every term (the compressed real and imaginary MSEs,
    the compressed magnitude MSE, and the negative log SI-SNR over the iSTFT) scores a realistic pair.
    `output` is the loss the reference module returns; the terms and both waveforms are seams.
    """
    import os

    sys.path.insert(0, os.environ.get("IK_GTCRN_SRC", "."))
    from loss import HybridLoss

    n_fft, hop = 512, 256
    samples = 8000
    t = np.arange(samples, dtype=np.float32) / 16000.0
    gen = np.random.default_rng(31)
    speech = sum(0.3 / (k + 1) * np.sin(2 * np.pi * 150 * (k + 1) * t) for k in range(5))
    clean = (speech * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)
    noisy = (clean + 0.1 * gen.standard_normal(samples)).astype(np.float32)

    window = torch.hann_window(n_fft).pow(0.5)
    def spectrogram(wave):
        spec = torch.stft(torch.from_numpy(wave).unsqueeze(0), n_fft, hop, n_fft, window,
                          center=True, return_complex=True)
        return torch.stack([spec.real, spec.imag], dim=-1)                  # [1, F, T, 2]
    pred, true = spectrogram(noisy), spectrogram(clean)

    loss = HybridLoss()(pred, true)

    # The terms, for localizing a disagreement. The assertion is on `output` alone.
    pred_mag = torch.sqrt(pred[..., 0] ** 2 + pred[..., 1] ** 2 + 1e-12)
    true_mag = torch.sqrt(true[..., 0] ** 2 + true[..., 1] ** 2 + 1e-12)
    real_term = torch.nn.MSELoss()(pred[..., 0] / pred_mag ** 0.7, true[..., 0] / true_mag ** 0.7)
    imag_term = torch.nn.MSELoss()(pred[..., 1] / pred_mag ** 0.7, true[..., 1] / true_mag ** 0.7)
    mag_term = torch.nn.MSELoss()(pred_mag ** 0.3, true_mag ** 0.3)
    y_pred = torch.istft(pred[..., 0] + 1j * pred[..., 1], n_fft, hop, n_fft, window=window)
    y_true = torch.istft(true[..., 0] + 1j * true[..., 1], n_fft, hop, n_fft, window=window)

    globals()["_extra"] = {
        "predicted": pred[0].contiguous(),                                  # [F, T, 2]
        "target": true[0].contiguous(),
        "real_term": real_term.reshape(1).contiguous(),
        "imag_term": imag_term.reshape(1).contiguous(),
        "magnitude_term": mag_term.reshape(1).contiguous(),
        "predicted_waveform": y_pred[0].contiguous(),
        "target_waveform": y_true[0].contiguous(),
    }
    return loss.reshape(1).contiguous()


def run_nuwave2_loss(image, checkpoint):
    """NU-Wave 2's training objective, `NuWave2.common_step` in `lightning_model.py`, on identical tensors.

    `--checkpoint` is the official Lightning checkpoint and `IK_NUWAVE2_SRC` the cloned repository,
    as for `nuwave2`. The wide-band clip is a deterministic 48 kHz tone stack reaching 20 kHz, trimmed
    to the training segment of 32,768 samples and peak-normalized, and the narrow-band input comes from
    `dataloader.py`'s own degradation at its validation settings (a Chebyshev type I low-pass of order
    8 and ripple 0.05 at 8 kHz through `sosfiltfilt`, then `resample_poly` down to 16 kHz and back).
    The diffusion time and noise are fixed and recorded, so both sides score the same draw:
    `Diffusion.diffusion` noises the clip, the network predicts the noise, and the loss is `nn.L1Loss`
    between the prediction and the noise.
    """
    from scipy.signal import cheby1, resample_poly, sosfiltfilt
    diffusion, hparams = _load_nuwave2(checkpoint)

    rate = hparams.audio.sampling_rate
    length = hparams.audio.length
    t = np.arange(length, dtype=np.float64) / rate
    wav = sum(0.4 / (k + 1) * np.sin(2 * np.pi * 220 * (k + 1) * t) for k in range(90))
    wav = (wav / np.max(np.abs(wav))).astype(np.float32)

    highcut = 8000
    hi = highcut / (0.5 * rate)
    sos = cheby1(8, 0.05, hi, btype="lowpass", output="sos")
    wav_l = resample_poly(resample_poly(sosfiltfilt(sos, wav), highcut * 2, rate), rate, highcut * 2)
    wav_l = wav_l[:length].astype(np.float32)
    fft_size = hparams.audio.filter_length // 2 + 1
    band = torch.zeros(fft_size, dtype=torch.int64)
    band[: int(hi * fft_size)] = 1

    y = torch.from_numpy(wav).unsqueeze(0)
    y_l = torch.from_numpy(wav_l.copy()).unsqueeze(0)
    band = band.unsqueeze(0)
    time = torch.tensor([0.37])
    generator = torch.Generator().manual_seed(11)
    z = torch.randn(y.shape, generator=generator)

    with torch.no_grad():
        _, _, noised = diffusion.diffusion(y, z, time)
        estimate, logsnr, _ = diffusion(noised, y_l, band, time)
        loss = torch.nn.L1Loss()(estimate, z)

    globals()["_extra"] = {
        "waveform": y[0].contiguous(),
        "waveform_low": y_l[0].contiguous(),
        "band": band[0].to(torch.int32).contiguous(),
        "time": time.contiguous(),
        "noise": z[0].contiguous(),
        "noised": noised[0].contiguous(),
        "estimate": estimate[0].contiguous(),
        "logsnr": logsnr.reshape(1).contiguous(),
    }
    return loss.reshape(1).contiguous()


def run_convtasnet_loss(image):
    """Conv-TasNet's training objective, asteroid v0.5.2's `PITLossWrapper(pairwise_neg_sisdr,
    pit_from="pw_mtx")`, as `egs/librimix/ConvTasNet/train.py` builds it, on identical tensors.

    `IK_ASTEROID_SRC` holds the tag's `asteroid/losses/sdr.py` and `pit_wrapper.py`, loaded by path
    so the package's other dependencies stay out. Two speakers: the estimates are the sources swapped,
    scaled, shifted by a constant, and noised, so the best assignment is the crossed one, the zero-mean
    step matters, and the scale does not. `output` is the loss; `pairwise` is the matrix it minimizes over.
    """
    import importlib.util
    import os

    losses = os.path.join(os.environ.get("IK_ASTEROID_SRC", "."), "asteroid", "losses")
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    sdr = load("asteroid_sdr", os.path.join(losses, "sdr.py"))
    pit = load("asteroid_pit_wrapper", os.path.join(losses, "pit_wrapper.py"))

    samples = 8000
    t = np.arange(samples, dtype=np.float32) / 16000.0
    generator = np.random.default_rng(37)
    first = 0.4 * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    second = 0.3 * np.sign(np.sin(2 * np.pi * 95 * t)) * (0.5 + 0.5 * np.cos(2 * np.pi * 2 * t))
    sources = torch.from_numpy(np.stack([first, second]).astype(np.float32))[None]           # [1, 2, T]
    noise = torch.from_numpy(generator.standard_normal((1, 2, samples)).astype(np.float32))
    estimates = torch.stack([1.7 * sources[:, 1] + 0.2, 0.6 * sources[:, 0] - 0.1], dim=1) + 0.05 * noise

    loss_func = pit.PITLossWrapper(sdr.pairwise_neg_sisdr, pit_from="pw_mtx")
    loss = loss_func(estimates, sources)
    globals()["_extra"] = {
        "estimates": estimates[0].contiguous(),
        "sources": sources[0].contiguous(),
        "pairwise": sdr.pairwise_neg_sisdr(estimates, sources)[0].contiguous(),
    }
    return loss.reshape(1).contiguous()

def run_sgmse(image, checkpoint):
    """SGMSE+ (sp-uhh/sgmse) score-based speech dereverberation / enhancement.

    Set IK_SGMSE_SRC to the cloned repository (holding `sgmse/`). `--checkpoint` is a released Lightning
    `.ckpt` (dereverb = WSJ0-REVERB, enhancement = VoiceBank-DEMAND, …). Inference reads the EMA weights,
    which `model.eval()` copies into `model.dnn` via torch_ema.

    Records the DETERMINISTIC net seam — a fixed `x_t`, the compressed observation `y`, the time `t`, and
    the RAW NCSN++ output (the score is `-` this) plus the input-conv seam — which is the numeric ground,
    since a sampled clip's random stream is not reproducible across implementations. Also records the
    reference enhanced waveform for the end-to-end signal check.
    """
    import os

    sys.path.insert(0, os.environ.get("IK_SGMSE_SRC", "."))
    from sgmse.model import ScoreModel
    from sgmse.util.other import pad_spec

    # torch >= 2.6 defaults weights_only=True and refuses the pickled data module; restore pre-2.6.
    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

    model = ScoreModel.load_from_checkpoint(checkpoint, map_location="cpu", base_dir="/tmp", batch_size=1, num_workers=0)
    model.eval()                                                          # copies EMA weights into model.dnn
    dnn = model.dnn

    # A deterministic noisy clip, the GTCRN/MP-SENet recipe.
    sr = int(getattr(model, "sr", 16000))
    samples = sr
    t_axis = np.arange(samples, dtype=np.float32) / float(sr)
    gen = np.random.default_rng(11)
    speech = sum(0.3 / (k + 1) * np.sin(2 * np.pi * 130 * (k + 1) * t_axis) for k in range(5))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t_axis)
    reverb = np.convolve(speech * envelope, np.exp(-np.arange(400) / 120.0).astype(np.float32))[:samples]
    wave = (0.7 * speech * envelope + 0.3 * reverb + 0.02 * gen.standard_normal(samples)).astype(np.float32)
    y_wave = torch.from_numpy(wave).unsqueeze(0)                          # [1, samples]

    norm_factor = y_wave.abs().max()
    y_norm = y_wave / norm_factor
    Y = torch.unsqueeze(model._forward_transform(model._stft(y_norm)), 0)  # [1, 1, F, T] complex
    Y = pad_spec(Y)

    # Deterministic x_t: the OUVE marginal about the observation at a fixed t, so it is a physical state.
    t_val = 0.5
    vec_t = torch.ones(Y.shape[0]) * t_val
    torch.manual_seed(0)
    z = torch.randn_like(Y)
    std = model.sde._std(vec_t)[:, None, None, None]
    x_t = Y + std * z                                                     # a plausible perturbed state

    seams = {}
    handle = dnn.all_modules[3].register_forward_hook(
        lambda _m, _i, o: seams.__setitem__("conv_in", o))
    with torch.no_grad():
        net_out = dnn(torch.cat([x_t, Y], dim=1), vec_t)                # [B, 1, F, T] complex (raw dnn output)
    handle.remove()

    def ri(spec):                                                        # complex [B,1,F,T] -> [F,T,2]
        s = spec[0, 0]
        return torch.stack([s.real, s.imag], dim=-1).contiguous()

    conv_in = seams["conv_in"][0].permute(1, 2, 0).contiguous()          # [C,F,T] -> [F,T,C]

    # Reference end-to-end enhancement (random noise — recorded for the signal check, not compared bitwise).
    with torch.no_grad():
        sampler = model.get_pc_sampler("reverse_diffusion", "ald", Y, N=30, corrector_steps=1, snr=0.5)
        sample, _ = sampler()
        x_hat = model.to_audio(sample.squeeze(), samples)

    # Record the net geometry + front-end params so the Swift test builds a matching config: the released
    # SGMSE+ variants differ (classic 'ncsnpp' = progressive output_skip + attn@16 + sqrt-Hann; the
    # 'ncsnpp_48k' variant = progressive 'none', no attention, plain Hann, and the output projection
    # applied before the sigma division).
    # Read the geometry from the DNN's OWN attributes, not the saved hparams: a checkpoint that took the
    # backbone defaults (classic 'ncsnpp' → attn [16], progressive 'output_skip') carries neither in its
    # hparams, so an hparams fallback would mis-record them. ch_mult is not stored on the module, so it
    # comes from hparams with the backbone default.
    hp = dict(model.hparams)
    progressive = 1 if getattr(dnn, "progressive", "output_skip") == "output_skip" else 0
    window_power = 0.5 if hp.get("window", "hann") == "sqrthann" else 1.0
    ch_mult = list(hp.get("ch_mult", [1, 1, 2, 2, 2, 2, 2]))
    attn = list(getattr(dnn, "attn_resolutions", []) or [])
    nf_val = int(getattr(dnn, "nf", hp.get("nf", 128)))
    num_res = int(getattr(dnn, "num_res_blocks", hp.get("num_res_blocks", 2)))
    image_size = int(getattr(dnn, "all_resolutions", [256])[0])
    globals()["_extra"] = {
        "in_xt": ri(x_t),
        "in_y": ri(Y),
        "t": torch.tensor([t_val], dtype=torch.float32),
        "net_out": ri(net_out),
        "conv_in": conv_in,
        "waveform": y_wave[0].contiguous(),
        "enhanced": x_hat.reshape(-1).contiguous(),
        "cfg_nf": torch.tensor([nf_val], dtype=torch.int32),
        "cfg_num_res_blocks": torch.tensor([num_res], dtype=torch.int32),
        "cfg_ch_mult": torch.tensor(ch_mult, dtype=torch.int32),
        "cfg_attn": torch.tensor(attn if attn else [0], dtype=torch.int32),
        "cfg_attn_len": torch.tensor([len(attn)], dtype=torch.int32),
        "cfg_image_size": torch.tensor([image_size], dtype=torch.int32),
        "cfg_progressive": torch.tensor([progressive], dtype=torch.int32),
        "cfg_fourier_scale": torch.tensor([float(hp.get("fourier_scale", 16))], dtype=torch.float32),
        "cfg_window_power": torch.tensor([window_power], dtype=torch.float32),
        "cfg_n_fft": torch.tensor([int(hp.get("n_fft", 510))], dtype=torch.int32),
        "cfg_hop": torch.tensor([int(hp.get("hop_length", 128))], dtype=torch.int32),
        "cfg_spec_factor": torch.tensor([float(hp.get("spec_factor", 0.15))], dtype=torch.float32),
        "cfg_spec_abs_exponent": torch.tensor([float(hp.get("spec_abs_exponent", 0.5))], dtype=torch.float32),
        "cfg_theta": torch.tensor([float(hp.get("theta", 1.5))], dtype=torch.float32),
        "cfg_sigma_min": torch.tensor([float(hp.get("sigma_min", 0.05))], dtype=torch.float32),
        "cfg_sigma_max": torch.tensor([float(hp.get("sigma_max", 0.5))], dtype=torch.float32),
    }
    return ri(net_out)                                                    # the seam the parity test scores


def run_storm(image):
    """StoRM (sp-uhh/storm) stochastic-regeneration at a TINY RANDOM configuration.

    Set IK_STORM_SRC to the cloned repository (holding `sgmse/`). StoRM's two NCSN++ networks — a
    DISCRIMINATIVE predictor (`discriminative=True` → no time embedding, no sigma scaling, 2 input
    channels) and a conditioned SCORE net (`input_channels=6` for `condition='both'`) — are built directly
    from the backbone registry and randomized, so the architecture is verified with no download (the
    released combined checkpoints are GDrive-only; the individual NCSN++ backbone is already at
    released-weight parity via SGMSE+). Records the denoiser seam (y → y_denoised), the score seam
    ([x_t, y, y_denoised] → the RAW score-net output, the score being `-` this), the geometry, and both
    networks' weights (under `w::denoiser_net.*` / `w::score_net.*`). `image` unused.
    """
    import os
    import types
    import importlib.util

    src = os.environ.get("IK_STORM_SRC", ".")
    sys.path.insert(0, src)

    # StoRM's op package imports the FUSED upfirdn2d (a compiled CUDA/C++ extension needing ninja);
    # inject a shim exposing the repo's pure-Python native upfirdn2d and a plain leaky-ReLU so the
    # backbone runs on the CPU with no compilation. Parent packages are imported first so the shim
    # replaces only the leaf `op` module.
    import torch.nn as _nn
    import torch.nn.functional as _Fn

    # The pure-Python upfirdn2d (StyleGAN2's `upfirdn2d_native`, which the storm clone omits): insert
    # zeros to upsample, pad, convolve the flipped kernel, then stride to downsample. CPU-clean.
    def _upfirdn2d_native(inp, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
        _, channel, in_h, in_w = inp.shape
        inp = inp.reshape(-1, in_h, in_w, 1)
        _, in_h, in_w, minor = inp.shape
        kernel_h, kernel_w = kernel.shape
        out = inp.view(-1, in_h, 1, in_w, 1, minor)
        out = _Fn.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
        out = out.view(-1, in_h * up_y, in_w * up_x, minor)
        out = _Fn.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
        out = out[:, max(-pad_y0, 0): out.shape[1] - max(-pad_y1, 0),
                  max(-pad_x0, 0): out.shape[2] - max(-pad_x1, 0), :]
        out = out.permute(0, 3, 1, 2)
        out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
        w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
        out = _Fn.conv2d(out, w)
        out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
                          in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
        out = out.permute(0, 2, 3, 1)
        out = out[:, ::down_y, ::down_x, :]
        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
        return out.view(-1, channel, out_h, out_w)

    class _NativeUpfirdn:
        def upfirdn2d(self, inp, kernel, up=1, down=1, pad=(0, 0)):
            return _upfirdn2d_native(inp, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])
    _native = _NativeUpfirdn()

    class _FusedLeakyReLU(_nn.Module):
        def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
            super().__init__()
            self.negative_slope, self.scale = negative_slope, scale
        def forward(self, x):
            return _nn.functional.leaky_relu(x, self.negative_slope) * self.scale

    def _fused_leaky_relu(x, bias=None, negative_slope=0.2, scale=2 ** 0.5):
        if bias is not None:
            x = x + bias.view(1, -1, *([1] * (x.ndim - 2)))
        return _nn.functional.leaky_relu(x, negative_slope) * scale

    _op = types.ModuleType("sgmse.backbones.ncsnpp_utils.op")
    _op.upfirdn2d = _native.upfirdn2d
    _op.FusedLeakyReLU = _FusedLeakyReLU
    _op.fused_leaky_relu = _fused_leaky_relu
    sys.modules["sgmse.backbones.ncsnpp_utils.op"] = _op

    from sgmse.backbones.shared import BackboneRegistry

    ncsnpp = BackboneRegistry.get_by_name("ncsnpp")
    common = dict(nf=8, ch_mult=[1, 2], num_res_blocks=1, attn_resolutions=[8], image_size=16,
                  fourier_scale=16, centered=True)
    denoiser = _randomized(ncsnpp(input_channels=2, discriminative=True, **common), seed=31)
    score = _randomized(ncsnpp(input_channels=6, discriminative=False, **common), seed=32)

    gen = torch.Generator().manual_seed(7)
    freq, time = 16, 16
    def crand():
        return (torch.randn(1, 1, freq, time, generator=gen) + 1j * torch.randn(1, 1, freq, time, generator=gen))
    Y = crand()
    x_t = crand()
    t_val = 0.5
    vec_t = torch.tensor([t_val])

    with torch.no_grad():
        Y_denoised = denoiser(Y, time_cond=None)                        # [1,1,F,T] complex (discriminative)
        score_out = score(torch.cat([x_t, Y, Y_denoised], dim=1), vec_t)  # RAW score-net output (score = -this)

    def ri(spec):
        s = spec[0, 0]
        return torch.stack([s.real, s.imag], dim=-1).contiguous()

    extra = {
        "in_y": ri(Y),
        "in_xt": ri(x_t),
        "denoiser_out": ri(Y_denoised),
        "t": torch.tensor([t_val], dtype=torch.float32),
        "score_out": ri(score_out),
        "cfg_nf": torch.tensor([8], dtype=torch.int32),
        "cfg_num_res_blocks": torch.tensor([1], dtype=torch.int32),
        "cfg_ch_mult": torch.tensor([1, 2], dtype=torch.int32),
        "cfg_attn": torch.tensor([8], dtype=torch.int32),
        "cfg_attn_len": torch.tensor([1], dtype=torch.int32),
        "cfg_image_size": torch.tensor([16], dtype=torch.int32),
        "cfg_progressive": torch.tensor([1], dtype=torch.int32),
        "cfg_condition": torch.tensor([2], dtype=torch.int32),          # both
        "cfg_fourier_scale": torch.tensor([16.0], dtype=torch.float32),
        "cfg_window_power": torch.tensor([1.0], dtype=torch.float32),
        "cfg_n_fft": torch.tensor([510], dtype=torch.int32),
        "cfg_hop": torch.tensor([128], dtype=torch.int32),
        "cfg_spec_factor": torch.tensor([0.15], dtype=torch.float32),
        "cfg_spec_abs_exponent": torch.tensor([0.5], dtype=torch.float32),
    }
    for key, value in denoiser.state_dict().items():
        extra[f"w::denoiser_net.{key}"] = value.float().contiguous()
    for key, value in score.state_dict().items():
        extra[f"w::score_net.{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return ri(score_out)


def run_mossformer2_se(image, checkpoint):
    """MossFormer2 SE 48K (modelscope/ClearerVoice-Studio) speech enhancement, seam by seam.

    Set IK_MOSSFORMER2_SE_SRC to the ClearerVoice source dir that holds `models/mossformer2_se/`
    (e.g. `.../ClearerVoice-Studio/clearvoice/clearvoice`). `--checkpoint` is `last_best_checkpoint.pt`.
    The fbank front end forces **dither=0** (the released `compute_fbank` uses dither=1.0, which is
    random noise and makes parity impossible). Records the 180-dim network feature, the masking STFT,
    the encoder output, the first and last block outputs, the 961-bin mask, and the waveform.
    """
    import os
    import torchaudio

    sys.path.insert(0, os.environ.get("IK_MOSSFORMER2_SE_SRC", "."))
    from models.mossformer2_se.mossformer2 import MossFormer_MaskNet

    net = MossFormer_MaskNet(in_channels=180, out_channels=512, out_channels_final=961).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("model", state) if isinstance(state, dict) else state
    stripped = {k[len("mossformer."):]: v for k, v in state.items() if k.startswith("mossformer.")}
    net.load_state_dict(stripped if stripped else state, strict=True)

    sr, n_fft, hop, win = 48000, 1920, 384, 1920
    samples = sr
    t = np.arange(samples, dtype=np.float32) / sr
    gen = np.random.default_rng(9)
    speech = sum(0.3 / (k + 1) * np.sin(2 * np.pi * 140 * (k + 1) * t) for k in range(5))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    wave = (speech * envelope + 0.05 * gen.standard_normal(samples)).astype(np.float32)
    y = torch.from_numpy(wave).unsqueeze(0)

    fbank = torchaudio.compliance.kaldi.fbank(y, dither=0.0, frame_length=40.0, frame_shift=8.0,
                                              num_mel_bins=60, sample_frequency=sr, window_type="hamming")
    delta = torchaudio.functional.compute_deltas(fbank.transpose(0, 1)).transpose(0, 1)
    delta2 = torchaudio.functional.compute_deltas(
        torchaudio.functional.compute_deltas(fbank.transpose(0, 1))).transpose(0, 1)
    feat = torch.cat([fbank, delta, delta2], dim=1)                    # [S, 180]

    seams = {}
    def hook(name):
        def fn(_m, _i, o):
            seams[name] = o[0] if isinstance(o, tuple) else o
        return fn
    blocks = net.mdl.intra_mdl.mossformerM.layers
    handles = [net.conv1d_encoder.register_forward_hook(hook("encoder")),
               blocks[0].register_forward_hook(hook("block0")),
               blocks[len(blocks) - 1].register_forward_hook(hook("block_last"))]
    with torch.no_grad():
        out = net(feat.unsqueeze(0).transpose(1, 2))                  # wrapper transposes [B,S,180]->[B,180,S]
    mask = out[0] if isinstance(out, (list, tuple)) else out          # [B, 961, S] or [B, S, 961]
    for handle in handles:
        handle.remove()

    window = torch.hamming_window(win, periodic=False)
    spec = torch.stft(y, n_fft, hop, win, window, center=False, return_complex=False)   # [1, F, T, 2]
    mask_t = mask.squeeze(0)
    if mask_t.shape[0] != spec.shape[1]:                              # want [F, T]
        mask_t = mask_t.transpose(0, 1)
    masked = spec.squeeze(0) * mask_t.unsqueeze(-1)                   # [F, T, 2]
    complex_spec = torch.complex(masked[..., 0], masked[..., 1])
    wav = torch.istft(complex_spec, n_fft, hop, win, window, center=False, length=samples)

    globals()["_extra"] = {
        "waveform": y[0].contiguous(),
        "feature": feat.contiguous(),                                # [S, 180] the net input
        "spectrum": spec.squeeze(0).contiguous(),                    # [F, T, 2]
        "encoder": seams["encoder"][0].contiguous(),
        "block0": seams["block0"][0].contiguous(),
        "block_last": seams["block_last"][0].contiguous(),
        "mask": mask.squeeze(0).contiguous(),
    }
    return wav.contiguous()                                          # [samples]


def run_deepfilternet(image, checkpoint):
    """DeepFilterNet3 (Rikorose/DeepFilterNet, dual MIT/Apache-2.0) real-time 48 kHz speech denoising.

    `--checkpoint` is the `DeepFilterNet3` model directory (the `init_df` cache, holding
    `checkpoints/` + `config.ini`). Needs the `deepfilternet` + `deepfilterlib` pip packages (torch,
    torchaudio, numpy<2). Records the whole net boundary (`spec`/`feat_erb`/`feat_spec` in, the encoder
    seams `e0..e3`/`emb`/`c0`, the ERB mask `m`, the deep-filter coefficients, `spec_e`, `lsnr`) plus the
    libdf DSP I/O (the ERB banks and the enhanced waveform) so the MLX port validates the net seam by seam
    and the DSP end to end. Set `IK_DEEPFILTERNET_WEIGHTS_OUT` to also dump the model's state dict as
    `weights.safetensors` beside the record (dropping the `num_batches_tracked` counters).
    """
    import os
    import numpy as np
    import torch
    from df.enhance import init_df, df_features, enhance
    from df.model import ModelParams
    from df.utils import get_norm_alpha, as_complex

    model, df_state, _ = init_df(checkpoint)
    model.eval()
    p = ModelParams()
    sr, nb_df = p.sr, p.nb_df

    # A deterministic 0.5 s clip (the restoration-family recipe, at 48 kHz).
    samples = sr // 2
    t = np.arange(samples, dtype=np.float32) / sr
    gen = np.random.default_rng(7)
    speech = sum(0.3 / (k + 1) * np.sin(2 * np.pi * 160 * (k + 1) * t) for k in range(5))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    wave = (0.8 * speech * env + 0.05 * gen.standard_normal(samples)).astype(np.float32)
    audio = torch.from_numpy(wave).unsqueeze(0)

    spec, erb_feat, spec_feat = df_features(audio, df_state, nb_df)

    seams = {}
    def enc_hook(_m, _i, o):
        seams["enc"] = [x.detach().clone() for x in o]
    handles = [model.enc.register_forward_hook(enc_hook),
               model.enc.df_conv1.register_forward_hook(lambda _m, _i, o: seams.__setitem__("c1", o.detach().clone())),
               model.enc.df_fc_emb.register_forward_hook(lambda _m, _i, o: seams.__setitem__("cemb", o.detach().clone())),
               model.enc.emb_gru.register_forward_pre_hook(lambda _m, i: seams.__setitem__("emb_in", i[0].detach().clone()))]
    with torch.no_grad():
        spec_e, m, lsnr, df_coefs = model(spec.clone(), erb_feat, spec_feat)
    for h in handles:
        h.remove()
    e0, e1, e2, e3, emb, c0, lsnr2 = seams["enc"]

    spec_m = model.mask(spec.clone(), m)                               # the ERB-masked spectrum

    # Snapshot BEFORE synthesis: `df_state.synthesis(as_complex(spec_e).numpy())` writes in place through
    # the view and would corrupt the recorded `spec_e`.
    spec_e_rec = spec_e.detach().clone()
    spec_m_rec = spec_m.detach().clone()

    # The reference `output` is the full `enhance()` path (pad to compensate the STFT delay → analysis →
    # net → synthesis → trim `[n_fft-hop : orig+n_fft-hop]`), which is what the MLX backend reproduces.
    df_state.reset()
    audio_out = enhance(model, df_state, audio)

    weights_out = os.environ.get("IK_DEEPFILTERNET_WEIGHTS_OUT")
    if weights_out:
        from safetensors.torch import save_file
        sd = {k: v.contiguous() for k, v in model.state_dict().items()
              if not k.endswith("num_batches_tracked")}
        save_file(sd, weights_out)
        print(f"wrote weights {weights_out} ({len(sd)} tensors)")

    globals()["_extra"] = {
        "waveform": audio[0].clone().contiguous(),
        "spec": spec.squeeze(1).squeeze(0).clone().contiguous(),       # [T, F, 2] the net input
        "feat_erb": erb_feat.squeeze(1).squeeze(0).contiguous(),       # [T, 32]
        "feat_spec": spec_feat.squeeze(1).squeeze(0).contiguous(),     # [T, 96, 2]
        "e0": e0[0].contiguous(), "e1": e1[0].contiguous(),
        "e2": e2[0].contiguous(), "e3": e3[0].contiguous(),            # [C, T, F]
        "emb": emb[0].contiguous(),                                    # [T, 512]
        "c1": seams["c1"][0].contiguous(),                             # [C, T, 48]
        "cemb": seams["cemb"][0].contiguous(),                         # [T, 512] df_fc_emb output
        "emb_in": seams["emb_in"][0].contiguous(),                     # [T, 512] combined, pre emb_gru
        "c0": c0[0].contiguous(),                                      # [C, T, 96]
        "m": m.squeeze(1)[0].contiguous(),                             # [T, 32]
        "df_coefs": df_coefs[0].contiguous(),                          # [O, T, 96, 2]
        "spec_e": spec_e_rec.squeeze(1)[0].contiguous(),               # [T, F, 2]
        "spec_m": spec_m_rec.squeeze(1)[0].contiguous(),               # [T, F, 2] the ERB-masked spectrum
        "lsnr": lsnr[0].contiguous(),                                  # [T, 1]
        "erb_fb": model.erb_fb.detach().contiguous(),                  # [F, 32]
        "erb_inv_fb": model.mask.erb_inv_fb.detach().contiguous(),     # [32, F] (the enhanced waveform is `output`)
    }
    return (audio_out[0] if audio_out.dim() > 1 else audio_out).contiguous()


def run_voicerestore(image, checkpoint):
    """VoiceRestore (skirdey/voicerestore, MIT) — the E2-TTS flow-matching TRANSFORMER, seam by seam.

    Set IK_VR_SRC to the source dir (voice_restore.py + tensor_typing.py). `--checkpoint` is the released
    transformer (`pytorch_model.bin`, keyed `transformer.*`/`proj_in`/`cond_proj`/`to_pred`). Needs
    x-transformers==1.34.0, gateloop-transformer==0.2.5, torchdiffeq, jaxtyping. Records the pre-transformer
    additive condition, every block's gateloop/attn/ff seam over the packed sequence (32 registers + mel),
    and the final velocity — and the null (cfg cond=None) velocity for the CFG path. The hooks are removed
    BEFORE the null pass so the recorded seams stay the CONDITIONED ones.
    """
    import os
    sys.path.insert(0, os.environ["IK_VR_SRC"])
    from voice_restore import VoiceRestore

    model = VoiceRestore(sigma=0.0, transformer=dict(dim=768, depth=20, heads=16, dim_head=64,
                         skip_connect_type="concat", max_seq_len=2000), num_channels=100)
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    model.load_state_dict(sd, strict=True)
    model.eval()

    torch.manual_seed(0)
    x_t = torch.randn(1, 64, 100); cond = torch.randn(1, 64, 100); t = torch.tensor(0.5)
    seams = {}
    handles = []
    for i, layer in enumerate(model.transformer.layers):
        gl, skip, an, at, az, fn, ff, fz = layer
        handles.append(at.register_forward_hook(lambda m, inp, o, i=i: seams.__setitem__(f"attn{i}", o.detach().clone())))
        if i in (0, 19):
            handles.append(gl.register_forward_hook(lambda m, inp, o, i=i: seams.__setitem__(f"gl{i}", o.detach().clone())))
            handles.append(ff.register_forward_hook(lambda m, inp, o, i=i: seams.__setitem__(f"ff{i}", o.detach().clone())))
    with torch.no_grad():
        x_in = model.proj_in(x_t) + model.cond_proj(cond)
        velocity = model.transformer_with_pred_head(x_t, times=t, cond=cond)
    for h in handles:
        h.remove()
    with torch.no_grad():
        null_vel = model.transformer_with_pred_head(x_t, times=t, cond=None)

    globals()["_extra"] = {"x_t": x_t[0], "cond": cond[0], "t": t.reshape(1),
                           "x_in": x_in[0], "null_velocity": null_vel[0]}
    for i in range(20):
        globals()["_extra"][f"attn{i}"] = seams[f"attn{i}"][0]
    for i in (0, 19):
        globals()["_extra"][f"gl{i}"] = seams[f"gl{i}"][0]
        globals()["_extra"][f"ff{i}"] = seams[f"ff{i}"][0]
    return velocity[0].contiguous()


def run_bigvgan(image, checkpoint):
    """BigVGAN v2 (`nvidia/bigvgan_v2_24khz_100band_256x`, MIT) — mel → waveform. Set IK_VR_SRC to the dir
    holding the vendored `BigVGAN/` package. `--checkpoint` is `bigvgan_generator.pt`."""
    import os, json
    sys.path.insert(0, os.environ["IK_VR_SRC"])
    from BigVGAN.bigvgan import BigVGAN
    from BigVGAN.env import AttrDict

    h = AttrDict(json.load(open(os.environ["IK_VR_SRC"] + "/BigVGAN/configs/bigvgan_v2_24khz_100band_256x.json")))
    model = BigVGAN(h, use_cuda_kernel=False)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ck = ck.get("generator", ck) if isinstance(ck, dict) else ck
    model.load_state_dict(ck, strict=True)
    model.eval(); model.remove_weight_norm()
    torch.manual_seed(1)
    mel = torch.randn(1, 100, 64)
    with torch.no_grad():
        wav = model(mel)
    globals()["_extra"] = {"mel": mel[0]}
    return wav[0, 0].contiguous()


def run_voicerestore_e2e(image, checkpoint):
    """VoiceRestore end to end: the BigVGAN mel front end, the CFM midpoint sampler seeded with a recorded
    `y0`, and the vocoder. `--checkpoint` is the transformer; set IK_VR_BIGVGAN to `bigvgan_generator.pt`."""
    import os, json
    sys.path.insert(0, os.environ["IK_VR_SRC"])
    from voice_restore import VoiceRestore
    from BigVGAN.bigvgan import BigVGAN
    from BigVGAN.env import AttrDict
    from BigVGAN.meldataset import get_mel_spectrogram
    from torchdiffeq import odeint

    model = VoiceRestore(sigma=0.0, transformer=dict(dim=768, depth=20, heads=16, dim_head=64,
                         skip_connect_type="concat", max_seq_len=2000), num_channels=100)
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    model.load_state_dict(sd, strict=True); model.eval()
    h = AttrDict(json.load(open(os.environ["IK_VR_SRC"] + "/BigVGAN/configs/bigvgan_v2_24khz_100band_256x.json")))
    bv = BigVGAN(h, use_cuda_kernel=False)
    bck = torch.load(os.environ["IK_VR_BIGVGAN"], map_location="cpu", weights_only=False)
    bck = bck.get("generator", bck) if isinstance(bck, dict) else bck
    bv.load_state_dict(bck, strict=True); bv.eval(); bv.remove_weight_norm()

    sr = 24000; n = sr // 2; ta = np.arange(n, dtype=np.float32) / sr
    g = np.random.default_rng(3); speech = sum(0.3 / (k + 1) * np.sin(2 * np.pi * 150 * (k + 1) * ta) for k in range(4))
    wave = (0.7 * speech * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * ta)) + 0.05 * g.standard_normal(n)).astype(np.float32)
    audio = torch.from_numpy(wave).unsqueeze(0)
    mel = get_mel_spectrogram(audio, h)
    processed = mel.transpose(1, 2)
    steps = 8; torch.manual_seed(5); y0 = torch.randn_like(processed)
    times = torch.linspace(0, 1, steps)
    def ode_fn(tt, x): return model.cfg_transformer_with_pred_head(x, times=tt, cond=processed, cfg_strength=0.5)
    with torch.no_grad():
        restored = odeint(ode_fn, y0, times, atol=1e-5, rtol=1e-5, method="midpoint")[-1]
        wav = bv(restored.transpose(1, 2))
    globals()["_extra"] = {"audio": audio[0].contiguous(), "mel": mel[0].contiguous(),
                           "processed": processed[0].contiguous(), "y0": y0[0].contiguous(),
                           "restored": restored[0].contiguous()}   # the waveform is `output`
    return wav[0, 0].contiguous()


def _reenhance_src():
    import os
    src = os.environ.get("IK_REENHANCE_SRC", os.path.expanduser("~/.inferkit-validation/reference-sources/resemble-enhance"))
    if src not in sys.path:
        sys.path.insert(0, src)


def _reenhance_hp(checkpoint=None):
    _reenhance_src()
    from resemble_enhance.enhancer.hparams import HParams
    if checkpoint:
        import os
        run_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(checkpoint))))
        yaml = os.path.join(run_dir, "hparams.yaml")
        if os.path.exists(yaml):
            return HParams.from_yaml(yaml)   # the released z_scale is 6, not the default 5
    return HParams()


def _reenhance_wav(seconds=0.4, sr=44100, seed=3):
    n = int(sr * seconds)
    ta = np.arange(n, dtype=np.float32) / sr
    g = np.random.default_rng(seed)
    speech = sum(0.3 / (k + 1) * np.sin(2 * np.pi * 150 * (k + 1) * ta) for k in range(4))
    wave = (0.7 * speech * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * ta)) + 0.05 * g.standard_normal(n)).astype(np.float32)
    return torch.from_numpy(wave)


def _wn_old_to_new(state):
    """The released checkpoint stores weight-norm as `weight_g`/`weight_v` (the old API); the current
    source builds modules with the parametrization API, whose state keys are `parametrizations.weight.
    original0`/`original1`. Rename so a strict load succeeds."""
    out = {}
    for k, v in state.items():
        if k.endswith(".weight_g"):
            out[k[: -len(".weight_g")] + ".parametrizations.weight.original0"] = v
        elif k.endswith(".weight_v"):
            out[k[: -len(".weight_v")] + ".parametrizations.weight.original1"] = v
        else:
            out[k] = v
    return out


def _reenhance_sub(checkpoint, prefix):
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["module"]
    n = len(prefix)
    return _wn_old_to_new({k[n:]: v for k, v in sd.items() if k.startswith(prefix)})


def run_reenhance_mel(image, checkpoint):
    """Resemble Enhance (resemble-ai, MIT) — the MelSpectrogram front end (torchaudio magnitude mel,
    preemphasis, amp-to-db, headroom normalize). `--checkpoint` is the enhancer_stage2 model states."""
    _reenhance_src()
    from resemble_enhance.melspec import MelSpectrogram
    hp = _reenhance_hp(checkpoint)
    mel_fn = MelSpectrogram(hp)
    wav = _reenhance_wav()[None]
    with torch.no_grad():
        mel = mel_fn(wav)          # [1, 128, frames]
    globals()["_extra"] = {"wav": wav[0].contiguous()}
    return mel[0, :, :-1].contiguous()   # to_mel drops the last frame


def run_reenhance_univnet(image, checkpoint):
    """The UnivNet LVC vocoder: acoustic features [1, 160, t] + a recorded noise z → waveform."""
    _reenhance_src()
    from resemble_enhance.enhancer.univnet import UnivNet
    hp = _reenhance_hp(checkpoint)
    d_input = hp.num_mels + hp.vocoder_extra_dim
    voc = UnivNet(hp, d_input)
    voc.load_state_dict(_reenhance_sub(checkpoint, "vocoder."), strict=True)
    voc.eval()
    torch.manual_seed(2)
    x = torch.randn(1, d_input, 16)
    cap, orig = {}, torch.randn
    def fake(*a, **k):
        r = orig(*a, **k); cap["z"] = r.clone(); return r
    torch.randn = fake
    seams = {}
    for j, b in enumerate(voc.blocks):
        b.register_forward_hook(lambda m, i, o, j=j: seams.__setitem__(f"block{j}", o.detach().clone()))
    with torch.no_grad():
        wav = voc(x)
    torch.randn = orig
    globals()["_extra"] = {"x": x[0].contiguous(), "z": cap["z"][0].contiguous()}
    for j in range(len(voc.blocks)):
        globals()["_extra"][f"block{j}"] = seams[f"block{j}"][0].contiguous()
    return wav[0].contiguous()


def run_reenhance_irmae(image, checkpoint):
    """The IRMAE autoencoder: mel [1, 128, t] → latent [1, 64, t] (encode) → [1, 160, t] (decode)."""
    _reenhance_src()
    from resemble_enhance.enhancer.lcfm import IRMAE
    hp = _reenhance_hp(checkpoint)
    ae = IRMAE(input_dim=hp.num_mels, output_dim=hp.num_mels + hp.vocoder_extra_dim, latent_dim=hp.lcfm_latent_dim)
    ae.load_state_dict(_reenhance_sub(checkpoint, "lcfm.ae."), strict=True)
    ae.eval()
    torch.manual_seed(4)
    x = torch.randn(1, hp.num_mels, 20)
    with torch.no_grad():
        z = ae.encode(x)
        h = ae.decode(z)
    globals()["_extra"] = {"x": x[0].contiguous(), "z": z[0].contiguous()}
    return h[0].contiguous()


def run_reenhance_cfm(image, checkpoint):
    """The CFM velocity net (WN) + the exponential-decay midpoint sampler. Records a velocity seam at a
    fixed (psi_t, x, t) and a full sample from a recorded psi_0, plus the solver time schedule `ts`."""
    _reenhance_src()
    from resemble_enhance.enhancer.lcfm import CFM
    hp = _reenhance_hp(checkpoint)
    cfm = CFM(cond_dim=hp.num_mels, output_dim=hp.lcfm_latent_dim, solver_nfe=hp.cfm_solver_nfe,
              solver_method=hp.cfm_solver_method, time_mapping_divisor=hp.cfm_time_mapping_divisor)
    cfm.load_state_dict(_reenhance_sub(checkpoint, "lcfm.cfm."), strict=True)
    cfm.eval()
    torch.manual_seed(6)
    T = 20
    psit = torch.randn(1, hp.lcfm_latent_dim, T)
    x = torch.randn(1, hp.num_mels, T)
    tval = torch.tensor(0.5)
    with torch.inference_mode():
        v = cfm._to_v(**{"ψt": psit, "t": 0.5, "x": x})
        cfm.solver.configurate_(nfe=32, method="midpoint")
        psi0 = torch.randn(1, hp.lcfm_latent_dim, T)
        psi1 = cfm.sample(x, **{"ψ0": psi0})
    ts = cfm.solver.time_mapping(np.linspace(0, 1, cfm.solver.n_steps + 1))
    globals()["_extra"] = {"psit": psit[0].contiguous(), "x": x[0].contiguous(), "t": tval.reshape(1),
                           "v": v[0].contiguous(), "psi0": psi0[0].contiguous(),
                           "ts": torch.tensor(np.asarray(ts), dtype=torch.float32)}
    return psi1[0].contiguous()


def run_reenhance_denoiser(image, checkpoint):
    """The stage-1 denoiser: a mixed waveform → a cleaned waveform through an STFT mask UNet."""
    _reenhance_src()
    from resemble_enhance.denoiser.denoiser import Denoiser
    from resemble_enhance.denoiser.hparams import HParams as DHP
    den = Denoiser(DHP())
    den.load_state_dict(_reenhance_sub(checkpoint, "denoiser."), strict=True)
    den.eval()
    wav = _reenhance_wav()[None]
    with torch.inference_mode():
        x = wav / (wav.abs().max(dim=-1, keepdim=True).values + 1e-7)
        mag, cos, sin = den._stft(x)
        net_out = den.net(torch.stack([mag, cos, sin], dim=1))
        o = den(wav)
    globals()["_extra"] = {"wav": wav[0].contiguous(), "mag": mag[0].contiguous(),
                           "net_out": net_out[0].contiguous()}
    return o[0].contiguous()


def run_reenhance_e2e(image, checkpoint):
    """The full enhance path (nfe=32, lambd=0.5, tau=0.5), replicated without importing the Enhancer
    (which pulls deepspeed). Records every seam and the recorded noises, returns the vocoder waveform."""
    _reenhance_src()
    import torch.nn.functional as F
    from resemble_enhance.enhancer.lcfm import LCFM, IRMAE, CFM
    from resemble_enhance.enhancer.univnet import UnivNet
    from resemble_enhance.denoiser.denoiser import Denoiser
    from resemble_enhance.denoiser.hparams import HParams as DHP
    from resemble_enhance.melspec import MelSpectrogram
    from resemble_enhance.common import Normalizer
    hp = _reenhance_hp(checkpoint)
    n_mels = hp.num_mels
    voc_in = n_mels + hp.vocoder_extra_dim
    ae = IRMAE(input_dim=n_mels, output_dim=voc_in, latent_dim=hp.lcfm_latent_dim)
    cfm = CFM(cond_dim=n_mels, output_dim=hp.lcfm_latent_dim, solver_nfe=hp.cfm_solver_nfe,
              solver_method=hp.cfm_solver_method, time_mapping_divisor=hp.cfm_time_mapping_divisor)
    lcfm = LCFM(ae, cfm, z_scale=hp.lcfm_z_scale)
    lcfm.set_mode_("cfm")
    voc = UnivNet(hp, voc_in)
    den = Denoiser(DHP())
    norm = Normalizer()
    mel_fn = MelSpectrogram(hp)

    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["module"]
    def sub(pfx):
        n = len(pfx)
        return _wn_old_to_new({k[n:]: v for k, v in sd.items() if k.startswith(pfx)})
    lcfm.load_state_dict(sub("lcfm."), strict=True)
    voc.load_state_dict(sub("vocoder."), strict=True)
    den.load_state_dict(sub("denoiser."), strict=True)
    norm.load_state_dict(sub("normalizer."), strict=True)
    lcfm.eval(); voc.eval(); den.eval(); norm.eval()

    nfe, lambd, tau = 32, 0.5, 0.5
    cfm.solver.configurate_(nfe, "midpoint")
    lcfm.eval_tau_(tau)

    def to_mel(x):
        return mel_fn(x)[..., :-1]
    def norm_wav(x):
        return x / (x.abs().max(dim=-1, keepdim=True).values + 1e-7)

    wav = _reenhance_wav()[None]
    with torch.inference_mode():
        x = norm_wav(wav)
        x = F.pad(x, (0, 441))          # inference_chunk npad
        x = norm_wav(x)                 # enhancer.forward
        x_mel_original = norm(to_mel(x), update=False)
        den_wav = den(x)
        x_mel_den = norm(to_mel(den_wav), update=False)
        x_mel_denoised = lambd * x_mel_den + (1 - lambd) * x_mel_original
        psi0_enc = lcfm._scale(lcfm.ae.encode(x_mel_original))
        torch.manual_seed(7)
        tau_noise = torch.randn_like(psi0_enc)
        psi0 = tau * tau_noise + (1 - tau) * psi0_enc
        z = lcfm._unscale(cfm.sample(x_mel_denoised, **{"ψ0": psi0}))
        h = lcfm.ae.decode(z)
        cap, orig = {}, torch.randn
        def fake(*a, **k):
            r = orig(*a, **k); cap["z"] = r.clone(); return r
        torch.randn = fake
        o = voc(h)
        torch.randn = orig
    globals()["_extra"] = {
        "wav": wav[0].contiguous(), "x_mel_original": x_mel_original[0].contiguous(),
        "den_wav": den_wav[0].contiguous(), "x_mel_denoised": x_mel_denoised[0].contiguous(),
        "psi0_enc": psi0_enc[0].contiguous(), "tau_noise": tau_noise[0].contiguous(),
        "psi0": psi0[0].contiguous(), "z": z[0].contiguous(), "h": h[0].contiguous(),
        "voc_z": cap["z"][0].contiguous(),
    }
    return o[0].contiguous()


def basic_pitch_clip(seconds=4.0, sample_rate=22050, seed=5):
    """A deterministic clip with notes in it: a held chord, a two-note melody over it, and a little
    noise. Basic Pitch scores pitch, so unstructured noise would give both sides an empty
    transcription and measure nothing."""
    total = int(seconds * sample_rate)
    t = np.arange(total, dtype=np.float32) / sample_rate
    generator = np.random.default_rng(seed)

    def tone(midi, start, end, amplitude):
        frequency = 440.0 * 2.0 ** ((midi - 69) / 12.0)
        window = ((t >= start) & (t < end)).astype(np.float32)
        # A short fade keeps the note from clicking, which would read as an onset of its own.
        fade = np.minimum(np.minimum(t - start, end - t) / 0.02, 1.0).clip(0.0, 1.0).astype(np.float32)
        partials = sum((1.0 / (k + 1)) * np.sin(2 * np.pi * frequency * (k + 1) * t) for k in range(4))
        return amplitude * window * fade * partials.astype(np.float32)

    wave = tone(60, 0.2, 3.6, 0.20) + tone(64, 0.2, 3.6, 0.16) + tone(67, 0.2, 3.6, 0.13)
    wave = wave + tone(72, 0.5, 1.4, 0.25) + tone(76, 1.6, 2.6, 0.25)
    wave = wave + 0.002 * generator.standard_normal(total).astype(np.float32)
    return np.ascontiguousarray(wave / np.abs(wave).max() * 0.9, dtype=np.float32)


def run_basic_pitch(image, checkpoint):
    """Basic Pitch (spotify/basic-pitch) over a deterministic clip, seam by seam.

    `--checkpoint` is the released `nmp.onnx` inside the `basic_pitch` package. The graph holds the
    whole pipeline, so the record is taken from the reference's own artifact through onnxruntime:
    the CQT magnitude, the normalized log, the harmonic stack, the three posteriorgrams per window,
    the stitched posteriorgrams, and the notes the reference's own `note_creation` reads out of them.
    Runs under the `basic_pitch` oracle environment (`bpvenv`).
    """
    import onnx
    import onnxruntime
    from basic_pitch.constants import AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE, FFT_HOP
    from basic_pitch import note_creation
    from basic_pitch.inference import unwrap_output

    graph = onnx.load(checkpoint).graph
    # The intermediates are named by node index rather than by name: the export fuses whole chains of
    # TensorFlow names into one identifier, and the indices are fixed for the released file.
    seams = {"cqt": 189, "logspec": 212, "stack": 228}
    for name, index in seams.items():
        graph.output.append(onnx.helper.make_empty_tensor_value_info(graph.node[index].output[0]))
    model = onnx.load(checkpoint)
    model.graph.CopyFrom(graph)
    session = onnxruntime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])

    overlap = 30 * FFT_HOP
    hop = AUDIO_N_SAMPLES - overlap
    wave = basic_pitch_clip()
    padded = np.concatenate([np.zeros(overlap // 2, dtype=np.float32), wave])

    windows = []
    start = 0
    while start < len(padded):
        window = padded[start:start + AUDIO_N_SAMPLES]
        if len(window) < AUDIO_N_SAMPLES:
            window = np.pad(window, [[0, AUDIO_N_SAMPLES - len(window)]])
        windows.append(window)
        start += hop
    batch = np.stack(windows)[..., None].astype(np.float32)

    names = [output.name for output in session.get_outputs()]
    values = session.run(names, {session.get_inputs()[0].name: batch})
    named = dict(zip(names, values))
    contour, note, onset = values[2], values[1], values[0]
    assert contour.shape[-1] == 264 and note.shape[-1] == 88 and onset.shape[-1] == 88

    unwrap = lambda x: unwrap_output(x, len(wave), 30)
    outputs = {"contour": unwrap(contour), "note": unwrap(note), "onset": unwrap(onset)}
    _, events = note_creation.model_output_to_notes(outputs, onset_thresh=0.5, frame_thresh=0.3,
                                                    min_note_len=11, infer_onsets=True,
                                                    melodia_trick=True, include_pitch_bends=True)

    rows = np.array([[start, end, pitch, amplitude] for start, end, pitch, amplitude, _ in events],
                    dtype=np.float32).reshape(-1, 4)
    lengths = np.array([0 if bends is None else len(bends) for _, _, _, _, bends in events], dtype=np.int32)
    flattened = np.array([bend for _, _, _, _, bends in events for bend in (bends or [])], dtype=np.int32)

    globals()["_extra"] = {
        "waveform": torch.from_numpy(wave),
        "windows": torch.from_numpy(batch[..., 0].copy()),
        "cqt": torch.from_numpy(named[graph.node[seams["cqt"]].output[0]].copy()),
        "logspec": torch.from_numpy(named[graph.node[seams["logspec"]].output[0]][..., 0].copy()),
        "stack": torch.from_numpy(named[graph.node[seams["stack"]].output[0]].copy()),
        "window_contour": torch.from_numpy(contour.copy()),
        "window_note": torch.from_numpy(note.copy()),
        "window_onset": torch.from_numpy(onset.copy()),
        "contour": torch.from_numpy(outputs["contour"].copy()),
        "note": torch.from_numpy(outputs["note"].copy()),
        "notes": torch.from_numpy(rows),
        "bend_lengths": torch.from_numpy(lengths),
        "bend_values": torch.from_numpy(flattened),
    }
    return torch.from_numpy(outputs["onset"].copy())


def _allin1_natten_shim():
    """A torch neighborhood attention with the relative-position bias, standing in for the NATTEN
    kernels All-In-One imports.

    NATTEN dropped the biased 1D/2D kernels after 0.14, so the released model's own dependency no
    longer installs. The neighbor and bias index rules are transcribed from NATTEN 0.14.6's
    `natten_cpu_commons.h`, and the neighbor rule is checked against the installed NATTEN's own
    kernel before anything is recorded (`_allin1_check_natten`), so the substitution is verified
    rather than assumed.
    """
    import types

    def window_start(index, length, kernel, dilation):
        half = kernel // 2
        if dilation <= 1:
            return max(index - half, 0) + (length - index - half - 1 if index + half >= length else 0)
        neighbor = index - half * dilation
        if neighbor < 0:
            return index % dilation
        if index + half * dilation >= length:
            remainder = index % dilation
            whole = (length // dilation) * dilation
            leftover = length - whole
            if remainder < leftover:
                return length - leftover + remainder - 2 * half * dilation
            return whole + remainder - kernel * dilation
        return neighbor

    def bias_start(index, length, kernel, dilation):
        half = kernel // 2
        if dilation <= 1:
            return (half + (half - index if index < half else 0)
                    + (length - index - 1 - half if index + half >= length else 0))
        if index - half * dilation < 0:
            return kernel - 1 - (index // dilation)
        if index + half * dilation >= length:
            return (length - index - 1) // dilation
        return half

    def tables(length, kernel, dilation, device):
        starts = torch.tensor([window_start(i, length, kernel, dilation) for i in range(length)], device=device)
        biases = torch.tensor([bias_start(i, length, kernel, dilation) for i in range(length)], device=device)
        steps = torch.arange(kernel, device=device)
        return starts[:, None] + steps[None, :] * dilation, biases[:, None] + steps[None, :]

    def qkrpb1d(query, key, rpb, kernel, dilation):
        # query/key: [B, heads, length, dim]
        length = query.shape[2]
        neighbors, biases = tables(length, kernel, dilation, query.device)
        gathered = key[:, :, neighbors, :]                                  # [B, heads, L, K, dim]
        scores = (query.unsqueeze(3) * gathered).sum(-1)
        return scores + rpb[None, :, :, :].expand(query.shape[0], -1, -1, -1).gather(
            3, biases[None, None].expand(query.shape[0], rpb.shape[0], length, kernel)
        ) if rpb.dim() == 3 else scores + rpb[:, biases][None]

    def av1d(attn, value, kernel, dilation):
        length = value.shape[2]
        neighbors, _ = tables(length, kernel, dilation, value.device)
        gathered = value[:, :, neighbors, :]
        return (attn.unsqueeze(-1) * gathered).sum(3)

    def qkrpb2d(query, key, rpb, kernel, dilation):
        # query/key: [B, heads, rows, columns, dim]
        rows, columns = query.shape[2], query.shape[3]
        row_neighbors, row_biases = tables(rows, kernel, dilation, query.device)
        column_neighbors, column_biases = tables(columns, kernel, dilation, query.device)
        gathered = key[:, :, row_neighbors, :, :][:, :, :, :, column_neighbors, :]
        # [B, heads, rows, K, columns, K, dim] -> [B, heads, rows, columns, K, K, dim]
        gathered = gathered.permute(0, 1, 2, 4, 3, 5, 6)
        scores = (query[:, :, :, :, None, None, :] * gathered).sum(-1)
        bias = rpb[:, row_biases[:, :, None, None], column_biases[None, None, :, :]]
        bias = bias.permute(0, 1, 3, 2, 4)[None]                            # [1, heads, rows, columns, K, K]
        return (scores + bias).reshape(*scores.shape[:4], kernel * kernel)

    def av2d(attn, value, kernel, dilation):
        rows, columns = value.shape[2], value.shape[3]
        row_neighbors, _ = tables(rows, kernel, dilation, value.device)
        column_neighbors, _ = tables(columns, kernel, dilation, value.device)
        gathered = value[:, :, row_neighbors, :, :][:, :, :, :, column_neighbors, :]
        gathered = gathered.permute(0, 1, 2, 4, 3, 5, 6).reshape(*value.shape[:4], kernel * kernel, -1)
        return (attn.unsqueeze(-1) * gathered).sum(4)

    module = types.ModuleType("natten.functional")
    module.natten1dqkrpb = qkrpb1d
    module.natten1dav = av1d
    module.natten2dqkrpb = qkrpb2d
    module.natten2dav = av2d
    return module


def _allin1_install_natten_shim(module):
    """Puts the stand-in where All-In-One's `dinat.py` looks for it. Installed only after the check
    above has read the real NATTEN, which this replaces."""
    import types
    package = types.ModuleType("natten")
    package.functional = module
    sys.modules["natten"] = package
    sys.modules["natten.functional"] = module


def _allin1_check_natten(shim):
    """Checks the stand-in's neighbor selection against the installed NATTEN kernel, with the bias
    zeroed. A mismatch means the edge rule moved and the record would be worthless."""
    try:
        from natten.backends import na1d_flex, na2d_flex
    except ImportError:
        print("NATTEN is not installed; the neighborhood attention stand-in is UNCHECKED")
        return
    torch.manual_seed(0)
    kernel = 5
    # The installed NATTEN runs this on torch's flex attention, which takes only power-of-two head
    # dimensions. The head dimension does not enter the neighbor rule, so the check runs at 16 while
    # the model itself runs at 12.
    def na1d(q, k, v, kernel_size, dilation, scale):
        return na1d_flex(q, k, v, kernel_size=(kernel_size,), stride=(1,), dilation=(dilation,),
                         is_causal=(False,), scale=scale)

    def na2d(q, k, v, kernel_size, dilation, scale):
        return na2d_flex(q, k, v, kernel_size=(kernel_size, kernel_size), stride=(1, 1),
                         dilation=(dilation, dilation), is_causal=(False, False), scale=scale)

    for dilation in (1, 2, 4):
        q = torch.randn(1, 41, 2, 16)                                       # [B, length, heads, dim]
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        reference = na1d(q, k, v, kernel_size=kernel, dilation=dilation, scale=1.0)
        scores = shim.natten1dqkrpb(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3),
                                    torch.zeros(2, 2 * kernel - 1), kernel, dilation)
        mine = shim.natten1dav(scores.softmax(-1), v.permute(0, 2, 1, 3), kernel, dilation)
        difference = (mine.permute(0, 2, 1, 3) - reference).abs().max().item()
        assert difference < 1e-4, f"1D neighborhood attention differs from NATTEN at dilation {dilation}: {difference}"
    q = torch.randn(1, 5, 41, 2, 16)                                        # [B, rows, columns, heads, dim]
    k, v = torch.randn_like(q), torch.randn_like(q)
    reference = na2d(q, k, v, kernel_size=kernel, dilation=1, scale=1.0)
    scores = shim.natten2dqkrpb(q.permute(0, 3, 1, 2, 4), k.permute(0, 3, 1, 2, 4),
                                torch.zeros(2, 2 * kernel - 1, 2 * kernel - 1), kernel, 1)
    mine = shim.natten2dav(scores.softmax(-1), v.permute(0, 3, 1, 2, 4), kernel, 1)
    difference = (mine.permute(0, 2, 3, 1, 4) - reference).abs().max().item()
    assert difference < 1e-4, f"2D neighborhood attention differs from NATTEN: {difference}"
    print("the neighborhood attention stand-in matches NATTEN's own kernel")


def allin1_stems(seconds=12.0, sample_rate=44100, seed=11):
    """Four deterministic stems with a beat in them: a kick on every beat, a bass note per beat, a
    chord bed, and a vocal-like tone that enters halfway, so the section head has a boundary to find."""
    total = int(seconds * sample_rate)
    t = np.arange(total, dtype=np.float32) / sample_rate
    generator = np.random.default_rng(seed)
    beat = 0.5                                                              # 120 BPM
    phase = np.mod(t, beat)

    kick = (np.exp(-phase * 40) * np.sin(2 * np.pi * 60 * phase)).astype(np.float32)
    hat = (np.exp(-np.mod(t, beat / 2) * 200) * generator.standard_normal(total) * 0.3).astype(np.float32)
    drums = 0.8 * kick + 0.2 * hat
    bass = (np.exp(-phase * 6) * np.sin(2 * np.pi * 55 * t)).astype(np.float32)
    other = sum(0.2 * np.sin(2 * np.pi * f * t) for f in (220.0, 277.2, 330.0)).astype(np.float32)
    vocals = np.where(t > seconds / 2, 0.4 * np.sin(2 * np.pi * (440 + 20 * np.sin(2 * np.pi * 5 * t)) * t), 0)

    stems = [bass, drums, other, vocals.astype(np.float32)]                 # the model's own order
    return [np.ascontiguousarray(s / max(1e-6, np.abs(s).max()) * 0.8, dtype=np.float32) for s in stems]


def run_allin1_training(image):
    """All-In-One's training path, from the authors' own sources, in three parts.

    `IK_ALLIN1_SRC` is the vendored package (config.py, models/, and training/ from
    mir-aidj/all-in-one at 18e7890) and `IK_TIMM_OPTIM_SRC` holds timm 0.9.16's `radam.py`.

    Targets: a fixed annotation (beats, downbeats, section boundaries and labels) over 500 frames
    goes through `training/data/eventconverters` and `widen_temporal_events` as `DatasetBase`
    applies them. Loss: `AllInOneTrainer.compute_losses`, taken from `trainer.py` as written and
    run on seeded logits against those targets with `Config`'s default weights, without importing
    the Lightning, timm, and madmom modules the rest of the file needs. Optimizer: twelve steps of
    timm's `RAdam` at the configuration's rate and weight decay over a weight in the decay group and
    a bias outside it, as `param_groups_weight_decay` splits them, from recorded gradients.
    """
    import ast
    import importlib.util
    import os
    import types

    source = os.environ.get("IK_ALLIN1_SRC", ".")
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    # config.py registers itself with hydra's ConfigStore, which the oracle does not need.
    for name in ["hydra", "hydra.core", "hydra.core.config_store"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    class _Store:
        @staticmethod
        def instance():
            return _Store()
        def store(self, *args, **kwargs):
            pass
    sys.modules["hydra.core.config_store"].ConfigStore = _Store
    config = load("allin1_training_config", os.path.join(source, "config.py"))
    cfg = config.Config()
    labels = list(config.HARMONIX_LABELS)

    utils = load("allin1_training_utils", os.path.join(source, "training", "data", "utils.py"))
    converters = load("allin1_training_converters",
                      os.path.join(source, "training", "data", "eventconverters", "eventconverters.py"))

    frames = 500
    # Every time is exact in float32, because the record stores floats that way: 1.23 lands on frame
    # 122 in float64 and on 123 once rounded to float32. The fractions still fall between samples and
    # between frames, so the truncation and the floor division are both exercised.
    beat_times = np.array([0.0, 0.50390625, 1.0, 1.4990234375, 2.0, 2.5, 3.0078125, 3.5, 4.0, 4.5,
                           4.998046875, 5.0])
    downbeat_times = np.array([0.0, 2.0, 4.0])
    section_times = np.array([0.0, 1.228515625, 3.0703125, 4.955078125])
    section_labels = ["start", "intro", "verse", "chorus", "end"]
    common = dict(segment_frames=frames, sr=cfg.sample_rate, hop=cfg.hop_size)
    beat = converters.BeatConverter(beat_times, **common)
    downbeat = converters.DownbeatConverter(downbeat_times, **common)
    section = converters.SectionConverter(section_times, section_labels, labels, beat_times, **common)
    true_beat = beat.of_frames(encode=True)
    true_downbeat = downbeat.of_frames(encode=True)
    true_section = section.of_frames(encode=True, return_labels=False)
    true_function = section.of_frames(encode=True, return_labels=True)
    widen_beat = utils.widen_temporal_events(true_beat, num_neighbors=1)
    widen_downbeat = utils.widen_temporal_events(true_downbeat, num_neighbors=1)
    widen_section = utils.widen_temporal_events(true_section, num_neighbors=2)

    trainer_path = os.path.join(source, "training", "trainer.py")
    tree = ast.parse(open(trainer_path).read())
    trainer_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AllInOneTrainer")
    method = next(node for node in trainer_class.body if isinstance(node, ast.FunctionDef) and node.name == "compute_losses")
    namespace = {"F": torch.nn.functional, "torch": torch, "Dict": dict, "AllInOneOutput": object,
                 "prefix_dict": lambda d, p: {p + k: v for k, v in d.items()}}
    exec(compile(ast.Module(body=[method], type_ignores=[]), trainer_path, "exec"), namespace)
    compute_losses = namespace["compute_losses"]

    generator = torch.Generator().manual_seed(17)
    outputs = types.SimpleNamespace(
        logits_beat=torch.randn(1, frames, generator=generator) * 2,
        logits_downbeat=torch.randn(1, frames, generator=generator) * 2,
        logits_section=torch.randn(1, frames, generator=generator) * 2,
        logits_function=torch.randn(1, len(labels), frames, generator=generator) * 2)
    batch = {"widen_true_beat": torch.from_numpy(widen_beat).unsqueeze(0).float(),
             "widen_true_downbeat": torch.from_numpy(widen_downbeat).unsqueeze(0).float(),
             "widen_true_section": torch.from_numpy(widen_section).unsqueeze(0).float(),
             "true_function": torch.from_numpy(np.asarray(true_function)).unsqueeze(0).long(),
             "mask": torch.ones(1, frames)}
    losses = compute_losses(types.SimpleNamespace(cfg=cfg), outputs, batch)

    radam = load("timm_radam", os.path.join(os.environ["IK_TIMM_OPTIM_SRC"], "radam.py"))
    weight = torch.nn.Parameter(torch.randn(3, 4, generator=generator))
    bias = torch.nn.Parameter(torch.randn(4, generator=generator))
    optimizer = radam.RAdam([{"params": [bias], "weight_decay": 0.0},
                             {"params": [weight], "weight_decay": cfg.weight_decay}], lr=cfg.lr)
    extra = {"radam_weight_start": weight.detach().clone(), "radam_bias_start": bias.detach().clone(),
             "radam_rate": torch.tensor([cfg.lr]), "radam_weight_decay": torch.tensor([cfg.weight_decay])}
    for step in range(12):
        weight.grad = torch.randn(3, 4, generator=generator)
        bias.grad = torch.randn(4, generator=generator)
        extra[f"radam_weight_grad_{step}"] = weight.grad.clone()
        extra[f"radam_bias_grad_{step}"] = bias.grad.clone()
        optimizer.step()
        extra[f"radam_weight_{step}"] = weight.detach().clone()
        extra[f"radam_bias_{step}"] = bias.detach().clone()

    extra.update({
        "beat_times": torch.from_numpy(beat_times).float(),
        "downbeat_times": torch.from_numpy(downbeat_times).float(),
        "section_times": torch.from_numpy(section_times).float(),
        "section_labels": torch.tensor([labels.index(label) for label in section_labels], dtype=torch.int32),
        "widen_true_beat": batch["widen_true_beat"][0],
        "widen_true_downbeat": batch["widen_true_downbeat"][0],
        "widen_true_section": batch["widen_true_section"][0],
        "true_function": batch["true_function"][0].to(torch.int32),
        "logits_beat": outputs.logits_beat[0], "logits_downbeat": outputs.logits_downbeat[0],
        "logits_section": outputs.logits_section[0], "logits_function": outputs.logits_function[0],
        "loss_beat": losses["loss_beat"].reshape(1), "loss_downbeat": losses["loss_downbeat"].reshape(1),
        "loss_section": losses["loss_section"].reshape(1), "loss_function": losses["loss_function"].reshape(1),
    })
    globals()["_extra"] = {key: value.contiguous() for key, value in extra.items()}
    return losses["loss"].reshape(1).contiguous()

def run_allin1(image, checkpoint):
    """All-In-One music structure analysis (mir-aidj/all-in-one) over four deterministic stems.

    `--checkpoint` is a released `.pth` (`taejunkim/allinone`). Set IK_ALLIN1_SRC to the directory
    holding the vendored `models/` and `postprocessing/` sources. Records the madmom spectrograms,
    the embedding, every block's output, the four logits, and the sections and beats the reference's
    own post-processing reads from them. Runs under the `allin1` oracle environment.
    """
    import os
    import types

    shim = _allin1_natten_shim()
    _allin1_check_natten(shim)
    _allin1_install_natten_shim(shim)

    source = os.environ.get("IK_ALLIN1_SRC", ".")
    sys.path.insert(0, source)
    from omegaconf import OmegaConf
    # `allinone.py` reaches for its package's config and typings modules; the checkpoint carries the
    # config, and the output type is a plain container, so both are supplied rather than imported.
    package = types.ModuleType("allin1_ref")
    package.__path__ = [source]
    sys.modules["allin1_ref"] = package
    config_module = types.ModuleType("allin1_ref.config")
    config_module.Config = object
    config_module.HARMONIX_LABELS = ["start", "end", "intro", "outro", "break", "bridge", "inst",
                                     "solo", "verse", "chorus"]
    typings_module = types.ModuleType("allin1_ref.typings")

    class AllInOneOutput(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.__dict__.update(kwargs)

    class Segment:
        def __init__(self, start, end, label):
            self.start, self.end, self.label = start, end, label

    typings_module.AllInOneOutput = AllInOneOutput
    typings_module.Segment = Segment
    sys.modules["allin1_ref.config"] = config_module
    sys.modules["allin1_ref.typings"] = typings_module

    import importlib.util

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, os.path.join(source, path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    package.config = config_module
    package.typings = typings_module
    load("allin1_ref.models", os.path.join("models", "__init__.py")) if False else None
    models_package = types.ModuleType("allin1_ref.models")
    models_package.__path__ = [os.path.join(source, "models")]
    sys.modules["allin1_ref.models"] = models_package
    load("allin1_ref.models.utils", os.path.join("models", "utils.py"))
    load("allin1_ref.models.dinat", os.path.join("models", "dinat.py"))
    allinone = load("allin1_ref.models.allinone", os.path.join("models", "allinone.py"))

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(state["config"])
    model = allinone.AllInOne(cfg).eval()
    model.load_state_dict(state["state_dict"])

    # The front end is madmom's, exactly as the reference's preprocessing builds it.
    from madmom.audio.signal import FramedSignalProcessor, Signal
    from madmom.audio.stft import ShortTimeFourierTransformProcessor
    from madmom.processors import SequentialProcessor
    from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor

    processor = SequentialProcessor([
        FramedSignalProcessor(frame_size=cfg.window_size, fps=cfg.fps),
        ShortTimeFourierTransformProcessor(),
        FilteredSpectrogramProcessor(num_bands=cfg.num_bands, fmin=cfg.fmin, fmax=cfg.fmax, norm_filters=True),
        LogarithmicSpectrogramProcessor(mul=1, add=1),
    ])
    stems = allin1_stems()
    spectrograms = np.stack([np.asarray(processor(Signal(stem, sample_rate=cfg.sample_rate, num_channels=1)))
                             for stem in stems])                            # [instruments, frames, bands]

    seams = {}
    handles = [model.embeddings.register_forward_hook(
        lambda _m, _i, o: seams.__setitem__("embeddings", o))]
    for index, layer in enumerate(model.encoder.layers):
        handles.append(layer.register_forward_hook(
            lambda _m, _i, o, index=index: seams.__setitem__(f"block{index}", o[0])))

    with torch.no_grad():
        logits = model(torch.from_numpy(spectrograms).unsqueeze(0))
    for handle in handles:
        handle.remove()

    functional = load("allin1_ref.postprocessing.helpers", os.path.join("postprocessing", "helpers.py"))
    sys.modules["allin1_ref.postprocessing"] = types.ModuleType("allin1_ref.postprocessing")
    sys.modules["allin1_ref.postprocessing"].__path__ = [os.path.join(source, "postprocessing")]
    sections_module = load("allin1_ref.postprocessing.functional", os.path.join("postprocessing", "functional.py"))
    metrical_module = load("allin1_ref.postprocessing.metrical", os.path.join("postprocessing", "metrical.py"))

    sections = sections_module.postprocess_functional_structure(logits, cfg)
    metrical = metrical_module.postprocess_metrical_structure(logits, cfg)

    # The decoded path itself, so a beat that lands one frame off can be traced to the state it came
    # from rather than guessed at. This repeats the reference's own activation assembly.
    beat_probability = torch.sigmoid(logits.logits_beat[0])
    downbeat_probability = torch.sigmoid(logits.logits_downbeat[0])
    off_beat = torch.maximum(torch.tensor(1e-8), beat_probability - downbeat_probability)
    neither = ((1 - beat_probability) + (1 - downbeat_probability)) / 2
    combined = torch.stack([off_beat, downbeat_probability, neither], dim=-1)
    combined = (combined / combined.sum(-1, keepdim=True)).numpy()
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor
    processor = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], threshold=None, fps=cfg.fps)
    paths = []
    for hmm in processor.hmms:
        path, probability = hmm.viterbi(combined[:, :2].astype(np.float32))
        paths.append((probability, hmm.transition_model.state_space.state_positions[path],
                      hmm.observation_model.pointers[path]))
    best_path = max(paths, key=lambda item: item[0])
    labels = config_module.HARMONIX_LABELS
    section_rows = np.array([[s.start, s.end, labels.index(s.label)] for s in sections], dtype=np.float32).reshape(-1, 3)
    beat_rows = np.array([[time, position] for time, position
                          in zip(metrical["beats"], metrical["beat_positions"])], dtype=np.float32).reshape(-1, 2)

    globals()["_extra"] = {
        "stems": torch.from_numpy(np.stack(stems)),
        "spectrograms": torch.from_numpy(spectrograms),
        "embeddings": seams["embeddings"].contiguous(),
        "block0": seams["block0"].contiguous(),
        "block5": seams["block5"].contiguous(),
        "block10": seams["block10"].contiguous(),
        "logits_beat": logits.logits_beat.contiguous(),
        "logits_downbeat": logits.logits_downbeat.contiguous(),
        "logits_function": logits.logits_function.contiguous(),
        "sections": torch.from_numpy(section_rows),
        "beats": torch.from_numpy(beat_rows),
        "dbn_activations": torch.from_numpy(combined.astype(np.float32)),
        "dbn_positions": torch.from_numpy(best_path[1].astype(np.float32)),
        "dbn_pointers": torch.from_numpy(best_path[2].astype(np.int32)),
        "dbn_probability": torch.tensor([float(best_path[0])]),
    }
    return logits.logits_section.contiguous()


def run_gemma4_shared_kv(image):
    """The Gemma 4 decoder with keys and values SHARED on its full-attention layers
    (`attention_k_eq_v`), at a tiny random configuration, from transformers' own Gemma4ForCausalLM.

    The 26B-A4B, the 12B unified, and the 31B releases all set the flag, and none of them fits this
    machine. Two behaviors ride on it, and neither one appears in a release that leaves it off: a
    full-attention layer carries no `v_proj` and takes its value from the key projection, read
    BEFORE the key norm and the rotary; and a full-attention layer runs
    `num_global_key_value_heads` in place of `num_key_value_heads`. The two counts are set apart
    here (2 sliding, 1 full) so that a net ignoring the override builds the wrong shape rather than
    the right one by coincidence. Every hidden state is recorded so a divergence lands on a layer.
    """
    from transformers import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    config = Gemma4TextConfig(
        hidden_size=64, num_hidden_layers=2, vocab_size=131, num_attention_heads=4,
        num_key_value_heads=2, num_global_key_value_heads=1, attention_k_eq_v=True,
        head_dim=16, global_head_dim=32, intermediate_size=96,
        hidden_size_per_layer_input=0, vocab_size_per_layer_input=131,
        num_kv_shared_layers=0, sliding_window=64,
        layer_types=["sliding_attention", "full_attention"], rms_norm_eps=1e-6,
        max_position_embeddings=64, tie_word_embeddings=True,
        hidden_activation="gelu_pytorch_tanh", final_logit_softcapping=None)
    model = _randomized(Gemma4ForCausalLM(config), seed=23)
    tokens = torch.tensor([[3, 17, 42, 99, 7, 61, 12, 5]], dtype=torch.long)
    with torch.no_grad():
        out = model(tokens, output_hidden_states=True)
    extra = {"tokens": tokens[0].to(torch.int32).contiguous()}
    for index, hidden in enumerate(out.hidden_states):
        extra[f"hidden.{index}"] = hidden[0].float().contiguous()
    # The full-attention layer has no v_proj to save, which is what the record has to show.
    for key, value in model.state_dict().items():
        if key == "lm_head.weight":
            continue
        extra[f"w::{key}"] = (value.float() if value.is_floating_point() else value).contiguous()
    globals()["_extra"] = extra
    return out.logits[0].float().contiguous()


def muscriptor_clip(seconds=1.0, sample_rate=16000, seed=17):
    """A deterministic clip of struck notes.

    The clip opens with silence and each note has a hard attack and a decay, so the model has ONSETS
    to transcribe. A tone that is already sounding at t=0 reads as sustained from before the window,
    which the model reports as a tie section with no note-on events — true to the audio, and a
    degenerate thing to measure a decoder against."""
    total = int(seconds * sample_rate)
    t = np.arange(total, dtype=np.float32) / sample_rate
    generator = np.random.default_rng(seed)

    def struck(frequency, onset, decay=2.5):
        since = t - onset
        envelope = np.where(since >= 0, np.exp(-np.maximum(since, 0) * decay), 0.0)
        partials = sum((0.6 ** k) * np.sin(2 * np.pi * frequency * (k + 1) * t) for k in range(3))
        return (envelope * partials).astype(np.float32)

    # A C major triad struck together, then a melody note, then the triad again.
    wave = struck(261.63, 0.25) + struck(329.63, 0.25) + struck(392.0, 0.25)
    if seconds > 1.5:
        wave = wave + struck(523.25, 1.6) + struck(261.63, 3.0) + struck(329.63, 3.0)
    wave = wave + 0.002 * generator.standard_normal(total).astype(np.float32)
    return np.ascontiguousarray(wave / np.abs(wave).max() * 0.8, dtype=np.float32)


def run_muscriptor(image):
    """MuScriptor (muscriptor/muscriptor) at a TINY random configuration, seam by seam.

    The released weights are CC BY-NC 4.0 behind a gated repository, and the inference code is MIT
    with the released geometry in it, so the architecture is measured here the way the SD3 and FLUX
    transformers were: a small random model, built by the reference's own `_build_model`, whose
    weights ride in the record under `w::` so both sides run identical parameters.

    Records the mel magnitudes, the log mel, the conditioning prefix with and without an instrument
    class, the prefill logits, and a short greedy continuation. Runs under the `muscriptor` oracle
    environment. `image` unused.
    """
    import torch as _torch
    from muscriptor.transcription_model import _build_model, _ModelConfig, _SAMPLE_RATE
    from muscriptor.modules.conditioners import WavCondition

    _torch.manual_seed(20260921)
    cfg = _ModelConfig(dim=64, num_heads=4, num_layers=2, card=1395)
    model = _build_model(_torch.device("cpu"), cfg).eval()

    wave = muscriptor_clip()
    wav = _torch.from_numpy(wave).view(1, 1, -1)
    condition = WavCondition(wav=wav, length=_torch.tensor([wav.shape[-1]]),
                             sample_rate=[_SAMPLE_RATE], path=[None], seek_time=[None])

    conditioners = model.condition_provider.conditioners
    mel_conditioner = conditioners["self_wav"]
    with _torch.no_grad():
        mel = mel_conditioner.mel_spec_transform(wav)                        # [1, 1, mels, frames]
        log_mel = _torch.log(mel.squeeze(1).transpose(1, 2) + mel_conditioner.eps)
        mel_embed, _ = mel_conditioner(condition)

        # The prefix the LM actually sees. The provider's dict orders the class conditions first and
        # the wav last, and `forward` PREPENDS each in turn, so the wav ends up first.
        instrument = conditioners["instrument_group"]
        dataset = conditioners["dataset_name"]
        unspecified = instrument(instrument.tokenize([None]))[0]
        dataset_embed = dataset(dataset.tokenize([None]))[0]
        specified = instrument(instrument.tokenize(["5"]))[0]

        tokens = _torch.tensor([[model.initial_token_id]], dtype=_torch.long)
        condition_tensors = {
            "instrument_group": (unspecified, _torch.ones(unspecified.shape[:2])),
            "dataset_name": (dataset_embed, _torch.ones(dataset_embed.shape[:2])),
            "self_wav": (mel_embed, _torch.ones(mel_embed.shape[:2])),
        }
        logits = model(tokens, condition_tensors, first_step=True, model_state=None)

        # A short greedy continuation, recomputed from scratch each step so the record does not
        # depend on the reference's KV cache.
        sequence = tokens
        generated = []
        for _ in range(8):
            step_logits = model(sequence, condition_tensors, first_step=True, model_state=None)
            scores = step_logits[:, -1, :].float()
            scores[:, 1393:] = -float("inf")
            nxt = int(scores.argmax(dim=-1)[0])
            generated.append(nxt)
            sequence = _torch.cat([sequence, _torch.tensor([[nxt]], dtype=_torch.long)], dim=1)

    extra = {
        "audio": _torch.from_numpy(wave),
        "mel": mel.squeeze(0).squeeze(0).contiguous(),                        # [mels, frames]
        "log_mel": log_mel[0].contiguous(),                                   # [frames, mels]
        "mel_embed": mel_embed[0].contiguous(),                               # [frames, dim]
        "instrument_unspecified": unspecified[0].contiguous(),
        "instrument_specified": specified[0].contiguous(),
        "dataset_embed": dataset_embed[0].contiguous(),
        "tokens": _torch.tensor(generated, dtype=_torch.int32),
    }
    for name, tensor in model.state_dict().items():
        extra["w::" + name] = tensor.contiguous().float()
    globals()["_extra"] = extra
    return logits[0].contiguous()


def run_muscriptor_real(image, checkpoint):
    """MuScriptor on the RELEASED weights, seam by seam.

    `--checkpoint` is a downloaded release directory (`config.json` + `model.safetensors`) or the
    safetensors itself beside its config. The weights are CC BY-NC 4.0 behind a gated repository, so
    this mode runs only where a token has accepted the license; the tiny-configuration mode measures
    the architecture without them.

    Records the mel, the conditioning prefix, the prefill logits, and a greedy continuation long
    enough to contain note events. Runs under the `muscriptor` oracle environment.
    """
    import json
    import os
    import torch as _torch
    from pathlib import Path
    from muscriptor.transcription_model import (
        _build_model, _ModelConfig, _SAMPLE_RATE, _remap_single_codebook_keys,
    )
    from muscriptor.modules.conditioners import WavCondition
    from safetensors.torch import load_file

    path = Path(checkpoint)
    directory = path if path.is_dir() else path.parent
    weights = path if path.is_file() else directory / "model.safetensors"
    config = json.loads((directory / "config.json").read_text())
    cfg = _ModelConfig(dim=config["dim"], num_heads=config["num_heads"],
                       num_layers=config["num_layers"], card=config["card"])

    model = _build_model(_torch.device("cpu"), cfg).eval()
    # The release stores the embedding and the head as the first entry of a module list, which the
    # reference's own loader flattens before it loads.
    state = _remap_single_codebook_keys(load_file(str(weights)))
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The mel window and filterbank are buffers the release carries; anything else missing is a
    # mismatch worth failing on rather than measuring around.
    assert not unexpected, f"unexpected keys in the release: {sorted(unexpected)[:5]}"
    assert not missing, f"the release does not fill: {sorted(missing)[:5]}"

    wave = muscriptor_clip(seconds=5.0)
    wav = _torch.from_numpy(wave).view(1, 1, -1)
    condition = WavCondition(wav=wav, length=_torch.tensor([wav.shape[-1]]),
                             sample_rate=[_SAMPLE_RATE], path=[None], seek_time=[None])

    conditioners = model.condition_provider.conditioners
    mel_conditioner = conditioners["self_wav"]
    with _torch.no_grad():
        mel = mel_conditioner.mel_spec_transform(wav)
        mel_embed, _ = mel_conditioner(condition)
        instrument = conditioners["instrument_group"]
        dataset = conditioners["dataset_name"]
        instrument_embed = instrument(instrument.tokenize([None]))[0]
        dataset_embed = dataset(dataset.tokenize([None]))[0]
        condition_tensors = {
            "instrument_group": (instrument_embed, _torch.ones(instrument_embed.shape[:2])),
            "dataset_name": (dataset_embed, _torch.ones(dataset_embed.shape[:2])),
            "self_wav": (mel_embed, _torch.ones(mel_embed.shape[:2])),
        }

        tokens = _torch.tensor([[model.initial_token_id]], dtype=_torch.long)
        logits = model(tokens, condition_tensors, first_step=True, model_state=None)

        sequence = tokens
        generated = []
        for _ in range(int(os.environ.get("IK_MUSCRIPTOR_STEPS", "24"))):
            step_logits = model(sequence, condition_tensors, first_step=True, model_state=None)
            scores = step_logits[:, -1, :].float()
            scores[:, 1393:] = -float("inf")
            nxt = int(scores.argmax(dim=-1)[0])
            generated.append(nxt)
            if nxt == 1:                                                  # EOS
                break
            sequence = _torch.cat([sequence, _torch.tensor([[nxt]], dtype=_torch.long)], dim=1)

    # The reference's own decode state machine over those tokens, so the note path is compared and
    # not only the token stream.
    from muscriptor.events import ChunkBoundary, OpenNoteTracker, _StartNote, _EndNote, _DrumHit
    from muscriptor.tokenizer.mt3 import MT3Tokenizer

    tracker = OpenNoteTracker(MT3Tokenizer()._vocab, frame_rate=100)
    actions = list(tracker.feed(ChunkBoundary(seek_time=0.0, next_seek_time=None)))
    for token in generated:
        if token == 1:
            break
        actions.extend(tracker.feed(token))
    actions.extend(tracker.finish())

    rows = []
    for action in actions:
        if isinstance(action, _StartNote):
            rows.append([0, action.program, action.pitch, action.time])
        elif isinstance(action, _EndNote):
            rows.append([1, action.program, action.pitch, action.time])
        elif isinstance(action, _DrumHit):
            rows.append([2, 128, action.pitch, action.time])

    globals()["_extra"] = {
        "audio": _torch.from_numpy(wave),
        "mel": mel.squeeze(0).squeeze(0).contiguous(),
        "mel_embed": mel_embed[0].contiguous(),
        "prefix": _torch.cat([mel_embed, dataset_embed, instrument_embed], dim=1)[0].contiguous(),
        "tokens": _torch.tensor(generated, dtype=_torch.int32),
        "note_actions": _torch.tensor(rows, dtype=_torch.float32).reshape(-1, 4),
    }
    return logits[0].contiguous()


def run_hft(image, checkpoint):
    """hFT-Transformer (sony/hFT-Transformer, MIT) on the released MAESTRO weights, seam by seam.

    `--checkpoint` is the released `model_016_003.pkl`. Set IK_HFT_SRC to a directory holding
    `model/model_spec2midi.py` and `model/amt.py`: the release is a plain pickle of the live module,
    so its classes must be importable, and it was saved on CUDA, so the nested storage load is
    patched onto the CPU.

    Records the log-mel feature, the first segment's input, the encoder output, both output levels
    with the decoder's note-to-frequency attention, the stitched posteriorgrams, and the notes the
    reference's own `mpe2note` reads from them. Runs under the `hft` oracle environment.
    """
    import io
    import json
    import os
    import pickle
    import sys
    import tempfile
    import wave

    import torch as _torch
    import torch.storage as _torch_storage

    source = os.environ.get("IK_HFT_SRC", ".")
    sys.path.insert(0, source)
    _torch_storage._load_from_bytes = lambda data: _torch.load(io.BytesIO(data), map_location="cpu",
                                                               weights_only=False)
    # torchaudio 2.11 routes `load` through torchcodec, which is not installed here; the reference's
    # feature code is unchanged, only the file read is substituted.
    import torchaudio as _torchaudio

    def _load_wav(path, *args, **kwargs):
        with wave.open(path, "rb") as handle:
            frames = handle.readframes(handle.getnframes())
            data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
            return _torch.from_numpy(data.copy()).unsqueeze(0), handle.getframerate()

    _torchaudio.load = _load_wav
    from model import amt as amt_module

    config = {
        "feature": {"sr": 16000, "hop_sample": 256, "mel_bins": 256, "n_bins": 256, "fft_bins": 2048,
                    "window_length": 2048, "log_offset": 1e-8, "window": "hann", "pad_mode": "constant"},
        "input": {"margin_b": 32, "margin_f": 32, "num_frame": 128,
                  "min_value": float(np.log(1e-8))},
        "midi": {"note_min": 21, "note_max": 108, "num_note": 88, "num_velocity": 128},
    }

    amt = amt_module.AMT(config, checkpoint, batch_size=1)
    # The pickled module remembers the CUDA device it was trained on and builds its position indices
    # there; the weights are already on the CPU, so only the recorded device needs correcting.
    for part in (amt.model.encoder_spec2midi, amt.model.decoder_spec2midi):
        part.device = "cpu"
        for name in ("scale_freq", "scale_time"):
            if hasattr(part, name):
                setattr(part, name, getattr(part, name).cpu())

    # A deterministic piano-like clip: struck notes with harmonics and decay, long enough to cross a
    # segment boundary (4 seconds is 250 frames, so two 128-frame segments).
    sample_rate = 16000
    seconds = 4.0
    total = int(seconds * sample_rate)
    t = np.arange(total, dtype=np.float32) / sample_rate
    generator = np.random.default_rng(23)

    def struck(midi_pitch, onset, decay=3.0):
        frequency = 440.0 * 2.0 ** ((midi_pitch - 69) / 12.0)
        since = t - onset
        envelope = np.where(since >= 0, np.exp(-np.maximum(since, 0) * decay), 0.0)
        partials = sum((0.55 ** k) * np.sin(2 * np.pi * frequency * (k + 1) * t) for k in range(4))
        return (envelope * partials).astype(np.float32)

    wave_data = (struck(60, 0.2) + struck(64, 0.2) + struck(67, 0.2)
                 + struck(72, 1.2) + struck(55, 2.1) + struck(60, 2.1) + struck(64, 3.0))
    wave_data = wave_data + 0.001 * generator.standard_normal(total).astype(np.float32)
    wave_data = np.ascontiguousarray(wave_data / np.abs(wave_data).max() * 0.85, dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        path = handle.name
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((np.clip(wave_data, -1, 1) * 32767).astype("<i2").tobytes())

    feature = np.array(amt.wav2feature(path), dtype=np.float32)
    # The reference reads a 16-bit file, so the samples it actually saw are quantized. Recording the
    # pre-quantization floats would make both sides run different audio and read as a front-end
    # divergence of about 1e-5.
    loaded, _ = _load_wav(path)
    wave_data = loaded[0].numpy().astype(np.float32)
    os.unlink(path)

    # One segment through the model, the way `transcript` feeds it.
    padded = np.concatenate([
        np.full([config["input"]["margin_b"], config["feature"]["n_bins"]], config["input"]["min_value"], dtype=np.float32),
        feature,
        np.full([config["input"]["margin_f"] + 128, config["feature"]["n_bins"]], config["input"]["min_value"], dtype=np.float32),
    ], axis=0)
    segment = _torch.from_numpy(padded[0:32 + 128 + 32]).T.unsqueeze(0)

    with _torch.no_grad():
        encoded = amt.model.encoder_spec2midi(segment)
        outputs = amt.model(segment)
    (onset_f, offset_f, mpe_f, velocity_f, attention, onset_t, offset_t, mpe_t, velocity_t) = outputs

    # The whole clip, then the notes the reference reads from it.
    a_onset, a_offset, a_mpe, a_velocity, b_onset, b_offset, b_mpe, b_velocity = amt.transcript(feature)
    notes = amt.mpe2note(a_onset=b_onset, a_offset=b_offset, a_mpe=b_mpe, a_velocity=b_velocity)
    rows = np.array([[n["pitch"], n["onset"], n["offset"], n["velocity"]] for n in notes],
                    dtype=np.float32).reshape(-1, 4)

    globals()["_extra"] = {
        "audio": _torch.from_numpy(wave_data),
        "feature": _torch.from_numpy(feature),
        "segment": segment[0].contiguous(),
        "encoded": encoded[0].contiguous(),
        "onset_freq": onset_f[0].contiguous(),
        "offset_freq": offset_f[0].contiguous(),
        "mpe_freq": mpe_f[0].contiguous(),
        "velocity_freq": velocity_f[0].contiguous(),
        "attention": attention[0].contiguous(),
        "offset_time": offset_t[0].contiguous(),
        "mpe_time": mpe_t[0].contiguous(),
        "velocity_time": velocity_t[0].contiguous(),
        "clip_onset": _torch.from_numpy(b_onset),
        "clip_offset": _torch.from_numpy(b_offset),
        "clip_mpe": _torch.from_numpy(b_mpe),
        "clip_velocity": _torch.from_numpy(b_velocity.astype(np.int32)),
        "notes": _torch.from_numpy(rows),
    }
    return onset_t[0].contiguous()


def run_chatterbox_mtl_tokens(image, checkpoint):
    """The multilingual Chatterbox TEXT layer: `MTLTokenizer` over the released
    `grapheme_mtl_merged_expanded_v1.json`, recording the token ids for one line per language.

    `--checkpoint` is the unpacked multilingual release directory. The reference reaches for an
    OPTIONAL package for four languages and passes the text through unchanged when one is absent:
    pykakasi for Japanese, spacy_pkuseg for Chinese segmentation, dicta_onnx for Hebrew, and
    russian_text_stresser for Russian. The first two are installed in this environment and the last
    two are not, so this record disables the first two explicitly: the port implements the fallback
    path for all four, and a record taken with them active would measure a text layer the port does
    not have. Chinese Cangjie encoding and Korean Jamo decomposition ARE ported and stay active.
    Runs under the `chatterbox` oracle env.
    """
    import chatterbox.models.tokenizers.tokenizer as tk
    from chatterbox.models.tokenizers import MTLTokenizer
    from chatterbox.mtl_tts import punc_norm

    tokenizer = MTLTokenizer(os.path.join(checkpoint, "grapheme_mtl_merged_expanded_v1.json"))
    tokenizer.cangjie_converter.segmenter = None
    tk.hiragana_normalize = lambda text: text

    cases = [
        ("en", "The quick brown fox jumps over the lazy dog."),
        ("fr", "Le renard brun rapide saute par-dessus le chien paresseux."),
        ("de", "Gr\u00f6\u00dfe und Wei\u00df, sagte er."),
        ("es", "El veloz zorro marr\u00f3n salta sobre el perro perezoso."),
        ("ru", "\u0411\u044b\u0441\u0442\u0440\u0430\u044f \u043b\u0438\u0441\u0430."),
        ("he", "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd."),
        ("ko", "\uc548\ub155\ud558\uc138\uc694 \uc138\uacc4."),
        ("ja", "\u3053\u3093\u306b\u3061\u306f\u4e16\u754c\u3002"),
        ("zh", "\u5feb\u901f\u7684\u68d5\u8272\u72d0\u72f8\u3002"),
    ]
    extra = {}
    for language, text in cases:
        ids = tokenizer.text_to_tokens(punc_norm(text), language_id=language)[0].tolist()
        extra[f"tokens.{language}"] = torch.tensor(ids, dtype=torch.int32)
        print(f"chatterbox mtl {language}: {len(ids)} tokens")
    globals()["_extra"] = extra
    return torch.tensor([float(len(cases))])


def run_chatterbox_mtl_t3(image, checkpoint):
    """The MULTILINGUAL T3 on the released `t3_mtl23ls_v3.safetensors`, teacher-forced so nothing is
    sampled.

    `--checkpoint` is the unpacked multilingual release directory. The network is the English T3's
    graph at `T3Config.multilingual()`, whose only difference is the text embedding's width (2454
    against 704), so this measures the release rather than the architecture. Recorded: the text tokens
    the multilingual tokenizer produces, the conditioning the reference builds from the validation
    clip (`speaker_emb`, `cond_tokens`, `cond_emb`), and the speech logits over a FIXED speech-token
    sequence for both classifier-free-guidance rows (`tf_logits`). Runs under the `chatterbox` env.
    """
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    from chatterbox.models.tokenizers import MTLTokenizer
    from chatterbox.models.voice_encoder import VoiceEncoder
    from chatterbox.models.s3tokenizer import S3Tokenizer, S3_SR
    from chatterbox.mtl_tts import punc_norm
    import chatterbox.models.tokenizers.tokenizer as tk

    directory = checkpoint
    t3 = T3(T3Config.multilingual())
    t3.load_state_dict(load_file(os.path.join(directory, "t3_mtl23ls_v3.safetensors")))
    t3.eval()
    ve = VoiceEncoder()
    ve.load_state_dict(load_file(os.path.join(directory, "ve.safetensors")))
    ve.eval()
    s3gen_state = load_file(os.path.join(directory, "s3gen_v3.safetensors"))
    speech_tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
    speech_tokenizer.load_state_dict(
        {k[len("tokenizer."):]: v for k, v in s3gen_state.items() if k.startswith("tokenizer.")},
        strict=False)
    speech_tokenizer.eval()
    del s3gen_state

    text_tokenizer = MTLTokenizer(os.path.join(directory, "grapheme_mtl_merged_expanded_v1.json"))
    text_tokenizer.cangjie_converter.segmenter = None
    tk.hiragana_normalize = lambda text: text

    wav24, wav16 = _chatterbox_reference_audio()
    with torch.inference_mode():
        ve_embed = torch.from_numpy(ve.embeds_from_wavs([wav16], sample_rate=S3_SR)).mean(axis=0, keepdim=True)
        cond_tokens, _ = speech_tokenizer.forward([wav16[: 6 * S3_SR]], max_len=t3.hp.speech_cond_prompt_len)
    cond_tokens = torch.atleast_2d(cond_tokens)
    t3_cond = T3Cond(speaker_emb=ve_embed, cond_prompt_speech_tokens=cond_tokens,
                     emotion_adv=0.5 * torch.ones(1, 1, 1))

    text = punc_norm("The quick brown fox jumps over the lazy dog.")
    text_tokens = text_tokenizer.text_to_tokens(text, language_id="en")
    text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
    text_tokens = F.pad(text_tokens, (1, 0), value=t3.hp.start_text_token)
    text_tokens = F.pad(text_tokens, (0, 1), value=t3.hp.stop_text_token).long()

    # A fixed speech prefix, so the seam is teacher-forced and carries no sampling.
    forced = torch.tensor([[t3.hp.start_speech_token, 137, 2048, 511, 4096]], dtype=torch.long)
    forced = torch.cat([forced, forced], dim=0)
    with torch.inference_mode():
        cond_emb = t3.prepare_conditioning(t3_cond)
        embeds, _ = t3.prepare_input_embeds(t3_cond=t3_cond, text_tokens=text_tokens,
                                            speech_tokens=forced, cfg_weight=0.5)
        out = t3.tfmr(inputs_embeds=embeds, output_hidden_states=True)
        logits = t3.speech_head(out.hidden_states[-1])

    globals()["_extra"] = {
        "text_tokens": text_tokens[0].to(torch.int32).contiguous(),
        "speaker_emb": ve_embed[0].float().contiguous(),
        "cond_tokens": cond_tokens[0].to(torch.int32).contiguous(),
        "cond_emb": cond_emb[0].float().contiguous(),
        "forced": forced[0].to(torch.int32).contiguous(),
    }
    print(f"chatterbox mtl t3: {text_tokens.shape[1]} text tokens, cond {cond_emb.shape[1]}, "
          f"logits {tuple(logits.shape)}")
    return logits.float().contiguous()


def run_rf_detr_seg(image, checkpoint):
    """RF-DETR INSTANCE SEGMENTATION on released weights, from transformers'
    `RfDetrForInstanceSegmentation` and its image processor.

    `--checkpoint` is a `Roboflow/rf-detr-seg-*` snapshot directory. The detector underneath is the
    one already at parity here, so this measures the mask head: the projector's output resampled to a
    quarter of the input, one ConvNeXt-style block per decoder layer, and that layer's queries
    projected and multiplied against the block's output. Records the preprocessed pixel values (so the
    port runs on the identical input), the projector output the head reads, every decoder layer's
    queries, and the masks of every layer. The reference's prediction is the last layer's.
    Runs under the `rfdetr` oracle env.
    """
    from transformers import RfDetrForInstanceSegmentation, AutoImageProcessor
    from PIL import Image

    model = RfDetrForInstanceSegmentation.from_pretrained(checkpoint, torch_dtype=torch.float32).eval()
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    pil = Image.fromarray((image * 255).astype(np.uint8)) if image.dtype != np.uint8 else Image.fromarray(image)
    pixel_values = processor(images=pil, return_tensors="pt")["pixel_values"]

    with torch.no_grad():
        base = model.model.model(pixel_values)
        image_size = pixel_values.shape[-2:]
        masks = model.segmentation_head(base.backbone_features, base.intermediate_hidden_states, image_size)
        out = model(pixel_values)

    extra = {
        "pixels": pixel_values[0].permute(1, 2, 0).contiguous(),
        "proj": base.backbone_features[0].permute(1, 2, 0).contiguous(),
        "pred_boxes": out.pred_boxes[0].contiguous(),
        "logits": out.logits[0].contiguous(),
    }
    for layer, queries in enumerate(base.intermediate_hidden_states):
        extra[f"queries.{layer}"] = queries[0].contiguous()
    for layer, layer_masks in enumerate(masks):
        extra[f"masks.{layer}"] = layer_masks[0].clone().contiguous()
    globals()["_extra"] = extra
    print(f"rf-detr seg: pixels {tuple(pixel_values.shape)}, proj {tuple(base.backbone_features.shape)}, "
          f"{len(masks)} layers, masks {tuple(masks[-1].shape)}")
    return masks[-1][0].contiguous()


def run_chronos(image):
    """Amazon Chronos-Bolt (amazon/chronos-bolt-base, Apache-2.0) time-series forecaster: a patched T5
    encoder-decoder emitting quantile forecasts. Loads from the Hub (no --checkpoint). Records the
    instance-norm scale, the patched input embeddings, the encoder and (1-token) decoder outputs, and
    the quantile forecast before and after un-scaling."""
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base", device_map="cpu", torch_dtype=torch.float32)
    m = pipe.model.eval()
    torch.manual_seed(0)
    n = 512
    t = torch.arange(n, dtype=torch.float32)
    ctx = (0.01 * t + torch.sin(2 * torch.pi * t / 24) + 0.3 * torch.sin(2 * torch.pi * t / 168)
           + 0.1 * torch.randn(n)).unsqueeze(0)                                  # [1, 512]
    with torch.no_grad():
        hidden, loc_scale, input_embeds, attention_mask = m.encode(context=ctx)
        seq = m.decode(input_embeds, attention_mask, hidden)                     # [1, 1, d_model]
        pl, nq = m.chronos_config.prediction_length, m.num_quantiles
        qp_scaled = m.output_patch_embedding(seq).view(1, nq, pl)
        qp = m.instance_norm.inverse(qp_scaled.view(1, -1), loc_scale).view(1, nq, pl)
        preds = m(context=ctx).quantile_preds                                    # [1, 9, 64]
    loc, scale = loc_scale
    globals()["_extra"] = {
        "context": ctx[0].contiguous(),                       # [512]
        "loc": loc.reshape(-1).contiguous(),                  # [1]
        "scale": scale.reshape(-1).contiguous(),              # [1]
        "input_embeds": input_embeds[0].contiguous(),         # [nP+1, d_model]
        "attention_mask": attention_mask[0].to(torch.float32).contiguous(),
        "encoder_hidden": hidden[0].contiguous(),
        "decoder_out": seq[0].contiguous(),                   # [1, d_model]
        "quantile_preds_scaled": qp_scaled[0].contiguous(),   # [9, 64]
    }
    return preds[0].contiguous()                              # [9, 64]  (recorded as "output")


def run_mimi(image):
    """Kyutai Mimi (kyutai/mimi, CC-BY-4.0) neural codec: audio -> codes -> audio. Loads from the Hub
    (no --checkpoint). Records every seam -- encoder, encoder transformer, downsample, codes, quantizer
    decode, upsample, decoder transformer, and the reconstructed waveform -- plus the public encode/decode."""
    from transformers import MimiModel
    m = MimiModel.from_pretrained("kyutai/mimi").eval()
    torch.manual_seed(0)
    n = 24000
    t = torch.arange(n, dtype=torch.float32) / 24000.0
    x = (0.5 * torch.sin(2 * torch.pi * 220 * t) + 0.3 * torch.sin(2 * torch.pi * 440 * t)
         + 0.1 * torch.randn(n)).unsqueeze(0).unsqueeze(0)                       # [1, 1, N]
    with torch.no_grad():
        emb = m.encoder(x)                                                       # [1, 512, T@25]
        et = m.encoder_transformer(emb.transpose(1, 2))[0].transpose(1, 2)
        ds = m.downsample(et)                                                    # [1, 512, T@12.5]
        codes = m.quantizer.encode(ds)                                          # [K, 1, T]
        codes_bkt = codes.transpose(0, 1)                                        # [1, K, T]
        dq = m.quantizer.decode(codes_bkt)                                       # [1, 512, T@12.5]
        us = m.upsample(dq)                                                      # [1, 512, T@25]
        dt = m.decoder_transformer(us.transpose(1, 2))[0].transpose(1, 2)
        wav = m.decoder(dt)                                                      # [1, 1, samples]
        full_codes = m.encode(x).audio_codes                                     # [1, K, T]
        full_wav = m.decode(full_codes).audio_values                             # [1, 1, samples]
    globals()["_extra"] = {
        "audio": x[0].contiguous(),                          # [1, N]
        "encoder_out": emb[0].contiguous(),                  # [512, T@25]
        "transformer_out": et[0].contiguous(),
        "downsample_out": ds[0].contiguous(),                # [512, T@12.5]
        "codes": codes_bkt[0].to(torch.int32).contiguous(),  # [K, T]
        "quant_decode": dq[0].contiguous(),
        "upsample_out": us[0].contiguous(),
        "dtransformer_out": dt[0].contiguous(),
        "full_codes": full_codes[0].to(torch.int32).contiguous(),
        "full_waveform": full_wav[0, 0].contiguous(),
    }
    return wav[0, 0].contiguous()


def _qwen3vl_retrieval_model(checkpoint):
    """The released Qwen3-VL retrieval checkpoint and its processor, at float32."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(checkpoint, padding_side="right")
    model = AutoModelForImageTextToText.from_pretrained(checkpoint, dtype=torch.float32).eval()
    return model, processor


def run_qwen3vl_embedding(image, checkpoint):
    """Qwen3-VL-Embedding, from the release's own `Qwen3VLForEmbedding` recipe.

    `--checkpoint` is the released `Qwen/Qwen3-VL-Embedding-2B` directory. The reference class wraps
    `Qwen3VLModel` (the backbone with no output projection), so the embedding is the last position's
    hidden state, L2-normalized. The instruction goes in a system turn and the content in a user turn,
    and the chat template's generation prompt ends the sequence, which is the position that is pooled.

    The record carries two cases: a text-only query under the default instruction, and an image with
    text beside it. Each case records its input ids, the full hidden states, and the pooled embedding;
    the image case adds the processor's pixel values and grid plus the vision tower's output and its
    three deepstack features, so a mismatch localizes to a seam rather than to the pair.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    model, processor = _qwen3vl_retrieval_model(checkpoint)
    backbone = model.model
    instruction = "Represent the user's input."
    extra = {}

    def conversation(text, has_image):
        content = []
        if has_image:
            content.append({"type": "image"})
        if text:
            content.append({"type": "text", "text": text})
        return [{"role": "system", "content": [{"type": "text", "text": instruction}]},
                {"role": "user", "content": content}]

    # The text-only case.
    prompt = processor.apply_chat_template(conversation("A photograph of a red bicycle.", False),
                                           add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[prompt], return_tensors="pt")
    with torch.no_grad():
        hidden = backbone(**inputs).last_hidden_state
    extra["text_prompt_ids"] = inputs["input_ids"][0].to(torch.int32).contiguous()
    extra["text_hidden"] = hidden[0].contiguous()
    extra["text_embedding"] = F.normalize(hidden[0, -1], p=2, dim=-1).contiguous()

    # The image-and-text case.
    pil = Image.fromarray((image * 255).astype(np.uint8))
    prompt = processor.apply_chat_template(conversation("What is in this image?", True),
                                           add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[prompt], images=[pil], return_tensors="pt")
    with torch.no_grad():
        embeds, deepstack = backbone.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
        hidden = backbone(**inputs).last_hidden_state
    extra["image_prompt_ids"] = inputs["input_ids"][0].to(torch.int32).contiguous()
    extra["pixel_values"] = inputs["pixel_values"].contiguous()
    extra["image_grid_thw"] = inputs["image_grid_thw"].to(torch.int32).contiguous()
    extra["vision_output"] = embeds.contiguous()
    for index, feature in enumerate(deepstack):
        extra[f"deepstack_{index}"] = feature.contiguous()
    extra["image_hidden"] = hidden[0].contiguous()
    extra["image_embedding"] = F.normalize(hidden[0, -1], p=2, dim=-1).contiguous()

    globals()["_extra"] = extra
    # safetensors refuses aliased storage, so the returned output is its own tensor.
    return extra["text_embedding"].clone()


def run_qwen3vl_reranker(image, checkpoint):
    """Qwen3-VL-Reranker, from the release's own `Qwen3VLReranker` recipe.

    `--checkpoint` is the released `Qwen/Qwen3-VL-Reranker-2B` directory. The reference scores a pair
    by reading the last position's hidden state through a one-output linear layer holding
    `lm_head[yes] - lm_head[no]`, then a sigmoid. That equals the difference of the two logits, because
    the output projection carries no bias, which is what the port computes.

    The record carries a relevant text pair, an irrelevant one, and an image document, each with its
    input ids and its score, plus the two scored token ids and the raw logit difference.
    """
    import numpy as np
    import torch
    from PIL import Image

    model, processor = _qwen3vl_retrieval_model(checkpoint)
    backbone = model.model
    vocabulary = processor.tokenizer.get_vocab()
    yes_id, no_id = vocabulary["yes"], vocabulary["no"]
    weights = model.lm_head.weight.data
    direction = weights[yes_id] - weights[no_id]
    instruction = "Given a search query, retrieve relevant candidates that answer the query."
    judgement = ('Judge whether the Document meets the requirements based on the Query and the '
                 'Instruct provided. Note that the answer can only be "yes" or "no".')
    extra = {"scored_token_ids": torch.tensor([yes_id, no_id], dtype=torch.int32)}

    def pair(query, document, has_image):
        content = [{"type": "text", "text": "<Instruct>: " + instruction},
                   {"type": "text", "text": "<Query>:"},
                   {"type": "text", "text": query},
                   {"type": "text", "text": "\n<Document>:"}]
        if has_image:
            content.append({"type": "image"})
        if document:
            content.append({"type": "text", "text": document})
        return [{"role": "system", "content": [{"type": "text", "text": judgement}]},
                {"role": "user", "content": content}]

    def record(name, query, document, has_image):
        prompt = processor.apply_chat_template(pair(query, document, has_image),
                                               add_generation_prompt=True, tokenize=False)
        images = [Image.fromarray((image * 255).astype(np.uint8))] if has_image else None
        inputs = processor(text=[prompt], images=images, return_tensors="pt")
        with torch.no_grad():
            hidden = backbone(**inputs).last_hidden_state[0, -1]
            difference = torch.dot(hidden, direction)
        extra[f"{name}_ids"] = inputs["input_ids"][0].to(torch.int32).contiguous()
        extra[f"{name}_difference"] = difference.reshape(1).contiguous()
        extra[f"{name}_score"] = torch.sigmoid(difference).reshape(1).contiguous()
        if has_image:
            extra[f"{name}_pixel_values"] = inputs["pixel_values"].contiguous()
            extra[f"{name}_grid_thw"] = inputs["image_grid_thw"].to(torch.int32).contiguous()
        return extra[f"{name}_score"]

    relevant = record("relevant", "How tall is the Eiffel Tower?",
                      "The Eiffel Tower stands 330 metres tall, including its antennas.", False)
    record("irrelevant", "How tall is the Eiffel Tower?",
           "Sourdough bread needs a starter kept at room temperature.", False)
    record("image", "What is in this image?", "A photograph.", True)

    globals()["_extra"] = extra
    # safetensors refuses aliased storage, so the returned output is its own tensor.
    return relevant.clone()


def run_qwen3vl_retrieval_loss(image):
    """The two objectives the Qwen3-VL retrieval releases are trained with, from sentence-transformers.

    The releases package themselves for sentence-transformers (`modules.json` names its Transformer,
    Pooling, and Normalize modules for the embedder and its LogitScore module for the reranker), so
    that library's losses are the reference training code: `MultipleNegativesRankingLoss` at its
    defaults (in-batch negatives, cosine similarity, scale 20) for the embedder, and
    `BinaryCrossEntropyLoss` over the raw pair logit for the reranker.

    The record carries the inputs and both losses. Nothing here loads a checkpoint: the losses score
    tensors, and the port must agree on identical ones.
    """
    import torch
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    torch.manual_seed(7)
    batch, width = 6, 32
    queries = torch.randn(batch, width)
    positives = torch.randn(batch, width)
    negatives = torch.randn(batch, width)
    logits = torch.randn(2 * batch)
    labels = torch.tensor([1.0, 0.0] * batch)

    ranking = MultipleNegativesRankingLoss(None)
    paired = ranking.compute_loss_from_embeddings([queries, positives], labels=None)
    with_negatives = ranking.compute_loss_from_embeddings([queries, positives, negatives], labels=None)
    # The reference's BinaryCrossEntropyLoss refuses to construct without a CrossEncoder, and at its
    # defaults (an Identity activation, no positive weight) it is exactly this loss over the raw pair
    # logit, which is what its `__init__` builds and its `forward` calls.
    binary = BinaryCrossEntropyLoss.__init__.__globals__["nn"].BCEWithLogitsLoss()(logits, labels)

    globals()["_extra"] = {
        "queries": queries.contiguous(),
        "positives": positives.contiguous(),
        "negatives": negatives.contiguous(),
        "logits": logits.contiguous(),
        "labels": labels.contiguous(),
        "ranking_loss": paired.reshape(1).contiguous(),
        "ranking_loss_with_negatives": with_negatives.reshape(1).contiguous(),
        "binary_loss": binary.reshape(1).contiguous(),
    }
    return torch.stack([paired, with_negatives, binary]).contiguous()


def run_qwenimage21(image):
    """Qwen-Image 2.1's denoising transformer, from diffusers' own QwenImage21Transformer2DModel, at a
    tiny random configuration.

    The released transformer is 7B, which no float32 comparison on this machine holds, so the arithmetic
    is measured at a small geometry the reference builds from the port's own parameters, and the released
    weights take the structural check. Everything that distinguishes 2.1 is exercised here: the
    block-causal mask over a sequence carrying one condition image and one target image, the
    `causal_condition` split that modulates text and condition tokens from t=0, the three-axis rotary
    with its centered and negative positions, and the zero-centered RMS norm in the caption projection.

    The record carries the inputs, every weight in release naming, and the output over the whole joint
    sequence.
    """
    import torch
    from diffusers import QwenImage21Transformer2DModel

    model = QwenImage21Transformer2DModel(
        patch_size=1, in_channels=8, out_channels=8, num_layers=2, attention_head_dim=16,
        num_attention_heads=2, context_in_dim=12, mlp_ratio=3, axes_dims_rope=(4, 6, 6), eps=1e-6,
        causal_condition=True)
    _randomized(model, seed=5, scale=0.1)

    torch.manual_seed(3)
    # One condition image (2x2 latent tokens, so one vision-language slot) inside the caption, and a
    # target image of 4x4 latent tokens, whose four slots trail the caption.
    img_shapes = [[(1, 2, 2), (1, 4, 4)]]
    img_mask = torch.tensor([[False, False, False, True, False, False, False, False,
                              True, True, True, True]])
    encoder_hidden_states = torch.randn(1, 8, 12)
    # The last caption slot is padding, so it must never be attended to as a key.
    encoder_hidden_states_mask = torch.tensor([[True, True, True, True, True, True, True, False]])
    latents = torch.randn(1, 20, 8)
    timestep = torch.tensor([0.7])

    with torch.no_grad():
        output = model(
            hidden_states=latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_shapes=img_shapes,
            img_mask=img_mask,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            return_dict=False,
        )[0]

    image_pad_mask = torch.repeat_interleave(img_mask[0], torch.where(img_mask, 4, 1)[0])
    image_ids, target_mask = model.build_token_metadata(image_pad_mask, img_shapes[0])
    rotary = model.pos_embed(img_shapes[0], image_pad_mask, device=latents.device)

    extra = {
        "latents": latents[0].contiguous(),
        "encoder_hidden_states": encoder_hidden_states[0].contiguous(),
        "encoder_mask": encoder_hidden_states_mask[0].to(torch.int32).contiguous(),
        "img_mask": img_mask[0].to(torch.int32).contiguous(),
        "image_pad_mask": image_pad_mask.to(torch.int32).contiguous(),
        "image_ids": image_ids.to(torch.int32).contiguous(),
        "target_mask": target_mask.to(torch.int32).contiguous(),
        "timestep": timestep.contiguous(),
        "rotary_real": torch.view_as_real(rotary)[..., 0].contiguous(),
        "rotary_imag": torch.view_as_real(rotary)[..., 1].contiguous(),
    }
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return output[0].contiguous()


def run_qwenimage21_real(image, checkpoint):
    """Qwen-Image 2.1's transformer on the RELEASED weights, at the precision they ship in.

    `--checkpoint` is the release's `transformer/` directory. 7.1B parameters at float32 is 28 GB, which
    this machine cannot hold on both sides, so both sides run bfloat16 and the comparison describes the
    released precision rather than the arithmetic's ceiling; the tiny-configuration mode is where the
    arithmetic is measured exactly.

    The caption features are random rather than the vision-language encoder's, which keeps the
    transformer measured in isolation, the way the LTX DiT is. The sequence carries one condition image
    inside the caption and a small target image after it.
    """
    import torch
    from diffusers import QwenImage21Transformer2DModel

    model = QwenImage21Transformer2DModel.from_pretrained(checkpoint, dtype=torch.bfloat16).eval()

    torch.manual_seed(11)
    img_shapes = [[(1, 2, 2), (1, 8, 8)]]
    slots = [False] * 16
    slots[5] = True                                     # the condition image's slot, inside the caption
    img_mask = torch.tensor([slots + [True] * 16])      # the target image's sixteen slots follow
    encoder_hidden_states = (torch.randn(1, 16, 4096) * 0.5).to(torch.bfloat16)
    encoder_hidden_states_mask = torch.ones(1, 16, dtype=torch.bool)
    encoder_hidden_states_mask[0, -2:] = False          # two padded caption slots
    latents = (torch.randn(1, 68, 64) * 0.5).to(torch.bfloat16)
    timestep = torch.tensor([0.35])

    seams = {}

    def capture(name):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            seams[name] = value.detach()[0].float().contiguous()
        return hook

    handles = [model.txt_in.register_forward_hook(capture("txt_in")),
               model.img_in.register_forward_hook(capture("img_in")),
               model.modulation.register_forward_hook(capture("modulation")),
               model.time_text_embed.register_forward_hook(capture("temb")),
               model.norm_out.register_forward_hook(capture("norm_out"))]
    for index in (0, 7, 15, 23, 31):
        handles.append(model.transformer_blocks[index].register_forward_hook(capture(f"block_{index}")))

    with torch.no_grad():
        output = model(
            hidden_states=latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_shapes=img_shapes,
            img_mask=img_mask,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            return_dict=False,
        )[0]
    for handle in handles:
        handle.remove()

    globals()["_extra"] = {
        **{f"seam::{name}": value for name, value in seams.items()},
        "latents": latents[0].float().contiguous(),
        "encoder_hidden_states": encoder_hidden_states[0].float().contiguous(),
        "encoder_mask": encoder_hidden_states_mask[0].to(torch.int32).contiguous(),
        "img_mask": img_mask[0].to(torch.int32).contiguous(),
        "timestep": timestep.contiguous(),
    }
    return output[0].float().contiguous()


def run_qwenimage21_scheduler(image, checkpoint):
    """Qwen-Image 2.1's sampler schedule, from diffusers' own FlowMatchEulerDiscreteScheduler.

    `--checkpoint` is the release directory, whose `scheduler/` config carries the dynamic shifting
    (base 0.5 over 256 tokens to max 0.9 over 8192) and the 0.02 terminal stretch. The pipeline passes
    its own sigma ramp, `linspace(1, 1/steps, steps)`, rather than letting the scheduler build one from
    `1/num_train_timesteps`, so the ramp is part of the schedule being measured.

    The record carries sigmas and timesteps at three step counts and two sequence lengths.
    """
    import numpy as np
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift

    extra = {}
    for steps in (4, 20, 50):
        for sequence in (1024, 4096):
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint, subfolder="scheduler")
            mu = calculate_shift(
                sequence,
                scheduler.config.get("base_image_seq_len", 256),
                scheduler.config.get("max_image_seq_len", 4096),
                scheduler.config.get("base_shift", 0.5),
                scheduler.config.get("max_shift", 1.15),
            )
            sigmas = np.linspace(1.0, 1 / steps, steps)
            scheduler.set_timesteps(steps, device="cpu", sigmas=sigmas, mu=mu)
            extra[f"sigmas_{steps}_{sequence}"] = scheduler.sigmas.float().contiguous()
            extra[f"timesteps_{steps}_{sequence}"] = scheduler.timesteps.float().contiguous()
            extra[f"mu_{steps}_{sequence}"] = torch.tensor([mu], dtype=torch.float32)
    globals()["_extra"] = extra
    return extra["sigmas_20_4096"].clone()


def run_qwenimage21_vae(image, checkpoint):
    """Qwen-Image 2.1's autoencoder on the released weights, from diffusers' own
    AutoencoderKLQwenImage21.

    `--checkpoint` is the release directory. The model is the Wan 2.2 residual VAE specialized to one
    frame: its causal convolution subclasses `nn.Conv2d` and refuses a feature cache, so every
    convolution weight is 4-D and the temporal branches never run for a single frame.

    The record carries the input frame, the encoder's moments before the split, the latent the
    pipeline works in (normalized by the release's own per-channel statistics), and the decode of that
    latent back to pixels.
    """
    import numpy as np
    import torch
    from diffusers import AutoencoderKLQwenImage21

    vae = AutoencoderKLQwenImage21.from_pretrained(checkpoint, subfolder="vae", dtype=torch.float32).eval()

    # The autoencoder takes four channels, so the plate's three are carried with a fourth held at one,
    # which is what an opaque image gives.
    plate = torch.from_numpy(image).permute(2, 0, 1)[None, :, None]          # [1, 3, 1, H, W]
    plate = plate * 2 - 1
    frame = torch.cat([plate, torch.ones_like(plate[:, :1])], dim=1)         # [1, 4, 1, H, W]

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)

    with torch.no_grad():
        posterior = vae.encode(frame).latent_dist
        moments = torch.cat([posterior.mean, posterior.logvar], dim=1)
        latent = (posterior.mean - latents_mean) / latents_std
        decoded = vae.decode(latent * latents_std + latents_mean).sample

    globals()["_extra"] = {
        "frame": frame[0].permute(1, 2, 3, 0).contiguous(),                  # [T, H, W, C]
        "moments": moments[0].permute(1, 2, 3, 0).contiguous(),
        "latent": latent[0].permute(1, 2, 3, 0).contiguous(),
        "decoded": decoded[0].permute(1, 2, 3, 0).contiguous(),
        "latents_mean": latents_mean.reshape(-1).contiguous(),
        "latents_std": latents_std.reshape(-1).contiguous(),
    }
    return decoded[0].permute(1, 2, 3, 0).clone().contiguous()


def run_qwenimage21_pipeline(image, checkpoint):
    """Qwen-Image 2.1's text-to-image glue, from diffusers' own QwenImage21Pipeline.

    `--checkpoint` is the release directory, for its scheduler config alone. The transformer and the
    autoencoder are tiny random ones, so what this measures is the pipeline: the latent packing, the
    image mask the transformer reads, the sampling loop over the flow schedule, the slice that takes
    the target image's rows out of the joint sequence, and the latent normalization across the two
    models. Each of those models is measured against its own reference elsewhere.

    The record carries the inputs, both models' weights in release naming, the final packed latents,
    and the decoded image, with and without the prefix KV cache the reference uses by default.
    """
    import torch
    from diffusers import (AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler,
                           QwenImage21Pipeline, QwenImage21Transformer2DModel)

    transformer = QwenImage21Transformer2DModel(
        patch_size=1, in_channels=4, out_channels=4, num_layers=2, attention_head_dim=16,
        num_attention_heads=2, context_in_dim=12, mlp_ratio=3, axes_dims_rope=(4, 6, 6), eps=1e-6,
        causal_condition=True)
    _randomized(transformer, seed=5, scale=0.1)
    vae = AutoencoderKLQwenImage21(
        base_dim=8, decoder_base_dim=8, z_dim=4, dim_mult=[2, 2], num_res_blocks=1,
        temperal_downsample=[True], in_channels=4, out_channels=4, is_residual=True,
        latents_mean=[0.1, -0.2, 0.3, -0.4], latents_std=[1.1, 0.9, 1.3, 0.8])
    _randomized(vae, seed=7, scale=0.1)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint, subfolder="scheduler")

    # The processor is the release's own (tokenizer files, no weights): the pipeline reads the system
    # turn's token count from it, which is what it drops from the encoder's hidden states.
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(checkpoint, subfolder="processor")
    pipeline = QwenImage21Pipeline(scheduler=scheduler, vae=vae, text_encoder=None, processor=processor,
                                   transformer=transformer)

    torch.manual_seed(3)
    prompt_embeds = torch.randn(1, 8, 12)
    latents = torch.randn(1, 16, 4)                     # 64x64 pixels at the pipeline's 16x factor

    extra = {"prompt_embeds": prompt_embeds[0].contiguous(), "latents": latents[0].contiguous(),
             "drop_index": torch.tensor([pipeline._drop_idx], dtype=torch.int32),
             "image_token_id": torch.tensor([pipeline._img_token_id], dtype=torch.int32)}
    for name, cache in (("cached", True), ("uncached", False)):
        with torch.no_grad():
            final = pipeline(prompt_embeds=prompt_embeds, height=64, width=64, num_inference_steps=4,
                             latents=latents.clone(), output_type="latent", use_kv_cache=cache,
                             return_dict=False)[0]
            decoded = pipeline(prompt_embeds=prompt_embeds, height=64, width=64, num_inference_steps=4,
                               latents=latents.clone(), output_type="pt", use_kv_cache=cache,
                               return_dict=False)[0]
        extra[f"final_latents_{name}"] = final[0].contiguous()
        extra[f"image_{name}"] = decoded[0].permute(1, 2, 0).contiguous()      # [H, W, C] in 0…1

    for key, value in transformer.state_dict().items():
        extra[f"t::{key}"] = value.float().contiguous()
    for key, value in vae.state_dict().items():
        extra[f"v::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return extra["final_latents_uncached"].clone()


def run_qwenimage21_text(image, checkpoint):
    """Qwen-Image 2.1's prompt encoding on the released text encoder, the pipeline's own recipe.

    `--checkpoint` is the release directory. The text encoder is Qwen3-VL at the 8B geometry, run at
    the bfloat16 it ships in. Two details decide the features: the prompt is a RAW template string
    passed straight to the processor rather than the chat template's rendering of it, and the states
    read are the last decoder layer's output BEFORE the final normalization, which the pipeline obtains
    by neutralizing that norm with a forward hook. The system turn's leading tokens are then dropped.

    The record carries the token ids, the count dropped, and the resulting features.
    """
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(checkpoint, subfolder="processor")
    # Eager attention: a bf16 port is held to transformers' eager rounding, which torch's CPU SDPA
    # kernel does not reproduce.
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint, subfolder="text_encoder", dtype=torch.bfloat16, attn_implementation="eager").eval()

    system = "Comprehend and analyze the provided prompt."
    prompt = "A calico cat asleep on a stack of books, warm afternoon light"
    template = (f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n")
    inputs = processor(text=[template.format(prompt)], padding=True, padding_side="left",
                       return_tensors="pt")

    sys_message = [{"role": "system", "content": [{"type": "text", "text": system}]}]
    drop = len(processor.apply_chat_template(sys_message, tokenize=True, return_dict=False)[0])

    language = getattr(encoder.model, "language_model", encoder.model)
    handle = language.norm.register_forward_hook(lambda module, args, output: args[0])
    try:
        with torch.no_grad():
            outputs = encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                              output_hidden_states=True)
    finally:
        handle.remove()
    features = outputs.hidden_states[-1][0][drop:]

    extra = {
        "input_ids": inputs.input_ids[0].to(torch.int32).contiguous(),
        "drop_index": torch.tensor([drop], dtype=torch.int32),
        "prompt_embeds": features.float().contiguous(),
    }
    # Per-layer states, so a divergence localizes to a layer rather than to the whole encoder.
    for index in (0, 1, 9, 18, 27, 30, 33, 34, 35, len(outputs.hidden_states) - 1):
        extra[f"seam::hidden_{index}"] = outputs.hidden_states[index][0].float().contiguous()
    globals()["_extra"] = extra
    return features.float().clone().contiguous()


TRANSLATION_SENTENCES = [
    "Hello world! How are you?",
    "The quick brown fox jumps over the lazy dog.",
    "  spaced   out  text  ",
    "naïve café — résumé 2024",
    "ﬁne ½",
]
TRANSLATION_TARGET = "Der schnelle braune Fuchs springt über den faulen Hund."


def _translation_record(model, tokenizer, encode_source, encode_target, beams, generate_kwargs):
    """The seams a translation port is measured on: the tokenizer's ids for TRANSLATION_SENTENCES,
    the encoder output for sentence 1, the reference's greedy and beam outputs, the teacher-forced
    logits over the greedy output, and the training loss against TRANSLATION_TARGET."""
    source = encode_source(TRANSLATION_SENTENCES[1])
    input_ids = torch.tensor([source])
    attention = torch.ones_like(input_ids)
    with torch.no_grad():
        encoder_hidden = model.get_encoder()(input_ids=input_ids, attention_mask=attention).last_hidden_state
        greedy = model.generate(input_ids=input_ids, attention_mask=attention, num_beams=1, do_sample=False,
                                max_new_tokens=64, **generate_kwargs)
        beam = model.generate(input_ids=input_ids, attention_mask=attention, num_beams=beams, do_sample=False,
                              max_new_tokens=64, **generate_kwargs)
        decoder_input = greedy[:, :-1]
        logits = model(input_ids=input_ids, attention_mask=attention, decoder_input_ids=decoder_input).logits
        target = torch.tensor([encode_target(TRANSLATION_TARGET)])
        loss = model(input_ids=input_ids, attention_mask=attention, labels=target).loss
    extra = {
        "source_ids": input_ids[0].to(torch.int32).contiguous(),
        "encoder_hidden": encoder_hidden[0].contiguous(),
        "greedy": greedy[0].to(torch.int32).contiguous(),
        "beam": beam[0].to(torch.int32).contiguous(),
        "beams": torch.tensor([beams], dtype=torch.int32),
        "decoder_input": decoder_input[0].to(torch.int32).contiguous(),
        "target_ids": target[0].to(torch.int32).contiguous(),
        "loss": loss.reshape(1).contiguous(),
    }
    for index, sentence in enumerate(TRANSLATION_SENTENCES):
        extra[f"tokens_{index}"] = torch.tensor(encode_source(sentence), dtype=torch.int32)
    print("greedy:", tokenizer.decode(greedy[0], skip_special_tokens=True))
    print("beam:", tokenizer.decode(beam[0], skip_special_tokens=True))
    globals()["_extra"] = extra
    return logits[0].contiguous()


def run_marian(image, checkpoint):
    """OPUS-MT (Helsinki-NLP/opus-mt-en-de, `MarianMTModel`) on a release directory: the seams of
    `_translation_record`, beams from the release's generation config (4)."""
    from transformers import MarianMTModel, MarianTokenizer
    tokenizer = MarianTokenizer.from_pretrained(checkpoint)
    model = MarianMTModel.from_pretrained(checkpoint, torch_dtype=torch.float32).eval()
    beams = model.generation_config.num_beams or 4
    return _translation_record(model, tokenizer, lambda s: tokenizer(s).input_ids,
                               lambda s: tokenizer(text_target=s).input_ids, beams, {})


def run_m2m100(image, checkpoint):
    """M2M-100 (facebook/m2m100_418M, `M2M100ForConditionalGeneration`) on a release directory,
    English to German: the seams of `_translation_record` with the target marker forced first, beams
    from the release's generation config (5)."""
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
    tokenizer = M2M100Tokenizer.from_pretrained(checkpoint, src_lang="en", tgt_lang="de")
    model = M2M100ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.float32).eval()
    beams = model.generation_config.num_beams or 5
    return _translation_record(model, tokenizer, lambda s: tokenizer(s).input_ids,
                               lambda s: tokenizer(text_target=s).input_ids, beams,
                               {"forced_bos_token_id": tokenizer.get_lang_id("de")})


def run_madlad(image, checkpoint):
    """MADLAD-400 (google/madlad400-3b-mt, `T5ForConditionalGeneration`) on a release directory,
    into German through the `<2de>` marker and the release's fast tokenizer: the seams of
    `_translation_record`, beams 4."""
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.float32).eval()
    return _translation_record(model, tokenizer, lambda s: tokenizer("<2de> " + s).input_ids,
                               lambda s: tokenizer(text_target=s).input_ids, 4, {})


def run_florence2_loss(image, checkpoint):
    """Florence-2's fine-tuning objective: the release's own `labels=` loss
    (`Florence2LanguageForConditionalGeneration`: the labels shifted right behind the decoder start
    token, then `CrossEntropyLoss` over every position). Microsoft publishes no training script, so this
    is the reference objective. FLORENCE2_REPO picks the release, as in run_florence2; `checkpoint` is
    unused.

    The answer is the release's own `<CAPTION>` (its generation settings, `use_cache=False` as the
    generation mode needs), tokenized `<s> … </s>` as the processor's tokenizer gives it. Records the
    pixels (NHWC), the prompt ids, the answer ids, the teacher-forced logits, the release's loss (the
    output), and the same cross-entropy in float64 (`loss_f64`).
    """
    import os as _os
    from transformers import AutoModelForCausalLM, AutoProcessor
    from PIL import Image

    repo = _os.environ.get("FLORENCE2_REPO", "microsoft/Florence-2-large")
    model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.float32,
                                                 attn_implementation="eager").eval()
    processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
    pil = Image.fromarray((image * 255).astype("uint8")).convert("RGB")
    inputs = processor(text="<CAPTION>", images=pil, return_tensors="pt")
    with torch.no_grad():
        generated = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                                   max_new_tokens=40, use_cache=False)
        answer = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        labels = processor.tokenizer(answer, return_tensors="pt").input_ids
        out = model(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], labels=labels,
                    use_cache=False)
        exact = torch.nn.functional.cross_entropy(out.logits[0].double(), labels[0])
    print("florence2 answer:", answer)
    globals()["_extra"] = {
        "pixels": inputs["pixel_values"][0].permute(1, 2, 0).contiguous(),
        "prompt_ids": inputs["input_ids"][0].to(torch.int32).contiguous(),
        "answer_ids": labels[0].to(torch.int32).contiguous(),
        "answer_utf8": torch.tensor(list(answer.encode("utf-8")), dtype=torch.int32),
        "logits": out.logits[0].contiguous(),
        "loss_f64": exact.float().reshape(1).contiguous(),
    }
    return out.loss.reshape(1).contiguous()

def run_florence2(image, checkpoint):
    """Florence-2-large (microsoft/Florence-2-large, MIT) from transformers
    Florence2ForConditionalGeneration on the RELEASED weights: the DaViT vision tower (per block a
    windowed SPATIAL attention then a grouped CHANNEL attention), the multi-modal projector (learned
    2-D position embedding + cosine temporal embedding + mean-pooled-token prepend + projection and
    LayerNorm), and the BART encoder-decoder over the scattered image+prompt sequence. `checkpoint` is
    the local release directory. Records the preprocessed pixel values (NHWC, so the port runs on the
    identical input), the prompt input_ids, the stage-0 patch embedding, the first DaViT block, the
    vision-tower output, the projector output, the encoder last hidden state, and the first-step
    logits. Runs under llmvenv (transformers 4.57 has native florence2)."""
    # The native transformers Florence2 does not map the released original-davit checkpoint (it loads
    # all-random), so the authoritative reference is the repo's own code via trust_remote_code, which
    # matches the checkpoint layout the port loads from. `checkpoint` is unused; the Hub id supplies the
    # remote code and the cached weights. Needs timm.
    # FLORENCE2_REPO picks the release: microsoft/Florence-2-large (the default) or -base.
    import os as _os
    from transformers import AutoModelForCausalLM, AutoProcessor
    from PIL import Image

    repo = _os.environ.get("FLORENCE2_REPO", "microsoft/Florence-2-large")
    model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, torch_dtype=torch.float32,
                                                 attn_implementation="eager").eval()
    processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
    pil = Image.fromarray((image * 255).astype("uint8")).convert("RGB")
    inputs = processor(text="<OD>", images=pil, return_tensors="pt")
    pixel_values = inputs["pixel_values"]                                          # [1, 3, S, S]
    input_ids = inputs["input_ids"]

    # Canonicalize any seam to [tokens, C] in row-major (H, W) order so cosines align with the NHWC port.
    def canon(o):
        o = o[0] if o.dim() == 4 else o
        if o.dim() == 3 and o.shape[0] == 1:
            o = o[0]
        if o.dim() == 3:                                                           # [C, H, W]
            return o.permute(1, 2, 0).reshape(-1, o.shape[0]).contiguous()
        return o.contiguous()                                                      # [tokens, C]

    # The original davit ConvEmbed and blocks return (tokens[B, H*W, C], (H, W)) tuples, not NCHW maps.
    def grab(name):
        return lambda m, i, o: seams.__setitem__(name, (o[0] if isinstance(o, (tuple, list)) else o).detach())
    seams = {}
    model.vision_tower.convs[0].register_forward_hook(grab("conv0"))
    model.vision_tower.blocks[0][0].register_forward_hook(grab("block0"))
    start = model.config.text_config.decoder_start_token_id
    with torch.no_grad():
        vision = model.vision_tower.forward_features_unpool(pixel_values)          # [B, H*W, C] (unpooled)
        proj = model._encode_image(pixel_values)                                   # [B, 1+H*W, d_model]
        # The BART fusion concatenates [image_features, text_embeds] (image first), then encodes; the
        # decoder cross-attends and the head adds final_logits_bias. Record the encoder output and the
        # first-step logits for the fusion parity.
        out = model(input_ids=input_ids, pixel_values=pixel_values,
                    decoder_input_ids=torch.tensor([[start]]))
        # Greedy generation for the generation-loop parity check, via a manual argmax loop over the same
        # forward (the model's own generate override is incompatible with this transformers version). The
        # sequence begins with the decoder start token.
        greedy = [start]
        for _ in range(32):
            step = model(input_ids=input_ids, pixel_values=pixel_values,
                         decoder_input_ids=torch.tensor([greedy]))
            token = int(step.logits[0, -1].argmax())
            greedy.append(token)
            if token == model.config.text_config.eos_token_id:
                break
        generated = torch.tensor([greedy])

    print("shapes conv0", tuple(seams["conv0"].shape), "block0", tuple(seams["block0"].shape),
          "vision", tuple(vision.shape), "proj", tuple(proj.shape),
          "enc_last", tuple(out.encoder_last_hidden_state.shape), "logits", tuple(out.logits.shape))
    extra = {
        "pixels": pixel_values[0].permute(1, 2, 0).contiguous(),                   # [S, S, 3] NHWC
        "input_ids": input_ids[0].to(torch.int32).contiguous(),                    # [T]
        "conv0": canon(seams["conv0"]),                                            # [H0*W0, C0]
        "block0": canon(seams["block0"]),                                          # [H0*W0, C0]
        "vision": canon(vision),                                                   # [H*W, C]
        "proj": proj[0].contiguous(),                                              # [1+H*W, d_model]
        "enc_last": out.encoder_last_hidden_state[0].contiguous(),                 # [1+H*W+T, d_model]
        "logits": out.logits[0, 0].contiguous(),                                   # [vocab] first step
        "generated": generated[0].to(torch.int32).contiguous(),                    # [T_gen] incl. start token
    }
    globals()["_extra"] = extra
    return proj[0].clone().contiguous()


FLORENCE2_GENERATION_TASKS = ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<OCR>", "<OD>"]


def run_florence2_generate(image, checkpoint):
    """Florence-2's own `generate` under the release's generation settings (`text_config`: three beams,
    early stopping, no repeated 3-gram, `<s>` forced first, `</s>` forced at the length limit), per task
    in FLORENCE2_GENERATION_TASKS with max_new_tokens 1024. Also records `<MORE_DETAILED_CAPTION>` cut to
    16 new tokens (`generated_truncated`, where the forced `</s>` decides the ending) and greedy
    `<CAPTION>` under the same constraints (`generated_greedy`). Records the shared preprocessed pixels
    (NHWC) and, per task index i, `input_ids_<i>` and `generated_<i>` (the start token first).
    FLORENCE2_REPO picks the release, as in run_florence2; `checkpoint` is unused."""
    import os as _os
    from transformers import AutoModelForCausalLM, AutoProcessor
    from PIL import Image

    repo = _os.environ.get("FLORENCE2_REPO", "microsoft/Florence-2-large")
    model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.float32,
                                                 attn_implementation="eager").eval()
    processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
    pil = Image.fromarray((image * 255).astype("uint8")).convert("RGB")
    # The remote code's cached decode fails under transformers 4.57 (its BART reads a legacy tuple
    # cache), so generation runs uncached; the settings still come from the release's own config.
    def generate(task, **overrides):
        inputs = processor(text=task, images=pil, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                                 use_cache=False, **{"max_new_tokens": 1024, **overrides})
        print(task, overrides, processor.batch_decode(ids, skip_special_tokens=False)[0])
        return inputs, ids[0].to(torch.int32).contiguous()

    extra = {}
    for index, task in enumerate(FLORENCE2_GENERATION_TASKS):
        inputs, ids = generate(task)
        extra[f"input_ids_{index}"] = inputs["input_ids"][0].to(torch.int32).contiguous()
        extra[f"generated_{index}"] = ids
        extra["pixels"] = inputs["pixel_values"][0].permute(1, 2, 0).contiguous()
    extra["generated_truncated"] = generate("<MORE_DETAILED_CAPTION>", max_new_tokens=16)[1]
    extra["generated_greedy"] = generate("<CAPTION>", num_beams=1)[1]
    globals()["_extra"] = extra
    return extra["pixels"].clone()


def run_translategemma(image, checkpoint):
    """TranslateGemma (google/translategemma-4b-it, `Gemma3ForConditionalGeneration` driven text-only)
    on a release directory, under the gemma oracle interpreter. Records the ids the release's own
    chat template renders for one text translation item (English to German, TRANSLATION_SENTENCES[1]),
    the logits at the last 16 prompt positions, the greedy continuation, and the supervised
    fine-tuning loss of TRANSLATION_TARGET as the model turn (prompt positions masked)."""
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # The 12B is 24 GB of bfloat16, twice the machine's memory at float32, so TRANSLATEGEMMA_DTYPE=bfloat16
    # runs the reference at the release's own precision; the port is then compared at .checkpoint.
    dtype = getattr(torch, os.environ.get("TRANSLATEGEMMA_DTYPE", "float32"))
    model = AutoModelForImageTextToText.from_pretrained(checkpoint, dtype=dtype).eval()
    messages = [{"role": "user", "content": [{"type": "text", "source_lang_code": "en",
                                              "target_lang_code": "de", "text": TRANSLATION_SENTENCES[1]}]}]
    # The rendered template already spells `<bos>`, so it is tokenized without a second one.
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = torch.tensor([tokenizer(text, add_special_tokens=False).input_ids])
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
        logits = out.logits[0]
        generated = model.generate(input_ids=ids, max_new_tokens=48, do_sample=False)
        target_ids = tokenizer(TRANSLATION_TARGET + "<end_of_turn>", add_special_tokens=False).input_ids
        full = torch.tensor([ids[0].tolist() + target_ids])
        labels = full.clone()
        labels[0, : ids.shape[1]] = -100
        loss = model(input_ids=full, labels=labels).loss
    continuation = generated[0, ids.shape[1]:]
    print("greedy:", tokenizer.decode(continuation, skip_special_tokens=True))
    globals()["_extra"] = {
        "tokens": ids[0].to(torch.int32).contiguous(),
        "continuation": continuation.to(torch.int32).contiguous(),
        "target_ids": torch.tensor(target_ids, dtype=torch.int32),
        "loss": loss.reshape(1).float().contiguous(),
    }
    # The last prompt position's state entering the stack and after every layer (the final norm on
    # the last), so a half-precision drift is located to a layer rather than guessed at.
    for index, state in enumerate(out.hidden_states):
        globals()["_extra"][f"hidden_last.{index}"] = state[0, -1].float().contiguous()
    return logits[-16:].float().contiguous()


def run_trocr_loss(image, checkpoint):
    """TrOCR's fine-tuning objective on a released model (`checkpoint`, a release directory), as the
    authors trained it in fairseq (`microsoft/unilm/trocr`: `fairseq-train --task text_recognition`
    with the default `cross_entropy` criterion, whose token sum the trainer divides by the token count).

    The target is the text's pieces followed by the end token, fairseq's `encode_line` form and the
    sequence the releases generate after their start token (no `<s>`). The record holds the pixels
    (NHWC), the text as UTF-8, the target ids the release tokenizer gives, the teacher-forced logits, the
    loss transformers computes from `labels=` (`hf_loss`; `shift_tokens_right` behind the decoder start
    token, which the HF fine-tuning recipe sets on the top-level config), and fairseq's normalized
    cross-entropy on the same logits (the output, and `fairseq_loss_f64` computed in float64). transformers 4.57 scores these labels with
    `ForCausalLMLoss`, which shifts the logits against the labels a second time, so `hf_loss` is
    misaligned by one token and is recorded only to show the difference. The text is the release's own
    greedy transcription. `lr_iam` steps fairseq's own
    `InverseSquareRootSchedule` (`IK_FAIRSEQ_SRC`, the file the manifest pins) with the IAM recipe's
    settings: 2e-5, a 500-update warm-up from 1e-8, the rate read at update counts 0 through 1199.
    """
    import dataclasses
    import sys
    import types

    import torch.nn.functional as F
    from transformers import VisionEncoderDecoderModel, TrOCRProcessor
    from PIL import Image

    model = VisionEncoderDecoderModel.from_pretrained(checkpoint, torch_dtype=torch.float32).eval()
    processor = TrOCRProcessor.from_pretrained(checkpoint)
    decoder = model.config.decoder
    model.config.decoder_start_token_id = decoder.decoder_start_token_id
    model.config.pad_token_id = decoder.pad_token_id
    pil = Image.fromarray((image * 255).astype("uint8")).convert("RGB")
    pixel_values = processor(images=pil, return_tensors="pt").pixel_values
    # The release's own greedy transcription is the text, so the loss scores a sequence the model rates.
    greedy = [decoder.decoder_start_token_id]
    with torch.no_grad():
        for _ in range(48):
            token = int(model(pixel_values=pixel_values, decoder_input_ids=torch.tensor([greedy])).logits[0, -1].argmax())
            greedy.append(token)
            if token == decoder.eos_token_id:
                break
    text = processor.batch_decode(torch.tensor([greedy]), skip_special_tokens=True)[0]
    target = processor.tokenizer(text, add_special_tokens=False).input_ids + [decoder.eos_token_id]
    labels = torch.tensor([target])
    with torch.no_grad():
        out = model(pixel_values=pixel_values, labels=labels)
        lprobs = torch.log_softmax(out.logits[0], dim=-1)
        fairseq = F.nll_loss(lprobs, labels[0], ignore_index=decoder.pad_token_id, reduction="sum") / len(target)
        # The same criterion in float64: float32's log-sum-exp over the 64k-entry vocabulary rounds to
        # about 2e-5, which is larger than any difference the port could make.
        exact = F.nll_loss(torch.log_softmax(out.logits[0].double(), dim=-1), labels[0],
                           ignore_index=decoder.pad_token_id, reduction="sum") / len(target)

    root = os.path.expanduser(os.environ.get("IK_FAIRSEQ_SRC", "~/.inferkit-validation/sources/fairseq"))
    stubs = {name: types.ModuleType(name) for name in
             ["fairseq", "fairseq.dataclass", "fairseq.optim", "fairseq.optim.lr_scheduler"]}

    @dataclasses.dataclass
    class FairseqDataclass:
        pass

    class FairseqLRScheduler:
        def __init__(self, cfg, optimizer):
            self.cfg, self.optimizer = cfg, optimizer

    stubs["fairseq.dataclass"].FairseqDataclass = FairseqDataclass
    stubs["fairseq.optim.lr_scheduler"].FairseqLRScheduler = FairseqLRScheduler
    stubs["fairseq.optim.lr_scheduler"].register_lr_scheduler = lambda name, dataclass=None: (lambda cls: cls)
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        namespace = {"__name__": "fairseq_inverse_sqrt"}
        exec(compile(open(os.path.join(root, "inverse_square_root_schedule.py")).read(),
                     "inverse_square_root_schedule.py", "exec"), namespace)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    class _Optimizer:
        lr = 0.0

        def set_lr(self, lr):
            self.lr = lr

        def get_lr(self):
            return self.lr

    config = namespace["InverseSquareRootLRScheduleConfig"](warmup_updates=500, warmup_init_lr=1e-8, lr=[2e-5])
    schedule = namespace["InverseSquareRootSchedule"](config, _Optimizer())
    rates = [schedule.step_update(update) for update in range(1200)]

    globals()["_extra"] = {
        "pixel_values": pixel_values.permute(0, 2, 3, 1).contiguous(),
        "text_utf8": torch.tensor(list(text.encode("utf-8")), dtype=torch.int32),
        "target": torch.tensor(target, dtype=torch.int32),
        "logits": out.logits.contiguous(),
        "hf_loss": out.loss.reshape(1).contiguous(),
        "fairseq_loss_f64": exact.float().reshape(1).contiguous(),
        "lr_iam": torch.tensor(rates, dtype=torch.float64).to(torch.float32),
    }
    return fairseq.reshape(1).contiguous()

def run_trocr(image, checkpoint):
    """TrOCR (microsoft/trocr-*, MIT) VisionEncoderDecoder on the RELEASED weights: a ViT (base, large)
    or DeiT (small) image encoder and a BART-style `trocr` decoder that cross-attends the image
    features. `checkpoint` is the local release directory. Records the 8-bit image the processor
    received (`input_rgb`, so the port's processor is checked on identical bytes), the preprocessed
    pixel values (NHWC, so the network runs on the identical input), the embeddings output, the first
    encoder block, the encoder last hidden state (the decoder memory), the first-step logits, the greedy
    transcription ids, and their decoded text as UTF-8 (`text_utf8`). Runs under llmvenv."""
    from transformers import VisionEncoderDecoderModel, TrOCRProcessor
    from PIL import Image

    model = VisionEncoderDecoderModel.from_pretrained(checkpoint, torch_dtype=torch.float32).eval()
    processor = TrOCRProcessor.from_pretrained(checkpoint)
    pil = Image.fromarray((image * 255).astype("uint8")).convert("RGB")
    pixel_values = processor(images=pil, return_tensors="pt").pixel_values           # [1, 3, 384, 384]

    def grab(name):
        return lambda m, i, o: seams.__setitem__(name, (o[0] if isinstance(o, (tuple, list)) else o).detach())
    seams = {}
    model.encoder.embeddings.register_forward_hook(grab("emb"))
    model.encoder.encoder.layer[0].register_forward_hook(grab("block0"))
    start = model.config.decoder.decoder_start_token_id
    eos = model.config.decoder.eos_token_id
    with torch.no_grad():
        enc = model.encoder(pixel_values=pixel_values).last_hidden_state             # [1, 577, 768]
        out = model(pixel_values=pixel_values, decoder_input_ids=torch.tensor([[start]]))
        # Greedy generation for the generation-loop parity, a manual argmax loop over the same forward.
        greedy = [start]
        for _ in range(48):
            step = model(pixel_values=pixel_values, decoder_input_ids=torch.tensor([greedy]))
            token = int(step.logits[0, -1].argmax())
            greedy.append(token)
            if token == eos:
                break
        generated = torch.tensor([greedy])
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    print("greedy:", text)
    print("shapes emb", tuple(seams["emb"].shape), "block0", tuple(seams["block0"].shape),
          "enc_last", tuple(enc.shape), "logits", tuple(out.logits.shape))
    globals()["_extra"] = {
        "pixels": pixel_values[0].permute(1, 2, 0).contiguous(),                     # [384, 384, 3] NHWC
        "emb": seams["emb"][0].contiguous(),                                         # [577, 768]
        "block0": seams["block0"][0].contiguous(),                                   # [577, 768]
        "enc_last": enc[0].contiguous(),                                             # [577, 768] the memory
        "logits": out.logits[0, 0].contiguous(),                                     # [vocab] first step
        "generated": generated[0].to(torch.int32).contiguous(),                      # [T_gen] incl. start+eos
        "input_rgb": torch.from_numpy(np.asarray(pil)).to(torch.int32).contiguous(),  # [H, W, 3] 0...255
        "text_utf8": torch.tensor(list(text.encode("utf-8")), dtype=torch.int32),    # the decoded transcription
    }
    return enc[0].clone().contiguous()


def run_flux2(image):
    """The FLUX.2 transformer velocity at a tiny random configuration, from diffusers'
    Flux2Transformer2DModel, plus the modulation, double-block and single-block seams.

    FLUX.2 keeps FLUX.1's two block kinds and changes four things. The modulation lives on the MODEL,
    not in the blocks: three `Flux2Modulation` heads (double image, double text, single) are evaluated
    once from the timestep embedding and every block of that kind reads the same vector, so a released
    checkpoint carries three modulation tensors rather than one per block. The feed-forward is a SwiGLU
    whose gate is the first half of one fused projection. The single-stream block is a PARALLEL block:
    one projection produces q, k, v and both SwiGLU halves, and one projection takes the attention
    output concatenated with the gated MLP. The rotary runs over FOUR axes (t, h, w, l) at theta 2000,
    and the text ids carry the token index in the fourth axis rather than being all zero.

    There is no pooled text embedding: conditioning is the timestep plus the guidance scale. The text
    sequence is supplied directly (the release encodes it with Mistral-Small 3), so the DiT is verified
    in isolation, as the FLUX.1, SD3 and LTX DiTs are. Runs under the `qwenimage` oracle env, which
    carries the diffusers revision that has `Flux2Transformer2DModel`. `image` unused.
    """
    from diffusers import Flux2Transformer2DModel

    model = Flux2Transformer2DModel(
        patch_size=1, in_channels=8, num_layers=2, num_single_layers=2, attention_head_dim=8,
        num_attention_heads=2, joint_attention_dim=24, timestep_guidance_channels=16, mlp_ratio=3.0,
        axes_dims_rope=(2, 2, 2, 2), rope_theta=2000, eps=1e-6, guidance_embeds=True)
    model = _randomized(model, seed=31)

    generator = torch.Generator().manual_seed(8)
    lh, lw, txt = 2, 3, 5                                                  # 6 image tokens, 5 text tokens
    hidden = torch.randn(2, lh * lw, 8, generator=generator)               # [B, img_seq, in_channels]
    encoder = torch.randn(2, txt, 24, generator=generator)                 # [B, txt_seq, joint_attention_dim]
    t = torch.tensor([0.5, 0.5])
    guidance = torch.tensor([3.5, 3.5])

    # The pipeline's `_prepare_latent_ids` / `_prepare_text_ids`: image ids are (0, row, col, 0), text
    # ids are (0, 0, 0, token index).
    img_ids = torch.cartesian_prod(torch.arange(1), torch.arange(lh), torch.arange(lw),
                                   torch.arange(1)).float()                # [img_seq, 4]
    txt_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1),
                                   torch.arange(txt)).float()              # [txt_seq, 4]

    seams = {}
    model.time_guidance_embed.register_forward_hook(
        lambda m, i, o: seams.__setitem__("temb", o.detach().clone()))
    model.double_stream_modulation_img.register_forward_hook(
        lambda m, i, o: seams.__setitem__("mod_img", o.detach().clone()))
    model.single_stream_modulation.register_forward_hook(
        lambda m, i, o: seams.__setitem__("mod_single", o.detach().clone()))
    model.transformer_blocks[0].register_forward_hook(
        lambda m, i, o: seams.__setitem__("double0", (o[0].detach().clone(), o[1].detach().clone())))
    model.single_transformer_blocks[0].register_forward_hook(
        lambda m, i, o: seams.__setitem__("single0", o.detach().clone()))
    with torch.no_grad():
        output = model(hidden_states=hidden, encoder_hidden_states=encoder, timestep=t,
                       img_ids=img_ids, txt_ids=txt_ids, guidance=guidance, return_dict=False)[0]

    extra = {"hidden": hidden.contiguous(), "encoder": encoder.contiguous(),
             "timestep": t.contiguous(), "guidance": guidance.contiguous(),
             "img_ids": img_ids.contiguous(), "txt_ids": txt_ids.contiguous(),
             "temb": seams["temb"].contiguous(), "mod_img": seams["mod_img"].contiguous(),
             "mod_single": seams["mod_single"].contiguous(),
             "double0_txt": seams["double0"][0].contiguous(),
             "double0_img": seams["double0"][1].contiguous(),
             "single0": seams["single0"].contiguous()}
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return output.contiguous()                                             # [B, img_seq, out_channels]


def run_flux2_real(image, checkpoint):
    """The FLUX.2 [klein] transformer velocity on the RELEASED weights, at the precision they ship in.

    `--checkpoint` is the release's `transformer/` directory. `run_flux2` measures the arithmetic
    exactly at a tiny configuration; this measures the DECLARED geometry against a real checkpoint, so
    a preset that loads 169 tensors at the right shapes and still computes something else is caught.
    Both sides run bfloat16, the precision the release ships, so the figure describes that precision
    rather than the arithmetic's ceiling. klein 4B is 3.88B, about 7.8 GB in bfloat16, which this
    machine holds; the 9B sizes and [dev] do not reach a numeric run here.

    The spatial input is deliberately tiny (a 2x2 packed-latent grid, 4 image tokens, 8 text tokens) so
    only the WEIGHTS are large, not the activations. The text conditioning is random rather than the
    Qwen3 encoder's output, which keeps the transformer measured in isolation, the way `run_flux_real`
    does for FLUX.1. klein carries no guidance embedding, so `guidance` is None where the config says
    so. Runs under the `qwenimage` oracle env, which carries the diffusers revision that has
    `Flux2Transformer2DModel`. `image` unused.
    """
    return _flux2_real(checkpoint, "bfloat16")


def run_flux2_real_f32(image, checkpoint):
    """`run_flux2_real` at float32 on the same released weights and the same inputs.

    The bfloat16 record measures the release at the precision it ships; this one measures the
    arithmetic on real weights, which the tiny configuration cannot, because a precision-specific
    difference (a dtype promotion, an accumulation order) is invisible at float32 on random weights
    and at bfloat16 only shows as a lower cosine that could be read as noise. klein 4B is 15.5 GB at
    float32, which this machine holds in one process. The inputs are drawn exactly as the bfloat16
    mode draws them, so the two records differ in precision alone. `image` unused.
    """
    return _flux2_real(checkpoint, "float32")


def _flux2_real(checkpoint, precision):
    import torch
    from diffusers import Flux2Transformer2DModel

    dtype = getattr(torch, precision)
    model = Flux2Transformer2DModel.from_pretrained(checkpoint, torch_dtype=dtype).eval()

    generator = torch.Generator().manual_seed(8)
    lh, lw, txt = 2, 2, 8                                                  # 4 image tokens, 8 text tokens
    channels = model.config.in_channels                                    # 128 for klein 4B
    joint = model.config.joint_attention_dim                               # 7680 = 3 x 2560
    # Drawn at float32 and rounded to bfloat16 in BOTH modes, so the float32 record starts from the
    # exact inputs the bfloat16 one does and the two differ in the network's precision alone.
    hidden = torch.randn(1, lh * lw, channels, generator=generator).to(torch.bfloat16).to(dtype)
    encoder = torch.randn(1, txt, joint, generator=generator).to(torch.bfloat16).to(dtype)
    t = torch.tensor([0.5], dtype=dtype)
    guidance = (torch.tensor([3.5], dtype=dtype)
                if model.config.guidance_embeds else None)

    img_ids = torch.cartesian_prod(torch.arange(1), torch.arange(lh), torch.arange(lw),
                                   torch.arange(1)).float()                # (0, row, col, 0)
    txt_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1),
                                   torch.arange(txt)).float()              # (0, 0, 0, token index)

    with torch.no_grad():
        output = model(hidden_states=hidden, encoder_hidden_states=encoder, timestep=t,
                       img_ids=img_ids, txt_ids=txt_ids, guidance=guidance, return_dict=False)[0]

    extra = {"hidden": hidden.float().clone().contiguous(),
             "encoder": encoder.float().clone().contiguous(),
             "timestep": t.float().clone().contiguous(),
             "img_ids": img_ids.clone().contiguous(),
             "txt_ids": txt_ids.clone().contiguous(),
             "num_layers": torch.tensor([model.config.num_layers], dtype=torch.int32),
             "num_single_layers": torch.tensor([model.config.num_single_layers], dtype=torch.int32)}
    if guidance is not None:
        extra["guidance"] = guidance.float().clone().contiguous()
    globals()["_extra"] = extra
    return output.float().clone().contiguous()                             # [B, img_seq, out_channels]


def run_flux2_real_truncated(image, checkpoint):
    """A released FLUX.2 transformer cut to its first two double and first two single blocks, at
    float32 and at bfloat16, on the released weights and `_flux2_real`'s inputs.

    `--checkpoint` is the release's `transformer/` directory. It exists for the 9B release, which is
    36 GB at float32 and does not fit this machine whole, so `run_flux2_real_f32` cannot separate its
    bfloat16 figure's precision floor from a defect. The cut keeps everything specific to the
    geometry (the width, the head count, the 12288-wide text projection, the modulation heads, the
    output head) and drops only depth, which repeats the same blocks. Only the kept tensors are read,
    through `safe_open`, so the model never exists whole in memory. The record carries the float32
    output, the output of the SAME cut at bfloat16 (the reference's own precision gap at this depth),
    and the inputs.
    """
    import json
    import os
    import torch
    from diffusers import Flux2Transformer2DModel
    from safetensors import safe_open

    config = json.load(open(os.path.join(checkpoint, "config.json")))
    config = {k: v for k, v in config.items() if not k.startswith("_")}
    config.update(num_layers=2, num_single_layers=2)
    model = Flux2Transformer2DModel(**config).eval()
    wanted = set(model.state_dict())
    state = {}
    for name in sorted(os.listdir(checkpoint)):
        if name.endswith(".safetensors"):
            with safe_open(os.path.join(checkpoint, name), framework="pt") as f:
                for key in f.keys():
                    if key in wanted:
                        state[key] = f.get_tensor(key).float()
    model.load_state_dict(state, strict=True)

    generator = torch.Generator().manual_seed(8)
    lh, lw, txt = 2, 2, 8
    hidden = torch.randn(1, lh * lw, model.config.in_channels, generator=generator).to(torch.bfloat16)
    encoder = torch.randn(1, txt, model.config.joint_attention_dim, generator=generator).to(torch.bfloat16)
    t = torch.tensor([0.5])
    img_ids = torch.cartesian_prod(torch.arange(1), torch.arange(lh), torch.arange(lw),
                                   torch.arange(1)).float()
    txt_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1),
                                   torch.arange(txt)).float()

    def run(dtype):
        with torch.no_grad():
            return model.to(dtype)(hidden_states=hidden.to(dtype), encoder_hidden_states=encoder.to(dtype),
                                   timestep=t.to(dtype), img_ids=img_ids, txt_ids=txt_ids,
                                   guidance=None, return_dict=False)[0].float()

    output_f32 = run(torch.float32)
    output_bf16 = run(torch.bfloat16)
    a, b = output_f32.flatten().double(), output_bf16.flatten().double()
    print(f"  the reference's own bfloat16 against its float32 on the cut: {float(a @ b / a.norm() / b.norm()):.10f}")
    globals()["_extra"] = {
        "hidden": hidden.float().clone().contiguous(), "encoder": encoder.float().clone().contiguous(),
        "timestep": t.clone().contiguous(), "img_ids": img_ids.clone().contiguous(),
        "txt_ids": txt_ids.clone().contiguous(), "output_bf16": output_bf16.clone().contiguous()}
    return output_f32.clone().contiguous()


def run_flux2_kv_real(image, checkpoint):
    """FLUX.2 [klein] 9B KV's reference cache on its RELEASED transformer at bfloat16: the extracting
    step, a cached step, and ORDINARY reference conditioning on the same tokens as a control.

    `--checkpoint` is the release's `transformer/` directory. The tiny oracle (`run_flux2_kv`) had to
    raise its weight scale before the reference cache and ordinary conditioning came apart; on the
    released weights the control answers directly how far apart the two are. 18 GB. `image` unused.
    """
    from diffusers import Flux2Transformer2DModel

    model = Flux2Transformer2DModel.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).eval()
    globals()["_extra"] = _flux2_kv_forwards(model, torch.bfloat16)
    return globals()["_extra"]["extracted"].clone().contiguous()


def run_flux2_kv_real_truncated(image, checkpoint):
    """`run_flux2_kv_real` on the release cut to its first two double and two single blocks, at
    float32 and bfloat16. The whole release is 36 GB at float32; the cut keeps the geometry and drops
    repeated depth, and only its tensors are read. `image` unused.
    """
    import json
    import os
    from diffusers import Flux2Transformer2DModel
    from safetensors import safe_open

    config = {k: v for k, v in json.load(open(os.path.join(checkpoint, "config.json"))).items()
              if not k.startswith("_")}
    config.update(num_layers=2, num_single_layers=2)
    model = Flux2Transformer2DModel(**config).eval()
    wanted, state = set(model.state_dict()), {}
    for name in sorted(os.listdir(checkpoint)):
        if name.endswith(".safetensors"):
            with safe_open(os.path.join(checkpoint, name), framework="pt") as f:
                for key in f.keys():
                    if key in wanted:
                        state[key] = f.get_tensor(key).float()
    model.load_state_dict(state, strict=True)
    extra = _flux2_kv_forwards(model, torch.float32)
    for key, value in _flux2_kv_forwards(model.to(torch.bfloat16), torch.bfloat16).items():
        if key in ("extracted", "cached", "ordinary"):
            extra[f"{key}_bf16"] = value
            a, b = extra[key].flatten().double(), value.flatten().double()
            print(f"  the reference's own bfloat16 against its float32, {key}: "
                  f"{float(a @ b / a.norm() / b.norm()):.10f}")
    globals()["_extra"] = extra
    return extra["extracted"].clone().contiguous()


def _flux2_kv_forwards(model, dtype):
    """The extracting step, a cached step and ordinary conditioning on fixed tokens: a 2x3 reference
    grid, a 2x2 generated grid and 8 text tokens, drawn at float32 and rounded to bfloat16 so a float32
    and a bfloat16 run start from the same values."""
    from diffusers.models.transformers.transformer_flux2 import (
        Flux2KVAttnProcessor, Flux2KVParallelSelfAttnProcessor)

    for block in model.transformer_blocks:
        block.attn.set_processor(Flux2KVAttnProcessor())
    for block in model.single_transformer_blocks:
        block.attn.set_processor(Flux2KVParallelSelfAttnProcessor())
    generator = torch.Generator().manual_seed(29)
    def draw(*shape):
        return torch.randn(*shape, generator=generator).to(torch.bfloat16).to(dtype)
    channels, joint = model.config.in_channels, model.config.joint_attention_dim
    reference, latents, later = draw(1, 6, channels), draw(1, 4, channels), draw(1, 4, channels)
    embeds = draw(1, 8, joint)
    reference_ids = torch.cartesian_prod(torch.tensor([10]), torch.arange(2), torch.arange(3),
                                         torch.arange(1)).float()
    latent_ids = torch.cartesian_prod(torch.arange(1), torch.arange(2), torch.arange(2),
                                      torch.arange(1)).float()
    text_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1),
                                    torch.arange(8)).float()
    t, t_later = torch.tensor([0.8], dtype=dtype), torch.tensor([0.45], dtype=dtype)
    with torch.no_grad():
        extracted, cache = model(
            hidden_states=torch.cat([reference, latents], dim=1), encoder_hidden_states=embeds,
            timestep=t, img_ids=torch.cat([reference_ids, latent_ids]), txt_ids=text_ids,
            guidance=None, return_dict=False, kv_cache_mode="extract", num_ref_tokens=6)
        cached = model(hidden_states=later, encoder_hidden_states=embeds, timestep=t_later,
                       img_ids=latent_ids, txt_ids=text_ids, guidance=None, return_dict=False,
                       kv_cache=cache, kv_cache_mode="cached")[0]
        ordinary = model(hidden_states=torch.cat([latents, reference], dim=1),
                         encoder_hidden_states=embeds, timestep=t,
                         img_ids=torch.cat([latent_ids, reference_ids]), txt_ids=text_ids,
                         guidance=None, return_dict=False)[0][:, :4]
    e, o = extracted.flatten().double(), ordinary.flatten().double()
    print(f"  reference cache vs ordinary conditioning ({dtype}): {float(e @ o / e.norm() / o.norm()):.10f}")
    return {k: v.detach().float().clone().contiguous() for k, v in {
        "reference": reference, "latents": latents, "later": later, "embeds": embeds,
        "reference_ids": reference_ids, "latent_ids": latent_ids, "text_ids": text_ids,
        "timestep": t, "timestep_later": t_later, "extracted": extracted, "cached": cached,
        "ordinary": ordinary}.items()}


def _ltx2_tiny(**over):
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    settings = dict(
        in_channels=8, out_channels=8, patch_size=1, patch_size_t=1,
        num_attention_heads=2, attention_head_dim=16, cross_attention_dim=32,
        vae_scale_factors=(8, 32, 32), pos_embed_max_pos=20, base_height=64, base_width=64,
        gated_attn=True, cross_attn_mod=True,
        audio_in_channels=6, audio_out_channels=6, audio_patch_size=1, audio_patch_size_t=1,
        audio_num_attention_heads=2, audio_attention_head_dim=8, audio_cross_attention_dim=16,
        audio_scale_factor=4, audio_pos_embed_max_pos=20, audio_sampling_rate=16000,
        audio_hop_length=160, audio_gated_attn=True, audio_cross_attn_mod=True,
        num_layers=2, activation_fn="gelu-approximate", qk_norm="rms_norm_across_heads",
        norm_elementwise_affine=False, norm_eps=1e-6, caption_channels=14,
        attention_bias=True, attention_out_bias=True, rope_theta=10000.0,
        rope_double_precision=True, causal_offset=1, timestep_scale_multiplier=1000,
        cross_attn_timestep_scale_multiplier=1000, rope_type="split",
        use_prompt_embeddings=False, perturbed_attn=True, ff_bias=True, audio_ff_bias=True,
        use_prompt_adaln_single=True, use_keyframes_abs_pos_embedding=False)
    settings.update(over)
    return LTX2VideoTransformer3DModel(**settings)


def run_ltx2(image):
    """The LTX-2 audio-video transformer (`LTX2VideoTransformer3DModel`, Lightricks) at a tiny random
    configuration, in BOTH the arrangement the ungated LTX-2.3 release declares and the three switches
    LTX-2.5 changes.

    One transformer denoises a video latent and an audio latent together. Each block runs SIX
    attentions: video self-attention, audio self-attention, video-over-text and audio-over-text cross
    attention, and the two cross-modal directions (audio-to-video, where the video asks and the audio
    answers, and video-to-audio the other way). Every attention takes an ACROSS-HEADS RMS norm — the
    query and key are normalized over the whole projected width before the heads are split, not per
    head — and a per-head gate of `2 · sigmoid(linear(x))`, so a zero-initialized gate leaves the
    attention unchanged. The rotary is the `split` kind: the channel axis halves into (real, imaginary)
    blocks rather than interleaving adjacent pairs.

    Modulation is PixArt-alpha's adaptive-norm-single raised to ten heads. Six live on the model (the
    video and audio timestep embeddings, the two cross-modal scale/shift heads, the two cross-modal
    gates) and each block adds its own `scale_shift_table` on top, so a block's parameters are the
    per-layer DELTA of a globally computed vector.

    LTX-2.5 differs from LTX-2.3 in three declared switches: the video feed-forward drops its bias,
    the prompt cross-attention modulation becomes timestep-independent (`use_prompt_adaln_single`
    False, which drops the two prompt adaptive-norm heads and makes the text key/value cacheable
    across denoising steps), and a learned absolute-position embedding marks generated-keyframe
    tokens. Both arrangements are recorded here, under the `l23.` and `l25.` prefixes, because
    LTX-2.5's own `config.json` is behind Lightricks' gate: the arithmetic of every switch is measured
    even where which switch the release sets cannot be read. Runs under the `qwenimage` oracle env.
    `image` unused.
    """
    frames, height, width, audio_frames, text = 2, 2, 3, 4, 5
    tokens = frames * height * width

    generator = torch.Generator().manual_seed(3)
    hidden = torch.randn(1, tokens, 8, generator=generator)
    audio_hidden = torch.randn(1, audio_frames, 6, generator=generator)
    encoder = torch.randn(1, text, 32, generator=generator)
    audio_encoder = torch.randn(1, text, 16, generator=generator)
    timestep = torch.full((1, tokens), 500.0)
    audio_timestep = torch.full((1, audio_frames), 500.0)
    sigma = torch.tensor([0.5])

    extra = {"hidden": hidden.contiguous(), "audio_hidden": audio_hidden.contiguous(),
             "encoder": encoder.contiguous(), "audio_encoder": audio_encoder.contiguous(),
             "timestep": timestep.contiguous(), "audio_timestep": audio_timestep.contiguous(),
             "sigma": sigma.contiguous()}
    outputs = {}

    # `l20` carries the caption projections LTX-2.0 keeps inside the transformer, so that path is
    # measured too; its text arrives at `caption_channels` width rather than the attention's.
    caption = torch.randn(1, text, 14, generator=generator)
    audio_caption = torch.randn(1, text, 14, generator=generator)
    # The mask marks the tokens whose latent holds a single pixel frame; it broadcasts against the
    # embedding, so it carries a trailing axis of one.
    keyframes = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                             dtype=torch.float32).unsqueeze(-1)
    extra["caption"] = caption.contiguous()
    extra["audio_caption"] = audio_caption.contiguous()
    extra["keyframes"] = keyframes.contiguous()

    for prefix, settings in (("l20", dict(use_prompt_embeddings=True)),
                             ("l23", {}),
                             ("l25", dict(ff_bias=False, use_prompt_adaln_single=False,
                                          use_keyframes_abs_pos_embedding=True))):
        model = _randomized(_ltx2_tiny(**settings), seed=37)
        seams = {}
        model.rope.register_forward_hook(
            lambda m, i, o, s=seams: s.__setitem__("rope", (o[0].detach().clone(), o[1].detach().clone())))
        model.time_embed.register_forward_hook(
            lambda m, i, o, s=seams: s.__setitem__("temb", o[0].detach().clone()))
        model.audio_time_embed.register_forward_hook(
            lambda m, i, o, s=seams: s.__setitem__("temb_audio", o[0].detach().clone()))
        model.transformer_blocks[0].attn1.register_forward_hook(
            lambda m, i, o, s=seams: s.__setitem__("attn1", o.detach().clone()))
        model.transformer_blocks[0].audio_to_video_attn.register_forward_hook(
            lambda m, i, o, s=seams: s.__setitem__("a2v", o.detach().clone()))
        model.transformer_blocks[0].register_forward_hook(
            lambda m, i, o, s=seams: s.__setitem__("block0", (o[0].detach().clone(),
                                                              o[1].detach().clone())))
        with torch.no_grad():
            video, audio = model(
                hidden_states=hidden, audio_hidden_states=audio_hidden,
                encoder_hidden_states=caption if prefix == "l20" else encoder,
                audio_encoder_hidden_states=audio_caption if prefix == "l20" else audio_encoder,
                timestep=timestep, audio_timestep=audio_timestep, sigma=sigma,
                num_frames=frames, height=height, width=width, audio_num_frames=audio_frames,
                video_keyframes_mask=keyframes if prefix == "l25" else None,
                return_dict=False)

        extra[f"{prefix}.video"] = video.contiguous()
        extra[f"{prefix}.audio"] = audio.contiguous()
        extra[f"{prefix}.rope_cos"] = seams["rope"][0].contiguous()
        extra[f"{prefix}.rope_sin"] = seams["rope"][1].contiguous()
        extra[f"{prefix}.temb"] = seams["temb"].contiguous()
        extra[f"{prefix}.temb_audio"] = seams["temb_audio"].contiguous()
        extra[f"{prefix}.attn1"] = seams["attn1"].contiguous()
        extra[f"{prefix}.a2v"] = seams["a2v"].contiguous()
        extra[f"{prefix}.block0_video"] = seams["block0"][0].contiguous()
        extra[f"{prefix}.block0_audio"] = seams["block0"][1].contiguous()
        for key, value in model.state_dict().items():
            extra[f"w::{prefix}.{key}"] = value.float().contiguous()
        outputs[prefix] = video

    globals()["_extra"] = extra
    return outputs["l25"].clone().contiguous()


def run_flux2_vae(image):
    """FLUX.2's autoencoder at a tiny random configuration, and the latent codec the pipeline wraps
    it in, from diffusers' AutoencoderKLFlux2.

    The autoencoder itself is the ordinary `AutoencoderKL` the Stable Diffusion family uses, at 32
    latent channels with the quantization convolutions kept. What is new is outside it. FLUX.2 has no
    scalar `scaling_factor`/`shift_factor`: the latent is PATCHIFIED 2x2 into the channel axis and then
    whitened by a BatchNorm's RUNNING STATISTICS, which the release ships as `bn.running_mean` and
    `bn.running_var` beside the encoder and decoder. The transformer's 128 input channels are the 32
    latent channels times that 2x2 patch.

    The patchify order is the trap: `permute(0, 1, 3, 5, 2, 4)` puts the two sub-pixel axes directly
    after the channel, so a channel's four patch offsets are adjacent. A pixel-unshuffle that groups
    by spatial position instead produces the same shape and different contents. Runs under the
    `qwenimage` oracle env. `image` unused.
    """
    from diffusers import AutoencoderKLFlux2

    model = AutoencoderKLFlux2(
        in_channels=3, out_channels=3, block_out_channels=(8, 16), layers_per_block=1,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        latent_channels=4, norm_num_groups=4, sample_size=32, use_quant_conv=True,
        use_post_quant_conv=True, mid_block_add_attention=True, batch_norm_eps=1e-4,
        patch_size=(2, 2))
    model = _randomized(model, seed=41)
    # `_randomized` leaves the BatchNorm buffers alone, so give them values a whitening step would
    # actually use: a non-zero mean and a positive variance.
    torch.manual_seed(5)
    model.bn.running_mean.copy_(torch.randn(model.bn.running_mean.shape) * 0.3)
    model.bn.running_var.copy_(torch.rand(model.bn.running_var.shape) * 0.5 + 0.5)

    generator = torch.Generator().manual_seed(12)
    pixels = torch.randn(1, 3, 16, 16, generator=generator)

    # Decoder seams, so a disagreement inside the decode localizes to a stage rather than being
    # guessed at from the final image.
    seams = {}
    model.post_quant_conv.register_forward_hook(
        lambda m, i, o: seams.__setitem__("post_quant", o.detach().clone()))
    model.decoder.conv_in.register_forward_hook(
        lambda m, i, o: seams.__setitem__("dec_conv_in", o.detach().clone()))
    model.decoder.mid_block.register_forward_hook(
        lambda m, i, o: seams.__setitem__("dec_mid", o.detach().clone()))
    model.decoder.up_blocks[0].register_forward_hook(
        lambda m, i, o: seams.__setitem__("dec_up0", o.detach().clone()))
    model.decoder.conv_norm_out.register_forward_hook(
        lambda m, i, o: seams.__setitem__("dec_norm_out", o.detach().clone()))

    with torch.no_grad():
        posterior = model.encode(pixels).latent_dist
        latent = posterior.mode()                                          # [1, 4, 4, 4]
        patched = _flux2_patchify(latent)                                  # [1, 16, 2, 2]
        mean = model.bn.running_mean.view(1, -1, 1, 1)
        std = torch.sqrt(model.bn.running_var.view(1, -1, 1, 1) + model.config.batch_norm_eps)
        whitened = (patched - mean) / std
        packed = whitened.reshape(1, whitened.shape[1], -1).permute(0, 2, 1)   # [1, 4, 16]

        restored = packed.permute(0, 2, 1).reshape(whitened.shape)
        unwhitened = restored * std + mean
        unpatched = _flux2_unpatchify(unwhitened)
        decoded = model.decode(unpatched, return_dict=False)[0]

    extra = {f"seam.{k}": v.contiguous() for k, v in seams.items()}
    extra |= {"pixels": pixels.contiguous(), "latent": latent.contiguous(),
             "patched": patched.contiguous(), "whitened": whitened.contiguous(),
             "packed": packed.contiguous(), "unpatched": unpatched.contiguous(),
             "decoded": decoded.contiguous()}
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().contiguous()
    globals()["_extra"] = extra
    return decoded.clone().contiguous()


def run_flux2_vae_real(image, checkpoint):
    """FLUX.2's autoencoder and latent codec on the RELEASED weights, from diffusers'
    AutoencoderKLFlux2.

    `--checkpoint` is the release's `vae/` directory. `run_flux2_vae` measures the arithmetic at a
    tiny random configuration; this measures the shipped 32-channel autoencoder and the BatchNorm
    running statistics the release carries, which random buffers cannot stand in for. The input is
    the harness image resized to 64x64 and mapped to [-1, 1], an in-distribution picture rather than
    noise, because these are trained weights. 64x64 at the release's stride of 8 is an 8x8 latent, a
    4x4 patched grid, 16 tokens. The model is 84M parameters, so float32 costs nothing here.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKLFlux2

    model = AutoencoderKLFlux2.from_pretrained(checkpoint, torch_dtype=torch.float32).eval()

    picture = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
    pixels = F.interpolate(picture, size=(64, 64), mode="bilinear", align_corners=False) * 2 - 1

    with torch.no_grad():
        latent = model.encode(pixels).latent_dist.mode()                   # [1, 32, 8, 8]
        patched = _flux2_patchify(latent)                                  # [1, 128, 4, 4]
        mean = model.bn.running_mean.view(1, -1, 1, 1)
        std = torch.sqrt(model.bn.running_var.view(1, -1, 1, 1) + model.config.batch_norm_eps)
        whitened = (patched - mean) / std
        packed = whitened.reshape(1, whitened.shape[1], -1).permute(0, 2, 1)   # [1, 16, 128]

        restored = packed.permute(0, 2, 1).reshape(whitened.shape)
        unpatched = _flux2_unpatchify(restored * std + mean)
        decoded = model.decode(unpatched, return_dict=False)[0]

    globals()["_extra"] = {
        "pixels": pixels.clone().contiguous(), "latent": latent.clone().contiguous(),
        "patched": patched.clone().contiguous(), "whitened": whitened.clone().contiguous(),
        "packed": packed.clone().contiguous(), "unpatched": unpatched.clone().contiguous(),
        "decoded": decoded.clone().contiguous()}
    return decoded.clone().contiguous()


def _flux2_patchify(latents):
    batch, channels, height, width = latents.shape
    out = latents.view(batch, channels, height // 2, 2, width // 2, 2)
    out = out.permute(0, 1, 3, 5, 2, 4)
    return out.reshape(batch, channels * 4, height // 2, width // 2)


def _flux2_unpatchify(latents):
    batch, channels, height, width = latents.shape
    out = latents.reshape(batch, channels // 4, 2, 2, height, width)
    out = out.permute(0, 1, 4, 2, 5, 3)
    return out.reshape(batch, channels // 4, height * 2, width * 2)


def run_flux2_text(image):
    """FLUX.2 [klein]'s text front end at a tiny random Qwen3, from the reference's own
    `_get_qwen3_prompt_embeds`.

    The conditioning is not a language model's output. It is THREE of its intermediate hidden states
    concatenated per token: the release reads layers 9, 18 and 27 of a 36-layer Qwen3 and stacks them
    on the channel axis, which is why `joint_attention_dim` is three times the language model's
    width. `output.hidden_states[k]` is the state AFTER k decoder layers, with index 0 the embedding,
    so the three indices name layer outputs rather than blocks.

    The prompt is padded to `max_sequence_length` on the RIGHT and the whole padded sequence is
    encoded, attention mask included. That mask matters: a pad position attends to the real tokens and
    to the earlier pads, so masking the pad KEYS changes the pad positions' own states, and those
    states are part of the conditioning the transformer reads. The record carries the embedding with
    the mask and without it, so the difference is measured rather than argued about. Runs under the
    `qwenimage` oracle env. `image` unused.
    """
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        hidden_size=32, num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, intermediate_size=64, vocab_size=128, rms_norm_eps=1e-6, rope_theta=1000000.0,
        tie_word_embeddings=True, max_position_embeddings=64, attention_bias=False)
    model = _randomized(Qwen3ForCausalLM(config), seed=43)

    layers = [1, 3, 5]
    length, real = 12, 7
    tokens = torch.arange(1, real + 1, dtype=torch.long).unsqueeze(0)
    padding = torch.full((1, length - real), 0, dtype=torch.long)
    input_ids = torch.cat([tokens, padding], dim=1)                        # right padding
    attention_mask = torch.cat([torch.ones(1, real, dtype=torch.long),
                                torch.zeros(1, length - real, dtype=torch.long)], dim=1)

    def embedding(mask):
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True,
                        use_cache=False)
        stacked = torch.stack([out.hidden_states[k] for k in layers], dim=1)
        batch, channels, sequence, width = stacked.shape
        return (stacked.permute(0, 2, 1, 3).reshape(batch, sequence, channels * width),
                [h.detach().clone() for h in out.hidden_states])

    masked, states = embedding(attention_mask)
    unmasked, _ = embedding(None)

    # Every entry is cloned: `contiguous()` returns the tensor itself when it already is, and
    # safetensors refuses to write two names that alias one buffer.
    extra = {"input_ids": input_ids[0].to(torch.int32).clone().contiguous(),
             "attention_mask": attention_mask[0].to(torch.int32).clone().contiguous(),
             "embedding": masked.clone().contiguous(),
             "embedding_unmasked": unmasked.clone().contiguous()}
    for index, state in enumerate(states):
        extra[f"hidden.{index}"] = state.clone().contiguous()
    # A tied Qwen3 has `lm_head.weight` aliasing `model.embed_tokens.weight`, which safetensors
    # refuses to write twice; the clone keeps the record faithful rather than dropping a key.
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().clone().contiguous()
    globals()["_extra"] = extra
    return masked.clone().contiguous()


def run_flux2_text_mistral(image):
    """FLUX.2 [dev]'s text front end at a tiny random Mistral-Small 3, from transformers' own
    MistralForCausalLM.

    [dev] conditions on Mistral-Small 3 where [klein] conditions on Qwen3, and the front end is
    otherwise the one `run_flux2_text` records: THREE intermediate hidden states concatenated per
    token, right padding to the context length, and an attention mask that changes the pad positions'
    own states. The released decoder is 40 layers and `joint_attention_dim` 15360 is three times its
    5120 width, so the layers read are 10, 20 and 30; this configuration keeps that ratio at 8 layers
    and reads 2, 4 and 6.

    Two things differ from the Qwen3 front end and are the reason this record exists. Mistral does not
    normalize queries and keys per head, and its head width is STATED rather than implied: the release
    sets `head_dim` 128 against a 5120 residual over 32 heads, which divides to 160, so the attention
    projections are narrower than the residual. `head_dim` 8 against a 40-wide residual over 4 heads
    reproduces that. `sliding_window` is None because the release sets it null; a reader that took
    MistralConfig's default would apply a sliding mask the release does not. Runs under the `llm`
    oracle env. `image` unused.
    """
    from transformers import MistralConfig, MistralForCausalLM

    config = MistralConfig(
        hidden_size=40, num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, intermediate_size=64, vocab_size=128, rms_norm_eps=1e-5,
        rope_theta=1000000000.0, tie_word_embeddings=False, max_position_embeddings=64,
        attention_bias=False, sliding_window=None)
    model = _randomized(MistralForCausalLM(config), seed=47)

    layers = [2, 4, 6]
    length, real = 12, 7
    tokens = torch.arange(1, real + 1, dtype=torch.long).unsqueeze(0)
    padding = torch.full((1, length - real), 0, dtype=torch.long)
    input_ids = torch.cat([tokens, padding], dim=1)                        # right padding
    attention_mask = torch.cat([torch.ones(1, real, dtype=torch.long),
                                torch.zeros(1, length - real, dtype=torch.long)], dim=1)

    def embedding(mask):
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True,
                        use_cache=False)
        stacked = torch.stack([out.hidden_states[k] for k in layers], dim=1)
        batch, channels, sequence, width = stacked.shape
        return (stacked.permute(0, 2, 1, 3).reshape(batch, sequence, channels * width),
                [h.detach().clone() for h in out.hidden_states])

    masked, states = embedding(attention_mask)
    unmasked, _ = embedding(None)

    extra = {"input_ids": input_ids[0].to(torch.int32).clone().contiguous(),
             "attention_mask": attention_mask[0].to(torch.int32).clone().contiguous(),
             "embedding": masked.clone().contiguous(),
             "embedding_unmasked": unmasked.clone().contiguous()}
    for index, state in enumerate(states):
        extra[f"hidden.{index}"] = state.clone().contiguous()
    for key, value in model.state_dict().items():
        extra[f"w::{key}"] = value.float().clone().contiguous()
    globals()["_extra"] = extra
    return masked.clone().contiguous()


def run_flux2_inpaint(image):
    """FLUX.2 [klein] inpainting, from the reference's OWN `Flux2KleinInpaintPipeline.__call__`, at a
    tiny random transformer and autoencoder.

    The pipeline conditions on the source image twice. Its whitened latent is the first REFERENCE,
    appended to the token sequence at time coordinate 10, which is how FLUX.2 edits. It is also the
    starting point: `get_timesteps` starts the loop `int(N - N * strength)` steps in, from the image
    noised to that sigma, and after every Euler step the kept region is overwritten with the image
    noised to the NEXT sigma, with the noise drawn once at the start; the last step blends with the
    clean image. `scale_noise` picks its sigma by INDEX (`begin_index` before the loop, `step_index`
    after a step), not by the timestep value it is handed.

    The inputs are chosen to reach every trap. 50 steps at `strength` 0.8 are the pipeline's own
    defaults and a double-precision case: the loop starts at step 10, where a single-precision 0.8
    (0.80000001) starts at step 9. Of the step counts 4..50 and strengths 0.01..0.99, 78 pairs start
    differently in single precision; (10, 0.7) is not one of them, because `10 * 0.7` rounds to
    exactly 7.0. The mask's white rectangle (rows 2..9, columns 6..13 of 16) has edges that fall
    BETWEEN the 4x4 packing cells, so the bilinear downsample to the packed grid produces fractional
    blend weights rather than a clean 0/1 grid. Two runs share the image, mask and noise: `distilled`
    (klein 4B's own setting, no guidance) and `guided` (a base release, guidance 3 against a negative
    conditioning). The start noise, the packed mask, the source latent and the latents after every
    step are captured from inside the reference, so a disagreement names a step. `image` is the
    harness picture at 16x16. Runs under the `qwenimage` oracle env.
    """
    import numpy as np
    from PIL import Image
    from diffusers import (AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler,
                           Flux2Transformer2DModel)
    from diffusers.pipelines.flux2.pipeline_flux2_klein_inpaint import Flux2KleinInpaintPipeline

    transformer = _randomized(Flux2Transformer2DModel(
        patch_size=1, in_channels=16, num_layers=2, num_single_layers=2, attention_head_dim=8,
        num_attention_heads=2, joint_attention_dim=24, timestep_guidance_channels=16, mlp_ratio=3.0,
        axes_dims_rope=(2, 2, 2, 2), rope_theta=2000, eps=1e-6, guidance_embeds=False), seed=51)
    vae = _randomized(AutoencoderKLFlux2(
        in_channels=3, out_channels=3, block_out_channels=(8, 16), layers_per_block=1,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        latent_channels=4, norm_num_groups=4, sample_size=32, use_quant_conv=True,
        use_post_quant_conv=True, mid_block_add_attention=True, batch_norm_eps=1e-4,
        patch_size=(2, 2)), seed=53)
    torch.manual_seed(7)
    vae.bn.running_mean.copy_(torch.randn(vae.bn.running_mean.shape) * 0.3)
    vae.bn.running_var.copy_(torch.rand(vae.bn.running_var.shape) * 0.5 + 0.5)
    # The released klein scheduler config (`scheduler/scheduler_config.json`).
    scheduler_config = dict(base_image_seq_len=256, base_shift=0.5, max_image_seq_len=4096,
                            max_shift=1.15, num_train_timesteps=1000, shift=3.0,
                            use_dynamic_shifting=True, time_shift_type="exponential")

    picture = Image.fromarray((np.asarray(image) * 255).clip(0, 255).astype(np.uint8)).resize(
        (16, 16), Image.BICUBIC)
    mask_pixels = np.zeros((16, 16), dtype=np.uint8)
    mask_pixels[2:10, 6:14] = 255
    mask = Image.fromarray(mask_pixels, mode="L")

    generator = torch.Generator().manual_seed(21)
    embeds = torch.randn(1, 5, 24, generator=generator)
    negative = torch.randn(1, 5, 24, generator=generator)

    extra = {"embeds": embeds.clone().contiguous(), "negative": negative.clone().contiguous()}
    for name, distilled, guidance in [("distilled", True, 1.0), ("guided", False, 3.0)]:
        pipe = Flux2KleinInpaintPipeline(
            scheduler=FlowMatchEulerDiscreteScheduler(**scheduler_config), vae=vae,
            text_encoder=None, tokenizer=None, transformer=transformer, is_distilled=distilled)
        captured, steps = {}, []
        prepare_latents, prepare_mask = pipe.prepare_latents, pipe.prepare_mask_latents

        def capture_latents(image, timestep, *args, **kwargs):
            out = prepare_latents(image, timestep, *args, **kwargs)
            captured.update(init_image=image, start=out[0], noise=out[1], source=out[2])
            return out

        def capture_mask(mask, *args, **kwargs):
            out = prepare_mask(mask, *args, **kwargs)
            captured.update(mask_full=mask, mask_packed=out)
            return out

        pipe.prepare_latents, pipe.prepare_mask_latents = capture_latents, capture_mask

        def on_step(pipeline, index, timestep, tensors):
            steps.append(tensors["latents"].detach().clone())
            return {}

        with torch.no_grad():
            output = pipe(prompt_embeds=embeds,
                          negative_prompt_embeds=None if distilled else negative,
                          image=picture, mask_image=mask, strength=0.8, num_inference_steps=50,
                          guidance_scale=guidance, height=16, width=16,
                          generator=torch.Generator().manual_seed(9), output_type="pt",
                          callback_on_step_end=on_step,
                          callback_on_step_end_tensor_inputs=["latents"]).images
        for key, value in captured.items():
            extra[f"{name}.{key}"] = value.detach().float().clone().contiguous()
        extra[f"{name}.steps"] = torch.stack(steps).float().clone().contiguous()
        extra[f"{name}.start_step"] = torch.tensor([50 - len(steps)], dtype=torch.int32)
        extra[f"{name}.output"] = output.float().clone().contiguous()
        print(f"  {name}: start step {50 - len(steps)}, {len(steps)} steps, "
              f"mask cells {sorted(set(np.round(captured['mask_packed'].flatten().numpy(), 4)))}")

    for key, value in transformer.state_dict().items():
        extra[f"w::transformer.{key}"] = value.float().clone().contiguous()
    for key, value in vae.state_dict().items():
        if value.is_floating_point():
            extra[f"w::vae.{key}"] = value.float().clone().contiguous()
    globals()["_extra"] = extra
    return extra["distilled.output"].clone().contiguous()


def run_flux2_kv(image):
    """FLUX.2 [klein] 9B KV's reference cache, from the reference's own `Flux2Transformer2DModel` under
    its KV attention processors and its own `Flux2KleinKVPipeline.__call__`, at a tiny random
    transformer and autoencoder.

    On the first step the reference tokens LEAD the image stream (`[text, reference, image]` in the
    joint sequence). They take the modulation of a fixed timestep, 0 by default, spliced in per
    position; they attend only to one another while the text and generated tokens attend to
    everything; and their post-rotary keys and values are cached per layer and dropped from the
    output. Every later step runs the generated tokens alone, with the cached keys and values
    spliced between the text and the image.

    The transformer half records the extracting velocity, the first double and first single layer's
    cached keys and values (the reference's `(batch, tokens, heads, head_dim)` layout), and a cached
    step's velocity. As a control it also records the ORDINARY reference-conditioning velocity on the
    same tokens. The weights are drawn at scale 0.4 rather than the harness's usual 0.05, and that is
    load-bearing: at 0.05 the modulation and the attention pattern barely move the output, the
    reference cache and ordinary conditioning agree to 1e-12, and a port of the WRONG mechanism would
    pass. At 0.4 they agree only to 0.912, so the parity figure discriminates between them. The pipeline half runs
    `__call__` at its default of 4 steps with one 64x64 reference image (the pipeline refuses one
    under 64 pixels), capturing the starting
    latent, the preprocessed reference pixels and the packed reference tokens. `image` is the harness
    picture. Runs under the `qwenimage` oracle env.
    """
    import numpy as np
    from PIL import Image
    from diffusers import (AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler,
                           Flux2Transformer2DModel)
    from diffusers.models.transformers.transformer_flux2 import (
        Flux2KVAttnProcessor, Flux2KVParallelSelfAttnProcessor)
    from diffusers.pipelines.flux2.pipeline_flux2_klein_kv import Flux2KleinKVPipeline

    transformer = _randomized(Flux2Transformer2DModel(
        patch_size=1, in_channels=16, num_layers=2, num_single_layers=2, attention_head_dim=8,
        num_attention_heads=2, joint_attention_dim=24, timestep_guidance_channels=16, mlp_ratio=3.0,
        axes_dims_rope=(2, 2, 2, 2), rope_theta=2000, eps=1e-6, guidance_embeds=False),
        seed=61, scale=0.4)
    for block in transformer.transformer_blocks:
        block.attn.set_processor(Flux2KVAttnProcessor())
    for block in transformer.single_transformer_blocks:
        block.attn.set_processor(Flux2KVParallelSelfAttnProcessor())

    generator = torch.Generator().manual_seed(23)
    reference = torch.randn(1, 6, 16, generator=generator)                 # a 2x3 reference grid
    latents = torch.randn(1, 4, 16, generator=generator)                   # a 2x2 generated grid
    later = torch.randn(1, 4, 16, generator=generator)
    embeds = torch.randn(1, 5, 24, generator=generator)
    reference_ids = torch.cartesian_prod(torch.tensor([10]), torch.arange(2), torch.arange(3),
                                         torch.arange(1)).float()
    latent_ids = torch.cartesian_prod(torch.arange(1), torch.arange(2), torch.arange(2),
                                      torch.arange(1)).float()
    text_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1),
                                    torch.arange(5)).float()
    t, t_later = torch.tensor([0.8]), torch.tensor([0.45])

    with torch.no_grad():
        extracted, cache = transformer(
            hidden_states=torch.cat([reference, latents], dim=1), encoder_hidden_states=embeds,
            timestep=t, img_ids=torch.cat([reference_ids, latent_ids]), txt_ids=text_ids,
            guidance=None, return_dict=False, kv_cache_mode="extract", num_ref_tokens=6)
        cached = transformer(
            hidden_states=later, encoder_hidden_states=embeds, timestep=t_later, img_ids=latent_ids,
            txt_ids=text_ids, guidance=None, return_dict=False, kv_cache=cache,
            kv_cache_mode="cached")[0]
        ordinary = transformer(
            hidden_states=torch.cat([latents, reference], dim=1), encoder_hidden_states=embeds,
            timestep=t, img_ids=torch.cat([latent_ids, reference_ids]), txt_ids=text_ids,
            guidance=None, return_dict=False)[0][:, :4]

    extra = {"reference": reference, "latents": latents, "later": later, "embeds": embeds,
             "reference_ids": reference_ids, "latent_ids": latent_ids, "text_ids": text_ids,
             "timestep": t, "timestep_later": t_later, "extracted": extracted, "cached": cached,
             "ordinary": ordinary,
             "cache.double0.key": cache.get_double(0).k_ref, "cache.double0.value": cache.get_double(0).v_ref,
             "cache.single0.key": cache.get_single(0).k_ref, "cache.single0.value": cache.get_single(0).v_ref}
    e, o = extracted.flatten().double(), ordinary.flatten().double()
    print(f"  reference cache vs ordinary conditioning on the same tokens: cosine "
          f"{float(e @ o / e.norm() / o.norm()):.12f}")

    # The pipeline, through its own __call__.
    vae = _randomized(AutoencoderKLFlux2(
        in_channels=3, out_channels=3, block_out_channels=(8, 16), layers_per_block=1,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        latent_channels=4, norm_num_groups=4, sample_size=32, use_quant_conv=True,
        use_post_quant_conv=True, mid_block_add_attention=True, batch_norm_eps=1e-4,
        patch_size=(2, 2)), seed=63)
    torch.manual_seed(8)
    vae.bn.running_mean.copy_(torch.randn(vae.bn.running_mean.shape) * 0.3)
    vae.bn.running_var.copy_(torch.rand(vae.bn.running_var.shape) * 0.5 + 0.5)
    pipe = Flux2KleinKVPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(
            base_image_seq_len=256, base_shift=0.5, max_image_seq_len=4096, max_shift=1.15,
            num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True,
            time_shift_type="exponential"),
        vae=vae, text_encoder=None, tokenizer=None, transformer=transformer)
    captured = {}
    prepare_latents, prepare_image_latents = pipe.prepare_latents, pipe.prepare_image_latents

    def capture_latents(*args, **kwargs):
        out = prepare_latents(*args, **kwargs)
        captured["start"] = out[0]
        return out

    def capture_images(*args, **kwargs):
        out = prepare_image_latents(*args, **kwargs)
        captured.update(reference_pixels=kwargs["images"][0], reference_tokens=out[0])
        return out

    pipe.prepare_latents, pipe.prepare_image_latents = capture_latents, capture_images
    # The pipeline refuses a reference image under 64 pixels on a side.
    picture = Image.fromarray((np.asarray(image) * 255).clip(0, 255).astype(np.uint8)).resize(
        (64, 64), Image.BICUBIC)
    with torch.no_grad():
        output = pipe(image=[picture], prompt_embeds=embeds, height=16, width=16,
                      num_inference_steps=4, generator=torch.Generator().manual_seed(4),
                      output_type="pt").images
    for key, value in captured.items():
        extra[f"pipeline.{key}"] = value
    extra["pipeline.output"] = output

    extra = {k: v.detach().float().clone().contiguous() for k, v in extra.items()}
    for key, value in transformer.state_dict().items():
        extra[f"w::transformer.{key}"] = value.float().clone().contiguous()
    for key, value in vae.state_dict().items():
        if value.is_floating_point():
            extra[f"w::vae.{key}"] = value.float().clone().contiguous()
    globals()["_extra"] = extra
    return extra["extracted"].clone().contiguous()


def run_flux2_scheduler(image):
    """FLUX.2's sigma schedule, from the release's own scheduler config and the pipeline's ramp.

    FLUX.2 replaces `calculate_shift` with `compute_empirical_mu`, which depends on the STEP COUNT as
    well as the sequence length: two lines in sequence length are fitted at 10 and 200 steps and the
    shift interpolates linearly between them in the number of steps, with the 200-step line used alone
    above a sequence length of 4300. The released `scheduler_config.json` still carries `base_shift`
    0.5 and `max_shift` 1.15, which the empirical fit replaces rather than reads — a port that took
    the config at its word would produce a plausible schedule that is not this one.

    The ramp is the pipeline's `np.linspace(1.0, 1 / num_steps, num_steps)`, not the scheduler's own
    `sigma_min`. Several step counts and sequence lengths are recorded, including one above and one
    below the 4300 crossover. Runs under the `qwenimage` oracle env. `image` unused.
    """
    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu

    cases = [(4, 256), (20, 1024), (28, 4096), (50, 4300), (28, 6000)]
    extra = {}
    for steps, sequence in cases:
        scheduler = FlowMatchEulerDiscreteScheduler.from_config({
            "base_image_seq_len": 256, "base_shift": 0.5, "invert_sigmas": False,
            "max_image_seq_len": 4096, "max_shift": 1.15, "num_train_timesteps": 1000,
            "shift": 3.0, "shift_terminal": None, "stochastic_sampling": False,
            "time_shift_type": "exponential", "use_beta_sigmas": False,
            "use_dynamic_shifting": True, "use_exponential_sigmas": False,
            "use_karras_sigmas": False})
        mu = compute_empirical_mu(image_seq_len=sequence, num_steps=steps)
        ramp = np.linspace(1.0, 1 / steps, steps)
        scheduler.set_timesteps(sigmas=ramp, mu=mu, device="cpu")
        key = f"{steps}x{sequence}"
        extra[f"mu.{key}"] = torch.tensor([mu], dtype=torch.float32)
        extra[f"sigmas.{key}"] = scheduler.sigmas.clone().to(torch.float32).contiguous()
        extra[f"timesteps.{key}"] = scheduler.timesteps.clone().to(torch.float32).contiguous()
        print(f"  {key}: mu {mu:.9f}, first sigma {float(scheduler.sigmas[0]):.9f}, "
              f"last non-zero {float(scheduler.sigmas[-2]):.9f}")
    globals()["_extra"] = extra
    return extra["sigmas.20x1024"].clone().contiguous()


def run_flux2_text_real(image, checkpoint):
    """FLUX.2 [klein]'s text conditioning on the RELEASED encoder, from the reference pipeline's OWN
    `Flux2KleinPipeline._get_qwen3_prompt_embeds`.

    `--checkpoint` is the release ROOT (it reads `text_encoder/` and `tokenizer/`). The oracle calls
    the reference's static method rather than reconstructing it, so the chat template, the right
    padding to 512, the attention mask and the layer read (9, 18, 27) are all the reference's own.
    The encoder runs at float32, the precision `NFKMLXLanguage.loadedRelease` loads it at, so the
    figure measures the arithmetic on the shipped weights. That matters because this release's
    encoder is NOT byte-identical to `Qwen/Qwen3-4B` (different sharding, no shard hash in common),
    so the package's existing Qwen3-4B measurement does not cover it. 16 GB at float32, the largest
    Qwen3 this machine holds. The padded ids and the mask are recorded beside the embedding, so a
    pad-token disagreement shows as a failed id comparison rather than a lower cosine.
    """
    return _flux2_text_real(checkpoint, "float32")["embedding.0"].clone().contiguous()


def run_flux2_text_real_bf16(image, checkpoint):
    """`run_flux2_text_real` at bfloat16, for an encoder too large for float32 whole: the Qwen3-8B in
    FLUX.2 [klein] 9B is 30.5 GB at float32 and 15.3 GB as released. `--checkpoint` is the release
    ROOT. `image` unused.
    """
    return _flux2_text_real(checkpoint, "bfloat16")["embedding.0"].clone().contiguous()


def _qwen3_cut_config(config, layers):
    """`config` cut to its first `layers` layers. The installed transformers validates that
    `layer_types` has one entry a layer, so the two change together."""
    from transformers import Qwen3Config
    values = config.to_dict()
    values["num_hidden_layers"] = layers
    if values.get("layer_types"):
        values["layer_types"] = values["layer_types"][:layers]
    return Qwen3Config(**values)


def run_flux2_text_real_truncated(image, checkpoint):
    """FLUX.2 [klein]'s text conditioning at FLOAT32 on the released 9B encoder cut to 28 layers, and
    the same cut at bfloat16.

    `--checkpoint` is the release ROOT. The conditioning reads hidden states 9, 18 and 27 of a
    36-layer Qwen3 and nothing past them, so a cut to 28 layers computes the COMPLETE conditioning;
    only unread depth and the language-model head are dropped. That brings float32 from 30.5 GB to
    24 GB. The cut is 28 and not 27 because the last entry of the reference's hidden-state tuple is
    taken AFTER the final norm: a model cut to 27 layers would hand back a normed state 27. That is
    checked here on a tiny Qwen3 in the installed transformers before the release is touched. The
    encoder is built on the meta device and its tensors are read one by one through `safe_open` and
    assigned, so the release never exists whole in memory. `image` unused.
    """
    import os
    from safetensors import safe_open
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    tiny = Qwen3Config(hidden_size=32, num_hidden_layers=6, num_attention_heads=4,
                       num_key_value_heads=2, head_dim=8, intermediate_size=64, vocab_size=128,
                       tie_word_embeddings=False)
    full = _randomized(Qwen3ForCausalLM(tiny), seed=71)
    cut = Qwen3ForCausalLM(_qwen3_cut_config(tiny, 4)).eval()
    cut.load_state_dict({k: v for k, v in full.state_dict().items()
                         if not k.startswith(("model.layers.4.", "model.layers.5."))})
    ids, mask = torch.arange(1, 9).unsqueeze(0), torch.tensor([[1] * 6 + [0] * 2])
    with torch.no_grad():
        kept = cut.float()(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
        whole = full(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
    assert torch.equal(kept[3], whole[3]), "a cut to N+1 layers changed hidden state N"
    assert not torch.equal(kept[4], whole[4]), "the cut's last state should be the normed one"
    print("  a cut to N+1 layers keeps hidden state N exactly; its last state is the normed one")

    config = _qwen3_cut_config(Qwen3Config.from_pretrained(f"{checkpoint}/text_encoder"), 28)
    with torch.device("meta"):
        encoder = Qwen3ForCausalLM(config)
    encoder.lm_head = torch.nn.Identity()
    wanted = set(encoder.state_dict())
    state = {}
    directory = f"{checkpoint}/text_encoder"
    for name in sorted(os.listdir(directory)):
        if name.endswith(".safetensors"):
            with safe_open(os.path.join(directory, name), framework="pt") as f:
                for key in f.keys():
                    if key in wanted:
                        state[key] = f.get_tensor(key).float()
    assert wanted == set(state), f"the cut lacks {sorted(wanted - set(state))[:4]}"
    encoder.load_state_dict(state, strict=True, assign=True)
    encoder.model.rotary_emb = Qwen3RotaryEmbedding(config=config)
    encoder.eval()

    extra = _flux2_text_real(checkpoint, "float32", text_encoder=encoder)
    bf16 = _flux2_text_real(checkpoint, "bfloat16", text_encoder=encoder.to(torch.bfloat16))
    for key, value in bf16.items():
        if key.startswith("embedding."):
            extra[key.replace("embedding.", "embedding_bf16.")] = value
            a, b = extra[key].flatten().double(), value.flatten().double()
            print(f"  the reference's own bfloat16 against its float32, {key}: "
                  f"{float(a @ b / a.norm() / b.norm()):.10f}")
    globals()["_extra"] = extra
    return extra["embedding.0"].clone().contiguous()


def _flux2_text_real(checkpoint, precision, text_encoder=None):
    import torch
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

    dtype = getattr(torch, precision)
    if text_encoder is None:
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            f"{checkpoint}/text_encoder", torch_dtype=dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(f"{checkpoint}/tokenizer")
    prompts = ["a red fox in the snow", "An astronaut riding a horse on Mars, 35mm film still."]

    extra = {}
    for index, prompt in enumerate(prompts):
        with torch.no_grad():
            embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
                text_encoder, tokenizer, prompt, dtype=dtype, max_sequence_length=512)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True,
                           max_length=512)
        extra[f"embedding.{index}"] = embeds[0].float().clone().contiguous()
        extra[f"input_ids.{index}"] = inputs["input_ids"][0].to(torch.int32).clone().contiguous()
        extra[f"attention_mask.{index}"] = inputs["attention_mask"][0].to(torch.int32).clone().contiguous()
        extra[f"prompt.{index}"] = torch.tensor(list(prompt.encode("utf-8")), dtype=torch.uint8)
        print(f"  [{index}] {int(inputs['attention_mask'].sum())} real tokens of 512, "
              f"pad id {tokenizer.pad_token_id}")
    globals()["_extra"] = extra
    return extra


def run_flux2_prompt(image, checkpoint):
    """FLUX.2 [klein]'s prompt path: the release's own chat template and tokenizer, on real prompts.

    `--checkpoint` is the release's `tokenizer/` directory (FLUX.2 [klein] 4B is ungated, so this is
    Black Forest Labs' own). The reference wraps the prompt in a single user message, asks for the
    generation prompt, and passes `enable_thinking=False`, which makes a Qwen3 template append an
    EMPTY think block after the assistant header. A port that fed the encoder a bare prompt would read
    different hidden states and produce a different image with nothing to show for it, so the rendered
    text and its token ids are both recorded. Runs under the `qwenimage` oracle env.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    prompts = ["a red fox in the snow",
               "An astronaut riding a horse on Mars, 35mm film still.",
               ""]
    extra = {}
    for index, prompt in enumerate(prompts):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
        extra[f"ids.{index}"] = ids.to(torch.int32).clone().contiguous()
        extra[f"length.{index}"] = torch.tensor([len(ids)], dtype=torch.int32)
        print(f"  [{index}] {len(ids)} ids; rendered {text!r}")
        # The rendered text as UTF-8 bytes, so the Swift side can compare the template's output
        # directly rather than only its tokenization.
        extra[f"text.{index}"] = torch.tensor(list(text.encode("utf-8")), dtype=torch.uint8)
    globals()["_extra"] = extra
    return extra["ids.0"].to(torch.float32).clone().contiguous()


MODELS = {"qwen25vl_vision_tiny": run_qwen25vl_vision_tiny, "llava_tiny": run_llava_tiny, "flux2_scheduler": run_flux2_scheduler, "flux2_text": run_flux2_text,
          "flux2_inpaint": run_flux2_inpaint, "flux2_kv": run_flux2_kv,
          "flux2_text_mistral": run_flux2_text_mistral, "flux2_vae": run_flux2_vae, "ltx2": run_ltx2, "flux2": run_flux2, "muscriptor": run_muscriptor, "qwenimage21": run_qwenimage21, "qwen3vl_retrieval_loss": run_qwen3vl_retrieval_loss, "storm": run_storm, "qwen4_exp": run_qwen4_exp, "deepseek_v4_release_decode": run_deepseek_v4_release_decode, "deepseek_v4_release_decode_bf16": run_deepseek_v4_release_decode_bf16, "deepseek_v4_pro_release_decode": run_deepseek_v4_pro_release_decode, "deepseek_v4_pro_release_decode_bf16": run_deepseek_v4_pro_release_decode_bf16, "deepseek_v4_pro_dspark": run_deepseek_v4_pro_dspark, "deepseek_v4_pro_dspark_bf16": run_deepseek_v4_pro_dspark_bf16, "deepseek_v4_release": run_deepseek_v4_release, "deepseek_v4_release_bf16": run_deepseek_v4_release_bf16, "deepseek_v4_pro_release": run_deepseek_v4_pro_release, "deepseek_v4_pro_release_bf16": run_deepseek_v4_pro_release_bf16, "deepseek_v41": run_deepseek_v41, "deepseek_v41_quantized": run_deepseek_v41_quantized, "deepseek_v41_bf16": run_deepseek_v41_bf16, "deepseek_v41_bf16_plain": run_deepseek_v41_bf16_plain, "deepseek_v41_decode": run_deepseek_v41_decode, "deepseek_v41_decode_bf16": run_deepseek_v41_decode_bf16, "deepseek_v41_dspark_bf16": run_deepseek_v41_dspark_bf16, "deepseek_v41_vision_bf16": run_deepseek_v41_vision_bf16, "deepseek_v41_vl_router": run_deepseek_v41_vl_router, "deepseek_v41_vl_router_bf16": run_deepseek_v41_vl_router_bf16, "deepseek_v41_vision": run_deepseek_v41_vision, "deepseek_v41_image": run_deepseek_v41_image, "deepseek_v41_dspark": run_deepseek_v41_dspark, "deepseek_v41_dspark_quantized": run_deepseek_v41_dspark_quantized, "sd_scheduler": run_sd_scheduler, "clip": run_clip, "segformer": run_segformer, "zero_dce_losses": run_zero_dce_losses,
          "mimi": run_mimi, "chronos": run_chronos,
          "dcn": run_dcn,
          "segformer_loss": run_segformer_loss, "gtcrn_loss": run_gtcrn_loss, "yolo_training_setup": run_yolo_training_setup, "yolo_e2e_loss": run_yolo_e2e_loss, "yolo_loss": run_yolo_loss, "convtasnet_loss": run_convtasnet_loss, "allin1_training": run_allin1_training, "vjepa2_probe": run_vjepa2_probe,
          "clip_text": run_clip_text, "sd_tokenizer": run_sd_tokenizer,
          "rope_scaling": run_rope_scaling, "silero_vad": run_silero_vad, "dac": run_dac,
          "snac": run_snac, "siglip2": run_siglip2, "taesd": run_taesd, "ltx_vae": run_ltx_vae, "ltx_transformer": run_ltx_transformer, "ltx_t5": run_ltx_t5, "z_image": run_z_image, "sana": run_sana, "sd3": run_sd3, "flux": run_flux, "sd3_controlnet": run_sd3_controlnet, "sd3_controlnet_single": run_sd3_controlnet_single, "flux_controlnet": run_flux_controlnet, "flux_controlnet_hint": run_flux_controlnet_hint, "wan": run_wan, "wan_animate": run_wan_animate, "sam2_video": run_sam2_video, "sam3_vision": run_sam3_vision, "sam3_text": run_sam3_text, "sam3_detector": run_sam3_detector, "sam2_loss": run_sam2_loss, "sam3_loss": run_sam3_loss, "flux_vae": run_flux_vae, "dc_ae": run_dc_ae, "wan_vae": run_wan_vae, "dpm_solver": run_dpm_solver, "unipc": run_unipc, "gemma2": run_gemma2, "gemma3_tiny": run_gemma3_tiny, "gemma3n_tiny": run_gemma3n_tiny, "gemma3n_audio": run_gemma3n_audio, "gemma3_bidirectional_tiny": run_gemma3_bidirectional_tiny, "umt5": run_umt5, "wan_vae_21": run_wan_vae_21, "dc_ae_real": run_dc_ae_real, "ip_adapter": run_ip_adapter, "rtdetr": run_rtdetr, "rtdetr_v2": run_rtdetr_v2, "rf_detr": run_rf_detr,
          "gemma4_shared_kv": run_gemma4_shared_kv}
CHECKPOINT_MODELS = {"hf_layer_probe": run_hf_layer_probe, "flux2_prompt": run_flux2_prompt, "flux2_real": run_flux2_real, "flux2_real_f32": run_flux2_real_f32, "flux2_real_truncated": run_flux2_real_truncated, "flux2_kv_real": run_flux2_kv_real, "flux2_kv_real_truncated": run_flux2_kv_real_truncated, "flux2_vae_real": run_flux2_vae_real, "flux2_text_real": run_flux2_text_real, "flux2_text_real_bf16": run_flux2_text_real_bf16, "flux2_text_real_truncated": run_flux2_text_real_truncated, "laya": run_laya, "laya_loss": run_laya_loss, "laya_episode": run_laya_episode, "open_jev_deberta": run_open_jev_deberta, "open_jev_deberta_budget": run_open_jev_deberta_budget, "open_jev": run_open_jev, "translategemma": run_translategemma, "florence2": run_florence2, "florence2_generate": run_florence2_generate, "florence2_loss": run_florence2_loss, "trocr": run_trocr, "trocr_loss": run_trocr_loss, "marian": run_marian, "m2m100": run_m2m100, "madlad": run_madlad, "hft": run_hft, "qwenimage21_text": run_qwenimage21_text, "qwenimage21_pipeline": run_qwenimage21_pipeline, "qwenimage21_vae": run_qwenimage21_vae, "qwenimage21_scheduler": run_qwenimage21_scheduler, "qwenimage21_real": run_qwenimage21_real, "muscriptor_real": run_muscriptor_real, "basic_pitch": run_basic_pitch, "chatterbox_mtl_tokens": run_chatterbox_mtl_tokens, "rf_detr_seg": run_rf_detr_seg, "chatterbox_mtl_t3": run_chatterbox_mtl_t3, "allin1": run_allin1, "sam_encoder": run_sam_encoder, "sam2_encoder": run_sam2_encoder, "sa2va_teacher": run_sa2va_teacher, "sa2va_loss": run_sa2va_loss, "sa2va_qwen": run_sa2va_qwen, "sa2va_processor": run_sa2va_processor, "internvit_qknorm_tiny": run_internvit_qknorm_tiny, "internlm2_tiny": run_internlm2_tiny, "internlm2_tokenizer": run_internlm2_tokenizer, "sa2va_llava_teacher": run_sa2va_llava_teacher, "sam2_decoder": run_sam2_decoder, "sam2_memory": run_sam2_memory, "sam": run_sam, "sam_decoder": run_sam_decoder,
                     "swinir": run_swinir,
                     "sd_unet": run_sd_unet, "sd_vae": run_sd_vae, "sd_text_encoder": run_sd_text_encoder, "sd_text_to_image": run_sd_text_to_image, "convtasnet": run_convtasnet, "demucs": run_demucs, "htdemucs": run_htdemucs, "htdemucs_bag": run_htdemucs_bag, "denoiser": run_denoiser,
                     "vad": run_vad, "vad_training": run_vad_training, "deeplab": run_deeplab, "u2net": run_u2net, "isnet": run_isnet, "adain": run_adain, "hat": run_hat, "pose": run_pose,
                     "audio_tagger": run_audio_tagger, "raft": run_raft, "rvm": run_rvm,
                     "depth": run_depth, "depth_encoder": run_depth_encoder, "depth3": run_depth3,
                     "videosr": run_videosr, "yolo": run_yolo, "yolo_generation": run_yolo_generation, "nafnet": run_nafnet, "rife": run_rife, "rife_v4": run_rife_v4, "modnet": run_modnet, "bisenet": run_bisenet, "bisenetv2": run_bisenetv2, "siggraph17": run_siggraph17, "whisper": run_whisper, "lama": run_lama, "yolo_detections": run_yolo_detections, "codeformer": run_codeformer, "retinaface": run_retinaface, "qwen3": run_qwen3, "qwen3_embedding": run_qwen3_embedding, "embeddinggemma": run_embeddinggemma, "modernbert_reranker": run_modernbert_reranker, "smolvlm": run_smolvlm, "qwen3vl": run_qwen3vl, "qwen3vl_embedding": run_qwen3vl_embedding, "qwen3vl_reranker": run_qwen3vl_reranker, "gguf": run_gguf, "gguf_lm": run_gguf_lm, "qwen3_moe": run_qwen3_moe, "mixtral": run_mixtral, "mamba2": run_mamba2, "mamba2_real": run_mamba2_real, "granite_hybrid": run_granite_hybrid, "granite_hybrid_moe": run_granite_hybrid_moe, "granite_hybrid_loss": run_granite_hybrid_loss, "granite_hybrid_real": run_granite_hybrid_real, "nemotron_h": run_nemotron_h, "nemotron_h_loss": run_nemotron_h_loss, "nemotron_h_real": run_nemotron_h_real, "granite_speech": run_granite_speech, "granite_speech_real": run_granite_speech_real, "voxtral": run_voxtral, "voxtral_real": run_voxtral_real, "qwen2_moe": run_qwen2_moe, "gpt_oss": run_gpt_oss, "gemma4": run_gemma4, "gemma3": run_gemma3, "gemma3n": run_gemma3n, "gemma3n_conditional_real": run_gemma3n_conditional_real, "gemma3n_vision_real": run_gemma3n_vision_real, "gemma3n_audio_real": run_gemma3n_audio_real, "gemma3n_mel": run_gemma3n_mel, "gemma3_vision_real": run_gemma3_vision_real, "gemma3_conditional_real": run_gemma3_conditional_real, "gemma4_moe": run_gemma4_moe, "gemma4_unified": run_gemma4_unified, "gemma4_vision": run_gemma4_vision, "gemma4_audio": run_gemma4_audio, "gemma4_mel": run_gemma4_mel, "gemma4_embedder": run_gemma4_embedder, "gemma4_audio_real": run_gemma4_audio_real, "gemma4_conditional_real": run_gemma4_conditional_real, "gemma4_vision_real": run_gemma4_vision_real, "qwen3_5": run_qwen3_5, "deepseek_v41_tokens": run_deepseek_v41_tokens, "deepseek_v41_quant": run_deepseek_v41_quant, "deepseek_quant": run_deepseek_quant, "gpt_oss_quant": run_gpt_oss_quant, "metricgan": run_metricgan, "cmgan": run_cmgan, "frcrn": run_frcrn, "mossformer2_sr": run_mossformer2_sr, "nuwave2": run_nuwave2, "nuwave2_loss": run_nuwave2_loss, "apollo": run_apollo, "deepseek_v4": run_deepseek_v4, "hifigan": run_hifigan, "fastspeech2": run_fastspeech2, "music_vocoder": run_music_vocoder, "music_depth": run_music_depth, "music_condition": run_music_condition, "music_dit": run_music_dit, "music_ar": run_music_ar, "music_tokenizer": run_music_tokenizer,
                     "zero_dce": run_zero_dce, "style_transfer": run_style_transfer,
                     "realesrgan": run_realesrgan, "colorizer": run_colorizer, "rtdetr_real": run_rtdetr_real, "rtdetr_v2_real": run_rtdetr_v2_real, "vitpose": run_vitpose, "ddcolor": run_ddcolor, "realesrgan_compact": run_realesrgan_compact, "zero_dce_plus": run_zero_dce_plus, "rf_detr_real": run_rf_detr_real, "ip_adapter_unet": run_ip_adapter_unet, "kokoro": run_kokoro, "parakeet": run_parakeet, "canary": run_canary, "phi4mm": run_phi4mm, "phi4mm_bf16": run_phi4mm_bf16, "phi4mm_conversation": run_phi4mm_conversation,"table_transformer": run_table_transformer, "table_transformer_loss": run_table_transformer_loss, "vjepa2": run_vjepa2, "sa2va": run_sa2va, "cosmos_tokenizer": run_cosmos_tokenizer, "cosmos_tokenizer_loss": run_cosmos_tokenizer_loss,
                     "chatterbox_voice": run_chatterbox_voice,
                     "chatterbox_t3": run_chatterbox_t3,
                     "chatterbox_s3gen": run_chatterbox_s3gen,
                     "birefnet_backbone": run_birefnet_backbone,
                     "birefnet_neck": run_birefnet_neck,
                     "birefnet_decode": run_birefnet_decode,
                     "mpsenet": run_mpsenet,
                     "gtcrn": run_gtcrn,
                     "sgmse": run_sgmse,
                     "mossformer2_se": run_mossformer2_se,
                     "deepfilternet": run_deepfilternet,
                     "voicerestore": run_voicerestore,
                     "bigvgan": run_bigvgan,
                     "voicerestore_e2e": run_voicerestore_e2e,
                     "reenhance_mel": run_reenhance_mel,
                     "reenhance_univnet": run_reenhance_univnet,
                     "reenhance_irmae": run_reenhance_irmae,
                     "reenhance_cfm": run_reenhance_cfm,
                     "reenhance_denoiser": run_reenhance_denoiser,
                     "reenhance_e2e": run_reenhance_e2e,
                     "pixtral": run_pixtral,
                     "pixtral_tiny": run_pixtral_tiny,
                     "flux_real": run_flux_real,
                     "flux_text": run_flux_text}


def run_ltx_pipeline(image, checkpoint):
    """LTX-Video 0.9.0's text-to-video glue, from diffusers' own LTXPipeline, at a tiny random geometry.

    `--checkpoint` is a directory holding the T5 SentencePiece tokenizer (a FLUX.1 release's
    `tokenizer_2/` serves: it is the same T5 v1.1 vocabulary). The T5 encoder, the transformer and the
    autoencoder are tiny random ones, so what this measures is the pipeline: the tokenization padded to
    `max_sequence_length` with its attention mask, the cross-attention masking, the sigma ramp and its
    dynamic shift, the rope interpolation scale, classifier-free guidance, the latent denormalization by
    the autoencoder's stored statistics, and the decode. Each model is measured against its own
    reference elsewhere.

    The record carries the token ids and masks, the prompt features, the starting latents, the sigmas,
    the final latents, the decoded video, and the three models' weights in release naming.
    """
    import torch
    from diffusers import (AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler, LTXPipeline,
                           LTXVideoTransformer3DModel)
    from transformers import T5Config, T5EncoderModel, T5TokenizerFast

    # The fast tokenizer reads the same vocabulary from tokenizer.json; this environment carries no
    # sentencepiece for the slow one, and the two agree on plain text.
    tokenizer = T5TokenizerFast(tokenizer_file=os.path.join(checkpoint, "tokenizer.json"),
                                eos_token="</s>", unk_token="<unk>", pad_token="<pad>", extra_ids=0)
    text_encoder = T5EncoderModel(T5Config(
        vocab_size=32128, d_model=32, d_kv=16, d_ff=64, num_layers=2, num_heads=2,
        relative_attention_num_buckets=16, relative_attention_max_distance=32,
        feed_forward_proj="gated-gelu", layer_norm_epsilon=1e-6))
    _randomized(text_encoder, seed=21, scale=0.1)
    transformer = LTXVideoTransformer3DModel(
        in_channels=16, out_channels=16, patch_size=1, patch_size_t=1, num_attention_heads=2,
        attention_head_dim=8, cross_attention_dim=16, num_layers=2, caption_channels=32,
        qk_norm="rms_norm_across_heads")
    _randomized(transformer, seed=22, scale=0.1)
    vae = AutoencoderKLLTXVideo(
        in_channels=3, out_channels=3, latent_channels=16, block_out_channels=(8, 16, 16, 16),
        decoder_block_out_channels=(8, 16, 16, 16), layers_per_block=(1, 1, 1, 1, 1),
        decoder_layers_per_block=(1, 1, 1, 1, 1), spatio_temporal_scaling=(True, True, True, False),
        decoder_spatio_temporal_scaling=(True, True, True, False), decoder_inject_noise=(False,) * 5,
        upsample_residual=(False,) * 4, upsample_factor=(1,) * 4, timestep_conditioning=False,
        patch_size=4, patch_size_t=1, encoder_causal=True, decoder_causal=False)
    _randomized(vae, seed=23, scale=0.1)
    torch.manual_seed(24)
    vae.latents_mean.copy_(torch.randn(16) * 0.3)
    vae.latents_std.copy_(torch.rand(16) + 0.5)
    scheduler = FlowMatchEulerDiscreteScheduler(
        base_image_seq_len=1024, base_shift=0.95, max_image_seq_len=4096, max_shift=2.05, shift=1.0,
        shift_terminal=0.1, use_dynamic_shifting=True)
    pipeline = LTXPipeline(scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                           transformer=transformer)

    prompt, negative = "A red fox walking through fresh snow, cinematic", "worst quality, blurry"
    frames, height, width, steps = 9, 64, 64, 4
    torch.manual_seed(25)
    latents = torch.randn(1, 2 * 2 * 2, 16)
    arguments = dict(prompt=prompt, negative_prompt=negative, height=height, width=width,
                     num_frames=frames, num_inference_steps=steps, guidance_scale=3.0, frame_rate=25,
                     max_sequence_length=128)
    with torch.no_grad():
        embeds, mask, negative_embeds, negative_mask = pipeline.encode_prompt(
            prompt=prompt, negative_prompt=negative, do_classifier_free_guidance=True,
            max_sequence_length=128)
        final = pipeline(latents=latents.clone(), output_type="latent", return_dict=False, **arguments)[0]
        video = pipeline(latents=latents.clone(), output_type="pt", return_dict=False, **arguments)[0]
    ids = tokenizer(prompt, padding="max_length", max_length=128, truncation=True,
                    add_special_tokens=True, return_tensors="pt").input_ids

    extra = {"input_ids": ids[0].to(torch.int32).contiguous(), "mask": mask[0].to(torch.int32).contiguous(),
             "negative_mask": negative_mask[0].to(torch.int32).contiguous(),
             "prompt_embeds": embeds[0].contiguous(), "negative_embeds": negative_embeds[0].contiguous(),
             "latents": latents[0].contiguous(), "sigmas": scheduler.sigmas.float().contiguous(),
             "final_latents": final[0].contiguous(),
             "video": video[0].permute(0, 2, 3, 1).contiguous()}           # [F, H, W, C] in 0...1
    for prefix, model in (("t5::", text_encoder), ("t::", transformer), ("v::", vae)):
        for key, value in model.state_dict().items():
            # The T5 encoder ties `encoder.embed_tokens` to `shared`; the release stores `shared` alone.
            if key == "encoder.embed_tokens.weight":
                continue
            extra[prefix + key] = value.float().clone().contiguous()
    globals()["_extra"] = extra
    return extra["final_latents"].clone()


CHECKPOINT_MODELS["ltx_pipeline"] = run_ltx_pipeline


def run_wan_pipeline(image, checkpoint):
    """Wan 2.1's text-to-video glue, from diffusers' own WanPipeline, at a tiny random geometry.

    `--checkpoint` is a directory holding a T5 `tokenizer.json`. The release's own umT5 tokenizer has a
    256k vocabulary no tiny encoder can take; a T5 vocabulary exercises the same path, the cleanup, the
    padding, the end token, and the mask, with ids a tiny umT5 accepts. The umT5 encoder, the
    transformer and the Wan 2.1 autoencoder are tiny random ones, so what this measures is the pipeline:
    the masked umT5 encode, the features cut at the prompt's length and zero-padded, the UniPC flow
    schedule, classifier-free guidance, the latent statistics, and the decode. The loop is run twice,
    with and without `expand_timesteps` (Wan 2.2 TI2V's per-token timestep), which for text-to-video
    gives every token the same step.

    The record carries the token ids and masks, the prompt features, the starting latents, the final
    latents of both runs, the decoded video, and the three models' weights in release naming.
    """
    import types

    import torch
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel
    from diffusers.pipelines.wan import pipeline_wan
    from transformers import T5TokenizerFast, UMT5Config, UMT5EncoderModel

    # The pipeline's cleanup runs `ftfy.fix_text`, which repairs mis-decoded text and leaves well-formed
    # text as it is; no oracle environment carries ftfy, and this prompt is well formed.
    pipeline_wan.ftfy = types.SimpleNamespace(fix_text=lambda text: text)

    tokenizer = T5TokenizerFast(tokenizer_file=os.path.join(checkpoint, "tokenizer.json"),
                                eos_token="</s>", unk_token="<unk>", pad_token="<pad>", extra_ids=0)
    text_encoder = UMT5EncoderModel(UMT5Config(
        vocab_size=32128, d_model=32, d_kv=16, d_ff=64, num_layers=2, num_heads=2,
        relative_attention_num_buckets=16, relative_attention_max_distance=32,
        feed_forward_proj="gated-gelu", layer_norm_epsilon=1e-6))
    _randomized(text_encoder, seed=31, scale=0.1)
    transformer = WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=16, in_channels=4, out_channels=4,
        text_dim=32, freq_dim=256, ffn_dim=48, num_layers=2, cross_attn_norm=True,
        qk_norm="rms_norm_across_heads", eps=1e-6, rope_max_seq_len=1024)
    _randomized(transformer, seed=32, scale=0.1)
    vae = AutoencoderKLWan(base_dim=8, z_dim=4, dim_mult=[1, 2], num_res_blocks=1, temperal_downsample=[True],
                           latents_mean=[0.1, -0.2, 0.3, -0.4], latents_std=[1.1, 0.9, 1.3, 0.8])
    _randomized(vae, seed=33, scale=0.1)
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0, num_train_timesteps=1000,
        solver_order=2, solver_type="bh2", final_sigmas_type="zero", lower_order_final=True, predict_x0=True)

    prompt, negative = "  A red fox   walking through fresh snow, cinematic ", "worst quality, blurry"
    arguments = dict(prompt=prompt, negative_prompt=negative, height=16, width=16, num_frames=5,
                     num_inference_steps=4, guidance_scale=5.0, max_sequence_length=64)
    torch.manual_seed(34)
    latents = torch.randn(1, 4, 3, 8, 8)
    extra = {"latents": latents[0].contiguous()}
    for name, expand in (("", False), ("_expanded", True)):
        pipeline = WanPipeline(tokenizer=tokenizer, text_encoder=text_encoder, transformer=transformer, vae=vae,
                               scheduler=scheduler, expand_timesteps=expand)
        with torch.no_grad():
            final = pipeline(latents=latents.clone(), output_type="latent", return_dict=False, **arguments)[0]
            if not expand:
                video = pipeline(latents=latents.clone(), output_type="pt", return_dict=False, **arguments)[0]
                embeds, negative_embeds = pipeline.encode_prompt(
                    prompt=prompt, negative_prompt=negative, do_classifier_free_guidance=True,
                    max_sequence_length=64)
        extra[f"final_latents{name}"] = final[0].contiguous()
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    inputs = tokenizer(prompt_clean(prompt), padding="max_length", max_length=64, truncation=True,
                       add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
    extra.update({"input_ids": inputs.input_ids[0].to(torch.int32).contiguous(),
                  "mask": inputs.attention_mask[0].to(torch.int32).contiguous(),
                  "prompt_embeds": embeds[0].contiguous(), "negative_embeds": negative_embeds[0].contiguous(),
                  "video": video[0].permute(0, 2, 3, 1).contiguous()})       # [F, H, W, C] in 0...1
    for prefix, model in (("t5::", text_encoder), ("t::", transformer), ("v::", vae)):
        for key, value in model.state_dict().items():
            if key == "encoder.embed_tokens.weight":
                continue
            extra[prefix + key] = value.float().clone().contiguous()
    globals()["_extra"] = extra
    return extra["final_latents"].clone()


CHECKPOINT_MODELS["wan_pipeline"] = run_wan_pipeline


def run_z_image_pipeline(image, checkpoint):
    """Z-Image's text-to-image glue, from diffusers' own ZImagePipeline, at a tiny random geometry.

    `--checkpoint` is a Qwen3 tokenizer directory (the Qwen3-4B release's, which the Z-Image release
    ships as `tokenizer/`). The Qwen3 text encoder, the S3-DiT and the Flux autoencoder are tiny random
    ones, so what this measures is the pipeline: the chat template with thinking enabled, the padded
    encode read at the penultimate hidden state and cut to the prompt's tokens, the static-shift flow
    schedule ending at sigma 0, the `(1000 - t) / 1000` timestep, classifier-free guidance above a scale
    of 1, the negated velocity, and the centered-latent decode. The loop runs twice: guided (scale 4,
    with a negative prompt) and unguided (scale 0, the Turbo release's setting).

    The record carries the token ids, the prompt features, the starting latents, the final latents of
    both runs, the decoded image, and the three models' weights in release naming.
    """
    import torch
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImagePipeline, ZImageTransformer2DModel
    from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    text_encoder = Qwen3Model(Qwen3Config(
        vocab_size=len(tokenizer), hidden_size=24, intermediate_size=48, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=1, head_dim=12, rms_norm_eps=1e-6, rope_theta=1000000.0,
        tie_word_embeddings=True, attn_implementation="eager"))
    _randomized(text_encoder, seed=41, scale=0.1)
    transformer = ZImageTransformer2DModel(
        all_patch_size=(2,), all_f_patch_size=(1,), in_channels=4, dim=32, n_layers=2,
        n_refiner_layers=1, n_heads=2, n_kv_heads=2, norm_eps=1e-5, qk_norm=True, cap_feat_dim=24,
        rope_theta=256.0, t_scale=1000.0, axes_dims=[4, 6, 6], axes_lens=[1024, 512, 512])
    _randomized(transformer, seed=42, scale=0.1)
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, block_out_channels=[8, 16], layers_per_block=1,
        latent_channels=4, norm_num_groups=4, use_quant_conv=False, use_post_quant_conv=False,
        mid_block_add_attention=True, scaling_factor=0.3611, shift_factor=0.1159,
        down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"])
    _randomized(vae, seed=43, scale=0.1)
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=False)
    pipeline = ZImagePipeline(scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                              transformer=transformer)

    prompt, negative = "A red fox walking through fresh snow, cinematic", "blurry, low quality"
    arguments = dict(prompt=prompt, height=16, width=16, num_inference_steps=5, max_sequence_length=64)
    torch.manual_seed(44)
    latents = torch.randn(1, 4, 8, 8)
    extra = {"latents": latents[0].contiguous()}
    with torch.no_grad():
        guided = dict(arguments, negative_prompt=negative, guidance_scale=4.0)
        extra["final_latents"] = pipeline(latents=latents.clone(), output_type="latent", return_dict=False,
                                          **guided)[0][0].contiguous()
        picture = pipeline(latents=latents.clone(), output_type="pt", return_dict=False, **guided)[0]
        extra["final_latents_unguided"] = pipeline(latents=latents.clone(), output_type="latent",
                                                   return_dict=False, guidance_scale=0.0,
                                                   **arguments)[0][0].contiguous()
        embeds, negative_embeds = pipeline.encode_prompt(prompt=prompt, negative_prompt=negative,
                                                         do_classifier_free_guidance=True,
                                                         max_sequence_length=64)
    templated = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=True)
    extra.update({"input_ids": torch.tensor(tokenizer(templated).input_ids, dtype=torch.int32),
                  "prompt_embeds": embeds[0].contiguous(), "negative_embeds": negative_embeds[0].contiguous(),
                  "image": picture[0].permute(1, 2, 0).contiguous()})        # [H, W, C] in 0...1
    print("z_image_pipeline template:", repr(templated))
    for prefix, model in (("te::model.", text_encoder), ("t::", transformer), ("v::", vae)):
        for key, value in model.state_dict().items():
            extra[prefix + key] = value.float().clone().contiguous()
    globals()["_extra"] = extra
    return extra["final_latents"].clone()


CHECKPOINT_MODELS["z_image_pipeline"] = run_z_image_pipeline


def run_sd3_pipeline(image, checkpoint):
    """Stable Diffusion 3's text-to-image glue, from diffusers' own StableDiffusion3Pipeline, at a tiny
    random geometry.

    `--checkpoint` is a CLIP tokenizer directory padding with the end marker (CLIP-L's, as a FLUX.1
    release's `tokenizer/`). IK_SD3_CLIP_G_TOKENIZER names a CLIP tokenizer padding with `!` (bigG's
    convention; the store's `clip-bang-tokenizer/`, copied from Stable Diffusion 2.1's `tokenizer/`), and
    IK_SD3_T5_TOKENIZER a T5 v1.1 tokenizer (a FLUX.1 release's `tokenizer_2/`). The three text towers,
    the MMDiT and the autoencoder are tiny random ones, so what this measures is the pipeline: the two
    CLIP penultimate states concatenated on channels and zero-padded to T5's width, the T5 sequence
    appended, the two pooled projections concatenated, the static-shift schedule over the scheduler's
    own ramp, the raw timestep, classifier-free guidance above a scale of 1, and the decode. The loop
    runs guided (scale 5, with a negative prompt) and unguided (scale 1).

    The record carries the token ids, the prompt features, the starting latents, the final latents of
    both runs, the decoded image, and the five models' weights in release naming.
    """
    import torch
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel, StableDiffusion3Pipeline
    from transformers import (CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizer, T5Config,
                              T5EncoderModel, T5TokenizerFast)

    # At the usual 0.1 the towers encode both prompts almost alike and guidance moves the final
    # latents by 1e-6; at 0.5 the guided and unguided records are 0.94 apart.
    text_scale = 0.5
    clip_g_directory = os.environ.get("IK_SD3_CLIP_G_TOKENIZER",
                                      os.path.expanduser("~/.inferkit-validation/clip-bang-tokenizer"))
    t5_directory = os.environ.get("IK_SD3_T5_TOKENIZER",
                                  os.path.expanduser("~/.inferkit-validation/flux-schnell-release/tokenizer_2"))
    tokenizer = CLIPTokenizer.from_pretrained(checkpoint)
    tokenizer_2 = CLIPTokenizer.from_pretrained(clip_g_directory)
    # The fast tokenizer reads tokenizer.json directly; from_pretrained converts spiece.model, which
    # needs protobuf, and the oracle environment carries none.
    tokenizer_3 = T5TokenizerFast(tokenizer_file=os.path.join(t5_directory, "tokenizer.json"),
                                  eos_token="</s>", unk_token="<unk>", pad_token="<pad>", extra_ids=0)

    def clip(hidden, projection, act, seed):
        model = CLIPTextModelWithProjection(CLIPTextConfig(
            vocab_size=49408, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=3,
            num_attention_heads=2, max_position_embeddings=77, hidden_act=act, projection_dim=projection,
            bos_token_id=49406, eos_token_id=2, pad_token_id=1))
        return _randomized(model, seed=seed, scale=text_scale)

    text_encoder = clip(8, 8, "quick_gelu", 51)
    text_encoder_2 = clip(8, 12, "gelu", 52)
    text_encoder_3 = _randomized(T5EncoderModel(T5Config(
        vocab_size=32128, d_model=24, d_kv=12, d_ff=48, num_layers=2, num_heads=2,
        relative_attention_num_buckets=16, relative_attention_max_distance=32,
        feed_forward_proj="gated-gelu", layer_norm_epsilon=1e-6)), seed=53, scale=text_scale)
    transformer = _randomized(SD3Transformer2DModel(
        sample_size=16, patch_size=2, in_channels=4, out_channels=4, num_layers=2, attention_head_dim=8,
        num_attention_heads=2, joint_attention_dim=24, caption_projection_dim=16, pooled_projection_dim=20,
        pos_embed_max_size=8, dual_attention_layers=(0,), qk_norm="rms_norm"), seed=54, scale=0.3)
    vae = _randomized(AutoencoderKL(
        in_channels=3, out_channels=3, block_out_channels=[8, 16], layers_per_block=1, latent_channels=4,
        norm_num_groups=4, use_quant_conv=False, use_post_quant_conv=False, mid_block_add_attention=True,
        scaling_factor=1.5305, shift_factor=0.0609,
        down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"]), seed=55, scale=0.1)
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    pipeline = StableDiffusion3Pipeline(
        transformer=transformer, scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        text_encoder_2=text_encoder_2, tokenizer_2=tokenizer_2, text_encoder_3=text_encoder_3,
        tokenizer_3=tokenizer_3)

    prompt, negative = "A red fox walking through fresh snow, cinematic", "blurry, low quality"
    arguments = dict(prompt=prompt, height=16, width=16, num_inference_steps=5, max_sequence_length=32)
    torch.manual_seed(56)
    latents = torch.randn(1, 4, 8, 8)
    extra = {"latents": latents[0].contiguous()}
    with torch.no_grad():
        guided = dict(arguments, negative_prompt=negative, guidance_scale=5.0)
        extra["final_latents"] = pipeline(latents=latents.clone(), output_type="latent", return_dict=False,
                                          **guided)[0][0].contiguous()
        picture = pipeline(latents=latents.clone(), output_type="pt", return_dict=False, **guided)[0]
        extra["final_latents_unguided"] = pipeline(latents=latents.clone(), output_type="latent",
                                                   return_dict=False, guidance_scale=1.0,
                                                   **arguments)[0][0].contiguous()
        embeds, negative_embeds, pooled, negative_pooled = pipeline.encode_prompt(
            prompt=prompt, prompt_2=None, prompt_3=None, negative_prompt=negative,
            do_classifier_free_guidance=True, max_sequence_length=32)
    ids = lambda t, length: torch.tensor(t(prompt, padding="max_length", max_length=length, truncation=True).input_ids,
                                         dtype=torch.int32)
    extra.update({"ids_l": ids(tokenizer, 77), "ids_g": ids(tokenizer_2, 77), "ids_t5": ids(tokenizer_3, 32),
                  "prompt_embeds": embeds[0].contiguous(), "negative_embeds": negative_embeds[0].contiguous(),
                  "pooled": pooled[0].contiguous(), "negative_pooled": negative_pooled[0].contiguous(),
                  "image": picture[0].permute(1, 2, 0).contiguous()})        # [H, W, C] in 0...1
    for prefix, model in (("l::", text_encoder), ("g::", text_encoder_2), ("t5::", text_encoder_3),
                          ("t::", transformer), ("v::", vae)):
        for key, value in model.state_dict().items():
            if key == "encoder.embed_tokens.weight":
                continue
            extra[prefix + key] = value.float().clone().contiguous()
    globals()["_extra"] = extra
    return extra["final_latents"].clone()


CHECKPOINT_MODELS["sd3_pipeline"] = run_sd3_pipeline


UMT5_TOKENIZER_PROMPTS = [
    "A red fox walking through fresh snow, cinematic",
    "  Two   cats\tplaying\nin the sun  ",
    "一只红色的狐狸在雪地里行走，电影感",
    "Ein roter Fuchs läuft durch frischen Schnee, 4K, HDR!",
    "キツネが雪の中を歩く 🦊❄️",
    "Un zorro rojo — cámara lenta, 24 fps, f/1.8",
    "Лиса бежит по снегу. 1234567890",
    "naïve café résumé coöperate",
    "",
]


def run_umt5_tokenizer(image, checkpoint):
    """Wan's umT5 tokenization, from the release's own fast tokenizer, as WanPipeline hands it to umT5.

    `--checkpoint` is the release's `tokenizer/` directory. Each prompt is cleaned the way the pipeline
    cleans it (whitespace collapsed and trimmed; `ftfy.fix_text` leaves these well-formed prompts as
    they are), then tokenized with the end token, padded to 512, and masked. The record carries each
    prompt's ids and mask, `ids_<n>` and `mask_<n>`.
    """
    import re

    import torch
    from transformers import T5TokenizerFast

    tokenizer = T5TokenizerFast(tokenizer_file=os.path.join(checkpoint, "tokenizer.json"),
                                eos_token="</s>", unk_token="<unk>", pad_token="<pad>", extra_ids=0)
    extra = {}
    for index, prompt in enumerate(UMT5_TOKENIZER_PROMPTS):
        cleaned = re.sub(r"\s+", " ", prompt).strip()
        inputs = tokenizer(cleaned, padding="max_length", max_length=512, truncation=True,
                           add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        extra[f"ids_{index}"] = inputs.input_ids[0].to(torch.int32).contiguous()
        extra[f"mask_{index}"] = inputs.attention_mask[0].to(torch.int32).contiguous()
    globals()["_extra"] = extra
    return torch.tensor([float(len(UMT5_TOKENIZER_PROMPTS))])


CHECKPOINT_MODELS["umt5_tokenizer"] = run_umt5_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=sorted(MODELS) + sorted(CHECKPOINT_MODELS), help="which reference to run")
    parser.add_argument("output", help="path to write the .safetensors record")
    parser.add_argument("--size", type=int, default=224, help="input plate size (square)")
    parser.add_argument("--plate", choices=["blocks", "subject"], default="blocks",
                        help="blocks: unstructured texture. subject: a foreground shape, for the "
                             "saliency and matting models, whose reference output is near-constant "
                             "on unstructured input")
    parser.add_argument("--checkpoint", help="local checkpoint path, for references that need one (sam)")
    parser.add_argument("--image", help="a real photo to use as the plate, resized to --size. For "
                                        "models whose reference output is degenerate on synthetic "
                                        "plates (human matting finds no person in an ellipse)")
    args = parser.parse_args()

    if args.image:
        from PIL import Image
        photo = Image.open(args.image).convert("RGB").resize((args.size, args.size), Image.BILINEAR)
        image = np.ascontiguousarray(np.asarray(photo).astype(np.float32) / 255.0)
    else:
        image = (subject_image(args.size, args.size) if args.plate == "subject"
                 else deterministic_image(args.size, args.size))
    if args.model in CHECKPOINT_MODELS:
        if not args.checkpoint:
            raise SystemExit(f"{args.model} needs --checkpoint")
        result = CHECKPOINT_MODELS[args.model](image, args.checkpoint)
    else:
        result = MODELS[args.model](image)
    record = {"input_image": torch.from_numpy(image).contiguous(), "output": result.float()}
    for name, value in {**globals().get("_extra", {}), **_DIT_PROBES}.items():
        # Integer extras stay integer: class labels are indices, and casting them to float would make
        # the Swift side guess at the conversion back.
        record[name] = value.float() if value.is_floating_point() else value
    save_file(record, args.output)
    print(f"wrote {args.output}: input {tuple(image.shape)}, reference output {tuple(result.shape)}")


if __name__ == "__main__":
    sys.exit(main())
