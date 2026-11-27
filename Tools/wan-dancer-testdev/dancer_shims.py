"""Lets Wan-Video/Wan-Dancer's DiffSynth-Studio fork run in one process on the CPU, for the Wan-Dancer oracle and
test-dev builder.

- `xfuser` and `yunchang` are imported at module scope for the sequence-parallel path. The single process answers
  rank 0 of a world of one, and `LongContextAttention` is a placeholder that raises if built. The released scripts
  run sequence parallel, and `model_fn_wan_video` reads the world size only on that path, so the oracles pass
  `use_unified_sequence_parallel`; in a world of one its chunks are whole and its gathers are skipped.
- `MusicEncoderLayer` builds its attention with `device="cuda"`; without CUDA it is built on the host.
- `music_spans(dit, spans)` reproduces the released scripts' eight-way sequence parallelism where it changes the
  result: each rank projects and encodes its own eighth of the music features (truncated to a multiple of the world
  size), and the gather concatenates the eighths. Every other sequence-parallel step (the Ulysses attention, the
  split tokens, the split music injection) computes what one process computes.

Import this after `IK_WAN_DANCER_SRC` is on `sys.path`.
"""

import importlib.machinery
import sys
import types

import torch


def _module(name, **attributes):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


class _Placeholder:
    def __init__(self, *_, **__):
        raise RuntimeError("the sequence-parallel attention is not reached in one process")


def _sp_group():
    raise RuntimeError("the sequence-parallel group is not reached in one process")


for _name in ("xfuser", "xfuser.core", "xfuser.model_executor"):
    _module(_name)
_module("xfuser.core.distributed", get_sequence_parallel_rank=lambda: 0, get_sequence_parallel_world_size=lambda: 1,
        get_sp_group=_sp_group, initialize_model_parallel=_sp_group, init_distributed_environment=_sp_group)
_module("yunchang", LongContextAttention=_Placeholder, set_seq_parallel_pg=_sp_group)
_module("yunchang.kernels", AttnType=None)


class SpannedMusicEncoder(torch.nn.Module):
    """`music_encoder` as `spans` ranks run it: each encodes its own span of the projected features."""

    def __init__(self, encoder, spans):
        super().__init__()
        self.encoder = encoder
        self.spans = spans

    def forward(self, x):
        length = x.shape[-2] // self.spans * self.spans
        return torch.cat([self.encoder(chunk) for chunk in torch.chunk(x[..., :length, :], self.spans, dim=-2)], dim=-2)


def music_spans(dit, spans=8):
    if not isinstance(dit.music_encoder, SpannedMusicEncoder):
        dit.music_encoder = SpannedMusicEncoder(dit.music_encoder, spans)
    return dit


if not torch.cuda.is_available():
    _multihead_init = torch.nn.MultiheadAttention.__init__

    def _multihead_on_host(self, *arguments, device=None, **keywords):
        _multihead_init(self, *arguments, device=None if device == "cuda" else device, **keywords)

    torch.nn.MultiheadAttention.__init__ = _multihead_on_host
