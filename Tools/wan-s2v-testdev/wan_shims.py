"""Lets Wan-Video/Wan2.2's own code run on the CPU at float32, for the S2V oracle and test-dev builder.

- `decord` and `dashscope` are imported at module scope by readers and prompt extenders the S2V path never
  calls; each is a placeholder module whose attributes raise when used.
- `wan.modules.t5.T5EncoderModel` evaluates `torch.cuda.current_device()` as a default argument at import, so
  without CUDA it answers `"cpu"`; the oracle passes its device explicitly.
- `flash_attention` asserts a CUDA device and half precision. Every module that bound the name gets an exact
  float substitute: torch's attention at the query's dtype over all queries, keys past `k_lens[b]` masked, which
  is what `flash_attn_varlen_func` computes for one unpadded query block per batch entry. A `q_lens` shorter
  than the queries, a window, or dropout raise, since the substitute does not model them.

Import this after `IK_WAN22_SRC` is on `sys.path` and before calling the pipeline.
"""

import importlib.machinery
import sys
import types

import torch

PLACEHOLDER_MODULES = ("decord", "dashscope")


def _placeholder(name):
    module = types.ModuleType(name)

    def missing(attribute):
        if attribute.startswith("__"):
            raise AttributeError(attribute)

        def function(*_, **__):
            raise RuntimeError(f"{name}.{attribute} reached on the S2V oracle path")
        return function

    module.__getattr__ = missing
    # transformers 4.x probes `importlib.util.find_spec` for its optional readers.
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    return module


if not torch.cuda.is_available():
    torch.cuda.current_device = lambda: "cpu"

for _name in PLACEHOLDER_MODULES:
    if _name not in sys.modules:
        sys.modules[_name] = _placeholder(_name)


def float_flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None, q_scale=None,
                          causal=False, window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16,
                          version=None):
    if dropout_p != 0.0 or tuple(window_size) != (-1, -1):
        raise RuntimeError("the float attention substitute models no dropout or window")
    if q_lens is not None and any(int(n) != q.size(1) for n in q_lens):
        raise RuntimeError("the float attention substitute models no padded queries")
    if q_scale is not None:
        q = q * q_scale
    mask = None
    if k_lens is not None:
        positions = torch.arange(k.size(1))
        mask = (positions[None, :] < torch.as_tensor(k_lens).view(-1, 1))[:, None, None, :]
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.to(q.dtype).transpose(1, 2), v.to(q.dtype).transpose(1, 2), attn_mask=mask,
        is_causal=causal, scale=softmax_scale)
    return out.transpose(1, 2).contiguous()


def install():
    """Rebinds `flash_attention` in every loaded `wan` module that imported it."""
    import wan  # noqa: F401
    import wan.modules.s2v.model_s2v  # noqa: F401
    import wan.modules.s2v.motioner  # noqa: F401
    for name, module in list(sys.modules.items()):
        if name.startswith("wan") and getattr(module, "flash_attention", None) is not None:
            module.flash_attention = float_flash_attention
