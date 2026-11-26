"""A substitute for the one CUDA-only module VideoX-Fun imports at module scope, so its float path runs on
the CPU.

`videox_fun/models/attention_kernel.py` imports `triton` and defines fused sparse-attention kernels that
only the sparse-attention models call. The substitute module carries the four names `attention_utils`
imports from it, and each raises when called, so a run that reached a kernel fails rather than computing
something else. `triton` itself is left absent, as torch and diffusers expect it to be on this machine.
"""

import sys
import types

KERNEL_MODULE = "videox_fun.models.attention_kernel"
KERNEL_NAMES = ("_sparse_linear_attention", "get_block_map", "mean_pool", "sparse_linear_attention_varlen")


def _unreachable(name):
    def function(*_, **__):
        raise RuntimeError(f"{KERNEL_MODULE}.{name} reached on the CPU float path")
    function.__name__ = name
    return function


if KERNEL_MODULE not in sys.modules:
    module = types.ModuleType(KERNEL_MODULE)
    for kernel in KERNEL_NAMES:
        setattr(module, kernel, _unreachable(kernel))
    sys.modules[KERNEL_MODULE] = module
