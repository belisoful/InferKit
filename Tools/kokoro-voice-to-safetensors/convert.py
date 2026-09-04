#!/usr/bin/env python3
"""Convert a Kokoro voicepack `.pt` (a bare `[styles, 1, 256]` tensor) to a safetensors file with a
single `voice` tensor, which `NFKMLXKokoro.loadVoice` reads. A released voicepack is a bare tensor, not
a state_dict, so the native torch reader does not interpret it; this is the offline conversion.

    python3 convert.py voices/af_heart.pt voices/af_heart.safetensors

Needs only torch + safetensors (no `kokoro`/misaki/spaCy).
"""
import sys
import torch
from safetensors.torch import save_file


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    tensor = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise SystemExit(f"{sys.argv[1]} is not a bare tensor voicepack")
    save_file({"voice": tensor.contiguous()}, sys.argv[2])
    print(f"wrote {sys.argv[2]}: {tuple(tensor.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
