#!/usr/bin/env python3
"""Adds one property at a time to a plainly-placed transformer, to find what moves it off the ANE.

A GPT-2 converted by `Tools/inferkit-convert` places none of its operations on the Neural Engine,
while a plain attention block places all of them. Something between the two is responsible. Removing
properties from the failing model has no signal to follow — it is off the Neural Engine at every step
until the last one — so this goes the other way: start from a model that IS placed and add the
candidates one at a time.

The base is deliberately GPT-2 shaped (12 layers, 768 wide, 12 heads) so that "too small to bother
dispatching" is not a competing explanation, which is what spoiled an earlier attempt.

NOTE the file name: calling this `bisect.py` shadows the standard library module that torch imports,
which fails as a circular import before any of this runs.

    python3 add_one_property.py out/  # writes one .mlpackage per variant
    # then, per variant, on the Objective-C side which is the one that reports:
    #   INFERKIT_TEST_MLMODELC=out/<name>.mlmodelc swift test --filter NFKComputePlanTests
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn


class Block(nn.Module):
    """A pre-norm transformer block, the ordinary way round."""

    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.width = heads, width
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=True)
        self.project = nn.Linear(width, width, bias=True)
        self.up = nn.Linear(width, 4 * width, bias=True)
        self.down = nn.Linear(4 * width, width, bias=True)

    def attend(self, x, cache=None):
        b, s, c = x.shape
        head = c // self.heads
        q, k, v = self.qkv(self.norm1(x)).split(c, dim=-1)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=1)
            v = torch.cat([cache[1], v], dim=1)
        shape = lambda t: t.view(b, t.shape[1], self.heads, head).transpose(1, 2)
        scores = torch.softmax(shape(q) @ shape(k).transpose(-1, -2) / (head ** 0.5), dim=-1)
        merged = (scores @ shape(v)).transpose(1, 2).reshape(b, s, c)
        return self.project(merged), k, v

    def forward(self, x, cache=None):
        attended, k, v = self.attend(x, cache)
        x = x + attended
        return x + self.down(torch.nn.functional.gelu(self.up(self.norm2(x)))), k, v


class Variant(nn.Module):
    """The base stack, with each candidate property switchable on."""

    def __init__(self, width, heads, layers, vocabulary=None, cache_length=None):
        super().__init__()
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.embed = nn.Embedding(vocabulary, width) if vocabulary else None
        self.cache_length = cache_length
        if cache_length:
            # A Core ML state, the shape `Tools/inferkit-convert` produces.
            self.register_buffer("keys", torch.zeros(len(self.blocks), 1, cache_length, width))
            self.register_buffer("values", torch.zeros(len(self.blocks), 1, cache_length, width))

    def forward(self, x, position=None):
        if self.embed is not None:
            x = self.embed(x)
        for index, block in enumerate(self.blocks):
            if self.cache_length is None:
                x, _, _ = block(x)
                continue
            cached = (self.keys[index], self.values[index])
            x, k, v = block(x, cached)
            # Write back into the state, which is what read_state / slice_update come from.
            self.keys[index, :, 0:1, :] = k[:, -1:, :]
            self.values[index, :, 0:1, :] = v[:, -1:, :]
        return self.norm(x)


def convert(ct, model, example, inputs, output_name, states, path):
    with torch.no_grad():
        traced = torch.jit.trace(model, example, check_trace=False)
    converted = ct.convert(
        traced, inputs=inputs, outputs=[ct.TensorType(name=output_name, dtype=np.float16)],
        states=states or None,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram", skip_model_load=True)
    converted.save(path)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--sequence", type=int, default=64)
    parser.add_argument("--vocabulary", type=int, default=50257)
    args = parser.parse_args()

    import coremltools as ct
    os.makedirs(args.output, exist_ok=True)
    torch.manual_seed(0)
    w, s = args.width, args.sequence

    def out(name):
        return os.path.join(args.output, f"{name}.mlpackage")

    # The control: GPT-2 shaped, plain activations in, one function, no state.
    plain = Variant(w, args.heads, args.layers).eval()
    convert(ct, plain, (torch.randn(1, s, w),),
            [ct.TensorType(name="x", shape=(1, s, w), dtype=np.float16)], "y", None, out("a_plain"))

    # + a single-token sequence, which is the decode shape.
    convert(ct, plain, (torch.randn(1, 1, w),),
            [ct.TensorType(name="x", shape=(1, 1, w), dtype=np.float16)], "y", None, out("b_seq1"))

    # + an embedding gather, so token ids come in rather than activations.
    embedded = Variant(w, args.heads, args.layers, vocabulary=args.vocabulary).eval()
    convert(ct, embedded, (torch.zeros(1, s, dtype=torch.int32),),
            [ct.TensorType(name="ids", shape=(1, s), dtype=np.int32)], "y", None, out("c_embed"))

    # + a Core ML state holding the key-value cache.
    stateful = Variant(w, args.heads, args.layers, cache_length=args.sequence).eval()
    states = [ct.StateType(wrapped_type=ct.TensorType(shape=(args.layers, 1, args.sequence, w)),
                           name=name) for name in ("keys", "values")]
    convert(ct, stateful, (torch.randn(1, 1, w), torch.tensor([0])),
            [ct.TensorType(name="x", shape=(1, 1, w), dtype=np.float16),
             ct.TensorType(name="position", shape=(1,), dtype=np.int32)], "y", states,
            out("d_state"))

    # Everything together, which is the shape the real converter emits.
    whole = Variant(w, args.heads, args.layers, vocabulary=args.vocabulary,
                    cache_length=args.sequence).eval()
    convert(ct, whole, (torch.zeros(1, 1, dtype=torch.int32), torch.tensor([0])),
            [ct.TensorType(name="ids", shape=(1, 1), dtype=np.int32),
             ct.TensorType(name="position", shape=(1,), dtype=np.int32)], "y", states,
            out("e_whole"))


if __name__ == "__main__":
    main()
