#!/usr/bin/env python3
"""Builds two Core ML models that compute the same attention block in different layouts.

The Neural Engine guidance for transformers is specific: hold activations as 4-D `(B, C, 1, S)`,
express every linear as a 1x1 convolution, and split the heads inside the graph. The claim is that
this changes where Core ML places the operations. That is a measurable claim, and this makes it
measurable: the two models compute the same function from the same weights and differ only in layout.

Writes `naive.mlpackage` and `ane.mlpackage`, and with --report prints where Core ML plans to run each.

    python3 build_fixture.py out/ --report
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn


class NaiveBlock(nn.Module):
    """The ordinary shape: `(B, S, C)` activations and `nn.Linear` projections."""

    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.width = heads, width
        self.q, self.k, self.v = (nn.Linear(width, width, bias=False) for _ in range(3))
        self.out = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):                                   # [1, S, C]
        b, s, c = x.shape
        head = c // self.heads
        q = self.q(x).view(b, s, self.heads, head).transpose(1, 2)
        k = self.k(x).view(b, s, self.heads, head).transpose(1, 2)
        v = self.v(x).view(b, s, self.heads, head).transpose(1, 2)
        scores = torch.softmax(q @ k.transpose(-1, -2) / (head ** 0.5), dim=-1)
        merged = (scores @ v).transpose(1, 2).reshape(b, s, c)
        return self.norm(x + self.out(merged))


class ChannelLayerNorm(nn.Module):
    """`nn.LayerNorm(C)` over `(B, S, C)`, expressed over `(B, C, 1, S)`.

    NOT `nn.GroupNorm(1, C)`, which is the obvious-looking substitution and is a different function:
    a single group normalizes over the channels AND the positions together, where a layer norm
    normalizes the channels separately at each position. The two agree only when the sequence is one
    long. This is written out because the equivalence check in `main` caught exactly that mistake.
    """

    def __init__(self, source: nn.LayerNorm):
        super().__init__()
        self.eps = source.eps
        self.weight = nn.Parameter(source.weight.data.clone())
        self.bias = nn.Parameter(source.bias.data.clone())

    def forward(self, x):                                   # [1, C, 1, S]
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        normalized = (x - mean) * torch.rsqrt(variance + self.eps)
        shape = (1, -1, 1, 1)
        return normalized * self.weight.view(shape) + self.bias.view(shape)


class ANEBlock(nn.Module):
    """The Neural Engine shape: `(B, C, 1, S)` activations and 1x1 convolutions.

    The weights are the same numbers — a 1x1 convolution over `(B, C, 1, S)` computes exactly what a
    linear computes over `(B, S, C)`, with the weight reshaped. Splitting the heads is a chunk along
    the channel axis rather than a reshape-and-transpose, so nothing has to be permuted.
    """

    def __init__(self, source: NaiveBlock):
        super().__init__()
        width, self.heads = source.width, source.heads
        self.width = width

        def conv_from(linear):
            conv = nn.Conv2d(width, width, kernel_size=1, bias=False)
            conv.weight.data = linear.weight.data.view(width, width, 1, 1).clone()
            return conv

        self.q, self.k, self.v = (conv_from(m) for m in (source.q, source.k, source.v))
        self.out = conv_from(source.out)
        self.norm = ChannelLayerNorm(source.norm)

    def forward(self, x):                                   # [1, C, 1, S]
        head = self.width // self.heads
        scale = head ** -0.5
        q = torch.split(self.q(x), head, dim=1)
        k = torch.split(self.k(x), head, dim=1)
        v = torch.split(self.v(x), head, dim=1)
        attended = []
        for qi, ki, vi in zip(q, k, v):
            # Per head, all in 4-D: no reshape off the (B, C, 1, S) layout at any point.
            scores = torch.softmax(torch.einsum("bchq,bchk->bqhk", qi * scale, ki), dim=-1)
            attended.append(torch.einsum("bqhk,bchk->bchq", scores, vi))
        return self.norm(x + self.out(torch.cat(attended, dim=1)))


def describe(plan, structure, label):
    """Prints the per-device operation counts a compute plan reports."""
    program = structure.program
    if program is None:
        print(f"  {label}: not an ML program")
        return
    counts, off = {}, {}
    for function in program.functions.values():
        for operation in function.block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
            if usage is None:
                counts["unreported"] = counts.get("unreported", 0) + 1
                continue
            name = type(usage.preferred_compute_device).__name__
            name = name.replace("MLComputeDevice", "").replace("Ml", "") or name
            counts[name] = counts.get(name, 0) + 1
            if "Neural" not in name:
                off[operation.operator_name] = off.get(operation.operator_name, 0) + 1
    total = sum(counts.values())
    ordered = ", ".join(f"{count} {device}" for device, count in sorted(counts.items()))
    engine = next((c for d, c in counts.items() if "Neural" in d), 0)
    share = 100.0 * engine / total if total else 0.0
    print(f"  {label}: {total} operations — {ordered} ({share:.1f}% Neural Engine)")
    if off:
        listed = ", ".join(f"{name}×{count}" for name, count
                           in sorted(off.items(), key=lambda kv: -kv[1])[:8])
        print(f"      off the Neural Engine: {listed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", help="directory to write the two .mlpackage bundles into")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--sequence", type=int, default=64)
    parser.add_argument("--report", action="store_true", help="print where Core ML plans to run each")
    parser.add_argument("--stateful", action="store_true",
                        help="compare a stateless step against one carrying a Core ML state, instead "
                             "of comparing layouts")
    args = parser.parse_args()

    import coremltools as ct

    os.makedirs(args.output, exist_ok=True)
    torch.manual_seed(0)
    naive = NaiveBlock(args.width, args.heads).eval()
    ane = ANEBlock(naive).eval()

    naive_input = torch.randn(1, args.sequence, args.width)
    ane_input = naive_input.transpose(1, 2).unsqueeze(2).contiguous()

    # The two must compute the same function, or the placement comparison is between two models.
    with torch.no_grad():
        a = naive(naive_input)
        b = ane(ane_input).squeeze(2).transpose(1, 2)
    worst = (a - b).abs().max().item()
    print(f"layouts agree to {worst:.3e}")
    if worst > 2e-4:
        sys.exit(f"the two layouts do not compute the same function (worst {worst})")

    if args.stateful:
        written = build_state_comparison(ct, args)
    else:
        written = build_layout_comparison(ct, args, naive, ane, naive_input, ane_input)

    if args.report:
        from coremltools.models.compute_plan import MLComputePlan, MLModelStructure
        from coremltools.models.utils import compile_model
        print("\nwhere Core ML plans to run each (MLComputeUnits.ALL):")
        for label, path in written:
            compiled = compile_model(path)
            plan = MLComputePlan.load_from_path(compiled, compute_units=ct.ComputeUnit.ALL)
            describe(plan, MLModelStructure.load_from_path(compiled), label)


class StatelessStep(nn.Module):
    """One attention step reading a cache passed in as an ordinary input."""

    def __init__(self, width):
        super().__init__()
        self.project = nn.Linear(width, width, bias=False)

    def forward(self, x, cache):                            # [1, 1, C], [1, S, C]
        q = self.project(x)
        scores = torch.softmax(q @ cache.transpose(-1, -2), dim=-1)
        return scores @ cache


class StatefulStep(nn.Module):
    """The same step reading and writing a Core ML STATE instead.

    This is the shape `Tools/inferkit-convert` produces: the key-value cache lives in the model as
    state, so a step updates it in place rather than taking it as an argument.
    """

    def __init__(self, width, length):
        super().__init__()
        self.project = nn.Linear(width, width, bias=False)
        self.register_buffer("cache", torch.zeros(1, length, width))

    def forward(self, x, position):                         # [1, 1, C], [1]
        self.cache[:, 0:1, :] = x
        q = self.project(x)
        scores = torch.softmax(q @ self.cache.transpose(-1, -2), dim=-1)
        return scores @ self.cache


def build_state_comparison(ct, args):
    """The same step with and without a Core ML state, so the state is the only difference."""
    length = args.sequence
    stateless = StatelessStep(args.width).eval()
    stateful = StatefulStep(args.width, length).eval()
    stateful.project.weight.data = stateless.project.weight.data.clone()

    x = torch.randn(1, 1, args.width)
    cache = torch.randn(1, length, args.width)
    written = []

    with torch.no_grad():
        traced = torch.jit.trace(stateless, (x, cache))
    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=(1, 1, args.width), dtype=np.float16),
                ct.TensorType(name="cache", shape=(1, length, args.width), dtype=np.float16)],
        outputs=[ct.TensorType(name="y", dtype=np.float16)],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram", skip_model_load=True)
    path = os.path.join(args.output, "stateless.mlpackage")
    converted.save(path)
    written.append(("stateless", path))
    print(f"wrote {path}")

    with torch.no_grad():
        traced = torch.jit.trace(stateful, (x, torch.tensor([0])), check_trace=False)
    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=(1, 1, args.width), dtype=np.float16),
                ct.TensorType(name="position", shape=(1,), dtype=np.int32)],
        outputs=[ct.TensorType(name="y", dtype=np.float16)],
        states=[ct.StateType(wrapped_type=ct.TensorType(shape=(1, length, args.width)),
                             name="cache")],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram", skip_model_load=True)
    path = os.path.join(args.output, "stateful.mlpackage")
    converted.save(path)
    written.append(("stateful", path))
    print(f"wrote {path}")
    return written


def build_layout_comparison(ct, args, naive, ane, naive_input, ane_input):
    written = []
    for label, model, example, shape in (
            ("naive", naive, naive_input, (1, args.sequence, args.width)),
            ("ane", ane, ane_input, (1, args.width, 1, args.sequence))):
        with torch.no_grad():
            traced = torch.jit.trace(model, example)
        converted = ct.convert(
            traced,
            inputs=[ct.TensorType(name="x", shape=shape, dtype=np.float16)],
            outputs=[ct.TensorType(name="y", dtype=np.float16)],
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.ALL,
            convert_to="mlprogram",
            # Only the package on disk is wanted. Instantiating each model as it converts makes the
            # second conversion fail while the first is still loaded, and the compute plan is read
            # from the path regardless.
            skip_model_load=True)
        path = os.path.join(args.output, f"{label}.mlpackage")
        converted.save(path)
        written.append((label, path))
        print(f"wrote {path}")
    return written


if __name__ == "__main__":
    main()
