#!/usr/bin/env python3
"""Extract Basic Pitch's released ONNX graph into a safetensors file for InferKitMLX.

Spotify ships `nmp.onnx` (230 KB, Apache-2.0) inside the `basic-pitch` package. The graph holds the
whole pipeline, the constant-Q front end included: the CQT's complex kernels, the anti-aliasing filter
that downsamples one octave to the next, and the per-bin scale are frozen initializers, so nothing here
re-derives them from nnAudio's kernel construction. The Keras batch normalizations are already folded
into the convolutions they precede, which is why the module has none.

Tensors are written in PyTorch layout (`[out, in, kH, kW]`, `[out, in, k]`); `NFKMLXBasicPitch.loadWeights`
transposes them to MLX's at load.

The two CQT kernels are the real and the imaginary halves. The graph negates one of them, and the
magnitude squares both, so which is which does not reach any output; they are written as `a` and `b`.

Usage:
    python convert.py nmp.onnx basic-pitch.safetensors
    python convert.py --list-keys nmp.onnx

Requires: onnx, numpy, safetensors.
"""

import argparse
import sys

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file

# The weight of each convolution, keyed by the name this port gives it, found by the shape that is
# unique to it in the released graph.
WEIGHTS_BY_SHAPE = {
    (36, 1, 1, 256): None,          # the two CQT kernels, resolved apart below
    (1, 1, 1, 256): "cqt.lowpass",
    (8, 8, 3, 39): "contour_conv.weight",
    (1, 8, 5, 5): "contour_out.weight",
    (32, 1, 7, 7): "note_conv.weight",
    (1, 32, 7, 3): "note_out.weight",
    (32, 8, 5, 5): "onset_conv.weight",
    (1, 33, 3, 3): "onset_out.weight",
}

# A convolution's bias, keyed by the primary name the export gives it.
BIASES_BY_NAME = {
    "model_1/cq_t2010v2_1/conv1d_25": "cqt.kernel_bias",
    "model_1/conv2d_5/Conv2D": "cqt.lowpass_bias",
    "model_1/re_lu_1/Relu": "contour_conv.bias",
    "model_1/contours-reduced/BiasAdd/ReadVariableOp": "contour_out.bias",
    "model_1/conv2d_2/BiasAdd/ReadVariableOp": "note_conv.bias",
    "model_1/conv2d_3/BiasAdd/ReadVariableOp": "note_out.bias",
    "model_1/re_lu_3/Relu": "onset_conv.bias",
    "model_1/conv2d_5/BiasAdd/ReadVariableOp": "onset_out.bias",
}


def primary(name):
    """The first of the names an ONNX export fuses into one initializer with semicolons."""
    return name.split(";")[0]


def normalization(graph, initializers):
    """The folded batch normalization after the log: the scale a Mul applies and the bias an Add does.

    Both are one-element tensors under the same fused name, so they are told apart by which node reads
    them rather than by their names.
    """
    producers = {}
    for node in graph.node:
        for output in node.output:
            producers[output] = node

    for node in graph.node:
        if node.op_type != "Mul":
            continue
        scale = [name for name in node.input if name in initializers]
        if len(scale) != 1 or initializers[scale[0]].size != 1:
            continue
        for candidate in graph.node:
            if candidate.op_type != "Add" or node.output[0] not in candidate.input:
                continue
            bias = [name for name in candidate.input if name in initializers]
            if len(bias) == 1 and initializers[bias[0]].size == 1:
                return initializers[scale[0]], initializers[bias[0]]
    raise SystemExit("the graph has no folded normalization (a Mul by a scalar feeding an Add of one)")


def cqt_scale(initializers):
    """The per-bin scale the CQT applies before taking the magnitude, `[bins, 1, 1]` in the graph."""
    for name, value in initializers.items():
        if value.ndim == 3 and value.shape[1:] == (1, 1) and value.shape[0] > 100:
            return value.reshape(-1)
    raise SystemExit("the graph has no per-bin CQT scale")


def convert(path):
    graph = onnx.load(path).graph
    initializers = {t.name: numpy_helper.to_array(t) for t in graph.initializer}

    tensors = {}
    kernels = []
    for name, value in initializers.items():
        shape = tuple(value.shape)
        if shape == (36, 1, 1, 256):
            kernels.append((primary(name), value))
            continue
        key = WEIGHTS_BY_SHAPE.get(shape)
        if key is not None:
            tensors[key] = value.reshape(shape[0], shape[1], -1) if shape[2] == 1 and shape[3] == 256 else value
        bias_key = BIASES_BY_NAME.get(primary(name))
        if bias_key is not None:
            tensors[bias_key] = value.reshape(-1)

    if len(kernels) != 2:
        raise SystemExit(f"expected two CQT kernels, found {len(kernels)}")
    kernels.sort(key=lambda pair: pair[0])
    tensors["cqt.kernel_a"] = kernels[0][1].reshape(36, 1, 256)
    tensors["cqt.kernel_b"] = kernels[1][1].reshape(36, 1, 256)
    tensors["cqt.scale"] = cqt_scale(initializers)

    scale, bias = normalization(graph, initializers)
    tensors["norm_scale"] = scale.reshape(-1)
    tensors["norm_bias"] = bias.reshape(-1)

    expected = set(WEIGHTS_BY_SHAPE.values()) | set(BIASES_BY_NAME.values())
    expected.discard(None)
    expected |= {"cqt.kernel_a", "cqt.kernel_b", "cqt.scale", "norm_scale", "norm_bias"}
    missing = expected - set(tensors)
    if missing:
        raise SystemExit(f"the graph is missing {sorted(missing)}")

    return {key: np.ascontiguousarray(value, dtype=np.float32) for key, value in sorted(tensors.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="the released nmp.onnx")
    parser.add_argument("output", nargs="?", help="path to write the safetensors file")
    parser.add_argument("--list-keys", action="store_true", help="print the extracted keys and exit")
    args = parser.parse_args()

    tensors = convert(args.input)
    if args.list_keys:
        for key, value in tensors.items():
            print(f"{key:26s} {list(value.shape)}")
        return 0
    if args.output is None:
        parser.error("an output path is required unless --list-keys is given")

    save_file(tensors, args.output)
    total = sum(value.size for value in tensors.values())
    print(f"wrote {args.output}: {len(tensors)} tensors, {total} parameters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
