#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.23.2",
#   "onnxruntime",
#   "numpy",
# ]
# ///

# used to generate model: maxpool3d_int.onnx
#
# MaxPool 3D on integer inputs (int8/uint8 since opset 12):
# - y_i8: int8 [1, 2, 3, 4, 4] input, kernel 2x2x2, asymmetric pads (front 1, top 0,
#   left 0, back 0, bottom 1, right 1), with the optional Indices output omitted ("");
# - y_u8 / y_u8_neg_indices: uint8 [1, 2, 3, 4, 4] input, kernel 2x3x2, strides 1x2x2,
#   with the Indices output feeding a Neg node. The graph carries no value_info, so the
#   intermediate Indices type comes from type inference alone.
# Expected values from ONNX Runtime, checked against the brute-force `max_pool` below.
# (onnx.reference.ReferenceEvaluator crashes on padded stride-1 MaxPool, any dtype.)

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

SHAPE = [1, 2, 3, 4, 4]


def max_pool(x, kernel, strides, pads):
    """Values and row-major flattened indices, looping over every output position."""
    n, c, *spatial = x.shape
    begin, end = pads[:3], pads[3:]
    out = [(s + b + e - k) // st + 1 for s, b, e, k, st in zip(spatial, begin, end, kernel, strides)]
    values = np.empty([n, c, *out], x.dtype)
    indices = np.empty([n, c, *out], np.int64)
    for b, ch, *pos in np.ndindex(n, c, *out):
        best = None
        for offset in np.ndindex(*kernel):
            at = [p * st - pb + o for p, st, pb, o in zip(pos, strides, begin, offset)]
            if all(0 <= a < s for a, s in zip(at, spatial)) and (best is None or x[b, ch, *at] > best[0]):
                best = (x[b, ch, *at], np.ravel_multi_index((b, ch, *at), x.shape))
        values[b, ch, *pos], indices[b, ch, *pos] = best
    return values, indices


def main():
    inputs = [
        helper.make_tensor_value_info("x_i8", TensorProto.INT8, SHAPE),
        helper.make_tensor_value_info("x_u8", TensorProto.UINT8, SHAPE),
    ]
    nodes = [
        helper.make_node("MaxPool", ["x_i8"], ["y_i8", ""], kernel_shape=[2, 2, 2], pads=[1, 0, 0, 0, 1, 1]),
        helper.make_node("MaxPool", ["x_u8"], ["y_u8", "u8_indices"], kernel_shape=[2, 3, 2], strides=[1, 2, 2]),
        helper.make_node("Neg", ["u8_indices"], ["y_u8_neg_indices"]),
    ]
    names = ["y_i8", "y_u8", "y_u8_neg_indices"]
    types = [TensorProto.INT8, TensorProto.UINT8, TensorProto.INT64]
    # Rank only: symbolic dims, so no output shape is precomputed
    outputs = [helper.make_tensor_value_info(n, t, list("ncdhw")) for n, t in zip(names, types)]
    graph = helper.make_graph(nodes, "maxpool3d_int_graph", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 22)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    assert not model.graph.value_info
    onnx.save(model, "maxpool3d_int.onnx")

    rng = np.random.default_rng(42)
    feeds = {
        "x_i8": rng.integers(-128, 128, SHAPE).astype(np.int8),
        # Distinct values, so each window's max (and its index) is unique
        "x_u8": rng.choice(256, np.prod(SHAPE), replace=False).astype(np.uint8).reshape(SHAPE),
    }
    y_i8, _ = max_pool(feeds["x_i8"], [2, 2, 2], [1, 1, 1], [1, 0, 0, 0, 1, 1])
    y_u8, u8_indices = max_pool(feeds["x_u8"], [2, 3, 2], [1, 2, 2], [0] * 6)
    reference = [y_i8, y_u8, -u8_indices]
    runtime = ort.InferenceSession(model.SerializeToString()).run(None, feeds)
    for name, value in feeds.items():
        print(name, value.flatten().tolist())
    for name, ref, rt in zip(names, reference, runtime):
        assert ref.dtype == rt.dtype and ref.shape == rt.shape and (ref == rt).all(), name
        print(name, ref.dtype, ref.shape, ref.flatten().tolist())


if __name__ == "__main__":
    main()
