#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.23.0",
#   "numpy",
#   "onnxruntime",
# ]
# ///

# used to generate model: pool_ceil_asymmetric.onnx
#
# MaxPool, LpPool and AveragePool (no Indices output) with ceil_mode=1 and
# asymmetric pads, where ONNX keeps a partial last window on one axis and drops
# the last window on another because it would start inside the end padding.
# burn's pool ops now take (begin, end) pads and drop that window themselves, so
# the generated code no longer trims the output.
#
# - 1D (x: [1, 2, 7]): kernel 4, stride 4, pads 0/3 -> 2 (not 3)
# - 2D (xx: [1, 2, 7, 7]): kernel 3x4, stride 3x4, pads 0,0/2,3
#   -> height 3 (last window partial), width 2 (not 3)
#
# AveragePool runs with count_include_pad 0 and 1 (asymmetric pads with
# count_include_pad=0 used to be a no-op in burn, tracel-ai/burn#5450). No window
# reaches past the end padding, so count_include_pad=1 divides by the full kernel.
#
# Expected outputs come from onnx.reference, checked against onnxruntime.

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    f = TensorProto.FLOAT
    p1 = dict(kernel_shape=[4], strides=[4], pads=[0, 3], ceil_mode=1)
    p2 = dict(kernel_shape=[3, 4], strides=[3, 4], pads=[0, 0, 2, 3], ceil_mode=1)
    nodes = [
        helper.make_node("MaxPool", ["x"], ["max1d"], **p1),
        helper.make_node("MaxPool", ["xx"], ["max2d"], **p2),
        helper.make_node("LpPool", ["x"], ["lp1d"], p=2, **p1),
        helper.make_node("LpPool", ["xx"], ["lp2d"], p=2, **p2),
        helper.make_node("AveragePool", ["x"], ["avg1d"], count_include_pad=0, **p1),
        helper.make_node("AveragePool", ["xx"], ["avg2d"], count_include_pad=0, **p2),
        helper.make_node("AveragePool", ["xx"], ["avg2d_pad"], count_include_pad=1, **p2),
    ]
    shape1, shape2 = [1, 2, 2], [1, 2, 3, 2]
    outputs = [
        helper.make_tensor_value_info(name, f, shape)
        for name, shape in [
            ("max1d", shape1),
            ("max2d", shape2),
            ("lp1d", shape1),
            ("lp2d", shape2),
            ("avg1d", shape1),
            ("avg2d", shape2),
            ("avg2d_pad", shape2),
        ]
    ]
    inputs = [
        helper.make_tensor_value_info("x", f, [1, 2, 7]),
        helper.make_tensor_value_info("xx", f, [1, 2, 7, 7]),
    ]
    graph = helper.make_graph(nodes, "pool_ceil_asymmetric", inputs, outputs)
    # LpPool's ceil_mode needs opset 18.
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "pool_ceil_asymmetric.onnx")

    np.random.seed(42)
    x = np.random.permutation(14).astype(np.float32).reshape(1, 2, 7)
    xx = np.random.permutation(98).astype(np.float32).reshape(1, 2, 7, 7)
    feeds = {"x": x, "xx": xx}
    reference = ReferenceEvaluator(model).run(None, feeds)
    runtime = ort.InferenceSession(model.SerializeToString()).run(None, feeds)
    print(f"x: {x.astype(int).tolist()}")
    print(f"xx: {xx.astype(int).tolist()}")
    for out, ref, rt in zip(outputs, reference, runtime):
        np.testing.assert_allclose(ref, rt, rtol=1e-5)
        print(f"{out.name}: {np.round(ref, 4).tolist()}")


if __name__ == "__main__":
    main()
