#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: conv3d_asymmetric_padding.onnx
#
# Conv3d with pads that differ between the two ends of an axis, which burn-nn's
# symmetric `PaddingConfig3d` cannot express (issue #646):
#   y1: constant weight, no bias, pads [0, 1, 0, 1, 0, 1]
#   y2: constant weight and bias, a different (begin, end) per axis, strides [2, 1, 1]
#   y3: constant weight, SAME_UPPER over a mixed even/odd kernel, so only some axes
#       resolve to asymmetric pads
#   y4: weight fed at run time, asymmetric pads

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def seq(shape, scale):
    return np.arange(np.prod(shape), dtype=np.float32).reshape(shape) * scale - 1.0


def main():
    x_shape = [1, 2, 3, 3, 3]
    w1 = numpy_helper.from_array(seq([2, 2, 2, 2, 2], 0.05), "w1")
    w2 = numpy_helper.from_array(seq([2, 2, 3, 3, 3], 0.02), "w2")
    b2 = numpy_helper.from_array(np.array([0.5, -1.0], dtype=np.float32), "b2")
    w3 = numpy_helper.from_array(seq([2, 2, 2, 3, 2], 0.03), "w3")

    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
        helper.make_tensor_value_info("w4", TensorProto.FLOAT, [2, 2, 2, 2, 2]),
    ]
    outputs = [
        helper.make_tensor_value_info("y1", TensorProto.FLOAT, [1, 2, 3, 3, 3]),
        helper.make_tensor_value_info("y2", TensorProto.FLOAT, [1, 2, 1, 2, 4]),
        helper.make_tensor_value_info("y3", TensorProto.FLOAT, [1, 2, 3, 3, 3]),
        helper.make_tensor_value_info("y4", TensorProto.FLOAT, [1, 2, 3, 3, 4]),
    ]
    # pads are [front, top, left, back, bottom, right]
    nodes = [
        helper.make_node("Conv", ["x", "w1"], ["y1"], pads=[0, 1, 0, 1, 0, 1]),
        helper.make_node(
            "Conv", ["x", "w2", "b2"], ["y2"], pads=[1, 0, 2, 0, 1, 1], strides=[2, 1, 1]
        ),
        helper.make_node("Conv", ["x", "w3"], ["y3"], auto_pad="SAME_UPPER"),
        helper.make_node("Conv", ["x", "w4"], ["y4"], pads=[1, 0, 1, 0, 1, 1]),
    ]
    graph = helper.make_graph(
        nodes, "conv3d_asymmetric_padding", inputs, outputs, initializer=[w1, w2, b2, w3]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "conv3d_asymmetric_padding.onnx")

    feeds = {"x": seq(x_shape, 0.1), "w4": seq([2, 2, 2, 2, 2], 0.04)}
    names = ["y1", "y2", "y3", "y4"]
    for name, value in zip(names, ReferenceEvaluator(model).run(None, feeds)):
        print(f"{name} {list(value.shape)} sum={value.sum():.5f}: {np.round(value.astype(np.float64), 4).tolist()}")


if __name__ == "__main__":
    main()
