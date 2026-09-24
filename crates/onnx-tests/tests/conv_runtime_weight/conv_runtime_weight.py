#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: conv_runtime_weight.onnx
#
# Conv and ConvTranspose whose weights are graph inputs rather than
# initializers (as after a DequantizeLinear): Conv1d without bias, Conv2d with
# a constant bias and asymmetric pads, and ConvTranspose2d with a stride.

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def main():
    inputs = [
        helper.make_tensor_value_info("x1", TensorProto.FLOAT, [1, 2, 5]),
        helper.make_tensor_value_info("w1", TensorProto.FLOAT, [3, 2, 3]),
        helper.make_tensor_value_info("x2", TensorProto.FLOAT, [1, 2, 4, 4]),
        helper.make_tensor_value_info("w2", TensorProto.FLOAT, [3, 2, 3, 3]),
        helper.make_tensor_value_info("xt", TensorProto.FLOAT, [1, 2, 2, 2]),
        helper.make_tensor_value_info("wt", TensorProto.FLOAT, [2, 3, 2, 2]),
    ]
    outputs = [
        helper.make_tensor_value_info("y1", TensorProto.FLOAT, [1, 3, 3]),
        helper.make_tensor_value_info("y2", TensorProto.FLOAT, [1, 3, 2, 4]),
        helper.make_tensor_value_info("yt", TensorProto.FLOAT, [1, 3, 4, 4]),
    ]
    bias = numpy_helper.from_array(np.array([0.5, -1.0, 2.0], dtype=np.float32), "b2")
    nodes = [
        helper.make_node("Conv", ["x1", "w1"], ["y1"]),
        helper.make_node("Conv", ["x2", "w2", "b2"], ["y2"], pads=[0, 1, 1, 1], strides=[2, 1]),
        helper.make_node("ConvTranspose", ["xt", "wt"], ["yt"], strides=[2, 2]),
    ]
    graph = helper.make_graph(nodes, "conv_runtime_weight", inputs, outputs, initializer=[bias])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "conv_runtime_weight.onnx")

    def seq(shape, scale):
        return (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) * scale - 1.0)

    feeds = {
        "x1": seq([1, 2, 5], 0.1),
        "w1": seq([3, 2, 3], 0.2),
        "x2": seq([1, 2, 4, 4], 0.1),
        "w2": seq([3, 2, 3, 3], 0.05),
        "xt": seq([1, 2, 2, 2], 0.5),
        "wt": seq([2, 3, 2, 2], 0.1),
    }
    for name, value in zip(["y1", "y2", "yt"], ReferenceEvaluator(model).run(None, feeds)):
        print(f"{name} {list(value.shape)} sum={value.sum():.5f}: {np.round(value, 5).tolist()}")


if __name__ == "__main__":
    main()
