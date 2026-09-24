#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: maxpool2d_indices.onnx
#
# MaxPool with the optional Indices output, in both storage orders, over a
# batch of two with two channels so the per-plane offsets are exercised.

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2, 4, 5])
    outputs = [
        helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2, 3, 3]),
        helper.make_tensor_value_info("i", TensorProto.INT64, [2, 2, 3, 3]),
        helper.make_tensor_value_info("yc", TensorProto.FLOAT, [2, 2, 3, 3]),
        helper.make_tensor_value_info("ic", TensorProto.INT64, [2, 2, 3, 3]),
    ]
    attrs = dict(kernel_shape=[2, 2], strides=[2, 2], pads=[1, 1, 1, 1])
    nodes = [
        helper.make_node("MaxPool", ["x"], ["y", "i"], **attrs),
        helper.make_node("MaxPool", ["x"], ["yc", "ic"], storage_order=1, **attrs),
    ]
    graph = helper.make_graph(nodes, "maxpool2d_indices", [x], outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "maxpool2d_indices.onnx")

    np.random.seed(42)
    data = np.random.permutation(80).astype(np.float32).reshape(2, 2, 4, 5)
    y, i, yc, ic = ReferenceEvaluator(model).run(None, {"x": data})
    print(f"x: {data.astype(int).tolist()}")
    print(f"y: {y.astype(int).tolist()}")
    print(f"i: {i.tolist()}")
    print(f"ic: {ic.tolist()}")


if __name__ == "__main__":
    main()
