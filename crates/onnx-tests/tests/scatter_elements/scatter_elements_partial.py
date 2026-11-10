#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.23.2",
#   "onnxruntime",
#   "numpy",
# ]
# ///

# used to generate model: scatter_elements_partial.onnx
#
# ScatterElements where the indices are smaller than the data on non-axis dimensions,
# which ONNX allows, with every reduction. 2D: indices [2, 3] into data [3, 4] along
# axis 0, so the last column passes through and a stride taken from the data shape
# instead of the index shape lands in the wrong place. 3D: indices [1, 2, 3] into data
# [2, 3, 4] along axis 1, smaller on both non-axis dimensions. Targets are unique per
# position (burn leaves duplicates undefined for Assign) and one index is negative.
# Expected values from onnx.reference.ReferenceEvaluator, checked against ONNX Runtime.

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

REDUCTIONS = ["none", "add", "mul", "max", "min"]


def main():
    inputs = [
        helper.make_tensor_value_info("data", TensorProto.FLOAT, [3, 4]),
        helper.make_tensor_value_info("indices", TensorProto.INT64, [2, 3]),
        helper.make_tensor_value_info("updates", TensorProto.FLOAT, [2, 3]),
        helper.make_tensor_value_info("data3", TensorProto.FLOAT, [2, 3, 4]),
        helper.make_tensor_value_info("indices3", TensorProto.INT64, [1, 2, 3]),
        helper.make_tensor_value_info("updates3", TensorProto.FLOAT, [1, 2, 3]),
    ]
    nodes, outputs = [], []
    for reduction in REDUCTIONS:
        name = f"y_{reduction}"
        nodes.append(
            helper.make_node(
                "ScatterElements",
                ["data", "indices", "updates"],
                [name],
                axis=0,
                reduction=reduction,
            )
        )
        outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [3, 4]))
    for reduction in ("none", "max"):
        name = f"y3_{reduction}"
        nodes.append(
            helper.make_node(
                "ScatterElements",
                ["data3", "indices3", "updates3"],
                [name],
                axis=1,
                reduction=reduction,
            )
        )
        outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [2, 3, 4]))
    graph = helper.make_graph(nodes, "scatter_elements_partial_graph", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "scatter_elements_partial.onnx")

    feed = {
        "data": np.arange(1, 13, dtype=np.float32).reshape(3, 4),
        "indices": np.array([[-1, 0, 1], [0, 1, 2]], dtype=np.int64),
        "updates": np.array([[20.0, 0.5, 30.0], [-4.0, 40.0, 2.0]], dtype=np.float32),
        "data3": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        "indices3": np.array([[[0, 2, 1], [2, 0, 0]]], dtype=np.int64),
        "updates3": np.array([[[100.0, -1.0, 50.0], [7.0, 60.0, -3.0]]], dtype=np.float32),
    }
    names = [o.name for o in outputs]
    reference = dict(zip(names, ReferenceEvaluator(model).run(None, feed)))
    runtime = dict(zip(names, ort.InferenceSession(model.SerializeToString()).run(None, feed)))
    for name in names:
        assert np.array_equal(reference[name], runtime[name]), name
        print(name, reference[name].tolist())


if __name__ == "__main__":
    main()
