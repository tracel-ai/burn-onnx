#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: scatter_opset10.onnx
#
# The deprecated Scatter operator (opset 10), which ScatterElements replaced.

import numpy as np
import onnx
from onnx import TensorProto, helper


def main():
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [3, 3])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [2, 3])
    updates = helper.make_tensor_value_info("updates", TensorProto.FLOAT, [2, 3])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 3])

    node = helper.make_node(
        "Scatter",
        inputs=["data", "indices", "updates"],
        outputs=["output"],
        axis=0,
    )

    graph = helper.make_graph(
        [node], "scatter_opset10_graph", [data, indices, updates], [output]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "scatter_opset10.onnx")

    test_data = np.zeros((3, 3), dtype=np.float32)
    test_indices = np.array([[1, 0, 2], [0, 2, 1]], dtype=np.int64)
    test_updates = np.array([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], dtype=np.float32)

    # onnx.reference has no Scatter implementation. Scatter is ScatterElements
    # without a reduction, so the expected output matches scatter_elements.py for
    # the same inputs:
    #   [[2.0, 1.1, 0.0], [1.0, 0.0, 2.2], [0.0, 2.1, 1.2]]
    print("Test data: {}".format(test_data))
    print("Test indices: {}".format(test_indices))
    print("Test updates: {}".format(test_updates))

if __name__ == "__main__":
    main()
