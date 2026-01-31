#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: one_hot.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node(
        "Constant", [], ["/Constant_output_0"],
        value=numpy_helper.from_array(np.array([3], dtype=np.int64).reshape([]), name="value"))
    node1 = helper.make_node(
        "Constant", [], ["/Constant_1_output_0"],
        value=numpy_helper.from_array(np.array([0, 1], dtype=np.int64).reshape([2]), name="value"))
    node2 = helper.make_node(
        "OneHot", ["input", "/Constant_output_0", "/Constant_1_output_0"], ["one_hot_output"],
        axis=-1)

    inp_input = helper.make_tensor_value_info("input", TensorProto.INT64, [None])

    out_one_hot_output = helper.make_tensor_value_info("one_hot_output", TensorProto.INT64, [None, 3])

    graph = helper.make_graph(
        [node0, node1, node2],
        "main_graph",
        [inp_input],
        [out_one_hot_output],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)])

    onnx.save(model, "one_hot.onnx")
    print(f"Finished exporting model to one_hot.onnx")


if __name__ == "__main__":
    main()
